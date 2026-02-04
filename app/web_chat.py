try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# =========================================================
# 📚 2. CÁC THƯ VIỆN CẦN THIẾT
# =========================================================
import streamlit as st
import streamlit.components.v1 as components 
import os
import sys
import zipfile
import time

# --- CẤU HÌNH ĐƯỜNG DẪN ĐỂ IMPORT FILE ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from src import config, database 

load_dotenv()

# =========================================================
# ⚙️ 3. CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="Health Chatbot", 
    page_icon="🏥", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)

# --- CẬP NHẬT: ĐỒNG BỘ KEY TỪ SECRETS VÀO HỆ THỐNG ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    config.GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
# ---------------------------------------------------

# =========================================================
# BẢO VỆ GIAO DIỆN KHÔNG ẢNH HƯỞNG DO Google Translate CỦA WEB ẢNH HƯỞNG
# =========================================================
components.html("""
<script>
    function addProtection(element) {
        if (element && !element.classList.contains('notranslate')) {
            element.classList.add('notranslate');
            element.setAttribute('translate', 'no');
        }
    }
    const observer = new MutationObserver((mutations) => {
        // Bảo vệ Sidebar
        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        addProtection(sidebar);
        
        // Bảo vệ các nút bấm, ô nhập liệu
        const inputs = window.parent.document.querySelectorAll('input, textarea, button, select');
        inputs.forEach(addProtection);
        
        // CẬP NHẬT MỚI: Bảo vệ bong bóng chat và nội dung chat
        const chatMessages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
        chatMessages.forEach(addProtection);
        
        const chatContent = window.parent.document.querySelectorAll('.stMarkdown');
        chatContent.forEach(addProtection);
    });
    
    const config = { childList: true, subtree: true, attributes: false };
    observer.observe(window.parent.document.body, config);
</script>
""", height=0)

# =========================================================
# 🎨 5. CSS GIAO DIỆN
# =========================================================
st.markdown("""
<style>
    /* 1. RESET FONT */
    html, body, [class*="css"] {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 16px; 
    }

    /* 2. SIDEBAR */
    section[data-testid="stSidebar"] {
        padding-top: 1rem; 
        border-right: 1px solid #e0e0e0;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-left: 0.5rem; 
        padding-right: 0.5rem;
    }

    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        font-size: 18px !important;
    }

    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] li, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio div {
        font-size: 14px !important; 
    }
    
    section[data-testid="stSidebar"] button {
        font-size: 13px !important;
        padding: 0.25rem 0.5rem;
    }

    div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
    }
    
    /* 3. KHU VỰC CHAT CHÍNH */
    .stChatMessage p {
        font-size: 17px !important;
        line-height: 1.6;
    }

    [data-testid="InputInstructions"] {
        display: none !important;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 900px; 
    }

    #MainMenu {visibility: hidden;} 
    footer {visibility: hidden;}
    .stException { display: none !important; }
    div:has(> .stException) { display: none !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🧠 6. KHỞI TẠO LOGIC
# =========================================================
@st.cache_resource
def load_resources():
    db_path = os.path.join(config.CHROMA_DB_DIR, "chroma.sqlite3")
    
    zip_path = db_path + ".zip"
    if not os.path.exists(zip_path):
        zip_path_double = db_path + ".zip.zip"
        if os.path.exists(zip_path_double):
            zip_path = zip_path_double

    if not os.path.exists(db_path) and os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(config.CHROMA_DB_DIR)
        except Exception as e:
            st.error(f"Lỗi giải nén: {e}")

    # --- TẢI DATABASE & AI ---
    vector_db = None
    if os.path.exists(config.CHROMA_DB_DIR):
        try:
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
            vector_db = Chroma(persist_directory=config.CHROMA_DB_DIR, embedding_function=embedding_model)
        except Exception as e:
            pass
    
    llm = None
    if config.GOOGLE_API_KEY:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3, google_api_key=config.GOOGLE_API_KEY)
        
    return vector_db, llm

vector_db, llm = load_resources()




# --- HÀM XỬ LÝ TRẢ LỜI ---
def get_bot_response(user_query, history):
    # 1. Kiểm tra kết nối
    if not vector_db or not llm: return "⚠️ Hệ thống đang bảo trì (Chưa kết nối Database/AI)."
    
    try:
        # 2. RAG - TÌM KIẾM CÓ CHỌN LỌC (LỌC ĐỘ CHÍNH XÁC)
        # ---------------------------------------------------------
        # Thay vì chỉ tìm kiếm, ta lấy thêm cả điểm số (Score) để biết độ giống
        # ChromaDB Score: 0 là giống hệt. Càng lớn là càng khác.
        results = vector_db.similarity_search_with_score(user_query, k=4)
        
        docs = []
        # THRESHOLD (NGƯỠNG LỌC): 
        # Chỉ lấy tài liệu có điểm < 1.2 (Con số này đảm bảo chỉ lấy tài liệu thực sự liên quan)
        for doc, score in results:
            if score < 1.2: 
                docs.append(doc)
                
        sources = []
        context = ""
        citation_text = ""
        
        # Chỉ khi sau khi lọc mà vẫn còn tài liệu thì mới tạo list nguồn
        if docs:
            for doc in docs:
                src = doc.metadata.get("source", "Tài liệu y khoa")
                if src not in sources: sources.append(src)
            context = "\n".join([d.page_content for d in docs])
            citation_text = "\n\n---\n**📚 Tài liệu tham khảo:**\n" + "\n".join([f"- {s}" for s in sources])
        else:
            # Nếu lọc xong không còn gì (VD: Hỏi thất tình, hỏi xã giao) -> Dùng kiến thức chung
            context = "Không có dữ liệu cụ thể trong kho, dùng kiến thức y khoa tổng quát."
            citation_text = "" # Chuỗi này rỗng để tí nữa bộ lọc ở bước 5 bắt được
        
        # 3. PROMPT ENGINEERING (GIỮ NGUYÊN CẤU TRÚC ĐẦY ĐỦ NHƯ BẠN YÊU CẦU)
        # ---------------------------------------------------------
        prompt = f"""
        VAI TRÒ: Bạn là "Người Bạn Bác Sĩ" (Health Chatbot) - Giỏi chuyên môn nhưng nói chuyện ân cần như bạn thân (Xưng hô: Mình - Bạn/Cậu).

        DỮ LIỆU ĐẦU VÀO:
        - Lịch sử chat: {history}
        - Kiến thức Y khoa (RAG): {context}
        - User vừa nói: "{user_query}"

        NHIỆM VỤ: HÃY XÁC ĐỊNH Ý ĐỊNH CỦA USER ĐỂ CHỌN 1 TRONG 2 CHẾ ĐỘ TRẢ LỜI:

        ════════════════════════════════════════════════════
        MODE 1: XÃ GIAO / CẢM XÚC (Khi User chào, cảm ơn, khen, chê, tạm biệt)
        ════════════════════════════════════════════════════
        - Nếu User nói "Hello", "Hi", "Chào": Trả lời vui vẻ, ngắn gọn, hỏi thăm sức khỏe.
        - Nếu User nói "Cảm ơn", "Hay quá", "Ok": Đáp lại lịch sự (VD: "Không có chi, chúc bạn mau khỏe nhé!").
        - QUY TẮC CỨNG: KHÔNG được dùng cấu trúc khám bệnh (###) trong trường hợp này.

        ════════════════════════════════════════════════════
        MODE 2: TƯ VẤN Y KHOA (Khi User kể bệnh, hỏi thuốc, lo lắng)
        ════════════════════════════════════════════════════
        Áp dụng đúng cấu trúc chuẩn đoán 5 phần dưới đây.
        
        *LƯU Ý THÔNG MINH:* - Hãy kiểm tra Lịch sử chat. Nếu User ĐÃ cung cấp đủ thông tin (Vị trí đau, thời gian, mức độ, tiền sử...), thì ở phần "Hỏi thêm" hãy nói: "Dựa trên thông tin cậu kể, mình đã nắm khá rõ tình hình." và không cần đặt câu hỏi nữa.
        - Nếu thiếu thông tin, hãy đặt 2-3 câu hỏi quan trọng nhất.

        BẮT BUỘC SỬ DỤNG ĐỊNH DẠNG MARKDOWN SAU CHO MODE 2:

        (Lời dẫn dắt cảm thông, ân cần)

        ### 🔍 Phân tích sơ bộ:
        (Phân tích kỹ các triệu chứng user kể, kết hợp với dữ liệu RAG)

        ### 🩺 Để hiểu rõ hơn, cậu cho mình hỏi thêm nhé:
        (Nếu thiếu tin: Đặt câu hỏi / Nếu đủ tin: Ghi "Thông tin đã khá đầy đủ để chẩn đoán.")

        ### 💡 Có thể cậu đang gặp vấn đề về:
        (Đưa ra các giả thuyết bệnh lý, xếp theo thứ tự khả năng cao nhất)

        ### 👉 Chuyên khoa cậu nên ghé khám:
        **[TÊN CHUYÊN KHOA]** - (Giải thích ngắn gọn tại sao)

        ### 📝 Lưu ý cho cậu:
        - **Chăm sóc:** (Lời khuyên ăn uống, nghỉ ngơi)
        - **Thuốc men:** (Gợi ý nhóm thuốc không kê đơn nếu an toàn)
        - **Cảnh báo:** (Dấu hiệu nguy hiểm cần đi cấp cứu ngay)

        (Lời chúc sức khỏe cuối cùng)
        """
        
        # 4. GỌI AI
        response = llm.invoke(prompt)
        
        # Xử lý kết quả trả về
        content = response.content
        final_ans = ""
        if isinstance(content, list):
            for item in content:
                if isinstance(item, str): final_ans += item
                elif isinstance(item, dict) and 'text' in item: final_ans += item['text']
        else:
            final_ans = str(content)
            
        # 5. BỘ LỌC HIỂN THỊ NGUỒN (LOGIC CHUẨN XÁC)
        # ------------------------------------------------------------------
        # Điều kiện hiển thị nguồn:
        # 1. Has Source: Phải tìm thấy nguồn thực tế trong kho (Sau khi đã lọc score ở bước 2, len(sources) > 0)
        # 2. Is Diagnosis: Phải là bài tư vấn bệnh (Có chứa dấu hiệu chẩn đoán "###")
        
        has_sources = len(sources) > 0
        is_diagnosis = "###" in final_ans
        
        if has_sources and is_diagnosis:
            return final_ans + citation_text
        else:
            return final_ans # Xã giao hoặc không tìm thấy nguồn (do bị lọc hết) -> Không hiện

    except Exception as e: return f"❌ Lỗi xử lý: {e}"

# (Session state, Sidebar, và Main Content)

# --- SESSION STATE ---
if "user_info" not in st.session_state: st.session_state.user_info = None 
if "current_conv_id" not in st.session_state: st.session_state.current_conv_id = None
if "guest_messages" not in st.session_state: st.session_state.guest_messages = [] 
if "delete_confirm_id" not in st.session_state: st.session_state.delete_confirm_id = None

# ==========================================
# 🛑 7. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=70)
    
    if st.session_state.user_info:
        user = st.session_state.user_info
        st.success(f"👋 Hi, {user['full_name']}")
        
        if st.button("➕ Tạo đoạn chat mới", use_container_width=True):
            st.session_state.current_conv_id = None
            st.rerun()
            
        st.divider()
        st.write("📂 **Lịch sử:**")
        
        convs = database.get_user_conversations(user['id'])
        if not convs: st.caption("Trống.")
        
        for conv in convs:
            col1, col2 = st.columns([0.85, 0.15]) 
            
            with col1:
                if st.session_state.delete_confirm_id == conv['id']:
                    st.caption("⚠️ Xóa nhé?")
                    conf_col1, conf_col2 = st.columns(2)
                    with conf_col1:
                        if st.button("Có", key=f"yes_{conv['id']}", type="primary", use_container_width=True):
                            database.delete_conversation(conv['id'])
                            st.session_state.delete_confirm_id = None
                            if st.session_state.current_conv_id == conv['id']:
                                st.session_state.current_conv_id = None
                            st.rerun()
                    with conf_col2:
                        if st.button("Hủy", key=f"no_{conv['id']}", use_container_width=True):
                            st.session_state.delete_confirm_id = None
                            st.rerun()
                else:
                    icon = "📌 " if conv.get('is_pinned', 0) else "💬 "
                    label = f"{icon}{conv['title'][:18]}..."
                    if st.button(label, key=f"btn_{conv['id']}", use_container_width=True):
                        st.session_state.current_conv_id = conv['id']
                        st.rerun()
            
            with col2:
                if st.session_state.delete_confirm_id != conv['id']:
                    with st.popover("⋮", use_container_width=True):
                        is_pinned = conv.get('is_pinned', 0)
                        pin_label = "Bỏ ghim" if is_pinned else "Ghim"
                        
                        if st.button(pin_label, key=f"pin_{conv['id']}", use_container_width=True):
                            database.toggle_pin_conversation(conv['id'], is_pinned)
                            st.rerun()
                            
                        if st.button("Xóa", key=f"trig_del_{conv['id']}", use_container_width=True):
                            st.session_state.delete_confirm_id = conv['id']
                            st.rerun()

        st.divider()
        if st.button("🚪 Đăng Xuất"):
            st.session_state.user_info = None
            st.session_state.current_conv_id = None
            st.session_state.guest_messages = []
            st.rerun()
    else:
        st.info("Chế độ: **Khách**")
        st.caption("Chat với Bác sĩ AI.")
        st.divider()
        
        mode = st.radio("Tài khoản:", ["Đăng Nhập", "Đăng Ký"], horizontal=True)
        
        if mode == "Đăng Nhập":
            with st.form("login_form"):
                u = st.text_input("User")
                p = st.text_input("Pass", type="password")
                if st.form_submit_button("Vào chat", use_container_width=True):
                    user = database.login_user(u, p)
                    if user:
                        st.session_state.user_info = user
                        st.rerun()
                    else: st.error("Sai rồi!")
        else: 
            with st.form("reg_form"):
                nu = st.text_input("User Mới")
                np = st.text_input("Pass Mới", type="password")
                nn = st.text_input("Họ Tên")
                if st.form_submit_button("Đăng ký", use_container_width=True):
                    ok, msg = database.register_user(nu, np, nn)
                    if ok: st.success("Xong!")
                    else: st.error(msg)

# =========================================================
# 8. MAIN CONTENT (GIAO DIỆN CHAT)
# =========================================================
st.title("🏥 Health Chatbot")

# Cảnh báo y tế
st.warning("⚠️ **Lưu ý:** AI chỉ hỗ trợ tư vấn sơ bộ, không thay thế chẩn đoán của bác sĩ chuyên khoa. Trong trường hợp khẩn cấp, hãy đến cơ sở y tế gần nhất.", icon="⚠️")

AVATAR_AI = "https://cdn-icons-png.flaticon.com/512/3774/3774299.png"
AVATAR_USER = "https://cdn-icons-png.flaticon.com/512/1144/1144760.png"

messages = []
if st.session_state.user_info:
    if st.session_state.current_conv_id:
        messages = database.load_messages(st.session_state.current_conv_id)
else:
    messages = st.session_state.guest_messages

# Màn hình chào
if not messages:
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 20px;'>
        <h3>👋 Xin chào! Mình có thể giúp gì cho sức khỏe của bạn?</h3>
        <p><i>Hãy kể cho mình nghe về triệu chứng, hoặc hỏi về thuốc men...</i></p>
    </div>
    """, unsafe_allow_html=True)

# Hiển thị lịch sử chat
for msg in messages:
    role = msg["role"]
    avatar = AVATAR_USER if role == "user" else AVATAR_AI
    with st.chat_message(role, avatar=avatar):
        st.write(msg["content"])

# --- XỬ LÝ NHẬP LIỆU (QUAN TRỌNG: ĐÃ SỬA LỖI RERUN) ---
if prompt := st.chat_input("Gõ triệu chứng vào đây..."):
    # 1. Hiện câu hỏi User ngay lập tức
    with st.chat_message("user", avatar=AVATAR_USER):
        st.write(prompt)
    
    # 2. Bot suy nghĩ và trả lời
    full_response = ""
    with st.chat_message("assistant", avatar=AVATAR_AI):
        with st.spinner("👨‍⚕️ Bác sĩ đang suy nghĩ..."):
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages[-4:]])
            full_response = get_bot_response(prompt, history_str)
            st.write(full_response)
    
    # 3. Lưu vào Database (Xử lý ngầm)
    if st.session_state.user_info:
        uid = st.session_state.user_info['id']
        
        # TRƯỜNG HỢP: CUỘC TRÒ CHUYỆN MỚI
        if st.session_state.current_conv_id is None:
            title = (prompt[:30] + '..') if len(prompt) > 30 else prompt
            new_id = database.create_conversation(uid, title)
            st.session_state.current_conv_id = new_id
            
            # Lưu tin nhắn đầu tiên
            database.save_message(new_id, "user", prompt)
            database.save_message(new_id, "assistant", full_response)
            
            # CHỈ RERUN KHI TẠO MỚI (Để cập nhật tên bên Sidebar)
            st.rerun()
            
        # TRƯỜNG HỢP: ĐANG CHAT TIẾP (KHÔNG RERUN)
        else:
            database.save_message(st.session_state.current_conv_id, "user", prompt)
            database.save_message(st.session_state.current_conv_id, "assistant", full_response)
            # Lưu xong thì thôi, KHÔNG gọi st.rerun() nữa -> Tin nhắn sẽ giữ nguyên trên màn hình
            
    else:
        # Chế độ khách
        st.session_state.guest_messages.append({"role": "user", "content": prompt})
        st.session_state.guest_messages.append({"role": "assistant", "content": full_response})