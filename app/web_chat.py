# =========================================================
# 🛠️ 1. THUỐC ĐẶC TRỊ LỖI SQLITE (BẮT BUỘC ĐỂ ĐẦU TIÊN)
# =========================================================
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

# --- CẤU HÌNH ĐƯỜNG DẪN ĐỂ IMPORT FILE TỪ THƯ MỤC KHÁC ---
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

# =========================================================
# 🛡️ 4. VẮC-XIN BẢO VỆ (CHỐNG LỖI removeChild KHI DỊCH TRANG)
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
        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        addProtection(sidebar);
        const radios = window.parent.document.querySelectorAll('[data-testid="stRadio"]');
        radios.forEach(addProtection);
        const buttons = window.parent.document.querySelectorAll('button');
        buttons.forEach(addProtection);
        const inputs = window.parent.document.querySelectorAll('input, textarea');
        inputs.forEach(addProtection);
        const forms = window.parent.document.querySelectorAll('[data-testid="stForm"]');
        forms.forEach(addProtection);
        const popovers = window.parent.document.querySelectorAll('[data-testid="stPopover"]');
        popovers.forEach(addProtection);
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
    html, body, [class*="css"] {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 16px; 
    }
    section[data-testid="stSidebar"] {
        padding-top: 1rem; 
        border-right: 1px solid #e0e0e0;
    }
    .stChatMessage p {
        font-size: 17px !important;
        line-height: 1.6;
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

    vector_db = None
    if os.path.exists(config.CHROMA_DB_DIR):
        try:
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
            vector_db = Chroma(persist_directory=config.CHROMA_DB_DIR, embedding_function=embedding_model)
        except Exception as e:
            pass
    
    llm = None
    if config.GOOGLE_API_KEY:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=config.GOOGLE_API_KEY)
        
    return vector_db, llm

vector_db, llm = load_resources()

def get_bot_response(user_query, history):
    if not vector_db or not llm: return "⚠️ Hệ thống đang bảo trì (Chưa kết nối Database/AI)."
    try:
        docs = vector_db.similarity_search(user_query, k=4)
        sources = []
        if docs:
            for doc in docs:
                src = doc.metadata.get("source", "Tài liệu y khoa")
                if src not in sources: sources.append(src)
            context = "\n".join([d.page_content for d in docs])
            citation_text = "\n\n---\n**📚 Tài liệu mình tham khảo:**\n" + "\n".join([f"- {s}" for s in sources])
        else:
            context = "Không có dữ liệu cụ thể."
            citation_text = "\n\n---\n*Theo kiến thức chuyên môn của mình.*"
        
        prompt = f"""
        VAI TRÒ: Bạn là một Bác sĩ giỏi, tận tâm và đang chat với BẠN THÂN (Xưng hô: Mình - Bạn). 

        DỮ LIỆU THAM KHẢO:
        - Lịch sử trò chuyện: {history}
        - Kiến thức y khoa tìm được: {context}
        - Câu hỏi hiện tại: "{user_query}"

        NHIỆM VỤ CỦA BẠN:
        1. THÂN THIỆN: Phản hồi bằng giọng điệu quan tâm, cảm thông.
        2. PHÂN TÍCH & HỎI THÊM (QUAN TRỌNG): Đừng vội kết luận. Hãy dựa vào thông tin người dùng đưa ra, phân tích các khả năng và ĐẶT 2-3 CÂU HỎI đào sâu để có thêm cơ sở chẩn đoán.
        3. CHẨN ĐOÁN TẠM THỜI: Đưa ra một vài giả thuyết về căn bệnh dựa trên kiến thức y khoa có sẵn.
        4. CHỈ ĐỊNH CHUYÊN KHOA: Tư vấn rõ ràng khoa nào cần khám và mức độ khẩn cấp.
        5. LỜI KHUYÊN: Các biện pháp chăm sóc tại nhà hoặc lưu ý an toàn.

        ĐỊNH DẠNG Markdown:
        (Câu chào cảm thông tự nhiên)

        ### 🔍 Phân tích sơ bộ:
        (Phân tích các triệu chứng người dùng vừa kể)

        ### 🩺 Để hiểu rõ hơn, cậu cho mình hỏi thêm nhé:
        (Đặt các câu hỏi thông minh để thu hẹp phạm vi chẩn đoán)

        ### 💡 Có thể cậu đang gặp vấn đề về:
        (Nêu các giả thuyết bệnh lý dựa trên kiến thức)

        ### 👉 Chuyên khoa cậu nên ghé khám:
        **[TÊN CHUYÊN KHOA]** - (Lý do vì sao chọn khoa này)

        ### 📝 Lưu ý cho cậu:
        (Dặn dò chăm sóc sức khỏe)

        (Câu chốt tình cảm)
        """
        response = llm.invoke(prompt)
        content = response.content
        final_ans = str(content)
        return final_ans + citation_text
    except Exception as e: return f"❌ Lỗi: {e}"

# --- SESSION STATE ---
if "user_info" not in st.session_state: st.session_state.user_info = None 
if "current_conv_id" not in st.session_state: st.session_state.current_conv_id = None
if "guest_messages" not in st.session_state: st.session_state.guest_messages = [] 
if "delete_confirm_id" not in st.session_state: st.session_state.delete_confirm_id = None

# ==========================================
# 🛑 7. SIDEBAR (GIAO DIỆN TRÁI)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=70)
    
    if st.session_state.user_info:
        user = st.session_state.user_info
        st.success(f"👋 Hi, {user['full_name']}")
        if st.button("➕ Ca mới", use_container_width=True):
            st.session_state.current_conv_id = None
            st.rerun()
        st.divider()
        if st.button("🚪 Đăng Xuất"):
            st.session_state.user_info = None
            st.rerun()
    else:
        st.info("Chế độ: **Khách**")
        mode = st.radio("Tài khoản:", ["Đăng Nhập", "Đăng Ký"], horizontal=True)
        with st.form("auth_form"):
            u = st.text_input("User")
            p = st.text_input("Pass", type="password")
            if st.form_submit_button("Xác nhận", use_container_width=True):
                if mode == "Đăng Nhập":
                    user = database.login_user(u, p)
                    if user: st.session_state.user_info = user; st.rerun()
                    else: st.error("Sai thông tin!")
                else:
                    ok, msg = database.register_user(u, p, u)
                    if ok: st.success("Đã đăng ký!"); st.rerun()
                    else: st.error(msg)

# ==========================================
# 🚀 8. MAIN CONTENT (GIAO DIỆN CHÍNH)
# ==========================================
st.title("🏥 Người Bạn Bác Sĩ (AI)")

messages = []
if st.session_state.user_info:
    if st.session_state.current_conv_id:
        messages = database.load_messages(st.session_state.current_conv_id)
else:
    messages = st.session_state.guest_messages

if not messages:
    st.markdown("👋 *Chào bạn! Có chỗ nào trong người thấy không ổn à? Kể mình nghe xem nào.*")

for msg in messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Kể triệu chứng cho mình nghe..."):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("👨‍⚕️ Đang ngẫm nghĩ xíu..."):
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages[-4:]])
            response = get_bot_response(prompt, history_str)
            st.write(response)
    
    if st.session_state.user_info:
        uid = st.session_state.user_info['id']
        if st.session_state.current_conv_id is None:
            st.session_state.current_conv_id = database.create_conversation(uid, prompt[:30])
        database.save_message(st.session_state.current_conv_id, "user", prompt)
        database.save_message(st.session_state.current_conv_id, "assistant", response)
    else:
        st.session_state.guest_messages.append({"role": "user", "content": prompt})
        st.session_state.guest_messages.append({"role": "assistant", "content": response})