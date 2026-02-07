try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# =========================================================
# 📚 CÁC THƯ VIỆN CẦN THIẾT
# =========================================================
import streamlit as st
import streamlit.components.v1 as components 
import os
import sys
import time
from PIL import Image 

# --- CẤU HÌNH ĐƯỜNG DẪN ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src import config, database 
from src.services import ai_service

load_dotenv()

# =========================================================
# ⚙️ CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="Health Chatbot", 
    page_icon="🏥", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)

# --- CẬP NHẬT KEY ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    config.GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# --- KEY QUẢN LÝ UPLOAD ẢNH ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# =========================================================
# 🛡️ 1. VACXIN CHỐNG DỊCH (LOẠI BỎ HOÀN TOÀN LỖI REMOVECHILD)
# =========================================================
# Script này chèn thẻ meta cấm dịch và gán thuộc tính cấm dịch cho toàn bộ HTML
components.html("""
<script>
    function injectAntiTranslate() {
        // 1. Chèn thẻ Meta cấm dịch vào HEAD (Cấp độ cao nhất của trình duyệt)
        const head = window.parent.document.head;
        if (!head.querySelector('meta[name="google"][content="notranslate"]')) {
            const meta = window.parent.document.createElement('meta');
            meta.name = "google";
            meta.content = "notranslate";
            head.appendChild(meta);
        }

        // 2. Khóa dịch ở thẻ HTML và BODY
        const html = window.parent.document.documentElement;
        html.setAttribute('translate', 'no');
        html.classList.add('notranslate');

        const body = window.parent.document.body;
        body.setAttribute('translate', 'no');
        body.classList.add('notranslate');

        // 3. Khóa dịch cụ thể Sidebar (Mục tiêu chính)
        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.setAttribute('translate', 'no');
            sidebar.classList.add('notranslate');
        }
    }
    
    // Chạy ngay lập tức
    injectAntiTranslate();
    
    // Chạy liên tục mỗi 300ms để đảm bảo trình duyệt không tự bật lại
    setInterval(injectAntiTranslate, 300);
</script>
""", height=0)

# =========================================================
# 🎨 2. CSS GIAO DIỆN & NÚT BẤM
# =========================================================
st.markdown("""
<style>
    /* Font chữ */
    html, body, [class*="css"] { font-family: 'Source Sans Pro', sans-serif; font-size: 16px; }

    /* Sidebar */
    section[data-testid="stSidebar"] { padding-top: 1rem; border-right: 1px solid #e0e0e0; }
    
    /* Khung chat chính */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 120px; 
        max-width: 900px;
        margin: 0 auto;
    }

    /* FIX LỖI THANH NHẬP LIỆU BỊ RỘNG */
    .stChatInput {
        max-width: 900px !important;
        margin: 0 auto !important;
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 99;
    }
    
    .stChatInput textarea {
        padding-left: 50px !important; 
    }

    /* --- NÚT KẸP GIẤY --- */
    /* Mặc định ẩn tất cả các nút popover để tránh hiện sai chỗ */
    [data-testid="stPopover"] {
        display: none; 
    }

    /* Riêng nút trong Sidebar thì PHẢI hiện (để bấm menu lịch sử) */
    [data-testid="stSidebar"] [data-testid="stPopover"] {
        display: inline-flex !important;
    }

    /* Style riêng cho nút Upload sau khi được JS tìm thấy và gắn class */
    .my-upload-btn {
        display: block !important;
        position: fixed !important;
        z-index: 100000 !important;
        width: 40px !important;
        height: 40px !important;
    }

    .my-upload-btn > button {
        background-color: transparent !important;
        border: none !important;
        color: #555 !important;
        border-radius: 50% !important;
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s !important;
        box-shadow: none !important;
    }

    .my-upload-btn > button:hover {
        background-color: #f0f2f6 !important;
        color: #ff4b4b !important;
    }
    
    .my-upload-btn > button::after {
        content: "📎";
        font-size: 24px;
        font-weight: bold;
    }
    .my-upload-btn > button > div { display: none !important; }

    #MainMenu, footer, .stDeployButton, [data-testid="InputInstructions"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🛠️ 3. JS THÔNG MINH: CHỌN ĐÚNG NÚT UPLOAD (KHÔNG NHẦM VỚI SIDEBAR)
# =========================================================
js_code = """
<script>
function alignButton() {
    try {
        const chatInput = window.parent.document.querySelector('.stChatInput');
        
        // 1. Lấy TẤT CẢ các nút Popover trên màn hình
        const allPopovers = window.parent.document.querySelectorAll('[data-testid="stPopover"]');
        let targetPopover = null;

        // 2. Duyệt qua từng nút để tìm đúng nút Upload (Nút KHÔNG nằm trong Sidebar)
        for (let i = 0; i < allPopovers.length; i++) {
            const p = allPopovers[i];
            // Kiểm tra: Nút này có tổ tiên là Sidebar không?
            const insideSidebar = p.closest('[data-testid="stSidebar"]');
            
            // Nếu KHÔNG nằm trong sidebar, thì đây chính là nút Upload chúng ta cần
            if (!insideSidebar) {
                targetPopover = p;
                break; // Tìm thấy rồi thì dừng lại, không cần tìm tiếp
            }
        }
        
        // 3. Nếu tìm thấy đúng nút và thanh chat
        if (chatInput && targetPopover) {
            // Đánh dấu class để CSS hiển thị nó lên (biến nó thành nút kẹp giấy)
            if (!targetPopover.classList.contains('my-upload-btn')) {
                targetPopover.classList.add('my-upload-btn');
            }

            const rect = chatInput.getBoundingClientRect();
            
            // Nếu thanh nhập liệu chưa hiện thì ẩn nút đi
            if (rect.width === 0) {
                targetPopover.style.display = 'none';
                return;
            }

            // Tính toán vị trí
            const targetLeft = rect.left + 5;
            const targetBottom = (window.parent.innerHeight - rect.bottom) + 15; 
            
            // Gán vị trí
            targetPopover.style.left = targetLeft + 'px';
            targetPopover.style.bottom = targetBottom + 'px';
            targetPopover.style.display = 'block'; // Hiện nút lên
        }
    } catch(e) {}
}
setInterval(alignButton, 50);
</script>
"""
components.html(js_code, height=0)

# =========================================================
# 🧠 KHỞI TẠO STATE
# =========================================================
if "user_info" not in st.session_state: st.session_state.user_info = None 
if "current_conv_id" not in st.session_state: st.session_state.current_conv_id = None
if "guest_messages" not in st.session_state: st.session_state.guest_messages = [] 
if "delete_confirm_id" not in st.session_state: st.session_state.delete_confirm_id = None

# ==========================================
# 🛑 SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=70)
    
    # Placeholder để reset nội dung sạch sẽ (Giúp tránh lỗi DOM cũ còn sót lại)
    sidebar_placeholder = st.empty()

    with sidebar_placeholder.container():
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
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("Có", key=f"yes_{conv['id']}", type="primary", use_container_width=True):
                                database.delete_conversation(conv['id'])
                                st.session_state.delete_confirm_id = None
                                if st.session_state.current_conv_id == conv['id']:
                                    st.session_state.current_conv_id = None
                                st.rerun()
                        with c2:
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
                        if ok: 
                            st.success("Xong!")
                        else: 
                            st.error(msg)

# =========================================================
# MAIN CONTENT (GIAO DIỆN CHAT)
# =========================================================
st.title("🏥 Health Chatbot")
st.warning("⚠️ **Lưu ý:** AI chỉ hỗ trợ tư vấn sơ bộ, không thay thế chẩn đoán của bác sĩ chuyên khoa.", icon="⚠️")

AVATAR_AI = "https://cdn-icons-png.flaticon.com/512/3774/3774299.png"
AVATAR_USER = "https://cdn-icons-png.flaticon.com/512/1144/1144760.png"

messages = []
if st.session_state.user_info:
    if st.session_state.current_conv_id:
        messages = database.load_messages(st.session_state.current_conv_id)
else:
    messages = st.session_state.guest_messages

if not messages:
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 20px;'>
        <h3>👋 Xin chào! Mình có thể giúp gì cho sức khỏe của bạn?</h3>
    </div>
    """, unsafe_allow_html=True)

for msg in messages:
    role = msg["role"]
    avatar = AVATAR_USER if role == "user" else AVATAR_AI
    with st.chat_message(role, avatar=avatar):
        st.write(msg["content"])

# ==========================================
# 📸 KHU VỰC NHẬP LIỆU
# ==========================================

image_input = None
upload_key = f"file_uploader_{st.session_state.uploader_key}"

# --- NÚT ĐÍNH KÈM (ĐÂY LÀ NÚT Ở MAIN CONTENT - SẼ ĐƯỢC JS CHỌN VÀ GẮN CSS) ---
with st.popover(" ", use_container_width=False):
    st.markdown("### 📸 Tải ảnh triệu chứng")
    uploaded_file = st.file_uploader(
        "Chọn ảnh:", 
        type=["jpg", "jpeg", "png"], 
        label_visibility="collapsed",
        key=upload_key
    )
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)
        st.image(image_input, width=200, caption="Đã chọn")
        st.success("Ảnh đã sẵn sàng! Gõ tin nhắn để gửi.")

# --- XỬ LÝ CHAT ---
if prompt := st.chat_input("Gõ triệu chứng vào đây..."):
    with st.chat_message("user", avatar=AVATAR_USER):
        st.write(prompt)
        if image_input: st.image(image_input, width=250)
    
    full_response = ""
    with st.chat_message("assistant", avatar=AVATAR_AI):
        with st.spinner("👨‍⚕️ Bác sĩ đang xem xét..."):
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages[-4:]])
            full_response = ai_service.get_bot_response(prompt, history_str, image=image_input)
            st.write(full_response)
    
    prompt_to_save = prompt
    if image_input: prompt_to_save += " \n\n[Người dùng đã gửi kèm một hình ảnh]"

    if st.session_state.user_info:
        uid = st.session_state.user_info['id']
        if st.session_state.current_conv_id is None:
            title = (prompt[:30] + '..') if len(prompt) > 30 else prompt
            new_id = database.create_conversation(uid, title)
            st.session_state.current_conv_id = new_id
            database.save_message(new_id, "user", prompt_to_save)
            database.save_message(new_id, "assistant", full_response)
            st.session_state.uploader_key += 1
            st.rerun()
        else:
            database.save_message(st.session_state.current_conv_id, "user", prompt_to_save)
            database.save_message(st.session_state.current_conv_id, "assistant", full_response)
            st.session_state.uploader_key += 1
            # KHÔNG RERUN ĐỂ TRÁNH LỖI MÀN HÌNH TRẮNG
    else:
        st.session_state.guest_messages.append({"role": "user", "content": prompt_to_save})
        st.session_state.guest_messages.append({"role": "assistant", "content": full_response})
        st.session_state.uploader_key += 1
        st.rerun()