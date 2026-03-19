try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import streamlit.components.v1 as components
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src import config, database
from src.services import ai_service

load_dotenv()

# =========================================================
# ⚙️ CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="Health AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)



# =========================================================
# 🧠 KHỞI TẠO STATE
# =========================================================
def init_session_state():
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = None
    if "guest_messages" not in st.session_state:
        st.session_state.guest_messages = []
    if "delete_confirm_id" not in st.session_state:
        st.session_state.delete_confirm_id = None

init_session_state()

# =========================================================
# 🛡️ CHỐNG DỊCH
# =========================================================
components.html("""
<script>
    function antiTranslate() {
        const head = window.parent.document.head;
        if (!head.querySelector('meta[name="google"][content="notranslate"]')) {
            const m = window.parent.document.createElement('meta');
            m.name = "google"; m.content = "notranslate";
            head.appendChild(m);
        }
        window.parent.document.documentElement.setAttribute('translate', 'no');
        window.parent.document.documentElement.classList.add('notranslate');
    }
    antiTranslate();
</script>
""", height=0)

# =========================================================
# 🎨 CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600&family=Google+Sans+Display:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Google Sans', 'Segoe UI', sans-serif;
        font-size: 14.5px;
        color: #1f1f1f;
    }

    #MainMenu, footer, .stDeployButton, [data-testid="InputInstructions"] { display: none !important; }

    header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
        box-shadow: none !important;
        height: 2.5rem !important;
    }
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        background-color: transparent !important;
    }

    .stApp { background-color: #f8f9fa; }

    section[data-testid="stSidebar"] {
        background-color: #f0f4f9 !important;
        border-right: 1px solid #e3e8f0 !important;
    }
    [data-testid="stSidebarContent"] {
        padding: 0 !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding: 2.5rem 0 0 0 !important;
        height: 100vh !important;
        display: flex;
        flex-direction: column;
        box-sizing: border-box !important;
    }
    [data-testid="stSidebarUserContent"] > div {
        display: flex;
        flex-direction: column;
        height: 100% !important;
        position: relative;
    }

    .sidebar-logo {
        display: flex; align-items: center; gap: 8px;
        padding: 0 10px 5px 10px;
        font-family: 'Google Sans Display', sans-serif;
        font-size: 17px; font-weight: 500; color: #1a73e8;
    }
    .sidebar-logo img { border-radius: 50%; width: 28px; }

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: #e8f0fe !important;
        color: #1a73e8 !important;
        border: none !important;
        border-radius: 24px !important;
        font-weight: 500 !important;
        padding: 0.4rem 1rem !important;
        margin-top: 5px;
    }

    hr { margin: 0.8rem 0 !important; border-color: #dde3ec !important; }

    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.history-marker) {
        height: calc(100vh - 140px) !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 4px 0 0 !important;
        margin-bottom: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.history-marker) > div {
        height: 100% !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.history-marker) .stButton > button {
        background-color: transparent !important;
        color: #3c4043 !important;
        border: none !important;
        text-align: left !important;
        padding: 4px 8px !important;
        justify-content: flex-start !important;
    }

    [data-testid="element-container"]:has(.logout-zone) {
        position: absolute;
        bottom: 0; left: 0; width: 100%;
    }
    .logout-zone {
        padding: 10px 0 10px 0;
        border-top: 1px solid #dde3ec;
        background: #f0f4f9;
    }

    [data-testid="stSidebar"]:has(.guest-mode-marker) [data-testid="stSidebarUserContent"] > div {
        overflow-y: auto !important;
        padding-bottom: 2rem !important;
    }
    [data-testid="stSidebar"]:has(.guest-mode-marker) .stTextInput input {
        padding: 0.4rem 0.8rem !important;
        min-height: 38px !important;
    }

    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #c4cdd6; border-radius: 4px; }

    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 120px;
        max-width: 800px;
    }
    [data-testid="stChatMessage"] { background: transparent !important; border: none !important; box-shadow: none !important; }
    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
        background-color: #e8f0fe !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 0.6rem 1rem !important;
        display: inline-block !important;
    }
    [data-testid="chatAvatarIcon-user"] img, [data-testid="chatAvatarIcon-assistant"] img {
        border-radius: 50% !important;
        border: 1px solid #e8eaed !important;
    }

    .stBottom { background: linear-gradient(to top, #f8f9fa 80%, transparent) !important; padding-bottom: 25px !important; }
    .stChatInput { max-width: 800px !important; margin: 0 auto !important; }
    .stChatInput > div {
        background: #ffffff !important;
        border: 1px solid #e3e8f0 !important;
        border-radius: 28px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
        padding: 5px 5px 5px 15px !important;
    }
    .stChatInput > div:focus-within { border-color: #1a73e8 !important; }
    .stChatInput [data-baseweb], .stChatInput textarea {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    .stChatInput textarea { font-family: 'Google Sans', sans-serif !important; }
    .stChatInput button {
        background: #1a73e8 !important;
        border-radius: 50% !important;
        width: 42px !important;
        height: 42px !important;
        margin: 0 !important;
    }

    .welcome-card {
        background: linear-gradient(135deg, #e8f0fe 0%, #eef6ff 100%);
        border-radius: 20px; padding: 2.5rem 2rem; text-align: center;
        border: 1px solid #d2e3fc; margin-bottom: 2rem;
    }
    .welcome-card h3 { color: #1a73e8; font-weight: 500; font-family: 'Google Sans Display', sans-serif; }
    .chips-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 1rem; }
    .chip { background: #fff; border: 1px solid #dadce0; border-radius: 20px; padding: 6px 14px; font-size: 13px; }

    /* ══════════════════════════════════════
       LOADING INDICATOR - HIỆU ỨNG ĐANG PHÂN TÍCH
    ══════════════════════════════════════ */
    .thinking-bubble {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: #ffffff;
        border: 1px solid #e3e8f0;
        border-radius: 18px 18px 18px 4px;
        padding: 10px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        font-size: 13.5px;
        color: #5f6368;
        margin-top: 2px;
    }
    /* Vòng xoay CSS thuần */
    .thinking-spinner {
        width: 16px;
        height: 16px;
        border: 2.5px solid #e3e8f0;
        border-top-color: #1a73e8;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        flex-shrink: 0;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    /* Ba chấm nhảy */
    .dots span {
        display: inline-block;
        width: 5px; height: 5px;
        margin: 0 2px;
        background: #1a73e8;
        border-radius: 50%;
        animation: bounce 1.2s infinite ease-in-out;
    }
    .dots span:nth-child(1) { animation-delay: 0s; }
    .dots span:nth-child(2) { animation-delay: 0.2s; }
    .dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
        0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
        40%            { transform: translateY(-5px); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 🗂️ RENDER SIDEBAR LOGIC
# =========================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <img src="https://cdn-icons-png.flaticon.com/512/3774/3774299.png" width="32">
            Health AI
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.user_info:
            st.markdown('<div class="logged-in-state"></div>', unsafe_allow_html=True)
            user = st.session_state.user_info

            st.markdown(f"👋 Xin chào, **{user['full_name']}**")
            if st.button("✦ Chat mới", use_container_width=True, type="primary"):
                st.session_state.current_conv_id = None
                st.rerun()

            st.divider()
            st.caption("📂 LỊCH SỬ GẦN ĐÂY")

            with st.container(height=280):
                st.markdown('<div class="history-marker"></div>', unsafe_allow_html=True)

                convs = database.get_user_conversations(user['id'])
                if not convs:
                    st.caption("Chưa có lịch sử.")
                else:
                    for conv in convs:
                        c1, c2 = st.columns([0.8, 0.2])
                        with c1:
                            if st.session_state.delete_confirm_id == conv['id']:
                                st.error("Xóa?", icon="⚠️")
                                d1, d2 = st.columns(2)
                                if d1.button("Có", key=f"y_{conv['id']}", use_container_width=True):
                                    database.delete_conversation(conv['id'])
                                    st.session_state.delete_confirm_id = None
                                    if st.session_state.current_conv_id == conv['id']:
                                        st.session_state.current_conv_id = None
                                    st.rerun()
                                if d2.button("Ko", key=f"n_{conv['id']}", use_container_width=True):
                                    st.session_state.delete_confirm_id = None
                                    st.rerun()
                            else:
                                icon  = "📌 " if conv.get('is_pinned', 0) else "💬 "
                                title = conv['title'][:20] + "..." if len(conv['title']) > 20 else conv['title']
                                if st.button(f"{icon}{title}", key=f"btn_{conv['id']}", use_container_width=True):
                                    st.session_state.current_conv_id = conv['id']
                                    st.rerun()
                        with c2:
                            if st.session_state.delete_confirm_id != conv['id']:
                                with st.popover("⋮"):
                                    is_pinned = conv.get('is_pinned', 0)
                                    if st.button("Bỏ ghim" if is_pinned else "📌 Ghim", key=f"pin_{conv['id']}", use_container_width=True):
                                        database.toggle_pin_conversation(conv['id'], is_pinned)
                                        st.rerun()
                                    if st.button("🗑️ Xóa", key=f"del_{conv['id']}", use_container_width=True):
                                        st.session_state.delete_confirm_id = conv['id']
                                        st.rerun()

            st.markdown('<div class="logout-zone">', unsafe_allow_html=True)
            if st.button("🚪 Đăng Xuất", use_container_width=True):
                st.session_state.user_info     = None
                st.session_state.current_conv_id = None
                st.session_state.guest_messages  = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="guest-mode-marker"></div>', unsafe_allow_html=True)
            st.info("🙂 Đang dùng chế độ Khách\n\nĐăng nhập để lưu lịch sử.")
            st.divider()

            mode = st.radio("Chế độ xác thực", ["Đăng Nhập", "Đăng Ký"], horizontal=True, label_visibility="collapsed")

            if mode == "Đăng Nhập":
                with st.form("login_form"):
                    u = st.text_input("Tài khoản", placeholder="Nhập username...")
                    p = st.text_input("Mật khẩu", type="password", placeholder="Nhập mật khẩu...")
                    if st.form_submit_button("Đăng nhập", use_container_width=True):
                        usr = database.login_user(u, p)
                        if usr:
                            st.session_state.user_info = usr
                            st.rerun()
                        else:
                            st.error("Sai tên đăng nhập hoặc mật khẩu.")
            else:
                with st.form("reg_form"):
                    nu = st.text_input("Tài khoản mới", placeholder="Chọn username...")
                    np = st.text_input("Mật khẩu", type="password", placeholder="Tạo mật khẩu...")
                    nn = st.text_input("Họ và tên", placeholder="Tên đầy đủ của bạn...")
                    if st.form_submit_button("Tạo tài khoản", use_container_width=True):
                        ok, msg = database.register_user(nu, np, nn)
                        if ok:
                            st.success("✅ Thành công! Hãy đăng nhập.")
                        else:
                            st.error(msg)


# =========================================================
# 💬 RENDER KHU VỰC CHAT CHÍNH
# =========================================================
def render_chat():
    st.title("Health AI ✦")
    st.warning("**Lưu ý:** Tư vấn sơ bộ từ AI, không thay thế chẩn đoán của bác sĩ.", icon="⚠️")

    AVATAR_AI   = "https://cdn-icons-png.flaticon.com/512/3774/3774299.png"
    AVATAR_USER = "https://cdn-icons-png.flaticon.com/512/1144/1144760.png"

    # Load tin nhắn
    messages = []
    if st.session_state.user_info:
        if st.session_state.current_conv_id:
            messages = database.load_messages(st.session_state.current_conv_id)
    else:
        messages = st.session_state.guest_messages

    # Box Welcome
    if not messages:
        st.markdown("""
        <div class="welcome-card">
            <h3>Xin chào! Tôi có thể giúp gì cho sức khỏe của bạn?</h3>
            <p>Mô tả triệu chứng, đặt câu hỏi hoặc tìm hiểu về y tế.</p>
            <div class="chips-row">
                <span class="chip">🤒 Bị sốt cao</span>
                <span class="chip">💊 Thuốc hạ sốt</span>
                <span class="chip">🫁 Ho kéo dài</span>
                <span class="chip">😴 Mất ngủ</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Hiển thị lịch sử chat
    for msg in messages:
        role   = msg["role"]
        avatar = AVATAR_USER if role == "user" else AVATAR_AI
        with st.chat_message(role, avatar=avatar):
            st.write(msg["content"])

    # ─── Xử lý input ───
    if prompt := st.chat_input("Hỏi về sức khỏe của bạn..."):

        # Hiển thị tin nhắn người dùng ngay lập tức
        with st.chat_message("user", avatar=AVATAR_USER):
            st.write(prompt)

        with st.chat_message("assistant", avatar=AVATAR_AI):
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages[-4:]])

            # ══════════════════════════════════════════════════════
            # [FIX CHÍNH] BƯỚC 1: Hiện loading bubble TRƯỚC
            # rồi chạy RAG (phần nặng nhất) bên trong spinner.
            # Spinner kết thúc → chữ bắt đầu stream ngay lập tức.
            # ══════════════════════════════════════════════════════

            # Placeholder để thay loading → stream text
            response_placeholder = st.empty()

            # Hiện bubble "đang phân tích" ngay lập tức
            response_placeholder.markdown("""
            <div class="thinking-bubble">
                <div class="thinking-spinner"></div>
                <span>Đang tra cứu tài liệu y khoa…</span>
                <span class="dots">
                    <span></span><span></span><span></span>
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Chạy RAG đồng bộ (bước chậm) - người dùng nhìn thấy bubble ở trên
            context, citation_text = ai_service._get_rag_context(prompt)
            built_prompt = ai_service._build_prompt(prompt, history_str, context)

            # Xoá bubble loading, thay bằng stream text
            response_placeholder.empty()

            # BƯỚC 2: Stream từ Groq (chữ hiện dần ngay từ token đầu tiên)
            full_response = st.write_stream(
                ai_service.stream_from_built_prompt(built_prompt, citation_text)
            )

        # ── Lưu DB ──
        if st.session_state.user_info:
            uid = st.session_state.user_info['id']
            if st.session_state.current_conv_id is None:
                title  = prompt[:30] + '..' if len(prompt) > 30 else prompt
                new_id = database.create_conversation(uid, title)
                st.session_state.current_conv_id = new_id
                database.save_message(new_id, "user",      prompt)
                database.save_message(new_id, "assistant", full_response)
                st.rerun()
            else:
                database.save_message(st.session_state.current_conv_id, "user",      prompt)
                database.save_message(st.session_state.current_conv_id, "assistant", full_response)
        else:
            st.session_state.guest_messages.append({"role": "user",      "content": prompt})
            st.session_state.guest_messages.append({"role": "assistant", "content": full_response})


# =========================================================
# 🚀 KHỞI CHẠY ỨNG DỤNG
# =========================================================
render_sidebar()
render_chat()