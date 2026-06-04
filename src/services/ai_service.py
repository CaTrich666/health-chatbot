"""
ai_service.py
"""


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
    page_title="Solar AI",
    page_icon="☀️",
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
            <img src="https://cdn-icons-png.flaticon.com/512/869/869869.png" width="32">
            Solar AI
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

            if st.button("✦ Chat mới", key="guest_new_chat", use_container_width=True, type="primary"):
                st.session_state.guest_messages = []
                st.rerun()

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
    st.title("Solar AI ✦")
    st.info("**Lưu ý:** Thông tin tư vấn chỉ mang tính tham khảo ban đầu. Để có phương án lắp đặt, công suất và chi phí chính xác, cần khảo sát thực tế bởi nhân viên kỹ thuật.", icon="☀️")

    AVATAR_AI   = "https://cdn-icons-png.flaticon.com/512/869/869869.png"
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
            <h3>Xin chào! Tôi có thể tư vấn gì về điện mặt trời?</h3>
            <p>Hãy hỏi về hệ thống áp mái, hòa lưới, hybrid, pin lưu trữ hoặc quy trình lắp đặt.</p>
            <div class="chips-row">
                <span class="chip">☀️ Điện mặt trời áp mái là gì?</span>
                <span class="chip">🔋 Có cần pin lưu trữ không?</span>
                <span class="chip">⚡ Hệ hòa lưới là gì?</span>
                <span class="chip">🏠 Nhà em phù hợp lắp không?</span>
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
    if prompt := st.chat_input("Hỏi về điện mặt trời..."):

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
                <span>Đang tra cứu tài liệu điện mặt trời…</span>
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
if __name__ == "__main__":
    render_sidebar()
    render_chat()

import os
import re
import zipfile
import time
import pickle
import streamlit as st
from typing import Generator, Optional, Tuple, List
from collections import defaultdict

from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src import config

# ── Tokenizer tiếng Việt ──────────────────────────────────
try:
    from underthesea import word_tokenize as vi_tokenize
    def _tokenize(text: str) -> list:
        return vi_tokenize(text.lower(), format="text").split()
    print("✅ Dùng underthesea tokenizer (tiếng Việt)")
except ImportError:
    def _tokenize(text: str) -> list:
        return text.lower().split()
    print("⚠️ underthesea chưa cài, dùng split() tạm thời")


# ═══════════════════════════════════════════════════════════
# GIẢI NÉN DB
# ═══════════════════════════════════════════════════════════
def _extract_db_if_needed():
    db_path = os.path.join(config.CHROMA_DB_DIR, "chroma.sqlite3")
    if os.path.exists(db_path):
        return
    for zip_name in [db_path + ".zip", db_path + ".zip.zip"]:
        if os.path.exists(zip_name):
            try:
                with zipfile.ZipFile(zip_name, 'r') as zf:
                    zf.extractall(config.CHROMA_DB_DIR)
                print("✅ Đã giải nén ChromaDB")
            except Exception as e:
                print(f"❌ Lỗi giải nén: {e}")
            break


# ═══════════════════════════════════════════════════════════
# THROTTLE
# ═══════════════════════════════════════════════════════════
_last_api_call: float = 0.0
_MIN_INTERVAL_SEC: float = 2.0

def _throttle_api():
    global _last_api_call
    elapsed = time.time() - _last_api_call
    if elapsed < _MIN_INTERVAL_SEC:
        time.sleep(_MIN_INTERVAL_SEC - elapsed)
    _last_api_call = time.time()


# ═══════════════════════════════════════════════════════════
# CACHE TÀI NGUYÊN
# ═══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_resources() -> Tuple[Optional[object], Optional[object], Optional[object], Optional[list]]:
    t_start = time.perf_counter()
    print("⏳ Đang khởi tạo tài nguyên AI tư vấn điện mặt trời (Chỉ chạy 1 lần)...")

    _extract_db_if_needed()

    api_key = os.environ.get("GROQ_API_KEY") or getattr(config, "GROQ_API_KEY", None)
    if not api_key:
        print("❌ Lỗi: Không tìm thấy GROQ_API_KEY.")
        return None, None, None, None

    llm = None
    try:
        llm = Groq(api_key=api_key)
        print("✅ Đã khởi tạo Groq LLM")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Groq: {e}")

    vector_db = None
    if os.path.exists(config.CHROMA_DB_DIR):
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            vector_db = Chroma(
                persist_directory=config.CHROMA_DB_DIR,
                embedding_function=embedding_model,
            )
            print("✅ Đã load xong VectorDB")
        except Exception as e:
            print(f"❌ Lỗi load ChromaDB: {e}")

    bm25_index = None
    bm25_docs  = None
    if os.path.exists(config.BM25_INDEX_PATH):
        try:
            with open(config.BM25_INDEX_PATH, "rb") as f:
                bm25_index, bm25_docs = pickle.load(f)
            print(f"✅ Đã load BM25 index ({len(bm25_docs)} documents)")
        except Exception as e:
            print(f"❌ Lỗi load BM25: {e}")
    else:
        print(f"⚠️ Không tìm thấy BM25 index tại {config.BM25_INDEX_PATH}")
        print("   → Hãy chạy script ingest để tạo file .pkl trước.")

    print(f"✅ Khởi tạo xong trong {time.perf_counter() - t_start:.2f}s")
    return vector_db, llm, bm25_index, bm25_docs


# ═══════════════════════════════════════════════════════════
# BM25 SEARCH
# ═══════════════════════════════════════════════════════════
def _bm25_search(query: str, k: int = None) -> List[Tuple[str, str, float]]:
    if k is None:
        k = config.TOP_K_RETRIEVAL
    _, _, bm25_index, bm25_docs = load_resources()
    if bm25_index is None or not bm25_docs:
        return []
    try:
        tokenized_query = _tokenize(query)
        scores      = bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            (bm25_docs[i].page_content,
             bm25_docs[i].metadata.get("source", "Tài liệu điện mặt trời"),
             float(scores[i]))
            for i in top_indices if scores[i] > 0
        ]
    except Exception as e:
        print(f"❌ Lỗi BM25 search: {e}")
        return []


# ═══════════════════════════════════════════════════════════
# RRF FUSION
# ═══════════════════════════════════════════════════════════
def _rrf_fusion(
    vector_results: List[Tuple[str, str]],
    bm25_results:   List[Tuple[str, str, float]],
    k: int = None,
) -> List[Tuple[str, str]]:
    if k is None:
        k = config.TOP_K_FINAL
    rrf_k       = config.RRF_K
    scores      = defaultdict(float)
    content_map : dict = {}
    for rank, (content, source) in enumerate(vector_results, start=1):
        scores[content]     += 1.0 / (rrf_k + rank)
        content_map[content] = source
    for rank, (content, source, _) in enumerate(bm25_results, start=1):
        scores[content]     += 1.0 / (rrf_k + rank)
        content_map[content] = source
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(content, content_map[content]) for content, _ in top_docs]


# ═══════════════════════════════════════════════════════════
# CACHE VECTOR SEARCH
# ═══════════════════════════════════════════════════════════
def _cached_similarity_search(query: str, k: int = None) -> tuple:
    if k is None:
        k = config.TOP_K_RETRIEVAL
    cache_key = f"vsearch_{query}_{k}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    vector_db = load_resources()[0]
    if vector_db is None:
        return ()
    try:
        docs   = vector_db.similarity_search(query, k=k)
        result = tuple(
            (d.page_content, d.metadata.get("source", "Tài liệu điện mặt trời"))
            for d in docs
        )
        st.session_state[cache_key] = result
        return result
    except Exception as e:
        print(f"❌ Lỗi tìm kiếm vector: {e}")
        return ()


# ═══════════════════════════════════════════════════════════
# RAG CONTEXT
# ═══════════════════════════════════════════════════════════
def _get_rag_context(user_query: str) -> Tuple[str, str]:
    vector_results = list(_cached_similarity_search(user_query, k=config.TOP_K_RETRIEVAL))
    bm25_results   = _bm25_search(user_query, k=config.TOP_K_RETRIEVAL)

    if bm25_results:
        fused_results = _rrf_fusion(vector_results, bm25_results, k=config.TOP_K_FINAL)
        print(f"🔀 RRF: vector={len(vector_results)}, bm25={len(bm25_results)} → fused={len(fused_results)}")
    else:
        fused_results = vector_results[:config.TOP_K_FINAL]
        _, _, bm25_index, _ = load_resources()
        if bm25_index is None:
            print("⚠️ BM25 index chưa load, dùng vector search thuần")

    if not fused_results:
        return "Không có dữ liệu cụ thể, dùng kiến thức tổng quát về điện năng lượng mặt trời.", ""

    context_parts = []
    sources       = []
    total_len     = 0
    MAX_CHARS     = 1500

    for content, source in fused_results:
        if total_len >= MAX_CHARS:
            break
        snippet = content[:MAX_CHARS - total_len]
        context_parts.append(snippet)
        total_len += len(snippet)
        if source not in sources:
            sources.append(source)

    context       = "\n---\n".join(context_parts)
    citation_text = "\n\n---\n**📚 Nguồn tham khảo:** " + " | ".join(sources)
    return context, citation_text


# ═══════════════════════════════════════════════════════════
# CẮT LỊCH SỬ AN TOÀN
# ═══════════════════════════════════════════════════════════
def _trim_history_safe(history: str, max_chars: int = 1500) -> str:
    """Cắt lịch sử an toàn theo số ký tự, không làm rách câu"""
    if len(history) <= max_chars:
        return history
    truncated = history[-max_chars:]
 
    idx = -1
    for marker in ["User:", "user:", "Người dùng:", "\nassistant:", "\nBot:", "Trợ lý:"]:
        idx = truncated.find(marker)
        if idx != -1:
            break
 
    if idx == -1:
        idx = truncated.find('\n')
 
    return "[...] " + truncated[idx:] if idx != -1 else "[...] " + truncated


# ═══════════════════════════════════════════════════════════
# LỌC HIDDEN CoT - Xóa nội dung trong thẻ <thinking>
# ═══════════════════════════════════════════════════════════
def _strip_thinking(text: str) -> str:
    """Xóa toàn bộ nội dung trong thẻ <thinking>...</thinking>"""
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    return cleaned.strip()


def _stream_strip_thinking(stream_generator) -> Generator[str, None, None]:
    """Lọc thẻ <thinking> trong luồng stream theo thời gian thực."""
    buffer       = ""
    inside_think = False
 
    for chunk in stream_generator:
        buffer += chunk
        while True:
            if not inside_think:
                start = buffer.find("<thinking>")
                if start == -1:
                    safe_len = max(0, len(buffer) - 9)
                    if safe_len > 0:
                        yield buffer[:safe_len]
                        buffer = buffer[safe_len:]
                    break
                else:
                    yield buffer[:start]
                    buffer       = buffer[start:]
                    inside_think = True
            else:
                end = buffer.find("</thinking>")
                if end == -1:
                    break
                else:
                    buffer       = buffer[end + len("</thinking>"):]
                    inside_think = False
 
    if buffer and not inside_think:
        yield buffer


# ═══════════════════════════════════════════════════════════
# BUILD PROMPT - HIDDEN CoT với XML Tag <thinking>
# Model suy luận trong <thinking>...</thinking>
# Backend lọc bỏ trước khi hiển thị ra UI
# ═══════════════════════════════════════════════════════════
def _build_prompt(user_query: str, history: str, context: str) -> str:
    history_trimmed = _trim_history_safe(history)
 
    return f"""Bạn là "Solar AI" - trợ lý tư vấn điện năng lượng mặt trời.
Bạn hỗ trợ người dùng tìm hiểu, lựa chọn và sử dụng hệ thống điện mặt trời một cách dễ hiểu, thực tế và an toàn.

PHONG CÁCH TRẢ LỜI:
- Trả lời tự nhiên, thân thiện, không quá máy móc.
- Luôn xưng là "mình", gọi người dùng là "bạn". Không xưng "tôi".
- Ưu tiên giải thích trực tiếp nếu người dùng hỏi kiến thức chung.
- Chỉ hỏi thêm thông tin khi thật sự cần để tư vấn phương án lắp đặt.
- Không ép người dùng phải cung cấp đủ thông tin trong mọi trường hợp.
- Không bịa thông tin về công ty, giá bán, bảo hành, sản phẩm hoặc thông số kỹ thuật nếu dữ liệu không có.
- Không đưa báo giá chính thức hoặc khẳng định công suất chính xác khi chưa khảo sát thực tế.
- Với câu hỏi kỹ thuật nguy hiểm, ưu tiên cảnh báo an toàn ngắn gọn và khuyến nghị liên hệ kỹ thuật viên.
- Không hiển thị nguồn RAG, tên category hoặc mã nguồn tham khảo trong câu trả lời cho người dùng.

LỊCH SỬ TRÒ CHUYỆN:
{history_trimmed}
 
DỮ LIỆU THAM KHẢO RAG:
{context}

LƯU Ý VỀ DỮ LIỆU:
- Dữ liệu RAG là nguồn tham khảo chính.
- Nếu dữ liệu RAG phù hợp, hãy dùng để trả lời.
- Nếu dữ liệu RAG không đủ, có thể dùng kiến thức tổng quát về điện mặt trời nhưng phải nói cẩn trọng.
- Nếu không chắc, hãy nói "mình chưa có đủ dữ liệu để khẳng định".
- Không tạo URL, link giả hoặc thông tin không có nguồn.
- Không nhắc lại tên nguồn nội bộ như: solution_selection, sizing_preliminary, chatbot_policy, safety_warning, maintenance, safety.

CÂU HỎI CỦA NGƯỜI DÙNG:
{user_query}
 
════════════════════════════════════════════════════════════
HƯỚNG DẪN ĐỊNH DẠNG ĐẦU RA:
Bắt buộc bắt đầu bằng thẻ <thinking> để suy luận nội bộ,
sau đó mới viết câu trả lời cho người dùng bên ngoài thẻ.
Người dùng sẽ không thấy nội dung trong thẻ <thinking>.

Cấu trúc:
<thinking>
Suy luận ngắn gọn nội bộ:
- Câu hỏi thuộc loại nào: chào hỏi / ngoài phạm vi / kiến thức chung / tư vấn lắp đặt / cảnh báo an toàn / báo giá / MKSolar
- Có cần hỏi thêm không?
- Có rủi ro an toàn không?
- Nguồn RAG có liên quan không?
</thinking>
Câu trả lời cho người dùng
════════════════════════════════════════════════════════════

QUY TẮC MARKDOWN BẮT BUỘC:
- Khi có tiêu đề phần, bắt buộc dùng Markdown heading cấp 3, ví dụ: ### Nhận xét sơ bộ
- Không được viết tiêu đề trần như "Nhận xét sơ bộ" nếu không có dấu ###
- Nếu có danh sách việc cần làm, dùng gạch đầu dòng "-"
- Không viết quá dài; ưu tiên rõ ràng, dễ đọc
- Luôn xưng "mình", gọi người dùng là "bạn"; không xưng "tôi"
- Không hiển thị dòng "Nguồn tham khảo" hoặc bất kỳ tên category nội bộ nào ở cuối câu trả lời.

PHÂN LOẠI CÂU HỎI VÀ CÁCH TRẢ LỜI:

1. CHÀO HỎI / CẢM ƠN
Nếu người dùng chỉ chào hỏi hoặc cảm ơn:
- Trả lời ngắn gọn, thân thiện.
- Gợi ý người dùng có thể hỏi về điện mặt trời áp mái, hòa lưới, hybrid, pin lưu trữ, chi phí tham khảo hoặc quy trình lắp đặt.
- Không dùng nhiều tiêu đề.

2. NGOÀI PHẠM VI ĐIỆN MẶT TRỜI
Nếu câu hỏi không liên quan đến điện mặt trời, hệ thống điện, lắp đặt, thiết bị, bảo trì hoặc sử dụng điện:
Trả lời ngắn:
"Mình chỉ hỗ trợ các nội dung liên quan đến điện năng lượng mặt trời. Bạn có thể hỏi về hệ thống áp mái, hòa lưới, hybrid, pin lưu trữ hoặc quy trình lắp đặt."

3. CÂU HỎI KIẾN THỨC CHUNG
Áp dụng khi người dùng hỏi dạng:
- "Điện mặt trời áp mái là gì?"
- "Hệ hòa lưới là gì?"
- "Hybrid là gì?"
- "Pin lưu trữ dùng để làm gì?"
- "Inverter là gì?"
- "Có cần vệ sinh tấm pin không?"

Cách trả lời:
- Trả lời trực tiếp ngay, không hỏi tiền điện hoặc diện tích mái trước.
- Giải thích dễ hiểu.
- Có thể thêm ví dụ ngắn.
- Cuối câu có thể gợi ý: "Nếu bạn muốn lắp thực tế, mình có thể tư vấn thêm dựa trên tiền điện, diện tích mái và nhu cầu sử dụng."

Định dạng bắt buộc với câu hỏi kiến thức chung:
### Giải thích ngắn gọn
Trả lời khái niệm chính.

### Khi nào nên quan tâm?
Nêu trường hợp áp dụng thực tế.

### Lưu ý
Nêu điểm cần khảo sát hoặc kiểm tra nếu muốn lắp đặt.

4. TƯ VẤN CHỌN HỆ THỐNG / LẮP ĐẶT
Áp dụng khi người dùng hỏi:
- "Nhà em tiền điện 2 triệu thì nên lắp hệ nào?"
- "Nên lắp 3kW hay 5kW?"
- "Nên chọn hòa lưới hay hybrid?"
- "Nhà xưởng nên lắp hệ gì?"
- "Có nên lắp pin lưu trữ không?"

Nếu thông tin còn thiếu:
- Vẫn đưa nhận xét sơ bộ.
- Sau đó hỏi thêm tối đa 1-2 thông tin quan trọng.
- Không chỉ hỏi mà không tư vấn gì.

Định dạng bắt buộc khi tư vấn lắp đặt:
### Nhận xét sơ bộ
Tóm tắt nhu cầu người dùng và nhận định ban đầu.

### Giải pháp có thể phù hợp
Gợi ý hướng như hòa lưới bám tải, hybrid hoặc có pin lưu trữ. Không khẳng định tuyệt đối.

### Cần thêm thông tin để tư vấn chính xác
Hỏi tối đa 2 ý, dùng gạch đầu dòng:
- Diện tích mái hoặc loại mái
- Khu vực lắp đặt hoặc mức dùng điện ban ngày/ban đêm

### Bước tiếp theo
Khuyến nghị khảo sát thực tế để tính công suất, thiết bị và chi phí phù hợp.

5. CẢNH BÁO AN TOÀN ĐIỆN
Nếu người dùng mô tả các dấu hiệu như:
chập điện, cháy nổ, mùi khét, inverter nóng bất thường, inverter báo lỗi nghiêm trọng, dây điện nóng, điện giật, tấm pin nứt vỡ, tia lửa, nước vào tủ điện hoặc khu vực điện bị ngập.

Trả lời đúng định dạng:
### 🚨 Cảnh báo an toàn điện
Nêu ngắn gọn rủi ro, không làm người dùng hoảng sợ quá mức.

### Việc nên làm ngay
- Ngừng sử dụng hệ thống.
- Ngắt nguồn nếu có thể thực hiện an toàn.
- Liên hệ kỹ thuật viên hoặc đơn vị lắp đặt để kiểm tra.

### Không nên làm
- Không tự tháo inverter.
- Không chạm vào dây dẫn, tủ điện hoặc khu vực có mùi khét, tia lửa, chập cháy hay ngập nước.

Kết thúc bằng đúng câu:
An toàn điện cần được ưu tiên trước, nên kiểm tra trực tiếp thay vì tiếp tục vận hành hệ thống.

6. CÂU HỎI VỀ GIÁ / BÁO GIÁ
- Không đưa báo giá chính thức nếu dữ liệu không có.
- Có thể nói các yếu tố ảnh hưởng chi phí: công suất, loại inverter, pin lưu trữ, loại mái, vật tư, khoảng cách thi công, yêu cầu an toàn, bảo hành.
- Khuyến nghị khảo sát để báo giá chính xác.

7. CÂU HỎI VỀ MKSOLAR
- Chỉ trả lời thông tin công ty nếu có trong dữ liệu RAG.
- Nếu không có dữ liệu, nói rõ: "Mình chưa có đủ dữ liệu để khẳng định thông tin này."
- Không tự bịa địa chỉ, số điện thoại, sản phẩm, chính sách hay giá bán.

YÊU CẦU CUỐI:
- Trả lời bằng tiếng Việt.
- Không quá dài nếu câu hỏi đơn giản.
- Không dùng giọng điệu bệnh viện/y khoa.
- Không dùng từ "chẩn đoán", "triệu chứng", "bác sĩ", "bệnh viện", "cấp cứu 115".
- Luôn phù hợp với bối cảnh chatbot tư vấn điện năng lượng mặt trời."""
 

# ═══════════════════════════════════════════════════════════
# STREAM VỚI RETRY + HIDDEN CoT FILTER
# ═══════════════════════════════════════════════════════════
def get_rag_context_sync(user_query: str, history_str: str) -> Tuple[str, str]:
    return _get_rag_context(user_query)


def stream_from_built_prompt(built_prompt: str, citation_text: str) -> Generator[str, None, None]:
    _, llm, _, _ = load_resources()
    if not llm:
        yield "⚠️ Hệ thống đang bảo trì. Vui lòng thử lại sau."
        return

    _throttle_api()
    MAX_RETRIES  = 3
    RETRY_DELAYS = [5, 15, 30]

    for attempt in range(MAX_RETRIES):
        try:
            raw_stream = llm.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=[{"role": "user", "content": built_prompt}],
                temperature=0.2,
                max_tokens=1024,
                stream=True
            )

            # Tạo generator trả về raw chunks
            def _raw_chunks():
                for chunk in raw_stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta

            # Lọc <thinking> trước khi yield ra UI
            full_response = ""
            for visible_chunk in _stream_strip_thinking(_raw_chunks()):
                full_response += visible_chunk
                yield visible_chunk

            return

        except Exception as e:
            err = str(e)
            if "429" in err:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAYS[attempt]
                    yield f"\n\n⏳ *AI đang bận, tự động thử lại sau {wait} giây...*\n\n"
                    time.sleep(wait)
                    global _last_api_call
                    _last_api_call = time.time()
                else:
                    yield "\n\n⚠️ AI quá tải, vui lòng gửi lại tin nhắn sau 1 phút."
            else:
                yield f"❌ Lỗi kết nối AI: {err}"
                return


def get_bot_response_stream(user_query: str, history: str) -> Generator[str, None, None]:
    _, llm, _, _ = load_resources()
    if not llm:
        yield "⚠️ Hệ thống đang bảo trì. Vui lòng thử lại sau."
        return
    context, citation_text = _get_rag_context(user_query)
    prompt = _build_prompt(user_query, history, context)
    yield from stream_from_built_prompt(prompt, citation_text)


def get_bot_response(user_query: str, history: str) -> str:
    _, llm, _, _ = load_resources()
    if not llm:
        return "⚠️ Hệ thống đang bảo trì. Vui lòng thử lại sau."

    context, citation_text = _get_rag_context(user_query)
    prompt = _build_prompt(user_query, history, context)
    _throttle_api()

    MAX_RETRIES  = 3
    RETRY_DELAYS = [5, 15, 30]

    for attempt in range(MAX_RETRIES):
        try:
            response  = llm.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            raw_text  = response.choices[0].message.content
            # Lọc <thinking> trong chế độ non-stream
            final_ans = _strip_thinking(raw_text)t
            return final_ans
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
            else:
                return f"❌ Lỗi khi gọi AI: {err}"

    return "⚠️ AI quá tải, vui lòng thử lại sau."


# ═══════════════════════════════════════════════════════════
# PRELOAD
# ═══════════════════════════════════════════════════════════
load_resources()