import os
import zipfile
import time
import pickle
import streamlit as st
from typing import Generator, Optional, Tuple, List
from collections import defaultdict

# 👇 [CẬP NHẬT] Import Groq thay vì genai
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src import config

# ── Tokenizer tiếng Việt ──────────────────────────────────
# Dùng underthesea nếu có, fallback về split() nếu chưa cài
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
# THROTTLE - TRÁNH GỌI QUÁ NHANH (Giữ nguyên cho an toàn)
# ═══════════════════════════════════════════════════════════
_last_gemini_call: float = 0.0
_MIN_INTERVAL_SEC: float = 2.0

def _throttle_gemini():
    global _last_gemini_call
    elapsed = time.time() - _last_gemini_call
    if elapsed < _MIN_INTERVAL_SEC:
        time.sleep(_MIN_INTERVAL_SEC - elapsed)
    _last_gemini_call = time.time()


# ═══════════════════════════════════════════════════════════
# CACHE TÀI NGUYÊN - CHỈ CHẠY 1 LẦN DUY NHẤT
# ═══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_resources() -> Tuple[Optional[object], Optional[object], Optional[object], Optional[list]]:
    t_start = time.perf_counter()
    print("⏳ Đang khởi tạo tài nguyên AI (Chỉ chạy 1 lần)...")

    _extract_db_if_needed()

    # 👇 [CẬP NHẬT] Khởi tạo Groq LLM thay vì Gemini
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

    # Vector DB (Giữ nguyên)
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

    # Load BM25 từ file .pkl (Giữ nguyên)
    bm25_index = None
    bm25_docs  = None
    if os.path.exists(config.BM25_INDEX_PATH):
        try:
            with open(config.BM25_INDEX_PATH, "rb") as f:
                bm25_index, bm25_docs = pickle.load(f)
            print(f"✅ Đã load BM25 index ({len(bm25_docs)} documents) từ {config.BM25_INDEX_PATH}")
        except Exception as e:
            print(f"❌ Lỗi load BM25: {e}")
    else:
        print(f"⚠️ Không tìm thấy BM25 index tại {config.BM25_INDEX_PATH}")
        print("   → Hãy chạy script ingest để tạo file .pkl trước.")

    print(f"✅ Khởi tạo xong trong {time.perf_counter() - t_start:.2f}s")
    return vector_db, llm, bm25_index, bm25_docs


# ═══════════════════════════════════════════════════════════
# BM25 SEARCH - load từ .pkl, tokenize bằng underthesea
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
            (
                bm25_docs[i].page_content,
                bm25_docs[i].metadata.get("source", "Tài liệu y khoa"),
                float(scores[i]),
            )
            for i in top_indices
            if scores[i] > 0
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
    k:              int = None,
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
            (d.page_content, d.metadata.get("source", "Tài liệu y khoa"))
            for d in docs
        )
        st.session_state[cache_key] = result
        return result
    except Exception as e:
        print(f"❌ Lỗi tìm kiếm vector: {e}")
        return ()


# ═══════════════════════════════════════════════════════════
# RAG CONTEXT - Vector + BM25 + RRF
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
            print("⚠️ BM25 index chưa được load, dùng vector search thuần")

    if not fused_results:
        return "Không có dữ liệu cụ thể, dùng kiến thức y khoa tổng quát.", ""

    context_parts     = []
    sources           = []
    total_len         = 0
    MAX_CONTEXT_CHARS = 1500

    for content, source in fused_results:
        if total_len >= MAX_CONTEXT_CHARS:
            break
        snippet = content[:MAX_CONTEXT_CHARS - total_len]
        context_parts.append(snippet)
        total_len += len(snippet)
        if source not in sources:
            sources.append(source)

    context       = "\n---\n".join(context_parts)
    citation_text = "\n\n---\n**📚 Nguồn tham khảo:** " + " | ".join(sources)
    return context, citation_text


# ═══════════════════════════════════════════════════════════
# BUILD PROMPT (👇 ĐÃ CẬP NHẬT THÀNH "TRỢ LÝ SÀNG LỌC CHUYÊN KHOA")
# ═══════════════════════════════════════════════════════════
def _build_prompt(user_query: str, history: str, context: str) -> str:
    history_trimmed = history[-500:] if len(history) > 500 else history
    return f"""Bạn là "Trợ lý ảo Sàng lọc Y tế" của một bệnh viện đa khoa lớn.
Nhiệm vụ của bạn KHÔNG PHẢI là chẩn đoán bệnh hay kê đơn thuốc, mà là LẮNG NGHE triệu chứng, CUNG CẤP thông tin sơ bộ và HƯỚNG DẪN BỆNH NHÂN ĐẾN ĐÚNG CHUYÊN KHOA.
Luôn xưng hô ân cần, lịch sự (xưng là "Trợ lý/Mình" và gọi người dùng là "Bạn/Anh/Chị").

LỊCH SỬ TRÒ CHUYỆN GẦN ĐÂY:
{history_trimmed}

DỮ LIỆU Y KHOA ĐÁNG TIN CẬY ĐƯỢC CUNG CẤP:
{context}

CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG:
{user_query}

HƯỚNG DẪN TRẢ LỜI:
1. Nếu câu hỏi là lời chào/cảm ơn/hỏi thăm thông thường: Trả lời ngắn gọn, thân thiện, KHÔNG dùng format có dấu ###.
2. Nếu câu hỏi về triệu chứng hoặc sức khỏe, BẮT BUỘC trả lời theo ĐÚNG format sau:

(1-2 câu mở đầu thể hiện sự đồng cảm với tình trạng của người dùng)

### 🔍 Phân tích sơ bộ:
(Dựa VÀO DỮ LIỆU ĐƯỢC CUNG CẤP, tóm tắt lại các triệu chứng và giải thích ngắn gọn nguyên nhân thông thường có thể gây ra tình trạng này).

### 🩺 Có thể bạn đang gặp phải:
(Chỉ liệt kê các hội chứng/vấn đề khả dĩ nhất dựa trên dữ liệu. Dùng từ ngữ cẩn trọng như "Có khả năng", "Có thể là", tuyệt đối không khẳng định chắc chắn 100%).

### 🚨 Dấu hiệu nguy hiểm cần lưu ý (Red flags):
(Nếu có các triệu chứng đi kèm nào cần cấp cứu ngay, hãy liệt kê ra. Nếu không, có thể bỏ qua phần này).

### 👉 Chuyên khoa đề xuất thăm khám:
**[TÊN CHUYÊN KHOA]** (Ví dụ: Khoa Tiêu hóa, Khoa Hô hấp, Khoa Thần kinh...)
- **Lý do:** (Giải thích ngắn gọn tại sao triệu chứng này lại cần bác sĩ chuyên khoa đó khám).
- **Lời khuyên trước khi đi khám:** (Ví dụ: Nhịn ăn sáng nếu cần xét nghiệm máu, mang theo sổ khám bệnh cũ...).

(Kết thúc bằng một lời khuyên chân thành: Nhấn mạnh rằng đây chỉ là tư vấn tham khảo, người dùng cần đến gặp bác sĩ trực tiếp để được thăm khám chính xác nhất)."""


# ═══════════════════════════════════════════════════════════
# STREAM VỚI RETRY + THROTTLE (👇 ĐÃ CẬP NHẬT GỌI GROQ)
# ═══════════════════════════════════════════════════════════
def get_rag_context_sync(user_query: str, history_str: str) -> Tuple[str, str]:
    return _get_rag_context(user_query)


def stream_from_built_prompt(built_prompt: str, citation_text: str) -> Generator[str, None, None]:
    _, llm, _, _ = load_resources()

    if not llm:
        yield "⚠️ Hệ thống đang bảo trì. Vui lòng thử lại sau."
        return

    _throttle_gemini()

    MAX_RETRIES  = 3
    RETRY_DELAYS = [5, 15, 30]

    for attempt in range(MAX_RETRIES):
        try:
            # 👇 Gọi API Chat của Groq (Llama 3) thay vì genai.generate_content
            response_stream = llm.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=[{"role": "user", "content": built_prompt}],
                temperature=0.2,
                max_tokens=1024,
                stream=True
            )
            
            full_response   = ""
            for chunk in response_stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    yield delta

            if citation_text and "###" in full_response:
                yield citation_text
            return

        except Exception as e:
            err = str(e)
            if "429" in err:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAYS[attempt]
                    yield f"\n\n⏳ *AI đang bận, tự động thử lại sau {wait} giây...*\n\n"
                    time.sleep(wait)
                    global _last_gemini_call
                    _last_gemini_call = time.time()
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

    _throttle_gemini()

    MAX_RETRIES  = 3
    RETRY_DELAYS = [5, 15, 30]

    for attempt in range(MAX_RETRIES):
        try:
            # 👇 Gọi API Đồng bộ của Groq
            response = llm.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            final_ans = response.choices[0].message.content
            
            if citation_text and "###" in final_ans:
                return final_ans + citation_text
            return final_ans
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
            else:
                return f"❌ Lỗi khi gọi AI: {err}"

    return "⚠️ AI quá tải, vui lòng thử lại sau."


# ═══════════════════════════════════════════════════════════
# PRELOAD KHI KHỞI ĐỘNG
# ═══════════════════════════════════════════════════════════
load_resources()