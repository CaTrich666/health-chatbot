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
    print("⏳ Đang khởi tạo tài nguyên AI (Chỉ chạy 1 lần)...")

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
             bm25_docs[i].metadata.get("source", "Tài liệu y khoa"),
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
            (d.page_content, d.metadata.get("source", "Tài liệu y khoa"))
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
        return "Không có dữ liệu cụ thể, dùng kiến thức y khoa tổng quát.", ""

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
# Đây là bước hậu xử lý ở backend, người dùng không thấy
# phần suy luận nội bộ, chỉ thấy câu trả lời sạch
# ═══════════════════════════════════════════════════════════
def _strip_thinking(text: str) -> str:
    """Xóa toàn bộ nội dung trong thẻ <thinking>...</thinking>"""
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    return cleaned.strip()


def _stream_strip_thinking(stream_generator) -> Generator[str, None, None]:
    """
    Lọc thẻ <thinking> trong luồng stream theo thời gian thực.
    Buffer tích lũy chunks cho đến khi thẻ đóng </thinking> xuất hiện
    thì bỏ toàn bộ, sau đó yield bình thường.
    """
    buffer        = ""
    inside_think  = False

    for chunk in stream_generator:
        buffer += chunk

        # Vòng lặp xử lý buffer cho đến khi không còn thẻ nào cần xử lý
        while True:
            if not inside_think:
                # Tìm thẻ mở
                start = buffer.find("<thinking>")
                if start == -1:
                    # Không có thẻ mở → yield toàn bộ buffer trừ 9 ký tự cuối
                    # (phòng trường hợp "<thinking" bị cắt giữa chunk)
                    safe_len = max(0, len(buffer) - 9)
                    if safe_len > 0:
                        yield buffer[:safe_len]
                        buffer = buffer[safe_len:]
                    break
                else:
                    # Yield phần trước thẻ mở, bắt đầu buffer từ thẻ mở
                    yield buffer[:start]
                    buffer       = buffer[start:]
                    inside_think = True
            else:
                # Đang trong thẻ thinking, tìm thẻ đóng
                end = buffer.find("</thinking>")
                if end == -1:
                    # Chưa thấy thẻ đóng → giữ buffer, chờ chunk tiếp theo
                    break
                else:
                    # Bỏ toàn bộ từ <thinking> đến </thinking>
                    buffer       = buffer[end + len("</thinking>"):]
                    inside_think = False

    # Yield phần còn lại sau khi stream kết thúc
    if buffer and not inside_think:
        yield buffer


# ═══════════════════════════════════════════════════════════
# BUILD PROMPT - HIDDEN CoT với XML Tag <thinking>
# Model suy luận trong <thinking>...</thinking>
# Backend lọc bỏ trước khi hiển thị ra UI
# ═══════════════════════════════════════════════════════════
def _build_prompt(user_query: str, history: str, context: str) -> str:
    history_trimmed = _trim_history_safe(history)

    return f"""Bạn là "Trợ lý ảo Sàng lọc Y tế" của một bệnh viện đa khoa uy tín.
Nhiệm vụ: LẮNG NGHE, ĐÁNH GIÁ mức độ nghiêm trọng, CUNG CẤP thông tin sơ bộ và HƯỚNG DẪN đến đúng chuyên khoa.
TUYỆT ĐỐI KHÔNG chẩn đoán xác định bệnh hay kê đơn thuốc. Xưng "Mình/Trợ lý" và gọi người dùng là "Bạn/Anh/Chị".

LỊCH SỬ TRÒ CHUYỆN:
{history_trimmed}

DỮ LIỆU Y KHOA RAG (Chỉ dùng SAU KHI đã đủ thông tin từ người dùng):
{context}
LƯU Ý: RAG có thể ở dạng tiếng Anh, hãy dịch tự nhiên sang tiếng Việt.
⛔ RAG TUYỆT ĐỐI KHÔNG được dùng để bù đắp thông tin còn thiếu từ người dùng.
Dù RAG có đầy đủ dữ liệu, nếu người dùng chỉ gõ "tôi bị đau đầu" thì vẫn PHẢI vào HƯỚNG 2.

CÂU HỎI CỦA NGƯỜI DÙNG:
{user_query}

════════════════════════════════════════════════════════════
HƯỚNG DẪN ĐỊNH DẠNG ĐẦU RA (QUAN TRỌNG):
Bắt buộc bắt đầu câu trả lời bằng thẻ <thinking> để suy luận nội bộ,
SAU ĐÓ mới viết câu trả lời cho người dùng bên ngoài thẻ.
Người dùng SẼ KHÔNG thấy nội dung trong thẻ <thinking>.

Ví dụ cấu trúc output:
<thinking>
[Toàn bộ suy luận nội bộ ở đây - người dùng không thấy]
</thinking>
[Câu trả lời thực sự cho người dùng ở đây]
════════════════════════════════════════════════════════════

BƯỚC 1 - SUY LUẬN NỘI BỘ (viết trong thẻ <thinking>, không hiện ra UI):
<thinking>
TỔNG HỢP ĐA LƯỢT: Gộp thông tin từ LỊCH SỬ + CÂU HỎI HIỆN TẠI.

BẢNG KIỂM TRA YẾU TỐ (chỉ đếm thông tin do NGƯỜI DÙNG cung cấp, KHÔNG đếm RAG):
• Yếu tố 1 - Triệu chứng cụ thể : CÓ/KHÔNG → [ghi rõ]
• Yếu tố 2 - Thời gian/Tần suất  : CÓ/KHÔNG → [ghi rõ]
• Yếu tố 3 - Dấu hiệu kèm/Đặc điểm: CÓ/KHÔNG → [ghi rõ]
• Dấu hiệu cấp cứu               : CÓ/KHÔNG → [ghi rõ]
• Tổng yếu tố: X/3
• Quyết định: HƯỚNG [0/1/2/3]
</thinking>

LUẬT PHÁ VÒNG LẶP: Nếu người dùng trả lời "không biết/không nhớ/không có"
→ Tính yếu tố đó là ĐÃ ĐÁP ỨNG, không hỏi lại.

NGOẠI LỆ CẤP CỨU (ghi đè mọi thứ): Nếu có từ khóa nguy hiểm
(khó thở đột ngột, đau ngực dữ dội, co giật, yếu liệt nửa người, xuất huyết ồ ạt)
→ Bỏ qua đếm yếu tố, vào HƯỚNG 3 với 🔴 KHẨN CẤP ngay.

BƯỚC 2 - CÂU TRẢ LỜI THỰC SỰ (viết bên ngoài thẻ <thinking>, người dùng sẽ thấy):

▶ HƯỚNG 0: NGOÀI PHẠM VI Y TẾ
"Mình chỉ hỗ trợ các vấn đề sức khỏe và y tế. Bạn có triệu chứng nào cần tư vấn không?"
(KHÔNG dùng ###)

▶ HƯỚNG 1: CHÀO HỎI / CẢM ƠN
Trả lời ngắn gọn thân thiện (KHÔNG dùng ###).

▶ HƯỚNG 2: THIẾU THÔNG TIN (Tổng < 2 VÀ không có dấu hiệu cấp cứu)
- KHÔNG đưa ra lời khuyên chuyên khoa.
- 1 câu đồng cảm + hỏi ĐÚNG 1 yếu tố còn thiếu:
  + Thiếu yếu tố 2 → "Triệu chứng này bắt đầu từ bao giờ? Liên tục hay từng cơn?"
  + Thiếu yếu tố 3 → "Ngoài ra có kèm sốt, buồn nôn hay dấu hiệu nào khác không?"
  + Thiếu cả 2 và 3 → chỉ hỏi yếu tố 2. Lượt sau mới hỏi yếu tố 3.
- KHÔNG hỏi quá 1 câu mỗi lượt.

▶ HƯỚNG 3: ĐẠT NGƯỠNG (Tổng >= 2 HOẶC có dấu hiệu cấp cứu)
⚠️ KHÔNG in ngoặc vuông [ ] hay ngoặc đơn ( ).

Viết 1-2 câu đồng cảm và trấn an tự nhiên.

### 🔍 Phân tích sơ bộ:
Tóm tắt triệu chứng và nguyên nhân dựa trên DỮ LIỆU RAG. Dùng: "Có khả năng", "Có thể là".

### 🚨 Mức độ & Cảnh báo:
Chỉ 1 dòng: icon màu + tên mức độ + lý do ngắn.
(🔴 Cấp cứu ngay / 🟡 Khám trong 24-48h / 🟢 Theo dõi tại nhà)

### 👉 Chuyên khoa đề xuất:
**Tên chuyên khoa ưu tiên 1**
- **Lý do:** Giải thích ngắn gọn.
- **Lưu ý:** Nhịn ăn sáng / Mang hồ sơ cũ / Cần người nhà đi cùng...

Tên chuyên khoa 2 nếu thực sự cần, nếu không thì bỏ qua.

⚠️ *Đây chỉ là thông tin hỗ trợ sàng lọc ban đầu, vui lòng đến cơ sở y tế để được Bác sĩ chẩn đoán chính xác nhất.*"""


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
            final_ans = _strip_thinking(raw_text)
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
# PRELOAD
# ═══════════════════════════════════════════════════════════
load_resources()