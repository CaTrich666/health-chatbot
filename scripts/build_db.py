import sys
import os
import pandas as pd
import pickle
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CẤU HÌNH ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

BATCH_SIZE = 100

# ==========================================
# TOKENIZER TIẾNG VIỆT
# PHẢI dùng cùng tokenizer với ai_service.py
# ==========================================
try:
    from underthesea import word_tokenize as vi_tokenize
    def _tokenize(text: str) -> list:
        return vi_tokenize(text.lower(), format="text").split()
    print("✅ Dùng underthesea tokenizer (tiếng Việt)")
except ImportError:
    def _tokenize(text: str) -> list:
        return text.lower().split()
    print("⚠️ underthesea chưa cài, dùng split() tạm thời")
    print("   → Cài bằng: pip install underthesea")


# ==========================================
# 1. EMBEDDING MODEL
# ==========================================
def get_embedding_model():
    print("🔄 Đang tải model Vector (HuggingFace)...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


# ==========================================
# 2. LƯU CHROMADB
# ==========================================
def save_batch_to_chroma(documents, embeddings):
    if not documents:
        return
    db = Chroma(
        persist_directory=config.CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    db.add_documents(documents)


# ==========================================
# 3. LOGIC CHÍNH
# ==========================================
def main():
    print("🚀 BẮT ĐẦU QUY TRÌNH NẠP DỮ LIỆU HYBRID (VECTOR + BM25)...")

    embeddings    = get_embedding_model()
    all_documents = []

    print(f"\n1️⃣  Đang đọc dữ liệu từ: {config.RAW_DATA_DIR}")

    if os.path.exists(config.PATH_MEDQUAD):
        print(f"   - Đang đọc MedQuad.csv...")
        try:
            df_med = pd.read_csv(config.PATH_MEDQUAD).dropna(subset=['question', 'answer'])
            for _, row in df_med.iterrows():
                text = f"Hỏi: {row['question']}\nĐáp: {row['answer']}"
                all_documents.append(Document(page_content=text, metadata={"source": "medquad"}))
            print(f"     -> Lấy được {len(df_med)} dòng.")
        except Exception as e:
            print(f"     ❌ Lỗi đọc MedQuad: {e}")

    if os.path.exists(config.PATH_SYMPTOM):
        print(f"   - Đang đọc train.jsonl...")
        try:
            df_sym = pd.read_json(config.PATH_SYMPTOM, lines=True).dropna(subset=['input_text', 'output_text'])
            for _, row in df_sym.iterrows():
                text = f"Triệu chứng: {row['input_text']}\nBệnh: {row['output_text']}"
                all_documents.append(Document(page_content=text, metadata={"source": "symptom"}))
            print(f"     -> Lấy được {len(df_sym)} dòng.")
        except Exception as e:
            print(f"     ❌ Lỗi đọc Jsonl: {e}")

    total_docs = len(all_documents)
    print(f"👉 TỔNG CỘNG: {total_docs} tài liệu cần xử lý.")

    if total_docs == 0:
        print("❌ Không có dữ liệu nào để nạp.")
        return

    # ── BƯỚC 1: Tạo BM25 index ──────────────────────────
    print("\n2️⃣  Đang tạo chỉ mục BM25 (tokenizer tiếng Việt)...")

    # ✅ FIX 1: dùng _tokenize() thay vì .split()
    tokenized_corpus = [_tokenize(doc.page_content) for doc in all_documents]

    # ✅ FIX 2: truyền k1, b từ config thay vì dùng mặc định
    bm25 = BM25Okapi(tokenized_corpus, k1=config.BM25_K1, b=config.BM25_B)

    os.makedirs(os.path.dirname(config.BM25_INDEX_PATH), exist_ok=True)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump((bm25, all_documents), f)
    print(f"   ✅ Đã lưu BM25 tại: {config.BM25_INDEX_PATH}")

    # ── BƯỚC 2: Lưu ChromaDB ────────────────────────────
    print("\n3️⃣  Bắt đầu Vector hóa & Lưu trữ ChromaDB...")
    total_batches = (total_docs // BATCH_SIZE) + 1

    for i in range(0, total_docs, BATCH_SIZE):
        batch             = all_documents[i : i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        print(f"   ⏳ Đang xử lý đợt {current_batch_num}/{total_batches} ({len(batch)} dòng)...", end='\r')
        save_batch_to_chroma(batch, embeddings)

    print(f"\n\n🎉 XONG TOÀN BỘ! Vector DB + BM25 ({config.BM25_INDEX_PATH}) đã sẵn sàng.")


if __name__ == "__main__":
    main()