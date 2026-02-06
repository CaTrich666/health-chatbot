import sys
import os
import time
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CẤU HÌNH ---
# Hack đường dẫn để import được src.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

# Kích thước mỗi gói nạp (để tránh tràn RAM hoặc lỗi)
BATCH_SIZE = 100 

# ==========================================
# 1. HÀM TỪ EMBEDDING SERVICE (CŨ)
# ==========================================
def get_embedding_model():
    print("🔄 Đang tải model Embedding Local (HuggingFace)...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

# ==========================================
# 2. HÀM TỪ DATABASE SERVICE (CŨ)
# ==========================================
def save_batch_to_chroma(documents, embeddings):
    """Nạp một gói dữ liệu vào ChromaDB"""
    if not documents: return
    
    # Kết nối DB
    db = Chroma(
        persist_directory=config.CHROMA_DB_DIR, 
        embedding_function=embeddings
    )
    
    # Thêm dữ liệu
    db.add_documents(documents)
    # Chroma bản mới tự động persist, nhưng gọi cho chắc nếu dùng bản cũ
    try: db.persist() 
    except: pass 

# ==========================================
# 3. LOGIC CHÍNH (TỪ INGEST DATA)
# ==========================================
def main():
    print("🚀 BẮT ĐẦU QUY TRÌNH NẠP DỮ LIỆU (All-in-One)...")
    
    # A. Khởi tạo Model
    embeddings = get_embedding_model()
    
    # B. Đọc dữ liệu thô
    all_documents = []
    print(f"\n1️⃣  Đang đọc dữ liệu từ: {config.RAW_DATA_DIR}")
    
    # --- Đọc MedQuad ---
    if os.path.exists(config.PATH_MEDQUAD):
        print(f"   - Đang đọc MedQuad.csv...")
        try:
            df_med = pd.read_csv(config.PATH_MEDQUAD)
            df_med = df_med.dropna(subset=['question', 'answer'])
            for _, row in df_med.iterrows():
                text = f"Hỏi: {row['question']}\nĐáp: {row['answer']}"
                all_documents.append(Document(page_content=text, metadata={"source": "medquad"}))
            print(f"     -> Lấy được {len(df_med)} dòng.")
        except Exception as e:
            print(f"     ❌ Lỗi đọc MedQuad: {e}")

    # --- Đọc Jsonl ---
    if os.path.exists(config.PATH_SYMPTOM):
        print(f"   - Đang đọc train.jsonl...")
        try:
            df_sym = pd.read_json(config.PATH_SYMPTOM, lines=True)
            df_sym = df_sym.dropna(subset=['input_text', 'output_text'])
            for _, row in df_sym.iterrows():
                text = f"Triệu chứng: {row['input_text']}\nBệnh: {row['output_text']}"
                all_documents.append(Document(page_content=text, metadata={"source": "symptom"}))
            print(f"     -> Lấy được {len(df_sym)} dòng.")
        except Exception as e:
            print(f"     ❌ Lỗi đọc Jsonl: {e}")

    total_docs = len(all_documents)
    print(f"👉 TỔNG CỘNG: {total_docs} tài liệu cần xử lý.")

    # C. Chia nhỏ và Nạp vào DB
    if total_docs > 0:
        print("\n2️⃣  Bắt đầu Vector hóa & Lưu trữ...")
        
        total_batches = (total_docs // BATCH_SIZE) + 1
        
        for i in range(0, total_docs, BATCH_SIZE):
            batch = all_documents[i : i + BATCH_SIZE]
            current_batch_num = (i // BATCH_SIZE) + 1
            
            print(f"   ⏳ Đang xử lý đợt {current_batch_num}/{total_batches} ({len(batch)} dòng)...", end='\r')
            
            # Gọi hàm lưu
            save_batch_to_chroma(batch, embeddings)
            
            # Nghỉ 1 chút để CPU thở (nếu máy yếu)
            time.sleep(0.1) 
            
        print(f"\n\n🎉 XONG! Đã lưu toàn bộ vào: {config.CHROMA_DB_DIR}")
    else:
        print("❌ Không có dữ liệu nào để nạp.")

if __name__ == "__main__":
    main()