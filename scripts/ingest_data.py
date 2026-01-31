import sys
import os
import pandas as pd
from langchain_core.documents import Document

# --- CẤU HÌNH ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.services import embedding_service, database_service

# Kích thước mỗi gói nạp
BATCH_SIZE = 500 

def process_in_batches(documents, db_function, total_docs):
    """Hàm xử lý chia nhỏ dữ liệu"""
    if not documents: return
    
    total_batches = (total_docs // BATCH_SIZE) + 1
    print(f"📦 Dữ liệu lớn! Sẽ chia thành {total_batches} đợt để nạp (Mỗi đợt {BATCH_SIZE} dòng)...")

    for i in range(0, total_docs, BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        print(f"   ⏳ Đang nạp đợt {current_batch_num}/{total_batches} ({len(batch)} dòng)...")
        
        # Gọi hàm lưu của database_service
        db_function(batch) 
        print(f"   ✅ Đã lưu đợt {current_batch_num} vào ổ cứng.")

def main():
    print("🚀 BẮT ĐẦU NẠP DỮ LIỆU LỚN (LOCAL EMBEDDING)...")
    
    # 1. Khởi tạo Embedding Local
    embeddings = embedding_service.get_embedding_model()
    
    # 2. Đọc dữ liệu
    all_documents = []
    
    print(f"\n1️⃣  Đang đọc dữ liệu từ: {config.RAW_DATA_DIR}")
    
    # Đọc MedQuad
    if os.path.exists(config.PATH_MEDQUAD):
        print(f"   - Đọc MedQuad...")
        df_med = pd.read_csv(config.PATH_MEDQUAD)
        # Lọc dữ liệu lỗi (nếu có)
        df_med = df_med.dropna(subset=['question', 'answer'])
        for _, row in df_med.iterrows():
            text = f"Hỏi: {row['question']}\nĐáp: {row['answer']}"
            all_documents.append(Document(page_content=text, metadata={"source": "medquad"}))

    # Đọc Jsonl
    if os.path.exists(config.PATH_SYMPTOM):
        print(f"   - Đọc Symptom...")
        df_sym = pd.read_json(config.PATH_SYMPTOM, lines=True)
        df_sym = df_sym.dropna(subset=['input_text', 'output_text'])
        for _, row in df_sym.iterrows():
            text = f"Triệu chứng: {row['input_text']}\nBệnh: {row['output_text']}"
            all_documents.append(Document(page_content=text, metadata={"source": "symptom"}))
            
    total_docs = len(all_documents)
    print(f"👉 Tổng cộng: {total_docs} dòng dữ liệu.")
    
    # 3. Nạp cuốn chiếu
    if total_docs > 0:
        print("\n2️⃣  Bắt đầu Vector hóa & Lưu trữ...")
        
        # Hàm wrapper để gọi database service
        def save_batch(batch_docs):
            database_service.create_db_from_documents(batch_docs, embeddings)
            
        process_in_batches(all_documents, save_batch, total_docs)
        print("\n🎉 HOÀN THÀNH XUẤT SẮC! TOÀN BỘ 17.000 DÒNG ĐÃ VÀO KHO.")
    else:
        print("❌ Không tìm thấy dữ liệu.")

if __name__ == "__main__":
    main()