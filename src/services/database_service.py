from langchain_community.vectorstores import Chroma
from src import config
import time

def get_db(embeddings):
    """Kết nối vào Database đã có"""
    return Chroma(persist_directory=config.CHROMA_DB_DIR, embedding_function=embeddings)

def create_db_from_documents(documents, embeddings):
    """Tạo Database mới từ dữ liệu (Có chế độ chờ 30s khi lỗi)"""
    print(f"\n⏳ Đang nạp {len(documents)} dòng dữ liệu vào ChromaDB...")
    
    db = Chroma(embedding_function=embeddings, persist_directory=config.CHROMA_DB_DIR)
    
    batch_size = 10
    total_docs = len(documents)
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        while True:
            try:
                db.add_documents(batch)
                print(f"   ✅ Đã nạp: {min(i+batch_size, total_docs)}/{total_docs}", end='\r')
                time.sleep(1.5) 
                break 
            except Exception as e:
                if "429" in str(e):
                    print(f"\n   😴 Google bận (Lỗi 429). Chờ 30s...   ", end='\r')
                    time.sleep(30)
                else:
                    print(f"\n   ❌ Lỗi lạ: {e}. Bỏ qua gói này.")
                    break
    
    print("\n🎉 Đã nạp xong toàn bộ!")
    return db