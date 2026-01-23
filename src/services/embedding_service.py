from langchain_huggingface import HuggingFaceEmbeddings
from src import config

def get_embedding_model():
    """
    Sử dụng model Local từ HuggingFace.
    Ưu điểm: KHÔNG CẦN API KEY, KHÔNG LO RATE LIMIT (Lỗi 429).
    Chạy trực tiếp trên CPU của máy tính.
    """
    print("🔄 Đang tải/khởi tạo model Embedding Local (HuggingFace)...")
    
    # Model này rất nhẹ (khoảng 80MB), tải lần đầu sẽ hơi lâu chút, lần sau chạy ngay lập tức.
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}, # Chạy trên CPU cho an toàn
        encode_kwargs={'normalize_embeddings': False}
    )
    
    print("✅ Đã tải xong Model Local!")
    return embeddings