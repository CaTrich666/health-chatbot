import sys
import os

# 1. Thêm đường dẫn gốc để Python nhìn thấy thư mục src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src import config
from src.services import embedding_service, database_service

def start_chat():
    print("🤖 HEALTH CHATBOT (MANUAL RAG MODE)")
    print("-" * 40)
    
    # 2. Khởi tạo Embedding & Database
    print("1️⃣  Đang kết nối bộ nhớ...", end="\r")
    embeddings = embedding_service.get_embedding_model()
    
    try:
        db = database_service.get_db(embeddings)
        # Kiểm tra dữ liệu
        if db._collection.count() == 0:
            print("\n❌ LỖI: Bộ nhớ rỗng! Chạy 'python scripts/ingest_data.py' trước.")
            return
    except Exception as e:
        print(f"\n❌ Lỗi DB: {e}")
        return

    # 3. Khởi tạo Gemini
    print("2️⃣  Đang khởi động não bộ AI...", end="\r")
    llm = ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.3
    )
    
    print("\n✅ ĐÃ SẴN SÀNG! (Gõ 'exit' để thoát)")
    print("-" * 40)

    # 4. Vòng lặp Chat
    while True:
        query = input("\n👤 Bạn: ")
        if query.lower() in ["exit", "thoát", "quit"]:
            print("👋 Tạm biệt!")
            break
        
        if not query.strip(): continue
        
        print("🤖 Bot đang suy nghĩ...", end="\r")
        
        try:
            # --- BƯỚC 1: TÌM KIẾM DỮ LIỆU (RETRIEVAL) ---
            # Tìm 3 đoạn văn bản liên quan nhất từ ChromaDB
            docs = db.similarity_search(query, k=3)
            
            if not docs:
                print(" " * 30, end="\r")
                print("🤖 Bot: Xin lỗi, tôi không tìm thấy thông tin liên quan trong dữ liệu.")
                continue

            # Ghép nội dung các đoạn tìm được thành 1 văn bản dài (Context)
            context_text = "\n\n---\n\n".join([d.page_content for d in docs])
            
            # --- BƯỚC 2: TẠO CÂU NHẮC (PROMPT) ---
            prompt = f"""Bạn là một trợ lý y tế thông minh. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp dưới đây.
            
            THÔNG TIN THAM KHẢO:
            {context_text}
            
            CÂU HỎI CỦA NGƯỜI DÙNG:
            {query}
            
            YÊU CẦU:
            - Trả lời ngắn gọn, chính xác dựa trên thông tin tham khảo.
            - Nếu thông tin tham khảo không đủ, hãy nói "Tôi không có dữ liệu về vấn đề này".
            
            TRẢ LỜI:"""
            
            # --- BƯỚC 3: GỬI CHO AI TRẢ LỜI (GENERATION) ---
            response = llm.invoke(prompt)
            
            # Xóa dòng "đang suy nghĩ"
            print(" " * 30, end="\r")
            
            # In câu trả lời
            print(f"🤖 Bot: {response.content}")
            
        except Exception as e:
            print(f"\n❌ Lỗi xử lý: {e}")

if __name__ == "__main__":
    start_chat()