import os
import zipfile
import sys
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
# Import config để lấy biến môi trường, tên model
from src import config

# --- BIẾN TOÀN CỤC ĐỂ CACHE MODEL (Tránh load lại nhiều lần) ---
_vector_db = None
_llm = None

def load_resources():
    """Hàm khởi tạo Model và Database"""
    global _vector_db, _llm
    
    # Nếu đã load rồi thì trả về luôn (Singleton Pattern)
    if _vector_db is not None and _llm is not None:
        return _vector_db, _llm

    # 1. Xử lý giải nén DB
    db_path = os.path.join(config.CHROMA_DB_DIR, "chroma.sqlite3")
    zip_path = db_path + ".zip"
    
    # Logic giải nén zip lồng nhau nếu có
    if not os.path.exists(zip_path):
        zip_path_double = db_path + ".zip.zip"
        if os.path.exists(zip_path_double):
            zip_path = zip_path_double

    if not os.path.exists(db_path) and os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(config.CHROMA_DB_DIR)
        except Exception as e:
            print(f"Lỗi giải nén: {e}")

    # 2. Load Vector DB
    if os.path.exists(config.CHROMA_DB_DIR):
        try:
            #Lấy tên model embedding từ config
            embedding_model = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL, 
                model_kwargs={'device': 'cpu'}
            )
            _vector_db = Chroma(persist_directory=config.CHROMA_DB_DIR, embedding_function=embedding_model)
        except Exception as e:
            print(f"Lỗi load ChromaDB: {e}")
    
    # 3. Load LLM (Gemini)
    # Ưu tiên lấy từ biến môi trường hệ thống (Streamlit Cloud), nếu không có thì lấy từ file .env qua config
    api_key = os.environ.get("GOOGLE_API_KEY") or config.GOOGLE_API_KEY
    
    if api_key:
        # CẬP NHẬT: Lấy tên model chat từ config
        _llm = ChatGoogleGenerativeAI(
            model=config.CHAT_MODEL, 
            temperature=0.3, 
            google_api_key=api_key
        )
        
    return _vector_db, _llm

def get_bot_response(user_query, history):
    """Hàm xử lý logic RAG và Prompt Engineering"""
    vector_db, llm = load_resources()
    
    # 1. Kiểm tra kết nối
    if not vector_db or not llm: 
        return "⚠️ Hệ thống đang bảo trì (Chưa kết nối Database/AI hoặc sai API Key)."
    
    try:
        # 2. RAG - Tìm kiếm dữ liệu y khoa
        docs = vector_db.similarity_search(user_query, k=4)
        sources = []
        context = ""
        citation_text = ""
        
        # Chỉ khi tìm thấy tài liệu thì mới tạo list nguồn
        if docs:
            for doc in docs:
                src = doc.metadata.get("source", "Tài liệu y khoa")
                if src not in sources: sources.append(src)
            context = "\n".join([d.page_content for d in docs])
            citation_text = "\n\n---\n**📚 Tài liệu tham khảo:**\n" + "\n".join([f"- {s}" for s in sources])
        else:
            context = "Không có dữ liệu cụ thể trong kho, dùng kiến thức y khoa tổng quát."
            citation_text = "" # Không có nguồn thì chuỗi rỗng
        
        # 3. PROMPT ENGINEERING
        prompt = f"""
        VAI TRÒ: Bạn là "Người Bạn Bác Sĩ" (Health Chatbot) - Giỏi chuyên môn nhưng nói chuyện ân cần như bạn thân (Xưng hô: Mình - Bạn/Cậu).

        DỮ LIỆU ĐẦU VÀO:
        - Lịch sử chat: {history}
        - Kiến thức Y khoa (RAG): {context}
        - User vừa nói: "{user_query}"

        NHIỆM VỤ: HÃY XÁC ĐỊNH Ý ĐỊNH CỦA USER ĐỂ CHỌN 1 TRONG 2 CHẾ ĐỘ TRẢ LỜI:

        ════════════════════════════════════════════════════
        MODE 1: XÃ GIAO / CẢM XÚC (Khi User chào, cảm ơn, khen, chê, tạm biệt)
        ════════════════════════════════════════════════════
        - Nếu User nói "Hello", "Hi", "Chào": Trả lời vui vẻ, ngắn gọn, hỏi thăm sức khỏe.
        - Nếu User nói "Cảm ơn", "Hay quá", "Ok": Đáp lại lịch sự (VD: "Không có chi, chúc bạn mau khỏe nhé!").
        - QUY TẮC CỨNG: KHÔNG được dùng cấu trúc khám bệnh (###) trong trường hợp này.

        ════════════════════════════════════════════════════
        MODE 2: TƯ VẤN Y KHOA (Khi User kể bệnh, hỏi thuốc, lo lắng)
        ════════════════════════════════════════════════════
        Áp dụng đúng cấu trúc chuẩn đoán 5 phần dưới đây.
        
        *LƯU Ý THÔNG MINH:* - Hãy kiểm tra Lịch sử chat. Nếu User ĐÃ cung cấp đủ thông tin (Vị trí đau, thời gian, mức độ, tiền sử...), thì ở phần "Hỏi thêm" hãy nói: "Dựa trên thông tin cậu kể, mình đã nắm khá rõ tình hình." và không cần đặt câu hỏi nữa.
        - Nếu thiếu thông tin, hãy đặt 2-3 câu hỏi quan trọng nhất.

        BẮT BUỘC SỬ DỤNG ĐỊNH DẠNG MARKDOWN SAU CHO MODE 2:

        (Lời dẫn dắt cảm thông, ân cần)

        ### 🔍 Phân tích sơ bộ:
        (Phân tích kỹ các triệu chứng user kể, kết hợp với dữ liệu RAG)

        ### 🩺 Để hiểu rõ hơn, cậu cho mình hỏi thêm nhé:
        (Nếu thiếu tin: Đặt câu hỏi / Nếu đủ tin: Ghi "Thông tin đã khá đầy đủ để chẩn đoán.")

        ### 💡 Có thể cậu đang gặp vấn đề về:
        (Đưa ra các giả thuyết bệnh lý, xếp theo thứ tự khả năng cao nhất)

        ### 👉 Chuyên khoa cậu nên ghé khám:
        **[TÊN CHUYÊN KHOA]** - (Giải thích ngắn gọn tại sao)

        ### 📝 Lưu ý cho cậu:
        - **Chăm sóc:** (Lời khuyên ăn uống, nghỉ ngơi)
        - **Thuốc men:** (Gợi ý nhóm thuốc không kê đơn nếu an toàn)
        - **Cảnh báo:** (Dấu hiệu nguy hiểm cần đi cấp cứu ngay)

        (Lời chúc sức khỏe cuối cùng)
        """
        
        # 4. GỌI AI
        response = llm.invoke(prompt)
        
        # Xử lý kết quả trả về
        content = response.content
        final_ans = ""
        if isinstance(content, list):
            for item in content:
                if isinstance(item, str): final_ans += item
                elif isinstance(item, dict) and 'text' in item: final_ans += item['text']
        else:
            final_ans = str(content)
            
        # 5. BỘ LỌC THÔNG MINH
        # Điều kiện hiển thị nguồn:
        # 1. Has Source: Phải tìm thấy nguồn thực tế trong kho (len(sources) > 0)
        # 2. Is Diagnosis: Phải là bài tư vấn bệnh (Có chứa dấu hiệu chẩn đoán "###")
        
        has_sources = len(sources) > 0
        is_diagnosis = "###" in final_ans
        
        if has_sources and is_diagnosis:
            return final_ans + citation_text
        else:
            return final_ans # Xã giao hoặc không tìm thấy nguồn -> Không hiện

    except Exception as e: return f"❌ Lỗi xử lý: {e}"