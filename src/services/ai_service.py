import os
import zipfile
import sys
import google.generativeai as genai  # Dùng thư viện gốc để xử lý ảnh tốt hơn
from langchain_huggingface import HuggingFaceEmbeddings 
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

    # 1. Cấu hình API Key cho Google GenAI
    api_key = os.environ.get("GOOGLE_API_KEY") or config.GOOGLE_API_KEY
    if not api_key:
        print("❌ Lỗi: Không tìm thấy Google API Key.")
        return None, None
        
    genai.configure(api_key=api_key)

    # 2. Xử lý giải nén DB
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

    # 3. Load Vector DB
    if os.path.exists(config.CHROMA_DB_DIR):
        try:
            # Lấy tên model embedding từ config
            embedding_model = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL, 
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False} # Fix lỗi dimension
            )
            _vector_db = Chroma(persist_directory=config.CHROMA_DB_DIR, embedding_function=embedding_model)
        except Exception as e:
            print(f"Lỗi load ChromaDB: {e}")
    
    # 4. Load LLM (Gemini Native)
    # Dùng thư viện gốc để hỗ trợ Multimodal (Text + Image) dễ dàng
    try:
        _llm = genai.GenerativeModel(config.CHAT_MODEL)
    except Exception as e:
        print(f"Lỗi khởi tạo Gemini: {e}")
        
    return _vector_db, _llm

def get_bot_response(user_query, history, image=None):
    """
    Hàm xử lý logic RAG và Prompt Engineering.
    Thêm tham số: image (Mặc định là None)
    """
    vector_db, llm = load_resources()
    
    # 1. Kiểm tra kết nối
    if not llm: 
        return "⚠️ Hệ thống đang bảo trì (Chưa kết nối Database/AI hoặc sai API Key)."
    
    try:
        # 2. RAG - Tìm kiếm dữ liệu y khoa
        # (Vẫn tìm kiếm văn bản để lấy kiến thức nền, kể cả khi có ảnh)
        context = ""
        citation_text = ""
        sources = []

        if vector_db:
            docs = vector_db.similarity_search(user_query, k=4)
            if docs:
                for doc in docs:
                    src = doc.metadata.get("source", "Tài liệu y khoa")
                    if src not in sources: sources.append(src)
                context = "\n".join([d.page_content for d in docs])
                citation_text = "\n\n---\n**📚 Tài liệu tham khảo:**\n" + "\n".join([f"- {s}" for s in sources])
            else:
                context = "Không có dữ liệu cụ thể trong kho, dùng kiến thức y khoa tổng quát."
        else:
            context = "Chế độ không có Database (Chỉ dùng kiến thức của AI)."
        
        # 3. PROMPT ENGINEERING (Điều chỉnh nhẹ để nhận diện ảnh)
        system_instruction = f"""
        VAI TRÒ: Bạn là "Người Bạn Bác Sĩ" (Health Chatbot) - Giỏi chuyên môn nhưng nói chuyện ân cần như bạn thân (Xưng hô: Mình - Bạn/Cậu).

        DỮ LIỆU ĐẦU VÀO:
        - Lịch sử chat: {history}
        - Kiến thức Y khoa tham khảo (RAG): {context}
        - User vừa nói: "{user_query}"
        {'- ⚠️ LƯU Ý: USER CÓ GỬI KÈM MỘT HÌNH ẢNH. HÃY QUAN SÁT KỸ ẢNH ĐỂ CHẨN ĐOÁN.' if image else ''}

        NHIỆM VỤ: HÃY XÁC ĐỊNH Ý ĐỊNH CỦA USER ĐỂ CHỌN 1 TRONG 2 CHẾ ĐỘ TRẢ LỜI:

        ════════════════════════════════════════════════════
        MODE 1: XÃ GIAO / CẢM XÚC (Khi User chào, cảm ơn, khen, chê, tạm biệt)
        ════════════════════════════════════════════════════
        - Nếu User nói "Hello", "Hi", "Chào": Trả lời vui vẻ, ngắn gọn.
        - Nếu User nói "Cảm ơn": Đáp lại lịch sự.
        - QUY TẮC CỨNG: KHÔNG được dùng cấu trúc khám bệnh (###) trong trường hợp này.

        ════════════════════════════════════════════════════
        MODE 2: TƯ VẤN Y KHOA (Khi User kể bệnh, hỏi thuốc, lo lắng HOẶC GỬI ẢNH)
        ════════════════════════════════════════════════════
        Áp dụng đúng cấu trúc chuẩn đoán 5 phần dưới đây.
        
        *LƯU Ý QUAN TRỌNG:* - Nếu có ảnh: Hãy mô tả những gì bạn thấy trong ảnh (sưng, đỏ, mủ, toa thuốc...) ở phần Phân tích sơ bộ.
        - Nếu User ĐÃ cung cấp đủ thông tin (Vị trí đau, thời gian, mức độ...), phần "Hỏi thêm" hãy nói: "Dựa trên thông tin và hình ảnh cậu gửi, mình đã nắm khá rõ tình hình."

        BẮT BUỘC SỬ DỤNG ĐỊNH DẠNG MARKDOWN SAU CHO MODE 2:

        (Lời dẫn dắt cảm thông, ân cần)

        ### 🔍 Phân tích sơ bộ:
        (Phân tích kỹ các triệu chứng user kể VÀ chi tiết hình ảnh nếu có, kết hợp với dữ liệu RAG)

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
        
        # 4. GỌI AI (XỬ LÝ ĐA PHƯƠNG THỨC)
        try:
            if image:
                # Nếu có ảnh, gửi cả prompt lẫn ảnh vào list
                response = llm.generate_content([system_instruction, image])
            else:
                # Nếu chỉ có text
                response = llm.generate_content(system_instruction)
            
            final_ans = response.text

        except Exception as e:
            return f"❌ Lỗi khi gọi Gemini: {str(e)}"

        # 5. BỘ LỌC THÔNG MINH
        has_sources = len(sources) > 0
        is_diagnosis = "###" in final_ans
        
        if has_sources and is_diagnosis:
            return final_ans + citation_text
        else:
            return final_ans 

    except Exception as e: return f"❌ Lỗi xử lý hệ thống: {e}"