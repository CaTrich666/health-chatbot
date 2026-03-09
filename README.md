# 🏥 Health Chatbot AI

link trang web: https://health-chatbot-tranhuutai.streamlit.app/

Dự án Chatbot tư vấn sức khỏe thông minh sử dụng kiến trúc **Hybrid RAG**.

Hệ thống được xây dựng dựa trên 3 trụ cột công nghệ chính:

1.  **Native Multimodal LLM (Đa phương thức):**
    * Sử dụng mô hình **Google Gemini 1.5 Flash** (`gemini-flash-latest`).
    * Hệ thống có thể xử lý **Văn bản** của người dùng.

2.  **Hybrid Search (Tìm kiếm Lai):**
    * **Vector Search (Đã hoàn thiện):** Sử dụng **ChromaDB** để tìm kiếm theo ý nghĩa câu hỏi (giúp AI hiểu được các mô tả triệu chứng, kể cả khi người dùng không dùng từ chuyên môn).
    * **Keyword Search & Fusion (Đã hoàn thiện):** Tích hợp thuật toán BM25 để bắt chính xác các danh từ riêng (tên thuốc, tên bệnh đặc thù). Hai luồng kết quả Vector và BM25 được gộp lại bằng thuật toán RRF (Reciprocal Rank Fusion) để cung cấp ngữ cảnh chính xác tuyệt đối cho AI.

3.  **Safety & Grounding (Kiểm soát Ảo giác):**
    * **Knowledge Base:** Chỉ sử dụng dữ liệu y văn uy tín (MedQuad) đã được kiểm chứng làm nguồn tham khảo.
    * **Robust Prompt Engineering:** Sử dụng kỹ thuật "Ra lệnh nghiêm ngặt" (Strict Prompting). Ép buộc AI chỉ được trả lời dựa trên dữ liệu được cung cấp, tuyệt đối không tự suy diễn hay bịa đặt thông tin y tế.

## 📂 Cấu Trúc Dự Án

```text
HEALTH_CHATBOT/
│
├── .streamlit/
│   └── secrets.toml          # Cấu hình bảo mật (API Key cho Cloud)
│
├── app/                      # GIAO DIỆN
│   ├── __init__.py
│   └── web_chat.py           # File Giao diện Chat
│
├── data/                     # DỮ LIỆU 
│   ├── chroma_db_diagnosis/  # Vector DB (Bộ nhớ AI - Kiến thức Y khoa)
│   ├── raw/                  # Dữ liệu thô (csv, jsonl)
│   └── chat_history.db       # SQLite: Lưu lịch sử chat & User
│
├── scripts/                  # CÔNG CỤ QUẢN TRỊ
│   ├── build_db.py           # Tool nạp dữ liệu vào Vector DB
│   └── check_models.py       # Tool kiểm tra trạng thái API Key
│
├── src/                      # LOGIC XỬ LÝ
│   ├── services/
│   │   ├── __init__.py
│   │   └── ai_service.py     # Xử lý AI
│   ├── utils/                # Các tiện ích mở rộng
│   ├── __init__.py
│   ├── config.py             # File cấu hình hệ thống (Model Name, Paths)
│   └── database.py           # Logic quản lý User, Đăng ký, Đăng nhập
│
├── .env                      # Biến môi trường (API Key Local)
├── requirements.txt          # Danh sách thư viện cần cài đặt
└── setup_database.py         # Script khởi tạo Database User ban đầu