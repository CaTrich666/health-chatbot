# 🏥 Health Chatbot AI

link trang web: https://health-chatbot-tranhuutai.streamlit.app/

Dự án Chatbot tư vấn sức khỏe thông minh sử dụng kiến trúc **Hybrid RAG**.

## Kiến trúc Hệ thống (3 Trụ cột Công nghệ)

Hệ thống được thiết kế theo kiến trúc RAG, xây dựng dựa trên 3 trụ cột công nghệ cốt lõi nhằm đảm bảo tốc độ, độ chính xác và tính an toàn trong y tế:

### 1. High-Performance LLM (Mô hình Ngôn ngữ lớn tốc độ cao)
   * **Động cơ cốt lõi:** Sử dụng mô hình **Llama 3.3 70B Versatile** (`llama-3.3-70b-versatile`) thông qua nền tảng phần cứng LPU của **Groq API**. Sự kết hợp này mang lại khả năng suy luận logic xuất sắc và tốc độ sinh văn bản siêu tốc (gần như tức thì).
   * **Xử lý Ngôn ngữ Tự nhiên:** Hệ thống tập trung xử lý văn bản tiếng Việt để thấu hiểu chính xác tình trạng, triệu chứng bệnh mà người dùng mô tả thông qua ngôn ngữ đời thường.

### 2. Hybrid Search (Truy xuất thông tin lai)
   * **Vector Search (Tìm kiếm Ngữ nghĩa):** Sử dụng cơ sở dữ liệu **ChromaDB** để tìm kiếm theo ý nghĩa câu hỏi. Khắc phục điểm yếu của tìm kiếm truyền thống, giúp AI hiểu được triệu chứng kể cả khi người dùng không dùng từ chuyên môn y khoa.
   * **Keyword Search (Tìm kiếm Từ khóa):** Tích hợp thuật toán **BM25** kết hợp cùng bộ xử lý NLP Tiếng Việt (`underthesea`) để bắt chính xác các danh từ riêng, "từ khóa cứng" (tên thuốc, tên hội chứng đặc thù).
   * **RRF Fusion:** Hai luồng kết quả độc lập (Vector và BM25) được gộp lại và xếp hạng chéo bằng thuật toán **Reciprocal Rank Fusion (RRF)**. Điều này giúp cung cấp đoạn ngữ cảnh (Context) tối ưu và chính xác tuyệt đối cho AI trước khi sinh câu trả lời.

### 3. Medical Triage & Safety (Sàng lọc An toàn & Kiểm soát Ảo giác)
   * **Knowledge Base & Grounding:** Chỉ sử dụng dữ liệu y văn uy tín (MedQuad) đã được kiểm chứng làm nền tảng tham khảo. Thông số sáng tạo của AI được khóa chặt (`temperature = 0.2`) để triệt tiêu hoàn toàn rủi ro bịa đặt thông tin y tế.
   * **Advanced Prompt Engineering:** Áp dụng kỹ thuật phân luồng đa kịch bản (Adaptive Routing) và tự kiểm duyệt ngầm (Self-Consistency) theo hướng **Sàng lọc chuyên khoa**. Ép buộc AI đánh giá dựa trên ngưỡng thông tin nghiêm ngặt và nhận diện dấu hiệu cấp cứu (Red Flags). Hệ thống **tuyệt đối không tự ý chẩn đoán hay kê đơn thuốc**, chỉ tập trung phân tích triệu chứng để điều hướng người bệnh đến đúng chuyên khoa một cách an toàn nhất.

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