import os
from dotenv import load_dotenv

# Nạp biến môi trường từ file .env (nếu chạy local)
load_dotenv()

# --- 1. CẤU HÌNH API KEY (AN TOÀN) ---
# Lấy từ biến môi trường. Trên Streamlit Cloud nó sẽ tự lấy từ Secrets.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- 2. CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Tên thư mục chứa Database (Khớp với máy bạn)
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db_diagnosis") 

# --- 3. ĐƯỜNG DẪN FILE DỮ LIỆU ---
PATH_MEDQUAD = os.path.join(RAW_DATA_DIR, "medquad.csv")
PATH_SYMPTOM = os.path.join(RAW_DATA_DIR, "train.jsonl")

# --- 4. CẤU HÌNH MODEL ---
# Chỉnh lại cho khớp với web_chat.py (Dùng HuggingFace cho ổn định)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "gemini-flash-latest"