import os
from dotenv import load_dotenv

# Nạp biến môi trường từ file .env
load_dotenv()

# --- 1. CẤU HÌNH API KEY ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- 2. CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Tên thư mục chứa Database
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db_diagnosis")

# Tên thư mục chứa file BM25:
BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")

# --- 3. ĐƯỜNG DẪN FILE DỮ LIỆU ---
PATH_MEDQUAD = os.path.join(RAW_DATA_DIR, "medquad.csv")
PATH_SYMPTOM = os.path.join(RAW_DATA_DIR, "train.jsonl")

# --- 4. CẤU HÌNH MODEL ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "llama-3.3-70b-versatile" 

# --- 5. CẤU HÌNH BM25 + RRF ---
BM25_K1 = 1.5        # độ nhạy với tần suất từ (cao hơn = ưu tiên từ xuất hiện nhiều)
BM25_B  = 0.75       # chuẩn hóa độ dài văn bản (0=không chuẩn hóa, 1=chuẩn hóa hoàn toàn)
RRF_K   = 60         # hằng số RRF, càng nhỏ càng ưu tiên top rank
TOP_K_RETRIEVAL = 5  # số doc lấy từ mỗi luồng trước khi fuse
TOP_K_FINAL     = 3  # số doc cuối cùng đưa vào prompt sau khi fuse