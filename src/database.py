import sqlite3
import datetime
import os
from src import config

# Đảm bảo thư mục data tồn tại
if not os.path.exists(config.DATA_DIR):
    os.makedirs(config.DATA_DIR)

DB_PATH = os.path.join(config.DATA_DIR, "chat_history.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Bảng user
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE, 
                  password TEXT, 
                  full_name TEXT)''')
    
    # Bảng conversations (Thêm cột is_pinned)
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  user_id INTEGER, 
                  title TEXT, 
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  is_pinned INTEGER DEFAULT 0)''') # 0: Không ghim, 1: Ghim
    
    # Bảng messages
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  conversation_id INTEGER, 
                  role TEXT, 
                  content TEXT, 
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # --- MIGRATION (Tự động thêm cột is_pinned nếu database cũ chưa có) ---
    try:
        c.execute("ALTER TABLE conversations ADD COLUMN is_pinned INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass # Cột đã tồn tại, bỏ qua
        
    conn.commit()
    conn.close()

# --- USER ---
def register_user(username, password, full_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, full_name) VALUES (?, ?, ?)", 
                  (username, password, full_name))
        conn.commit()
        return True, "Đăng ký thành công!"
    except sqlite3.IntegrityError:
        return False, "Tên đăng nhập đã tồn tại."
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    if user: return dict(user)
    return None

# --- CONVERSATION ---
def create_conversation(user_id, title):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO conversations (user_id, title) VALUES (?, ?)", (user_id, title))
    conv_id = c.lastrowid
    conn.commit()
    conn.close()
    return conv_id

def get_user_conversations(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    # Sắp xếp: Tin ghim lên trước (DESC), sau đó đến tin mới nhất
    c.execute("""
        SELECT * FROM conversations 
        WHERE user_id = ? 
        ORDER BY is_pinned DESC, created_at DESC
    """, (user_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# --- MESSAGE ---
def save_message(conv_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)", 
              (conv_id, role, content))
    conn.commit()
    conn.close()

def load_messages(conv_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC", (conv_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# --- NEW FEATURES: DELETE & PIN ---
def delete_conversation(conv_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    c.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()

def toggle_pin_conversation(conv_id, current_status):
    """Đổi trạng thái ghim: Nếu đang ghim -> bỏ ghim, và ngược lại"""
    new_status = 0 if current_status == 1 else 1
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE conversations SET is_pinned = ? WHERE id = ?", (new_status, conv_id))
    conn.commit()
    conn.close()