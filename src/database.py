import streamlit as st
from supabase import create_client, Client
import bcrypt
import uuid
from datetime import datetime
import os

# --- 1. KẾT NỐI SUPABASE TỪ SECRETS ---
try:
    # Lấy thông tin từ Secrets mà bạn vừa cấu hình
    url: str = st.secrets["SUPABASE_URL"]
    key: str = st.secrets["SUPABASE_KEY"]
    
    # Tạo kết nối
    supabase: Client = create_client(url, key)
except Exception as e:
    st.error(f"⚠️ Lỗi kết nối Supabase: {e}. Hãy kiểm tra lại Secrets!")
    supabase = None

# --- 2. XỬ LÝ TÀI KHOẢN (ĐĂNG KÝ/ĐĂNG NHẬP) ---
def register_user(username, password, full_name):
    if not supabase: return False, "Mất kết nối Database."
    try:
        # Mã hóa mật khẩu trước khi lưu
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        data = {
            "username": username,
            "password": hashed,
            "full_name": full_name
        }
        # Gửi lên mây
        response = supabase.table("users").insert(data).execute()
        
        if response.data:
            return True, "Đăng ký thành công!"
        return False, "Không thể tạo tài khoản."
            
    except Exception as e:
        if "duplicate key" in str(e): # Lỗi trùng tên đăng nhập
            return False, "Tên đăng nhập đã tồn tại."
        return False, f"Lỗi kỹ thuật: {e}"

def login_user(username, password):
    if not supabase: return None
    try:
        # Tìm user trên mây
        response = supabase.table("users").select("*").eq("username", username).execute()
        
        if not response.data:
            return None # Không tìm thấy user
            
        user = response.data[0]
        stored_hash = user['password']
        
        # So khớp mật khẩu
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            return user
        return None
    except Exception as e:
        print(f"Lỗi đăng nhập: {e}")
        return None

# --- 3. QUẢN LÝ HỘI THOẠI (LỊCH SỬ CHAT) ---
def get_user_conversations(user_id):
    if not supabase: return []
    try:
        # Lấy danh sách chat, ưu tiên Ghim lên đầu, sau đó đến Mới nhất
        response = supabase.table("conversations")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("is_pinned", desc=True)\
            .order("created_at", desc=True)\
            .execute()
        return response.data
    except: return []

def create_conversation(user_id, title):
    if not supabase: return None
    try:
        data = {"user_id": user_id, "title": title}
        response = supabase.table("conversations").insert(data).execute()
        if response.data:
            return response.data[0]['id']
        return None
    except: return None

def delete_conversation(conv_id):
    if not supabase: return
    try:
        supabase.table("conversations").delete().eq("id", conv_id).execute()
    except: pass

def toggle_pin_conversation(conv_id, current_status):
    if not supabase: return
    try:
        # Đảo ngược trạng thái ghim (True <-> False)
        new_status = not current_status
        supabase.table("conversations").update({"is_pinned": new_status}).eq("id", conv_id).execute()
    except: pass

# --- 4. XỬ LÝ TIN NHẮN CHI TIẾT ---
def load_messages(conv_id):
    if not supabase: return []
    try:
        # Load tin nhắn cũ nhất lên trước (để đọc từ trên xuống)
        response = supabase.table("messages")\
            .select("role, content")\
            .eq("conversation_id", conv_id)\
            .order("created_at", desc=False)\
            .execute()
        return response.data
    except: return []

def save_message(conv_id, role, content):
    if not supabase: return
    try:
        data = {
            "conversation_id": conv_id,
            "role": role,
            "content": content
        }
        supabase.table("messages").insert(data).execute()
    except Exception as e:
        print(f"Lỗi lưu tin nhắn: {e}")