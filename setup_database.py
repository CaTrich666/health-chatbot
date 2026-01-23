import os
import sys

# Thêm đường dẫn để tìm thấy thư mục src
sys.path.append(os.getcwd())

from src import database

print("🛠️  BẮT ĐẦU QUÁ TRÌNH CÀI ĐẶT DATABASE...")

# 1. Đường dẫn file DB
db_path = os.path.join("data", "chat_history.db")

# 2. Xóa file cũ nếu tồn tại (Làm sạch hoàn toàn)
if os.path.exists(db_path):
    try:
        os.remove(db_path)
        print(f"🗑️  Đã xóa file Database cũ tại: {db_path}")
    except Exception as e:
        print(f"❌ Lỗi khi xóa file: {e}")
else:
    print("ℹ️  Chưa có file Database cũ.")

# 3. Ép chạy lệnh tạo bảng
print("🔄 Đang khởi tạo các bảng dữ liệu (Users, Conversations, Messages)...")
try:
    database.init_db()
    print("✅ Đã chạy lệnh init_db() thành công.")
except Exception as e:
    print(f"❌ Lỗi khi khởi tạo: {e}")

# 4. Kiểm tra lại xem bảng đã có chưa
try:
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    
    table_names = [t[0] for t in tables]
    print(f"📊 Danh sách các bảng hiện có trong DB: {table_names}")
    
    if 'users' in table_names and 'conversations' in table_names:
        print("\n🎉 CÀI ĐẶT THÀNH CÔNG! BẠN CÓ THỂ CHẠY WEB ĐƯỢC RỒI.")
    else:
        print("\n⚠️ Cảnh báo: Vẫn thiếu bảng dữ liệu.")
        
except Exception as e:
    print(f"❌ Lỗi khi kiểm tra: {e}")