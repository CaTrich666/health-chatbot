import google.generativeai as genai
import os
import sys

# Thêm đường dẫn để lấy config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def check():
    print("🚀 ĐANG KIỂM TRA QUYỀN HẠN API KEY CHO HỆ THỐNG SOLAR AI...")
    genai.configure(api_key=config.GOOGLE_API_KEY)
    
    try:
        print("Danh sách Model được phép dùng cho chatbot tư vấn điện mặt trời:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"  ✅ {m.name}")
    except Exception as e:
        print(f"❌ LỖI API KEY: {e}")

if __name__ == "__main__":
    check()