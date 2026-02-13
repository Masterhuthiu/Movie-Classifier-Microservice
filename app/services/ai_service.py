import google.generativeai as genai
import os
from typing import List

class AIService:
    def __init__(self):
        # Lấy API Key từ biến môi trường
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ LỖI: GEMINI_API_KEY chưa được cấu hình!")
            self.enabled = False
            return
        
        try:
            genai.configure(api_key=api_key)
            # Dùng embedding-001 để đảm bảo 768 dimensions khớp MongoDB Index
            self.model_name = 'models/embedding-001'
            self.enabled = True
            print(f"✅ AIService initialized with model: {self.model_name}")
        except Exception as e:
            print(f"❌ Lỗi cấu hình Gemini: {e}")
            self.enabled = False

    async def get_embedding(self, text: str) -> List[float]:
        if not self.enabled:
            raise Exception("AI Service is not configured properly")

        try:
            # Gọi API đồng bộ trong thread (SDK của Google hiện chưa hỗ trợ async thuần)
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            
            # Đảm bảo trả về đúng định dạng list float
            if 'embedding' in result:
                return result['embedding']
            else:
                raise Exception("Phản hồi từ Gemini không chứa dữ liệu embedding")
            
        except Exception as e:
            print(f"🔥 Lỗi AI Embedding chi tiết: {str(e)}")
            raise Exception(f"Gemini API Error: {str(e)}")

# Khởi tạo instance duy nhất để dùng chung
ai_service = AIService()