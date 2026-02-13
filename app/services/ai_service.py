import google.generativeai as genai
import os
from typing import List

class AIService:
    def __init__(self):
        # Lấy API Key từ biến môi trường đã cấu hình trong K8s Secret
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables")
        
        genai.configure(api_key=api_key)
        # Sử dụng embedding-001 để đảm bảo 768 dimensions khớp với Atlas Index của bạn
        self.model = 'models/embedding-001'

    async def get_embedding(self, text: str) -> List[float]:
        try:
            # Gọi API của Google để lấy vector
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"
            )
            
            # Trả về mảng float (vector)
            return result['embedding']
            
        except Exception as e:
            print(f"Lỗi AI Embedding chi tiết: {str(e)}")
            # Raise lỗi để main.py bắt được và trả về 500 cho user
            raise Exception(f"Gemini API Error: {str(e)}")

# Khởi tạo instance duy nhất
ai_service = AIService()