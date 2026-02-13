import google.generativeai as genai
import os
from typing import List
import asyncio

class AIService:
    def __init__(self):
        # Lấy API Key từ Secret của Kubernetes
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ LỖI: GEMINI_API_KEY không tìm thấy trong biến môi trường!")
            self.enabled = False
            return
        
        try:
            genai.configure(api_key=api_key)
            # Model này trả về vector 768 chiều, khớp với cấu hình Index của bạn
            self.model_name = 'models/embedding-001'
            self.enabled = True
            print(f"🚀 AIService đã sẵn sàng với model: {self.model_name}")
        except Exception as e:
            print(f"❌ Lỗi khi cấu hình Google AI: {str(e)}")
            self.enabled = False

    async def get_embedding(self, text: str) -> List[float]:
        if not self.enabled:
            raise Exception("AI Service chưa được cấu hình đúng. Kiểm tra API Key.")

        try:
            # Chạy hàm embed_content (đồng bộ) trong một thread riêng để không chặn FastAPI
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_query"
                )
            )
            
            if 'embedding' in result:
                print(f"✅ Đã tạo thành công vector cho text: '{text[:30]}...'")
                return result['embedding']
            else:
                raise Exception("Phản hồi từ Gemini không có trường 'embedding'")
                
        except Exception as e:
            print(f"🔥 Lỗi AI Embedding chi tiết: {str(e)}")
            # Trả về lỗi rõ ràng để Gateway/User nhận diện được
            raise Exception(f"Gemini API Error: {str(e)}")

# Khởi tạo instance duy nhất (Singleton)
ai_service = AIService()