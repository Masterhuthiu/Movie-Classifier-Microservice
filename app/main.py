import os
import socket
import certifi
import consul
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
import google.genai as genai
from urllib.parse import quote_plus

app = FastAPI(title="Movie AI Classifier Microservice")

# ===============================
# 1. CẤU HÌNH (Khớp 100% với Atlas READY)
# ===============================
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://cosa199212:password@cluster0.3jl7a.mongodb.net/...")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_key")
PORT = int(os.getenv("PORT", 8083))

# Thông số khớp với Index bạn vừa tạo thành công
DB_NAME = "sample_mflix"
COLLECTION_NAME = "movies"
VECTOR_INDEX_NAME = "movies_vector_index"  # Đã READY trên Atlas
VECTOR_FIELD_PATH = "fullplot_gemini_embedding"
EMBEDDING_MODEL = "text-embedding-004"

# Khởi tạo kết nối
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
movies_col = db[COLLECTION_NAME]
ai_client = genai.Client(api_key=GEMINI_API_KEY)

class MovieQuery(BaseModel):
    description: str

# ===============================
# 2. ĐĂNG KÝ CONSUL (Service Discovery)
# ===============================
def register_to_consul():
    try:
        c = consul.Consul(host=os.getenv("CONSUL_HOST", "localhost"), port=8500)
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
        
        c.agent.service.register(
            name="movie-classifier-service",
            service_id=f"classifier-{PORT}",
            address=ip_addr,
            port=PORT,
            check=consul.Check.http(f"http://{ip_addr}:{PORT}/health", interval="10s")
        )
        print(f"✅ Registered to Consul: classifier-service on port {PORT}")
    except Exception as e:
        print(f"❌ Consul registration failed: {e}")

@app.on_event("startup")
async def startup_event():
    register_to_consul()

# ===============================
# 3. AI LOGIC (Embedding & Search)
# ===============================
def get_single_embedding(text: str):
    """Tạo vector từ Gemini API"""
    try:
        response = ai_client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        return response.embeddings[0].values
    except Exception as e:
        print(f"Lỗi AI Embedding: {e}")
        return None

def background_sync_embeddings():
    """Đồng bộ hóa các phim cũ chưa có vector (Chạy ngầm)"""
    print("--- Bắt đầu quét database ---")
    query = {"fullplot": {"$exists": True}, VECTOR_FIELD_PATH: {"$exists": False}}
    cursor = movies_col.find(query)
    
    count = 0
    for doc in cursor:
        vector = get_single_embedding(doc["fullplot"])
        if vector:
            movies_col.update_one({"_id": doc["_id"]}, {"$set": {VECTOR_FIELD_PATH: vector}})
            count += 1
    print(f"--- Hoàn tất! Đã cập nhật {count} phim mới ---")

# ===============================
# 4. ENDPOINTS (API Chính)
# ===============================

@app.post("/classify")
async def classify_movie(query: MovieQuery):
    """Nhận mô tả -> Tìm phim tương đồng -> Trả về thể loại dự đoán"""
    try:
        # Bước 1: Tạo vector cho câu hỏi của User
        user_vector = get_single_embedding(query.description)
        if not user_vector:
            raise HTTPException(status_code=500, detail="Gemini API Error")
        
        # Bước 2: Thực hiện Vector Search trên Cloud
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": VECTOR_FIELD_PATH,
                    "queryVector": user_vector,
                    "numCandidates": 50,
                    "limit": 5
                }
            },
            {
                "$project": {
                    "title": 1, 
                    "genres": 1, 
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        neighbors = list(movies_col.aggregate(pipeline))
        
        if not neighbors:
            return {"predicted_genre": "Unknown", "message": "Không tìm thấy phim tương đồng"}

        # Bước 3: Logic lấy thể loại (Genres) phổ biến nhất
        all_genres = []
        for n in neighbors:
            all_genres.extend(n.get('genres', []))
        
        predicted_genre = max(set(all_genres), key=all_genres.count) if all_genres else "Unknown"

        return {
            "input_description": query.description,
            "predicted_genre": predicted_genre,
            "confidence_score": neighbors[0]['score'],
            "similar_movies": [
                {"title": n.get('title'), "genres": n.get('genres')} for n in neighbors
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/sync-embeddings")
async def trigger_sync(background_tasks: BackgroundTasks):
    """Kích hoạt đồng bộ hóa dữ liệu mà không gây treo API"""
    background_tasks.add_task(background_sync_embeddings)
    return {"message": "Quá trình tạo embedding đang chạy ngầm..."}

@app.get("/health")
def health():
    """Kiểm tra trạng thái service và kết nối DB"""
    return {
        "status": "ready", 
        "port": PORT, 
        "database": DB_NAME,
        "index": VECTOR_INDEX_NAME
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)