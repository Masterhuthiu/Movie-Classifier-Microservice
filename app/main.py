import os
import socket
import certifi
import consul
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
import google.generativeai as genai
from typing import List

app = FastAPI(title="Movie AI Classifier Microservice")

# ===============================
# 1. CẤU HÌNH (Khớp với Secret & Atlas)
# ===============================
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://cosa199212:password@cluster0.3jl7a.mongodb.net/")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PORT = int(os.getenv("PORT", 8083))

DB_NAME = "sample_mflix"
COLLECTION_NAME = "movies"
VECTOR_INDEX_NAME = "movies_vector_index"
VECTOR_FIELD_PATH = "fullplot_gemini_embedding"
# Sử dụng embedding-001 để đảm bảo 768 dims (text-embedding-004 có thể trả về dims khác)
EMBEDDING_MODEL = "models/embedding-001"

# Khởi tạo kết nối MongoDB
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
movies_col = db[COLLECTION_NAME]

# ✅ FIX DÒNG 31: Sử dụng configure thay vì Client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully")
else:
    print("❌ ERROR: GEMINI_API_KEY is missing!")

class MovieQuery(BaseModel):
    description: str

# ===============================
# 2. ĐĂNG KÝ CONSUL (Service Discovery)
# ===============================
def register_to_consul():
    try:
        consul_host = os.getenv("CONSUL_HOST", "consul-server")
        c = consul.Consul(host=consul_host, port=8500)
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
        
        c.agent.service.register(
            name="movie-classifier-service",
            service_id=f"classifier-{PORT}",
            address=ip_addr,
            port=PORT,
            check=consul.Check.http(f"http://{ip_addr}:{PORT}/health", interval="10s")
        )
        print(f"✅ Registered to Consul: classifier-service at {ip_addr}:{PORT}")
    except Exception as e:
        print(f"❌ Consul registration failed: {e}")

@app.on_event("startup")
async def startup_event():
    register_to_consul()

# ===============================
# 3. AI LOGIC (Sử dụng google-generativeai syntax)
# ===============================
def get_single_embedding(text: str):
    """Tạo vector từ Gemini API sử dụng đúng syntax SDK"""
    try:
        if not text or not isinstance(text, str):
            return None
            
        # ✅ FIX: Cách gọi embedding chuẩn cho google-generativeai
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"🔥 Lỗi AI Embedding: {e}")
        return None

def background_sync_embeddings():
    """Đồng bộ hóa các phim cũ chưa có vector"""
    print("--- Bắt đầu quét database để tạo embedding ---")
    # Tìm phim có fullplot nhưng chưa có field embedding
    query = {"fullplot": {"$exists": True}, VECTOR_FIELD_PATH: {"$exists": False}}
    cursor = movies_col.find(query).limit(100) # Giới hạn mỗi lần chạy để tránh tràn quota API
    
    count = 0
    for doc in cursor:
        vector = get_single_embedding(doc["fullplot"])
        if vector:
            movies_col.update_one(
                {"_id": doc["_id"]}, 
                {"$set": {VECTOR_FIELD_PATH: vector}}
            )
            count += 1
    print(f"--- Hoàn tất! Đã cập nhật {count} phim mới ---")

# ===============================
# 4. ENDPOINTS
# ===============================

@app.post("/classify")
async def classify_movie(query: MovieQuery):
    try:
        # Bước 1: Tạo vector
        user_vector = get_single_embedding(query.description)
        if not user_vector:
            raise HTTPException(status_code=500, detail="Gemini API failed to generate embedding")
        
        # Bước 2: Vector Search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": VECTOR_FIELD_PATH,
                    "queryVector": user_vector,
                    "numCandidates": 100,
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
            return {"predicted_genre": "Unknown", "message": "No similar movies found"}

        # Bước 3: Dự đoán thể loại
        all_genres = []
        for n in neighbors:
            all_genres.extend(n.get('genres', []))
        
        predicted_genre = max(set(all_genres), key=all_genres.count) if all_genres else "Unknown"

        return {
            "input_description": query.description,
            "predicted_genre": predicted_genre,
            "confidence_score": neighbors[0]['score'],
            "similar_movies": [
                {"title": n.get('title'), "genres": n.get('genres'), "score": n.get('score')} 
                for n in neighbors
            ]
        }
    except Exception as e:
        print(f"❌ Classify Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/sync-embeddings")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_sync_embeddings)
    return {"message": "Background sync process started..."}

@app.get("/health")
def health():
    return {
        "status": "ready", 
        "port": PORT, 
        "db_connected": DB_NAME in client.list_database_names()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)