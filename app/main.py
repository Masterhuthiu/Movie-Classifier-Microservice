import os
import socket
import certifi
import consul
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
from google import genai  # 🔥 SDK google-genai mới nhất
from google.genai import types
from typing import List

# ===============================
# 1. CẤU HÌNH (CONFIG)
# ===============================
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://masterhuthiu:123456a%40A@cluster0.3jl7a.mongodb.net/?retryWrites=true&w=majority",
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PORT = int(os.getenv("PORT", 8083))

DB_NAME = "sample_mflix"
COLLECTION_NAME = "movies"
VECTOR_INDEX_NAME = "movies_vector_index"
VECTOR_FIELD_PATH = "fullplot_gemini_embedding"

# Model bạn chọn - Rất ổn định với 768 dims
EMBEDDING_MODEL = "gemini-embedding-001" 

# ===============================
# 2. KHỞI TẠO KẾT NỐI (SERVICES)
# ===============================
# Kết nối MongoDB
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
movies_col = db[COLLECTION_NAME]

# Khởi tạo Gemini Client (Dùng SDK google-genai 0.3.0)
if GEMINI_API_KEY:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"✅ Gemini AI Client initialized with: {EMBEDDING_MODEL}")
else:
    ai_client = None
    print("❌ ERROR: GEMINI_API_KEY is missing!")

class MovieQuery(BaseModel):
    description: str

# ===============================
# 3. ĐĂNG KÝ CONSUL (SERVICE DISCOVERY)
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
            check=consul.Check.http(f"http://{ip_addr}:{PORT}/health", interval="10s"),
        )
        print(f"✅ Registered to Consul at {ip_addr}:{PORT}")
    except Exception as e:
        print(f"❌ Consul registration failed: {e}")

# ===============================
# 4. LIFESPAN (QUẢN LÝ STARTUP/SHUTDOWN)
# ===============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khi app bắt đầu
    register_to_consul()
    yield
    # Khi app tắt
    print("🔻 Service shutting down")

app = FastAPI(title="Movie AI Classifier Microservice", lifespan=lifespan)

# ===============================
# 5. AI LOGIC (VECTOR EMBEDDING)
# ===============================
def get_single_embedding(text: str):
    """Tạo vector 768-dims sử dụng SDK google-genai 0.3.0"""
    try:
        if not text or ai_client is None:
            return None

        # Gọi API tạo vector
        result = ai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )

        # Trả về list các số float (768 dimensions)
        return result.embeddings[0].values

    except Exception as e:
        print(f"🔥 Gemini Error ({EMBEDDING_MODEL}): {e}")
        return None

def background_sync_embeddings():
    """Tự động tạo vector cho các phim chưa có trong database"""
    print("🔄 Starting background sync...")
    query = {"fullplot": {"$exists": True}, VECTOR_FIELD_PATH: {"$exists": False}}
    cursor = movies_col.find(query).limit(50)

    updated = 0
    for doc in cursor:
        vector = get_single_embedding(doc["fullplot"])
        if vector:
            movies_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {VECTOR_FIELD_PATH: vector}},
            )
            updated += 1
    print(f"✅ Background sync completed. Updated {updated} movies.")

# ===============================
# 6. API ENDPOINTS
# ===============================

@app.post("/classify")
async def classify_movie(query: MovieQuery):
    try:
        # Bước 1: Chuyển text mô tả phim thành vector
        user_vector = get_single_embedding(query.description)
        if not user_vector:
            raise HTTPException(status_code=500, detail="Gemini embedding failed")

        # Bước 2: Tìm kiếm Vector (Vector Search) trên MongoDB Atlas
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": VECTOR_FIELD_PATH,
                    "queryVector": user_vector,
                    "numCandidates": 100,
                    "limit": 5,
                }
            },
            {
                "$project": {
                    "title": 1,
                    "genres": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        neighbors = list(movies_col.aggregate(pipeline))

        if not neighbors:
            return {"predicted_genre": "Unknown", "message": "No similar movies found in database"}

        # Bước 3: Thuật toán KNN (Lấy thể loại xuất hiện nhiều nhất)
        all_genres = []
        for n in neighbors:
            all_genres.extend(n.get("genres", []))

        predicted = max(set(all_genres), key=all_genres.count) if all_genres else "Unknown"

        return {
            "input_description": query.description,
            "predicted_genre": predicted,
            "confidence_score": neighbors[0].get("score", 0),
            "similar_movies": [
                {
                    "title": n.get("title"),
                    "genres": n.get("genres"),
                    "score": n.get("score"),
                }
                for n in neighbors
            ],
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
    # Kiểm tra trạng thái DB và Model
    db_ok = False
    try:
        db_ok = DB_NAME in client.list_database_names()
    except:
        db_ok = False
        
    return {
        "status": "ready",
        "model": EMBEDDING_MODEL,
        "db_connected": db_ok,
    }

# ===============================
# 7. CHẠY SERVER
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)