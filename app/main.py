import os
import socket
import certifi
import consul
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
from google import genai
from google.genai import types

# ===============================
# 1. CẤU HÌNH (CONFIG)
# ===============================
# Lấy URI từ biến môi trường (Secret trong K8s)
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://masterhuthiu:123456a%40A@cluster0.3jl7a.mongodb.net/?retryWrites=true&w=majority",
)

# Lấy API Key từ biến môi trường - KHÔNG dán cứng vào đây để tránh bị Google khóa key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PORT = int(os.getenv("PORT", 8083))
DB_NAME = "sample_mflix"
COLLECTION_NAME = "movies"
VECTOR_INDEX_NAME = "movies_vector_index"
VECTOR_FIELD_PATH = "fullplot_gemini_embedding"

# Sử dụng model v001 để đảm bảo luôn ra đúng 768 dimensions (khớp với Index)
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ===============================
# 2. KHỞI TẠO DỊCH VỤ (INIT)
# ===============================
# Kết nối MongoDB
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
movies_col = db[COLLECTION_NAME]

# Khởi tạo Client Gemini
if GEMINI_API_KEY:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini Client initialized successfully.")
else:
    ai_client = None
    print("❌ ERROR: GEMINI_API_KEY is missing! API will fail.")

class MovieQuery(BaseModel):
    description: str

# ===============================
# 3. CONSUL & LIFESPAN (QUẢN LÝ DỊCH VỤ)
# ===============================
def register_to_consul():
    try:
        consul_host = os.getenv("CONSUL_HOST", "consul-server-service")
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
        print(f"✅ Registered to Consul: {ip_addr}:{PORT}")
    except Exception as e:
        print(f"❌ Consul Registration Error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Hành động khi khởi động (Startup)
    register_to_consul()
    yield
    # Hành động khi tắt máy (Shutdown)
    print("🔻 Service is shutting down...")

app = FastAPI(title="Movie AI Classifier (768 Dim)", lifespan=lifespan)

# ===============================
# 4. LOGIC XỬ LÝ VECTOR (GEMINI)
# ===============================
def get_single_embedding(text: str):
    """Tạo vector 768 chiều từ text bằng Gemini API"""
    try:
        if not text or ai_client is None:
            return None

        # Ép model 001 để luôn ra 768 chiều
        result = ai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        vector = result.embeddings[0].values
        
        # Log kiểm tra số chiều để debug nếu cần
        # print(f"DEBUG: Vector dimensions: {len(vector)}")
        return vector
    except Exception as e:
        print(f"🔥 Gemini Error: {e}")
        return None

def background_sync_embeddings():
    """Đồng bộ 50 phim chưa có vector mỗi lần gọi"""
    print("🔄 Syncing embeddings in background...")
    query = {"fullplot": {"$exists": True}, VECTOR_FIELD_PATH: {"$exists": False}}
    cursor = movies_col.find(query).limit(50)
    updated = 0
    for doc in cursor:
        vector = get_single_embedding(doc["fullplot"])
        if vector:
            movies_col.update_one(
                {"_id": doc["_id"]}, 
                {"$set": {VECTOR_FIELD_PATH: vector}}
            )
            updated += 1
    print(f"✅ Synced {updated} movies.")

# ===============================
# 5. CÁC API ENDPOINTS
# ===============================
@app.post("/classify")
async def classify_movie(query: MovieQuery):
    """Tìm phim tương đồng và dự đoán thể loại bằng Vector Search"""
    try:
        # 1. Chuyển mô tả thành vector
        user_vector = get_single_embedding(query.description)
        if not user_vector:
            raise HTTPException(status_code=500, detail="Gemini embedding failed. Check logs.")

        # 2. Pipeline Vector Search trên MongoDB Atlas
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
                    "score": {"$meta": "vectorSearchScore"}
                }
            },
        ]

        neighbors = list(movies_col.aggregate(pipeline))
        if not neighbors:
            return {"predicted_genre": "Unknown", "message": "No matching movies found."}

        # 3. Logic dự đoán thể loại (Genre)
        genres = []
        for n in neighbors:
            genres.extend(n.get("genres", []))
        
        predicted = max(set(genres), key=genres.count) if genres else "Unknown"

        return {
            "predicted_genre": predicted,
            "confidence": neighbors[0].get("score", 0),
            "matches": neighbors
        }
    except Exception as e:
        print(f"❌ API Classify Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/sync-embeddings")
async def trigger_sync(background_tasks: BackgroundTasks):
    """Admin endpoint để tạo vector cho dữ liệu cũ"""
    background_tasks.add_task(background_sync_embeddings)
    return {"message": "Background sync started."}

@app.get("/health")
def health():
    """Endpoint cho Consul Check & K8s Liveness Probe"""
    return {
        "status": "ok", 
        "model": EMBEDDING_MODEL, 
        "dimensions": 768,
        "api_key_status": "Loaded" if GEMINI_API_KEY else "Missing"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)