import os
import socket
import certifi
import consul
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
from google import genai  # 🔥 Dùng cho google-genai==0.3.0
from google.genai import types
from typing import List

# ===============================
# 1. CẤU HÌNH (CONFIG)
# ===============================
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://masterhuthiu:123456a%40A@cluster0.3jl7a.mongodb.net/?retryWrites=true&w=majority",
)
GEMINI_API_KEY ="AIzaSyDDlIjhAUI2H1tIxzzWguWKZ3IeEysAsME" #os.getenv("GEMINI_API_KEY")
PORT = int(os.getenv("PORT", 8083))

DB_NAME = "sample_mflix"
COLLECTION_NAME = "movies"
VECTOR_INDEX_NAME = "movies_vector_index"
VECTOR_FIELD_PATH = "fullplot_gemini_embedding"

# Tên model đầy đủ theo chuẩn SDK mới
EMBEDDING_MODEL = "models/gemini-embedding-001" 

# ===============================
# 2. KHỞI TẠO SERVICES
# ===============================
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
movies_col = db[COLLECTION_NAME]

# Khởi tạo Client
try:
    if GEMINI_API_KEY:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"✅ Gemini Client initialized. Model: {EMBEDDING_MODEL}")
    else:
        ai_client = None
        print("❌ CRITICAL: GEMINI_API_KEY is missing from Env!")
except Exception as e:
    ai_client = None
    print(f"❌ Failed to init Gemini Client: {e}")

class MovieQuery(BaseModel):
    description: str

# ===============================
# 3. CONSUL & LIFESPAN
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
        print(f"✅ Registered to Consul: {ip_addr}:{PORT}")
    except Exception as e:
        print(f"❌ Consul Error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    register_to_consul()
    yield
    print("🔻 Shutting down...")

app = FastAPI(title="Movie AI Classifier", lifespan=lifespan)

# ===============================
# 4. AI LOGIC (Embedding)
# ===============================
def get_single_embedding(text: str):
    """Tạo vector 768-dims và in lỗi chi tiết nếu thất bại"""
    try:
        if not text or ai_client is None:
            print("⚠️ Embedding skip: Text empty or Client not ready")
            return None

        # Gọi API tạo vector
        result = ai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )

        if result and result.embeddings:
            return result.embeddings[0].values
        
        print("⚠️ Gemini returned empty embeddings list")
        return None

    except Exception as e:
        # ⚡ ĐÂY LÀ DÒNG QUAN TRỌNG ĐỂ DEBUG TRÊN K8S
        print(f"🔥 Gemini Error Detail: {str(e)}")
        return None

def background_sync_embeddings():
    print("🔄 Background sync started...")
    query = {"fullplot": {"$exists": True}, VECTOR_FIELD_PATH: {"$exists": False}}
    cursor = movies_col.find(query).limit(50)
    updated = 0
    for doc in cursor:
        vector = get_single_embedding(doc["fullplot"])
        if vector:
            movies_col.update_one({"_id": doc["_id"]}, {"$set": {VECTOR_FIELD_PATH: vector}})
            updated += 1
    print(f"✅ Sync done. Updated {updated} docs.")

# ===============================
# 5. API ENDPOINTS
# ===============================
@app.post("/classify")
async def classify_movie(query: MovieQuery):
    try:
        # 1. Tạo embedding
        user_vector = get_single_embedding(query.description)
        if not user_vector:
            # Trả về lỗi chi tiết hơn thay vì 500 chung chung
            raise HTTPException(status_code=500, detail="Gemini failed. Check Pod logs for 🔥 error.")

        # 2. Vector Search
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
                    "title": 1, "genres": 1, "score": {"$meta": "vectorSearchScore"}
                }
            },
        ]

        neighbors = list(movies_col.aggregate(pipeline))

        if not neighbors:
            return {"predicted_genre": "Unknown", "message": "No matches in DB"}

        # 3. Predict Genre
        all_genres = []
        for n in neighbors:
            all_genres.extend(n.get("genres", []))
        
        predicted = max(set(all_genres), key=all_genres.count) if all_genres else "Unknown"

        return {
            "predicted_genre": predicted,
            "confidence": neighbors[0].get("score", 0),
            "matches": neighbors
        }

    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/sync-embeddings")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_sync_embeddings)
    return {"message": "Syncing..."}

@app.get("/health")
def health():
    return {"status": "ok", "model": EMBEDDING_MODEL, "api_ready": ai_client is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)