import os
import socket
import certifi
import consul
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from typing import List

# ===============================
# 1. CONFIG
# ===============================
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://masterhuthiu:123456a%40A@cluster0.3jl7a.mongodb.net/?retryWrites=true&w=majority",
)
PORT = int(os.getenv("PORT", 8083))

DB_NAME = "sample_mflix"
COLLECTION_NAME = "movies"
VECTOR_INDEX_NAME = "movies_vector_index"
VECTOR_FIELD_PATH = "fullplot_gemini_embedding"

# Model 768-dim chạy offline cực tốt
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DIM = 768

# ===============================
# 2. INIT SERVICES
# ===============================
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
movies_col = db[COLLECTION_NAME]

# Biến toàn cục để giữ model
embedding_model = None

class MovieQuery(BaseModel):
    description: str

# ===============================
# 3. CONSUL REGISTER
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

# ===============================
# 4. LIFESPAN (Load model tại đây)
# ===============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model
    # Load model khi startup
    try:
        print(f"⏳ Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"✅ Model loaded (Dim: {EMBEDDING_DIM})")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
    
    register_to_consul()
    yield
    print("🔻 Shutting down...")

app = FastAPI(title="Movie AI Classifier (Offline)", lifespan=lifespan)

# ===============================
# 5. EMBEDDING LOGIC
# ===============================
def get_single_embedding(text: str):
    try:
        if not text or embedding_model is None:
            return None

        # Tạo vector và chuyển sang list float
        vector = embedding_model.encode(text).tolist()

        if len(vector) != EMBEDDING_DIM:
            print(f"⚠️ Dim mismatch: {len(vector)} != {EMBEDDING_DIM}")
            return None

        return vector
    except Exception as e:
        print(f"🔥 Embedding error: {e}")
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
# 6. API ENDPOINTS
# ===============================
@app.post("/classify")
async def classify_movie(query: MovieQuery):
    try:
        user_vector = get_single_embedding(query.description)
        if not user_vector:
            raise HTTPException(status_code=500, detail="Embedding logic failed")

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
    return {"message": "Syncing in background..."}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": EMBEDDING_MODEL_NAME,
        "model_ready": embedding_model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)