import os
import socket
import certifi
import consul
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
from google import genai
from google.genai import types

# ===============================
# 1. CONFIGURATION
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

# ĐỔI THÀNH MODEL 001 ĐỂ TRÁNH LỖI 404
EMBEDDING_MODEL = "models/gemini-embedding-001" 

CONSUL_HOST = os.getenv("CONSUL_HOST", "consul-server")
SERVICE_NAME = "movie-classifier"

# ===============================
# 2. INIT SERVICES
# ===============================
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
movies_col = db[COLLECTION_NAME]

if GEMINI_API_KEY:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"✅ Gemini Client ready (Using {EMBEDDING_MODEL})")
else:
    ai_client = None
    print("⚠️ WARNING: GEMINI_API_KEY is missing!")

class MovieQuery(BaseModel):
    description: str

# ===============================
# 3. CONSUL (SERVICE DISCOVERY)
# ===============================
def register_to_consul():
    try:
        c = consul.Consul(host=CONSUL_HOST, port=8500)
        hostname = socket.gethostname()
        service_id = f"{SERVICE_NAME}-{hostname}"

        c.agent.service.register(
            name=SERVICE_NAME,
            service_id=service_id,
            address=hostname,
            port=PORT,
            check=consul.Check.http(
                url=f"http://{hostname}:{PORT}/health",
                interval="10s",
                timeout="5s",
                deregister="1m",
            ),
        )
        print(f"✅ Registered to Consul: {service_id}")
    except Exception as e:
        print(f"⚠️ Consul registration skipped: {e}")

def deregister_from_consul():
    try:
        c = consul.Consul(host=CONSUL_HOST, port=8500)
        hostname = socket.gethostname()
        service_id = f"{SERVICE_NAME}-{hostname}"
        c.agent.service.deregister(service_id)
        print("🔻 Deregistered from Consul")
    except Exception as e:
        print(f"⚠️ Consul deregistration failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    register_to_consul()
    yield
    deregister_from_consul()

app = FastAPI(title="Movie AI Classifier", lifespan=lifespan)

# ===============================
# 4. CORE LOGIC
# ===============================
def get_single_embedding(text: str):
    """Tạo embedding 768 chiều cho MongoDB Index"""
    try:
        if not text or ai_client is None:
            return None

        # Sử dụng config để ép dimensionality về 768
        result = ai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768
            )
        )
        vector = result.embeddings[0].values
        
        # In log để kiểm tra trong kubectl logs
        print(f"📏 Created vector: {len(vector)} dimensions for text: '{text[:30]}...'")
        
        return vector
    except Exception as e:
        print(f"🔥 Gemini API Error: {e}")
        return None

def background_sync_embeddings():
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
    print(f"✅ Background Sync Complete: Updated {updated} movies.")

# ===============================
# 5. API ENDPOINTS
# ===============================
@app.post("/classify")
async def classify_movie(query: MovieQuery):
    print(f"📩 Received request: {query.description}")
    
    user_vector = get_single_embedding(query.description)
    if not user_vector:
        raise HTTPException(status_code=500, detail="Gemini embedding failed")

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

    try:
        neighbors = list(movies_col.aggregate(pipeline))
        if not neighbors:
            return {"predicted_genre": "Unknown", "message": "No matches found"}

        all_genres = []
        for n in neighbors:
            all_genres.extend(n.get("genres", []))
        
        predicted = max(set(all_genres), key=all_genres.count) if all_genres else "Unknown"

        return {
            "predicted_genre": predicted,
            "confidence": neighbors[0].get("score", 0),
            "matches": neighbors,
        }
    except Exception as e:
        print(f"❌ DB Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/sync-embeddings")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_sync_embeddings)
    return {"message": "Sync started in background."}

@app.get("/health")
def health():
    return {"status": "ok", "dimensions": 768}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)