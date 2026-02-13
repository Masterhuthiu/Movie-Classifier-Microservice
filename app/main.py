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

# 🔥 Embedding model 768-dim
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# ===============================
# 2. INIT SERVICES
# ===============================
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
movies_col = db[COLLECTION_NAME]

# Load MiniLM model
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"✅ Embedding model loaded: {EMBEDDING_MODEL_NAME} (768 dim)")
except Exception as e:
    embedding_model = None
    print(f"❌ Failed to load embedding model: {e}")

class MovieQuery(BaseModel):
    description: str

# ===============================
# 3. CONSUL + LIFESPAN
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
# 4. EMBEDDING LOGIC (768 dim)
# ===============================
def get_single_embedding(text: str):
    try:
        if not text or embedding_model is None:
            print("⚠️ Embedding skipped: empty text or model not loaded")
            return None

        vector = embedding_model.encode(text).tolist()

        # đảm bảo đúng 768 chiều
        if len(vector) != 768:
            print(f"❌ Wrong embedding size: {len(vector)}")
            return None

        return vector

    except Exception as e:
        print(f"🔥 Embedding error: {e}")
        return None


def background_sync_embeddings():
    print("🔄 Background sync started...")

    query = {
        "fullplot": {"$exists": True},
        VECTOR_FIELD_PATH: {"$exists": False},
    }

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

    print(f"✅ Sync done. Updated {updated} docs.")


# ===============================
# 5. API
# ===============================
@app.post("/classify")
async def classify_movie(query: MovieQuery):
    try:
        # 1️⃣ embedding user query
        user_vector = get_single_embedding(query.description)

        if not user_vector:
            raise HTTPException(status_code=500, detail="Embedding failed")

        # 2️⃣ MongoDB Vector Search
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
            return {"predicted_genre": "Unknown", "message": "No matches in DB"}

        # 3️⃣ Predict genre bằng majority vote
        all_genres: List[str] = []

        for n in neighbors:
            all_genres.extend(n.get("genres", []))

        predicted = max(set(all_genres), key=all_genres.count) if all_genres else "Unknown"

        return {
            "predicted_genre": predicted,
            "confidence": neighbors[0].get("score", 0),
            "matches": neighbors,
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
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_dim": 768,
        "model_ready": embedding_model is not None,
    }


# ===============================
# 6. LOCAL RUN
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
