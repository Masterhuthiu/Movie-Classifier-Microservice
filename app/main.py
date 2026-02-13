import os
import socket
import certifi
import consul
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
from google import genai

# ===============================
# 1. CONFIG
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

EMBEDDING_MODEL = "models/gemini-embedding-001"  # 768 dim

CONSUL_HOST = os.getenv("CONSUL_HOST", "consul-server")
SERVICE_NAME = "movie-classifier"

# ===============================
# 2. INIT SERVICES
# ===============================
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
movies_col = db[COLLECTION_NAME]

# Gemini client
if GEMINI_API_KEY:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini Client ready (768-dim)")
else:
    ai_client = None
    print("❌ GEMINI_API_KEY missing")


class MovieQuery(BaseModel):
    description: str


# ===============================
# 3. CONSUL REGISTER
# ===============================
def register_to_consul():
    """
    Đăng ký service vào Consul khi start container
    """
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

        print(f"✅ Registered to Consul: {service_id}:{PORT}")

    except Exception as e:
        print(f"⚠️ Consul register failed: {e}")


def deregister_from_consul():
    """
    Hủy đăng ký khi pod shutdown
    """
    try:
        c = consul.Consul(host=CONSUL_HOST, port=8500)
        hostname = socket.gethostname()
        service_id = f"{SERVICE_NAME}-{hostname}"

        c.agent.service.deregister(service_id)
        print("🔻 Deregistered from Consul")

    except Exception as e:
        print(f"⚠️ Consul deregister failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    register_to_consul()
    yield
    deregister_from_consul()


app = FastAPI(title="Movie AI Classifier (768-dim)", lifespan=lifespan)


# ===============================
# 4. GEMINI EMBEDDING
# ===============================
def get_single_embedding(text: str):
    """
    Tạo embedding 768 chiều từ Gemini
    """
    try:
        if not text or ai_client is None:
            print("❌ Missing text or Gemini client")
            return None

        result = ai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config={
                "output_dimensionality": 768   # 🔥 QUAN TRỌNG
            }
        )

        vector = result.embeddings[0].values

        print(f"📏 Vector size = {len(vector)}")

        if len(vector) != 768:
            print(f"❌ Wrong vector size: {len(vector)}")
            return None

        return vector

    except Exception as e:
        print(f"🔥 REAL Gemini Error: {e}")
        return None



def background_sync_embeddings():
    """
    Sync 50 movie chưa có vector mỗi lần gọi
    """
    print("🔄 Background syncing embeddings...")

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

    print(f"✅ Synced {updated} movies")


# ===============================
# 5. API ENDPOINTS
# ===============================
@app.post("/classify")
async def classify_movie(query: MovieQuery):
    """
    Vector search + majority vote genre
    """
    try:
        user_vector = get_single_embedding(query.description)

        if not user_vector:
            raise HTTPException(status_code=500, detail="Embedding failed")

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
            return {"predicted_genre": "Unknown", "message": "No matches"}

        genres = []
        for n in neighbors:
            genres.extend(n.get("genres", []))

        predicted = max(set(genres), key=genres.count) if genres else "Unknown"

        return {
            "predicted_genre": predicted,
            "confidence": neighbors[0].get("score", 0),
            "matches": neighbors,
        }

    except Exception as e:
        print(f"❌ Classify error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/sync-embeddings")
async def trigger_sync(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_sync_embeddings)
    return {"message": "Background sync started"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedding_model": EMBEDDING_MODEL,
        "dimensions": 768,
        "api_ready": ai_client is not None,
    }


# ===============================
# 6. LOCAL RUN
# ===============================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
