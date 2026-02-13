import os
import socket
import certifi
import consul
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
import google.generativeai as genai

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

# 🔥 Model embedding đúng chuẩn SDK mới
EMBEDDING_MODEL = "models/embedding-001"

# ===============================
# 2. INIT SERVICES
# ===============================
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
movies_col = db[COLLECTION_NAME]

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully")
else:
    print("❌ GEMINI_API_KEY is missing")


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

        print(f"✅ Registered to Consul at {ip_addr}:{PORT}")

    except Exception as e:
        print(f"❌ Consul registration failed: {e}")


# ===============================
# 4. FASTAPI LIFESPAN (thay on_event)
# ===============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    register_to_consul()
    yield
    print("🔻 Service shutting down")


app = FastAPI(title="Movie AI Classifier Microservice", lifespan=lifespan)

# ===============================
# 5. GEMINI EMBEDDING
# ===============================
def get_single_embedding(text: str):
    try:
        if not text:
            return None

        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
        )

        return result["embedding"]

    except Exception as e:
        print(f"🔥 Gemini Error: {e}")
        return None


def background_sync_embeddings():
    print("🔄 Sync embeddings...")

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
# 6. API ENDPOINTS
# ===============================
@app.post("/classify")
async def classify_movie(query: MovieQuery):
    try:
        # Step 1: embedding
        user_vector = get_single_embedding(query.description)
        if not user_vector:
            raise HTTPException(status_code=500, detail="Gemini embedding failed")

        # Step 2: Mongo vector search
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
            return {"predicted_genre": "Unknown", "message": "No similar movies"}

        # Step 3: majority vote
        genres = []
        for n in neighbors:
            genres.extend(n.get("genres", []))

        predicted = max(set(genres), key=genres.count) if genres else "Unknown"

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
    return {"message": "Background sync started"}


@app.get("/health")
def health():
    return {
        "status": "ready",
        "model": EMBEDDING_MODEL,
        "db_connected": DB_NAME in client.list_database_names(),
    }


# ===============================
# 7. MAIN
# ===============================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
