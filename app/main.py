
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
# 1. CONFIG
# ===============================
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://masterhuthiu:123456a%40A@cluster0.3jl7a.mongodb.net/?retryWrites=true&w=majority")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PORT = int(os.getenv("PORT", 8083))

DB_NAME = "sample_mflix"
COLLECTION_NAME = "movies"
VECTOR_INDEX_NAME = "movies_vector_index"
VECTOR_FIELD_PATH = "fullplot_gemini_embedding"

# 🔥 FIX QUAN TRỌNG: model embedding đúng API 2025
EMBEDDING_MODEL = "models/embedding-001"

# MongoDB
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
movies_col = db[COLLECTION_NAME]

# Gemini config
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully with embedding-001")
else:
    print("❌ ERROR: GEMINI_API_KEY is missing!")

class MovieQuery(BaseModel):
    description: str


# ===============================
# 2. CONSUL REGISTER
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
# 3. GEMINI EMBEDDING
# ===============================
def get_single_embedding(text: str):
    """Generate embedding from Gemini"""
    try:
        if not text or not isinstance(text, str):
            return None

        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text
        )

        return result["embedding"]

    except Exception as e:
        print(f"🔥 Gemini Error Detail: {str(e)}")
        return None


def background_sync_embeddings():
    """Sync old movies without embedding"""
    print("--- Start scanning DB for missing embeddings ---")

    query = {"fullplot": {"$exists": True}, VECTOR_FIELD_PATH: {"$exists": False}}
    cursor = movies_col.find(query).limit(50)

    count = 0
    for doc in cursor:
        vector = get_single_embedding(doc["fullplot"])
        if vector:
            movies_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {VECTOR_FIELD_PATH: vector}}
            )
            count += 1

    print(f"--- Done! Updated {count} movies ---")


# ===============================
# 4. API ENDPOINTS
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
            return {
                "predicted_genre": "Unknown",
                "message": "No similar movies found"
            }

        # Step 3: majority vote genre
        all_genres = []
        for n in neighbors:
            all_genres.extend(n.get("genres", []))

        predicted_genre = max(set(all_genres), key=all_genres.count) if all_genres else "Unknown"

        return {
            "input_description": query.description,
            "predicted_genre": predicted_genre,
            "confidence_score": neighbors[0].get("score", 0),
            "similar_movies": [
                {
                    "title": n.get("title"),
                    "genres": n.get("genres"),
                    "score": n.get("score")
                }
                for n in neighbors
            ]
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
        "port": PORT,
        "model": EMBEDDING_MODEL,
        "db_connected": DB_NAME in client.list_database_names()
    }


# ===============================
# 5. MAIN
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
