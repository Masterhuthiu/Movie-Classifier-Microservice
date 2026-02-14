import os
import certifi
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from urllib.parse import quote_plus
from google import genai

# ===============================
# CONFIG
# ===============================

username = quote_plus("masterhuthiu")
password = quote_plus("123456a@A")

MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.3jl7a.mongodb.net/?retryWrites=true&w=majority"

DB_NAME = "sample_mflix"
COLLECTION = "movies"

VECTOR_INDEX = "movies_vector_index"
VECTOR_FIELD = "fullplot_gemini_embedding"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "PUT_KEY_HERE")

# ⚠️ GIỮ 768 để khớp Atlas
EMBED_MODEL = "models/gemini-embedding-001"
VECTOR_DIM = 768

# ===============================
# INIT SERVICES
# ===============================

client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
col = db[COLLECTION]

ai = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Movie Semantic Search Service")

print("✅ Microservice ready (MongoDB + Gemini)")

# ===============================
# REQUEST MODEL
# ===============================

class SearchRequest(BaseModel):
    query: str
    limit: int = 5


# ===============================
# CREATE EMBEDDING (CUT → 768)
# ===============================

def get_embedding(text: str) -> List[float] | None:
    if not text:
        return None

    try:
        res = ai.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )

        vec = res.embeddings[0].values

        # ⚠️ cắt về 768 để KHỚP index Atlas
        return vec[:VECTOR_DIM]

    except Exception as e:
        print("❌ Gemini error:", e)
        return None


# ===============================
# VECTOR SEARCH LOGIC
# ===============================

def semantic_search(query_text: str, limit: int):
    query_vec = get_embedding(query_text)

    if not query_vec:
        raise HTTPException(status_code=500, detail="Embedding failed")

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": VECTOR_FIELD,
                "queryVector": query_vec,
                "numCandidates": 100,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "title": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    return list(col.aggregate(pipeline))


# ===============================
# API ENDPOINT
# ===============================

@app.post("/search")
def search_movies(req: SearchRequest):
    results = semantic_search(req.query, req.limit)

    if not results:
        return []

    return results


@app.get("/health")
def health():
    return {"status": "ok"}
