import os
from dotenv import load_dotenv
from google import genai

# =============================
# LOAD ENV
# =============================
load_dotenv()
API_KEY = "AIzaSyDDlIjhAUI2H1tIxzzWguWKZ3IeEysAsME" #os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("❌ Missing GOOGLE_API_KEY in .env")

print("✅ API key loaded")

# =============================
# CREATE CLIENT
# =============================
client = genai.Client(api_key=API_KEY)

# =============================
# 1️⃣ TEST CHAT MODEL (Gemini 2.5)
# =============================
print("\n=== TEST CHAT MODEL ===")

try:
    chat_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in a friendly way."
    )

    print("✅ Chat OK")
    print("Response:", chat_response.text)

except Exception as e:
    print("❌ Chat FAILED:", e)


# =============================
# 2️⃣ TEST EMBEDDING MODEL (NEW)
# =============================
print("\n=== TEST EMBEDDING MODEL ===")

try:
    embedding_response = client.models.embed_content(
        model="gemini-embedding-001",
        contents="Hello world"
    )

    vector = embedding_response.embeddings[0].values
    print("✅ Embedding OK")
    print("Vector length:", len(vector))

except Exception as e:
    print("❌ Embedding FAILED:", e)


# =============================
# 3️⃣ LIST AVAILABLE MODELS
# =============================
print("\n=== LIST MODELS ===")

try:
    models = client.models.list()

    print("✅ List models OK")
    for m in models:
        print("-", m.name)

except Exception as e:
    print("❌ List models FAILED:", e)
