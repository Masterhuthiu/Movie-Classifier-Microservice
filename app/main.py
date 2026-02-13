from pymongo import MongoClient
from urllib.parse import quote_plus
import certifi

import google.genai as genai  # SDK Google Gemini
from google.genai.errors import APIError

# ===============================
# CẤU HÌNH MONGO & GEMINI API
# ===============================
username = quote_plus("masterhuthiu")  # Thay bằng username MongoDB
password = quote_plus("123456a@A")  # Thay bằng mật khẩu MongoDB

# Thay bằng Khóa API Google Gemini của bạn
GEMINI_API_KEY = "AIzaSyCHYvBTr8nsPIOeg-uxJZs5O9xxWsnodog
"  

uri = f"mongodb+srv://{username}:{password}@cluster0.3jl7a.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, tlsCAFile=certifi.where())
db = client.sample_mflix
movies_collection = db.movies

# ===============================
# KHỞI TẠO GEMINI CLIENT
# ===============================
EMBEDDING_MODEL = "text-embedding-004"  # 768 chiều

try:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    print("--- Khởi tạo Gemini Client thành công ---")
except Exception as e:
    print(f"LỖI: Không thể khởi tạo Gemini Client. Vui lòng kiểm tra GEMINI_API_KEY. Lỗi: {e}")
    client.close()
    exit()

# ===============================
# HÀM TẠO EMBEDDING
# ===============================
def create_embedding(text):
    """Tạo vector embedding 768 chiều dưới dạng list float."""
    if not text:
        return None
    try:
        response = ai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
        )
        # Lấy values của embedding đầu tiên (list float)
        embedding_vector = response.embeddings[0].values
        return embedding_vector

    except APIError as e:
        print(f"Lỗi khi tạo embedding từ Gemini API: {e.error.message}")
        return None
    except Exception as e:
        print(f"Lỗi không xác định khi tạo embedding: {e}")
        return None

# ===============================
# TẠO EMBEDDING CHO TOÀN BỘ DOCUMENT (CHỈ CHẠY 1 LẦN)
# ===============================
def generate_embeddings_for_collection():
    print("--- Bắt đầu tạo embeddings cho trường 'fullplot' ---")
    for doc in movies_collection.find():
        if not doc.get("fullplot"):
            continue
        vector = create_embedding(doc["fullplot"])
        if vector:
            movies_collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"fullplot_gemini_embedding": vector}}
            )
            print(f"✅ Đã tạo embedding cho: {doc['title']}")
    print("--- Hoàn tất tạo embeddings ---")

# ===============================
# HÀM TRUY VẤN VECTOR
# ===============================
VECTOR_INDEX_NAME = "gemini_vector_index"
VECTOR_FIELD_PATH = "fullplot_gemini_embedding"

def vector_search_query(user_query, index_name=VECTOR_INDEX_NAME, path=VECTOR_FIELD_PATH):
    """Thực hiện truy vấn tìm kiếm ngữ nghĩa."""
    
    # 1. Tạo vector embedding cho truy vấn
    query_embedding = create_embedding(user_query)
    
    if query_embedding is None:
        print("Truy vấn không thể thực hiện do không tạo được vector embedding.")
        return 

    # 2. Aggregation Pipeline cho Vector Search
    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name, 
                "path": path, 
                "queryVector": query_embedding, 
                "numCandidates": 100, 
                "limit": 5 
            }
        },
        {"$project": {"_id": 0, "title": 1, "fullplot": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]

    print(f"\n--- Tìm kiếm Vector cho: '{user_query}' ---")
    results = movies_collection.aggregate(pipeline)
    
    found = False
    for doc in results:
        found = True
        print(f"✅ [Score: {doc['score']:.4f}] Tiêu đề: {doc['title']}")
        print(f"  > Tóm tắt: {doc['fullplot'][:200]}...")

    if not found:
        print(f"❌ Không tìm thấy kết quả liên quan trong index '{index_name}'.")
        print(f"   Vui lòng kiểm tra cấu hình MongoDB Atlas: 1. Đã điền dữ liệu vector vào trường '{path}' chưa? 2. Index đã được tạo với dimensions: 768 chưa?")

# ===============================
# THỰC THI
# ===============================
if __name__ == "__main__":
    # 1. Tạo embedding cho tất cả movies (chỉ chạy 1 lần)
    generate_embeddings_for_collection()

    # 2. Ví dụ truy vấn vector
    vector_search_query("A sorrowful film about loss and heartbreak")
    vector_search_query("A fun, fast-paced comedy for the whole family")

    client.close()
    print("\n--- KẾT THÚC CHƯƠNG TRÌNH ---")
