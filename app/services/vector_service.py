from motor.motor_asyncio import AsyncIOMotorClient
import os

class VectorService:
    def __init__(self):
        uri = os.getenv("MONGO_URI")
        self.client = AsyncIOMotorClient(uri)
        # Sử dụng database và collection của bạn
        self.db = self.client['sample_mflix']
        self.collection = self.db['movies']

    async def search_movies(self, query_vector: list):
        pipeline = [
            {
                "\$vectorSearch": {
                    "index": "movies_vector_index", # Khớp với Atlas
                    "path": "fullplot_en_embedding", # Tìm trên bản tiếng Anh
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": 5
                }
            },
            {
                "\$project": {
                    "_id": 0,
                    "title": 1,
                    "plot": 1,
                    "fullplot": 1,
                    "score": {"\$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = []
        try:
            cursor = self.collection.aggregate(pipeline)
            async for doc in cursor:
                results.append(doc)
            return results
        except Exception as e:
            print(f"Lỗi Vector Search: {str(e)}")
            raise e

vector_service = VectorService()