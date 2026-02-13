import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_TITLE: str = "Movie Classifier Service"
    PORT: int = int(os.getenv("PORT", 8083))
    
    MONGO_URI: str = os.getenv("MONGO_URI")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    
    # Atlas Vector Search Config
    VECTOR_INDEX_NAME: str = "gemini_vector_index"

settings = Settings()