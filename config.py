import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Qdrant settings
    QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
    QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    
    # OpenAI settings (for embeddings)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://aipipe.org/openai/v1/')
    
    # Embedding settings
    EMBEDDING_MODEL = 'text-embedding-3-small'  # or 'text-embedding-ada-002'
    EMBEDDING_DIMENSION = 1536  # for text-embedding-3-small
    
    # Local embedding alternative
    LOCAL_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    LOCAL_EMBEDDING_DIMENSION = 384
    
    # Processing settings
    BATCH_SIZE = 100
    COLLECTION_NAME = 'forum_posts'