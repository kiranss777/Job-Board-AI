import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the RAG application"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY")  # NEW: JSearch API key
    
    # Snowflake Configuration
    SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
    SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
    SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
    SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
    SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
    SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
    SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE", "PUBLIC")
    SNOWFLAKE_TABLE = os.getenv("SNOWFLAKE_TABLE", "H1B_EMPLOYER_ANALYTICS_TABLE")  # Default table name
    
    # Pinecone Configuration
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ads7390")
    PINECONE_HOST = os.getenv("PINECONE_HOST")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
    
    # LLM Models
    
    LLM_MODELS = {
    "GPT-4o": "gpt-4o-mini",
    "Gemini Pro": "gemini-2.5-flash",
    "Gemini Flash": "gemini-2.5-flash",
    "DeepSeek": "deepseek-chat",
    }

    
    # Embedding Configuration - Using sentence-transformers
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model
    EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2
    
    # Chunking Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval Configuration
    TOP_K = 5  # Number of relevant chunks to retrieve
    
    # Batch Configuration
    BATCH_SIZE = 50
    MAX_RETRIES = 3
    
    # Job Search Configuration (NEW)
    JOB_SEARCH_MODEL = "DeepSeek"  # Use DeepSeek for job search features
    DEFAULT_JOB_RESULTS = 20  # Default number of jobs to fetch
    
    @staticmethod
    def validate():
        """Validate that all required API keys are set"""
        required_keys = [
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "DEEPSEEK_API_KEY",
            "PINECONE_API_KEY",
            "JSEARCH_API_KEY"  # NEW: Added to required keys
        ]
        
        missing_keys = []
        for key in required_keys:
            if not getattr(Config, key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        return True