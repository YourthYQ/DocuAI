import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """
    Application settings.
    Values are loaded from environment variables or .env file.
    """
    PROJECT_NAME: str = "DocuAI"
    VERSION: str = "0.1.0"

    # MongoDB Settings
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "docuai_db")

    # OpenAI API Key
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    # Pinecone Settings
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT") # e.g., "gcp-starter", "us-west1-gcp"
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "docuai-index")

    # Redis Settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_CONVERSATION_DB: int = int(os.getenv("REDIS_CONVERSATION_DB", 0))
    CONVERSATION_TIMEOUT_SECONDS: int = int(os.getenv("CONVERSATION_TIMEOUT_SECONDS", 3600)) # 1 hour

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_CONVERSATION_DB}"

    # LLM Settings
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4")

    # Other settings can be added here
    # For example, settings for the vector database, logging, etc.

settings = Settings()

# You can add helper functions here if needed, for example,
# to validate settings or to provide different configurations for
# different environments (dev, test, prod).

if __name__ == "__main__":
    # This part is for testing the settings loading
    print(f"Project Name: {settings.PROJECT_NAME}")
    print(f"Version: {settings.VERSION}")
    print(f"MongoDB URI: {settings.MONGO_URI}")
    print(f"MongoDB Database Name: {settings.MONGO_DB_NAME}")
    print(f"OpenAI API Key: {'*' * len(settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else 'Not set'}")
    print(f"Pinecone API Key: {'*' * len(settings.PINECONE_API_KEY) if settings.PINECONE_API_KEY else 'Not set'}")
    print(f"Pinecone Environment: {settings.PINECONE_ENVIRONMENT}")
    print(f"Pinecone Index Name: {settings.PINECONE_INDEX_NAME}")
    print(f"Redis Host: {settings.REDIS_HOST}")
    print(f"Redis Port: {settings.REDIS_PORT}")
    print(f"Redis Conversation DB: {settings.REDIS_CONVERSATION_DB}")
    print(f"Redis URL: {settings.REDIS_URL}")
    print(f"Conversation Timeout Seconds: {settings.CONVERSATION_TIMEOUT_SECONDS}")
    print(f"Default LLM Model: {settings.DEFAULT_LLM_MODEL}")
