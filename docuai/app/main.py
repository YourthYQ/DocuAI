from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api import endpoints as api_endpoints
from app.core.config import settings
from app.data import storage as mongo_storage
from app.retrieval import vector_retriever # To ensure clients are initialized on import
from app.services import conversation_manager as cm # To ensure Redis client is initialized

# --- App Lifespan Management (Startup/Shutdown Events) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("--- Application Startup ---")
    print(f"Project: {settings.PROJECT_NAME}, Version: {settings.VERSION}")

    # Initialize MongoDB Connection
    try:
        print("Initializing MongoDB connection...")
        mongo_storage.connect_to_db() # Uses settings from config.py
        # Perform a simple check
        mongo_storage.db_client.admin.command('ping')
        print("MongoDB connection successful.")
    except Exception as e:
        print(f"Error connecting to MongoDB during startup: {e}")
        # Depending on policy, you might want to prevent startup or allow it to continue
        # For now, we'll print the error and continue; health check will report it.

    # OpenAI client is initialized when vector_retriever is imported.
    if vector_retriever.openai_client:
        print("OpenAI client was initialized.")
    else:
        print("Warning: OpenAI client failed to initialize. Check OPENAI_API_KEY.")

    # Pinecone client is initialized when vector_retriever is imported.
    if vector_retriever.pinecone_client:
        print("Pinecone client was initialized.")
        # Optionally, ensure the main index exists or try to create it
        index_name = settings.PINECONE_INDEX_NAME
        if index_name:
            print(f"Checking/Creating Pinecone index: {index_name}...")
            # Dimension for text-embedding-ada-002 is 1536
            # This could take time if the index needs to be created.
            if not vector_retriever.get_pinecone_index(index_name):
                 if vector_retriever.create_pinecone_index_if_not_exists(index_name, dimension=1536):
                     print(f"Pinecone index '{index_name}' created or ensured.")
                 else:
                     print(f"Warning: Failed to create or ensure Pinecone index '{index_name}' during startup.")
            else:
                print(f"Pinecone index '{index_name}' is available.")
        else:
            print("Warning: PINECONE_INDEX_NAME not set. Pinecone index cannot be checked/created at startup.")
    else:
        print("Warning: Pinecone client failed to initialize. Check PINECONE_API_KEY and PINECONE_ENVIRONMENT.")

    # Redis client is initialized when conversation_manager is imported.
    if cm.redis_client:
        try:
            cm.redis_client.ping()
            print("Redis client was initialized and connection is responsive.")
        except Exception as e:
            print(f"Warning: Redis client initialized but ping failed: {e}")
    else:
        print("Warning: Redis client failed to initialize. Check Redis server and configuration.")

    yield # Application runs here

    # Shutdown
    print("--- Application Shutdown ---")
    mongo_storage.close_db_connection()
    print("MongoDB connection closed.")
    # Pinecone and OpenAI clients do not have explicit close methods in their current SDK versions.
    # Redis client connection pool will manage connections.
    print("Application shutdown complete.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="DocuAI: AI-powered document interaction and retrieval system.",
    lifespan=lifespan # Use the new lifespan context manager
)

# --- Include API Routers ---
app.include_router(api_endpoints.router, prefix="/api/v1") # Prefix all API routes

# --- Root Endpoint (Optional) ---
@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} v{settings.VERSION}. "
                   "Navigate to /docs for API documentation."
    }

if __name__ == "__main__":
    import uvicorn
    # This is for local development running this file directly.
    # In production, you'd use something like: uvicorn docuai.app.main:app --host 0.0.0.0 --port 8000
    print("Starting Uvicorn server for local development...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
