import os
import time
# Pinecone client for index management
from pinecone import Pinecone, ApiException as PineconeApiException
from pinecone.core.client.exceptions import NotFoundException as PineconeNotFoundException

from app.core.config import settings

# LangChain components
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore as LangchainPinecone  # Alias to avoid confusion
from langchain_core.documents import Document # For creating documents to upsert

# --- OpenAI Client Initialization (for direct use if needed, though LangChain components manage their own) ---
# The OpenAI client previously initialized here (openai_client) for get_embedding
# is no longer strictly necessary if all embedding is done via LangChain's OpenAIEmbeddings.
# However, LangChain's OpenAIEmbeddings will need OPENAI_API_KEY.
# We can remove the old openai_client global if it's not used elsewhere directly.
# For now, LangChain components will instantiate their own clients based on environment settings.
print(f"OpenAI API Key found: {'Yes' if settings.OPENAI_API_KEY else 'No'}")


# --- Pinecone Client Initialization (for index management) ---
# This client is used for administrative tasks like creating/deleting indexes.
# LangchainPinecone will use its own internal pinecone client logic for vector operations.
pinecone_admin_client = None # Renamed to distinguish from LangChain's internal client
if settings.PINECONE_API_KEY and settings.PINECONE_ENVIRONMENT:
    try:
        pinecone_admin_client = Pinecone(api_key=settings.PINECONE_API_KEY) # environment is implicitly handled or not needed for v3 control plane
        print(f"Pinecone Admin client initialized successfully.")
        # Example: list indexes to test
        # print(f"Available Pinecone indexes: {pinecone_admin_client.list_indexes().names}")
    except PineconeApiException as e:
        print(f"Error initializing Pinecone Admin client: {e}")
        pinecone_admin_client = None
    except Exception as e:
        print(f"An unexpected error occurred during Pinecone Admin client initialization: {e}")
        pinecone_admin_client = None
else:
    print("PINECONE_API_KEY or PINECONE_ENVIRONMENT not found/set in settings. Pinecone Admin client not initialized.")


# --- Embedding Model Initialization (LangChain) ---
# This single embeddings instance can be reused.
# It will pick up OPENAI_API_KEY from the environment (via settings).
lc_embeddings_model = None
if settings.OPENAI_API_KEY:
    try:
        lc_embeddings_model = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY, model="text-embedding-ada-002")
        print("LangChain OpenAIEmbeddings model initialized successfully.")
    except Exception as e:
        print(f"Error initializing LangChain OpenAIEmbeddings: {e}")
else:
    print("OPENAI_API_KEY not set. LangChain OpenAIEmbeddings model not initialized.")


def create_pinecone_index_if_not_exists(index_name: str, dimension: int = 1536, metric: str = 'cosine'):
    """
    Creates a Pinecone index if it doesn't already exist using the pinecone_admin_client.
    The dimension should match the OpenAI embedding dimension (e.g., 1536 for text-embedding-ada-002).
    """
    if not pinecone_admin_client:
        print("Pinecone Admin client not initialized. Cannot create index.")
        return False

    try:
        indexes = pinecone_admin_client.list_indexes()
        if index_name in [idx.name for idx in indexes.indexes]: # Updated for v3 client
            print(f"Pinecone index '{index_name}' already exists.")
            return True
    except PineconeApiException as e:
        print(f"Error listing Pinecone indexes: {e}. Assuming index might not exist.")
        # Continue to try creation if listing fails, maybe due to permissions for listing but not for describe/create
    
    print(f"Pinecone index '{index_name}' not found or listing failed. Attempting to create...")
    try:
        pinecone_admin_client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec={ # Using ServerlessSpec as an example, adjust if using PodSpec
                "serverless": {
                    "cloud": "aws", # or "gcp", "azure"
                    "region": settings.PINECONE_ENVIRONMENT or "us-east-1" # Ensure PINECONE_ENVIRONMENT is a valid region
                }
            }
            # For pod-based:
            # spec={"pod": {"environment": settings.PINECONE_ENVIRONMENT, "pod_type": "p1.x1"}}
        )
        # Wait for index to be ready
        max_retries = 15 # Increased retries
        delay = 10 # seconds
        for i in range(max_retries):
            try:
                index_description = pinecone_admin_client.describe_index(name=index_name)
                if index_description.status['ready']: # Updated for v3 client
                    print(f"Pinecone index '{index_name}' created and ready.")
                    return True
            except PineconeApiException as e_desc: # Catch error during describe_index specifically
                 print(f"Index '{index_name}' not fully ready or describe failed (Attempt {i+1}/{max_retries}): {e_desc}")

            print(f"Waiting for index '{index_name}' to be ready... Attempt {i+1}/{max_retries}")
            time.sleep(delay)
        
        print(f"Pinecone index '{index_name}' creation timed out or status check failed.")
        return False
    except PineconeApiException as e:
        if "already exists" in str(e).lower(): # Check if error is due to index already existing
            print(f"Pinecone index '{index_name}' likely already exists (based on error): {e}")
            return True # If it already exists, that's fine for our purpose
        print(f"Pinecone API error while creating index '{index_name}': {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while creating index '{index_name}': {e}")
        return False

# Global cache for vector store instances to avoid re-initialization
_vector_store_cache = {}

def get_vector_store(index_name: str = None) -> LangchainPinecone | None:
    """
    Initializes and returns a LangChain PineconeVectorStore instance.
    Caches instances to avoid repeated initializations for the same index.
    """
    global _vector_store_cache
    target_index_name = index_name or settings.PINECONE_INDEX_NAME

    if not target_index_name:
        print("Error: Pinecone index name not provided or found in settings.")
        return None
    if not lc_embeddings_model:
        print("Error: LangChain OpenAIEmbeddings model not initialized. Cannot create vector store.")
        return None
    # Pinecone client for data plane operations is managed by PineconeVectorStore itself using API keys from env.

    if target_index_name in _vector_store_cache:
        # print(f"Returning cached PineconeVectorStore for index '{target_index_name}'.")
        return _vector_store_cache[target_index_name]

    # Ensure the index exists before attempting to use it with LangChain
    # This relies on the admin client and our helper function.
    # create_pinecone_index_if_not_exists might be slow if the index needs creation.
    # Consider moving this to an application startup routine or ensuring index exists beforehand.
    if not create_pinecone_index_if_not_exists(target_index_name):
        print(f"Pinecone index '{target_index_name}' does not exist and could not be created. Cannot initialize vector store.")
        return None

    print(f"Initializing LangChain PineconeVectorStore for index: {target_index_name}...")
    try:
        vector_store = LangchainPinecone.from_existing_index(
            index_name=target_index_name,
            embedding=lc_embeddings_model
            # Pinecone API key and environment are typically picked up from environment variables
            # by the underlying Pinecone client used by LangchainPinecone.
            # If explicit passing is needed:
            # pinecone_api_key=settings.PINECONE_API_KEY,
            # pinecone_environment=settings.PINECONE_ENVIRONMENT # This might not be needed for from_existing_index with new client
        )
        _vector_store_cache[target_index_name] = vector_store
        print(f"LangChain PineconeVectorStore for index '{target_index_name}' initialized successfully.")
        return vector_store
    except Exception as e:
        print(f"Error initializing LangChain PineconeVectorStore for index '{target_index_name}': {e}")
        return None


def upsert_document_embedding(doc_id: str, content: str, metadata: dict = None):
    """
    Upserts a document (text and metadata) into the Pinecone index
    using LangChain's PineconeVectorStore.
    """
    vector_store = get_vector_store() # Uses default index from settings
    if not vector_store:
        print(f"Failed to get vector store. Cannot upsert document {doc_id}.")
        return False

    # LangChain expects a list of Document objects or texts.
    # Metadata should be a dictionary. Ensure `doc_id` is part of metadata for later retrieval.
    doc_metadata = metadata or {}
    doc_metadata["doc_id"] = doc_id # Ensure doc_id is in the metadata we store

    # Storing the full content in metadata is also good practice if needed directly from there
    doc_metadata["original_content"] = content

    try:
        # Using add_texts with explicit IDs.
        # Alternatively, use vector_store.add_documents([Document(page_content=content, metadata=doc_metadata, id=doc_id)])
        # but add_texts is more direct if you have texts, ids, and metadatas.
        vector_store.add_texts(texts=[content], ids=[doc_id], metadatas=[doc_metadata])
        print(f"Upserted document {doc_id} via LangChain PineconeVectorStore.")
        return True
    except Exception as e:
        print(f"LangChain error while upserting document {doc_id}: {e}")
        return False


def query_vector_store(query_text: str, top_k: int = 5, filter_criteria: dict = None) -> list[dict] | None:
    """
    Queries the Pinecone vector store using LangChain.
    Returns a list of results (doc_id, score, metadata, content).
    """
    vector_store = get_vector_store() # Uses default index
    if not vector_store:
        print("Failed to get vector store. Cannot query.")
        return None

    try:
        # Use similarity_search_with_score to get scores
        # The filter_criteria needs to be in Pinecone's metadata filter DSL if passed directly.
        # LangChain might have its own way to pass filters or this might be passed to underlying client.
        # For PineconeVectorStore, filter is usually passed directly.
        results_with_scores = vector_store.similarity_search_with_score(
            query=query_text,
            k=top_k,
            filter=filter_criteria
        )
        
        output_results = []
        for doc, score in results_with_scores:
            # doc is a LangChain Document object
            output_results.append({
                "doc_id": doc.metadata.get("doc_id", None), # Retrieve our custom doc_id
                "score": score,
                "content": doc.page_content, # LangChain Document stores text in page_content
                "metadata": doc.metadata
            })
        
        print(f"LangChain query returned {len(output_results)} results.")
        return output_results
    except Exception as e:
        print(f"LangChain error while querying vector store: {e}")
        return None

def delete_pinecone_index_lc(index_name: str): # Renamed to avoid conflict if old one is kept temporarily
    """Deletes a Pinecone index using the admin client."""
    if not pinecone_admin_client:
        print("Pinecone Admin client not initialized. Cannot delete index.")
        return False
    try:
        indexes = pinecone_admin_client.list_indexes()
        if index_name not in [idx.name for idx in indexes.indexes]:
             print(f"Pinecone index '{index_name}' not found, no need to delete.")
             return True

        print(f"Attempting to delete Pinecone index '{index_name}'...")
        pinecone_admin_client.delete_index(index_name)
        
        # Wait for deletion
        max_retries = 10
        delay = 10 # seconds
        for i in range(max_retries):
            time.sleep(delay)
            current_indexes = pinecone_admin_client.list_indexes()
            if index_name not in [idx.name for idx in current_indexes.indexes]:
                print(f"Pinecone index '{index_name}' deleted successfully.")
                if index_name in _vector_store_cache: # Clear from cache if deleted
                    del _vector_store_cache[index_name]
                return True
            print(f"Waiting for index '{index_name}' to be deleted... Attempt {i+1}/{max_retries}")
        print(f"Timeout waiting for Pinecone index '{index_name}' to be deleted.")
        return False
    except PineconeApiException as e:
        print(f"Pinecone API error while deleting index '{index_name}': {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while deleting index '{index_name}': {e}")
        return False

# --- Example Usage (Updated for LangChain) ---
if __name__ == "__main__":
    print("\n--- Running vector_retriever.py example (LangChain version) ---")

    if not lc_embeddings_model or not pinecone_admin_client :
        print("OpenAI Embeddings or Pinecone Admin client not initialized. Exiting example.")
        print("Please ensure OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_ENVIRONMENT are set.")
    else:
        # Use a dedicated test index name from settings or append -lc-test
        TEST_INDEX_NAME_LC = getattr(settings, 'PINECONE_INDEX_NAME', 'docuai-index') + "-lc-test"
        print(f"Using Pinecone test index: {TEST_INDEX_NAME_LC}")
        settings.PINECONE_INDEX_NAME = TEST_INDEX_NAME_LC # Override for this test session

        # 0. Ensure test index is clean or does not exist initially
        print(f"\n--- Ensuring test index '{TEST_INDEX_NAME_LC}' is clean for test run ---")
        delete_pinecone_index_lc(TEST_INDEX_NAME_LC) # Delete if exists
        time.sleep(5) # Give some time if deletion is slow

        # 1. Get Vector Store (this will also create the index if it doesn't exist)
        print(f"\n--- Getting LangChain Vector Store (and creating index '{TEST_INDEX_NAME_LC}' if needed) ---")
        vector_store_instance = get_vector_store(index_name=TEST_INDEX_NAME_LC)
        if not vector_store_instance:
            print(f"Failed to get/create vector store for index '{TEST_INDEX_NAME_LC}'. Exiting example.")
            exit()
        
        print(f"Successfully got vector store for index: {vector_store_instance.index_name}")

        # 2. Add Sample Documents using LangChain
        print("\n--- Adding Sample Documents (LangChain) ---")
        sample_lc_docs_data = [
            {"id": "lc_doc1", "content": "The quick brown fox jumps over the lazy dog in the context of LangChain.", "metadata": {"source": "lc_classic", "chapter": "1"}},
            {"id": "lc_doc2", "content": "LangChain provides abstractions for AI applications.", "metadata": {"source": "lc_tech", "framework": "LangChain"}},
            {"id": "lc_doc3", "content": "PineconeVectorStore integrates Pinecone with LangChain for vector search.", "metadata": {"source": "lc_pinecone_docs", "component": "PineconeVectorStore"}},
        ]

        for doc_data in sample_lc_docs_data:
            # We need to pass doc_id also in metadata if we want to retrieve it from Document.metadata later
            meta_with_id = {**doc_data["metadata"], "doc_id": doc_data["id"]}
            success = upsert_document_embedding(
                doc_id=doc_data["id"], 
                content=doc_data["content"], 
                metadata=meta_with_id
            )
            if success:
                print(f"Successfully upserted {doc_data['id']} via LangChain.")
            else:
                print(f"Failed to upsert {doc_data['id']} via LangChain.")
        
        # It might take a few moments for upserted vectors to be queryable
        print("Waiting a few seconds for embeddings to be indexed by Pinecone...")
        time.sleep(10) # Pinecone indexing can take a moment

        # 3. Perform a Sample Query using LangChain
        print("\n--- Performing Sample Query (LangChain) ---")
        query_lc = "What is LangChain?"
        query_results_lc = query_vector_store(query_text=query_lc, top_k=2)

        if query_results_lc:
            print(f"\nQuery: '{query_lc}'")
            print("Results (LangChain):")
            for res_lc in query_results_lc:
                print(f"  ID: {res_lc.get('doc_id')}, Score: {res_lc.get('score'):.4f}, Content: '{res_lc.get('content', '')[:50]}...'")
                print(f"  Metadata: {res_lc.get('metadata')}")
        else:
            print(f"No results for LangChain query: '{query_lc}' or an error occurred.")
        
        # Test with filter
        query_filtered_lc = "Tell me about PineconeVectorStore"
        # Pinecone metadata filter syntax: {"key": "value"}
        filter_criteria_lc = {"component": "PineconeVectorStore"}
        query_results_filtered_lc = query_vector_store(query_text=query_filtered_lc, top_k=1, filter_criteria=filter_criteria_lc)

        if query_results_filtered_lc:
            print(f"\nQuery (filtered): '{query_filtered_lc}' for component 'PineconeVectorStore'")
            print("Results (LangChain, filtered):")
            for res_f_lc in query_results_filtered_lc:
                print(f"  ID: {res_f_lc.get('doc_id')}, Score: {res_f_lc.get('score'):.4f}, Content: '{res_f_lc.get('content', '')[:50]}...'")
                print(f"  Metadata: {res_f_lc.get('metadata')}")
        else:
            print(f"No results for filtered LangChain query: '{query_filtered_lc}' or an error occurred.")


        # 4. Clean up by deleting the test index
        print(f"\n--- Deleting Test Index (LangChain): {TEST_INDEX_NAME_LC} ---")
        if delete_pinecone_index_lc(TEST_INDEX_NAME_LC):
            print(f"Test index '{TEST_INDEX_NAME_LC}' deleted successfully.")
        else:
            print(f"Failed to delete test index '{TEST_INDEX_NAME_LC}'. Please delete it manually if needed.")

    print("\n--- vector_retriever.py example (LangChain version) finished ---")
