from fastapi import APIRouter, HTTPException, Body, Depends
from typing import List, Optional

from .models import (
    DocumentInput, DocumentOutput, DocumentMinimalOutput,
    ChatMessageInput, ChatMessageOutput, RetrievedDocInfo, HealthStatus
)
from app.core.config import settings
from app.data import storage as mongo_storage
# vector_retriever is still needed for document upsert and index management via its LangChain compatible functions
from app.retrieval import vector_retriever
from app.services import conversation_manager as cm
from app.llm.rag_chain import invoke_rag_chain # New RAG chain
# from langchain_core.documents import Document as LangchainDocument # For type hinting, if needed

# --- Router Initialization ---
router = APIRouter()

# Note: The Health Check endpoint might need adjustments based on how clients (OpenAI, Pinecone)
# are now initialized or accessed, especially if the old global clients in vector_retriever
# are removed or changed due. LangChain components often manage their own clients.
# For OpenAI, we can check vector_retriever.lc_embeddings_model.
# For Pinecone, vector_retriever.pinecone_admin_client (for admin tasks) and
# the vector_store instance from get_vector_store (for data plane) can be checked.

# --- Helper Functions & Dependencies ---

# Dependency to ensure MongoDB is connected
# This is a simplified example. In a larger app, you might manage this
# with startup/shutdown events or a more sophisticated dependency system.
async def get_mongo_db():
    if mongo_storage.db is None:
        try:
            mongo_storage.connect_to_db()
        except ConnectionError as e:
            raise HTTPException(status_code=503, detail=f"MongoDB connection failed: {e}")
    return mongo_storage.db


@router.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    """
    Performs a health check of the application and its connected services.
    """
    status = HealthStatus(status="ok")

    # Check MongoDB
    try:
        if mongo_storage.db_client: # Check if client was initialized
            mongo_storage.db_client.admin.command('ping')
            status.mongo_status = "ok"
        else: # Try to connect if not initialized
            mongo_storage.connect_to_db() # This will raise ConnectionError if it fails
            mongo_storage.db_client.admin.command('ping')
            status.mongo_status = "ok"
    except Exception as e:
        status.mongo_status = f"error: {e}"

    # Check Redis
    try:
        if cm.redis_client and cm.redis_client.ping():
            status.redis_status = "ok"
        else:
            status.redis_status = "error: client not initialized or ping failed"
    except Exception as e:
        status.redis_status = f"error: {e}"

    # Check OpenAI (via LangChain embeddings model)
    try:
        if vector_retriever.lc_embeddings_model and hasattr(vector_retriever.lc_embeddings_model, 'client'):
            # Perform a lightweight check, e.g., try to embed a short string if there's no dedicated ping.
            # For now, just checking if the model object exists is a basic check.
            # A more robust check would involve a small API call if available and not costly.
            # vector_retriever.lc_embeddings_model.embed_query("health check") # Example, can be costly
            status.openai_status = "ok (LangChain OpenAIEmbeddings initialized)"
        elif vector_retriever.lc_embeddings_model:
             status.openai_status = "ok (LangChain OpenAIEmbeddings initialized, but client attribute not found for deeper check)"
        else:
            status.openai_status = "error: LangChain OpenAIEmbeddings model not initialized"
    except Exception as e:
        status.openai_status = f"error: {e}"

    # Check Pinecone Client (admin client for setup) and Index (via get_vector_store)
    try:
        if vector_retriever.pinecone_admin_client:
            # Try listing indexes as a health check for the admin client
            vector_retriever.pinecone_admin_client.list_indexes()
            status.pinecone_status = "ok (Admin client responsive)"
        else:
            status.pinecone_status = "error: Pinecone Admin client not initialized"

        index_name = settings.PINECONE_INDEX_NAME
        if index_name:
            # Check if vector store can be initialized (which implies index is accessible)
            # This also attempts to create the index if it doesn't exist, which can be slow.
            # For a health check, it might be better to just check existence without creation.
            # However, get_vector_store in vector_retriever.py already includes create_if_not_exists.
            vs = vector_retriever.get_vector_store(index_name)
            if vs : #and vs.index: # LangchainPinecone might not expose .index directly
                # To check if the index is truly operational, a small query could be attempted,
                # but that might be too much for a health check.
                # vs.similarity_search("health", k=1) # Example, too intensive
                status.pinecone_index_status = f"ok (VectorStore for '{index_name}' connectable)"
            else:
                status.pinecone_index_status = f"warning: VectorStore for '{index_name}' not connectable/creatable."
        else:
            status.pinecone_index_status = "warning: PINECONE_INDEX_NAME not set"
            
    except Exception as e:
        # If pinecone_admin_client check failed, this will also catch it.
        if status.pinecone_status == "pending": # Only update if not already set by admin client check
             status.pinecone_status = f"error: {e}"
        status.pinecone_index_status = f"error checking index: {e}"
        
    return status


@router.post("/documents/", response_model=DocumentMinimalOutput, status_code=201, tags=["Documents"])
async def add_document_endpoint(
    doc_input: DocumentInput = Body(...),
    # mongo_db_conn = Depends(get_mongo_db) # Example of DB dependency
):
    """
    Adds a new document to the MongoDB storage and its embedding to Pinecone.
    """
    # Ensure MongoDB is connected (if not using Depends for every route)
    if mongo_storage.db is None:
        try:
            mongo_storage.connect_to_db()
        except ConnectionError as e:
            raise HTTPException(status_code=503, detail=f"MongoDB connection error: {e}")

    # 1. Add document to MongoDB
    doc_id = mongo_storage.add_document(content=doc_input.content, metadata=doc_input.metadata)
    if not doc_id:
        raise HTTPException(status_code=500, detail="Failed to add document to MongoDB.")

    # 2. Upsert document embedding to Pinecone using the refactored vector_retriever
    # The get_vector_store function (called by upsert_document_embedding)
    # will ensure the index exists or attempt to create it.
    
    # The existing vector_retriever.upsert_document_embedding now uses LangChain.
    # It expects doc_id, content, and metadata.
    success_pinecone = vector_retriever.upsert_document_embedding(
        doc_id=doc_id, # Pass doc_id from MongoDB
        content=doc_input.content,
        metadata=doc_input.metadata or {} # Ensure metadata is a dict
    )

    if not success_pinecone:
        # Depending on desired behavior, you might raise an error or return a partial success
        # For now, return success for Mongo, but with a message about Pinecone failure.
        # The error logging is done within upsert_document_embedding.
        return DocumentMinimalOutput(doc_id=doc_id, message="Document added to MongoDB, but failed to upsert embedding to Pinecone.")

    return DocumentMinimalOutput(doc_id=doc_id, message="Document added to MongoDB and embedding upserted to Pinecone successfully.")


@router.post("/chat/", response_model=ChatMessageOutput, tags=["Chat"])
async def chat_endpoint(
    chat_input: ChatMessageInput = Body(...),
    # mongo_db_conn = Depends(get_mongo_db) # Ensure DB is available if needed by other parts
):
    """
    Handles a user's chat message. Retrieves relevant documents, generates a placeholder AI response,
    and stores the conversation turn.
    """
    session_id = chat_input.session_id
    user_message = chat_input.user_message

    # 1. Invoke the RAG chain
    rag_response = invoke_rag_chain(user_message)

    ai_response_text: str
    retrieved_docs_output: List[RetrievedDocInfo] = []

    if rag_response:
        ai_response_text = rag_response.get("answer", "Error: No answer found in RAG response.")
        
        # The 'context' from rag_response is a list of dicts,
        # each with 'page_content' and 'metadata'.
        # 'score' is not directly available from the basic RAG chain output unless customized.
        # 'doc_id' should be in metadata if added during upsert.
        raw_context_docs = rag_response.get("context", [])
        for doc_dict in raw_context_docs:
            retrieved_docs_output.append(
                RetrievedDocInfo(
                    doc_id=doc_dict.get("metadata", {}).get("doc_id"),
                    content=doc_dict.get("page_content"),
                    metadata=doc_dict.get("metadata"),
                    score=doc_dict.get("score") # Score might be None if not provided by chain
                )
            )
    else:
        # Fallback if RAG chain invocation itself fails critically
        ai_response_text = "I encountered an error trying to process your request. Please try again later."
        # No documents retrieved in this case
    
    # Ensure ai_response_text is never None for history saving.
    if ai_response_text is None:
        ai_response_text = "Error: LLM returned an empty response."


    # 2. Add user message and AI response to conversation history
    # cm.add_message_to_history now expects LangChain BaseMessage objects if we were to align strictly,
    # but it was refactored to accept strings (user_message, ai_message) and convert them internally.
    # So, this should still work.
    if not cm.add_message_to_history(session_id, user_message, ai_response_text):
        print(f"Warning: Failed to save message to history for session {session_id}.")

    return ChatMessageOutput(
        session_id=session_id,
        user_message=user_message,
        ai_response=ai_response_text,
        retrieved_docs=retrieved_docs_output
    )

# Example of how to get full document content (not required by current subtask spec but useful)
@router.get("/documents/{doc_id}", response_model=DocumentOutput, tags=["Documents"])
async def get_document_endpoint(
    doc_id: str,
    # mongo_db_conn = Depends(get_mongo_db)
):
    """
    Retrieves a document by its ID from MongoDB.
    """
    # Ensure MongoDB is connected
    if mongo_storage.db is None:
        try:
            mongo_storage.connect_to_db()
        except ConnectionError as e:
            raise HTTPException(status_code=503, detail=f"MongoDB connection error: {e}")

    document_data = mongo_storage.get_document(doc_id)
    if not document_data:
        raise HTTPException(status_code=404, detail=f"Document with ID '{doc_id}' not found.")
    
    # Convert MongoDB's _id to string if it's an ObjectId, and ensure all fields are present
    # Our current storage.py uses string _id, so this might not be strictly needed here.
    return DocumentOutput(
        doc_id=document_data.get("doc_id", str(document_data.get("_id"))), # Use doc_id if present
        content=document_data.get("content"),
        metadata=document_data.get("metadata", {})
    )
