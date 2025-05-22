import os
import shutil
import uuid
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, File, HTTPException, Path, Query, UploadFile

from app.core.config import settings
from app.data import storage as mongo_storage
# Import add_chat_log_entry, list_chat_sessions, get_chat_logs_for_session directly
from app.data.storage import add_chat_log_entry, list_chat_sessions, get_chat_logs_for_session
from app.llm.rag_chain import invoke_rag_chain
from app.retrieval import vector_retriever
from app.services import conversation_manager as cm
from app.services.document_processor import (
    extract_text_from_pdf,
    extract_text_from_txt,
    split_text_into_chunks
)
from .models import (
    ChatLogEntryModel,
    ChatMessageInput,
    ChatMessageOutput,
    DocumentInput,
    DocumentMinimalOutput,
    DocumentOutput,
    FileUploadResponse,
    HealthStatus,
    RetrievedDocInfo,
    SessionDetailModel
)


# --- Router Initialization ---
router = APIRouter()

TEMP_UPLOADS_DIR = "temp_uploads" # Ensure this directory exists and is in .gitignore
os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)

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
        print(f"Warning: Failed to save message to Redis history for session {session_id}.")

    # 3. Persist chat log to MongoDB
    try:
        retrieved_doc_ids_list = [
            doc.doc_id for doc in retrieved_docs_output if doc.doc_id is not None
        ]
        
        chat_log_data = ChatLogEntryModel(
            session_id=session_id, # from chat_input
            user_message=user_message, # from chat_input
            ai_response=ai_response_text,
            retrieved_doc_ids=retrieved_doc_ids_list if retrieved_doc_ids_list else None
            # timestamp and interaction_id will use Pydantic model defaults
        )
        
        # Use .model_dump() for Pydantic v2 (instead of .dict())
        if not add_chat_log_entry(chat_log_data.model_dump()):
            print(f"Warning: Failed to save chat log to MongoDB for session {session_id}, interaction {chat_log_data.interaction_id}.")
        else:
            print(f"Successfully saved chat log to MongoDB for interaction {chat_log_data.interaction_id}.")
            
    except Exception as e:
        # Catch any unexpected errors during chat log saving & print a warning
        # This ensures that chat log saving failure does not break the chat response to the user.
        print(f"Warning: An unexpected error occurred while saving chat log to MongoDB for session {session_id}: {e}")
        # import traceback; traceback.print_exc() # For more detailed debugging if needed

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


@router.post("/documents/upload_file/", response_model=FileUploadResponse, tags=["Documents"])
async def upload_document_file(file: UploadFile = File(...)):
    """
    Handles file uploads (PDF or TXT), extracts text, chunks it, and upserts embeddings.
    """
    # Ensure temp_uploads directory exists (though done at module level, good to double check if needed)
    # os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True) 

    temp_file_path = os.path.join(TEMP_UPLOADS_DIR, f"{uuid.uuid4()}_{file.filename}")

    try:
        # 1. Save Uploaded File Temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File '{file.filename}' saved temporarily to '{temp_file_path}'")

        # 2. Determine File Type and Extract Text
        extracted_text: str
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension == ".pdf":
            extracted_text = extract_text_from_pdf(temp_file_path)
        elif file_extension == ".txt":
            extracted_text = extract_text_from_txt(temp_file_path)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: '{file_extension}'. Supported types are .pdf and .txt."
            )
        
        if not extracted_text or extracted_text.isspace():
            return FileUploadResponse(
                filename=file.filename,
                message="No text content found in the document or document is empty.",
                total_chunks_processed=0,
                document_id=None
            )

        # 3. Chunk Text
        # Using default chunk_size and chunk_overlap from document_processor for now
        text_chunks = split_text_into_chunks(extracted_text)
        if not text_chunks:
            return FileUploadResponse(
                filename=file.filename,
                message="Text extracted but resulted in no processable chunks.",
                total_chunks_processed=0,
                document_id=None
            )

        # 4. Process and Upsert Chunks
        uploaded_doc_id = f"file_{uuid.uuid4()}" # Unique ID for the entire uploaded document
        successfully_processed_chunks = 0

        for i, chunk_content in enumerate(text_chunks):
            chunk_id = f"{uploaded_doc_id}_chunk_{i}"
            chunk_metadata = {
                "original_filename": file.filename,
                "uploaded_doc_id": uploaded_doc_id,
                "chunk_number": i,
                "total_chunks": len(text_chunks)
                # Add any other document-level metadata if available/needed
            }
            
            # Upsert using the LangChain compatible function in vector_retriever
            success = vector_retriever.upsert_document_embedding(
                doc_id=chunk_id,
                content=chunk_content,
                metadata=chunk_metadata
            )
            if success:
                successfully_processed_chunks += 1
            else:
                # Log this failure for monitoring, but continue processing other chunks
                print(f"Warning: Failed to upsert chunk {chunk_id} for document {uploaded_doc_id} ('{file.filename}').")

        if successfully_processed_chunks == 0 and text_chunks:
             # This means all chunk upserts failed
            raise HTTPException(
                status_code=500,
                detail=f"Extracted {len(text_chunks)} chunks from '{file.filename}', but failed to process any of them for vector storage."
            )

        return FileUploadResponse(
            filename=file.filename,
            message=f"Successfully processed '{file.filename}'. Extracted {len(text_chunks)} chunks, {successfully_processed_chunks} processed and sent for embedding.",
            total_chunks_processed=successfully_processed_chunks,
            document_id=uploaded_doc_id
        )

    except FileNotFoundError as e:
        print(f"Error during file upload processing: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as e: # Re-raise HTTPExceptions
        raise e
    except Exception as e:
        print(f"Unexpected error during file upload of '{file.filename}': {e}")
        # Log the full error for debugging: import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing file '{file.filename}': {str(e)}")
    finally:
        # 5. Cleanup: Delete the temporary uploaded file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Temporary file '{temp_file_path}' deleted successfully.")
            except Exception as e_clean:
                print(f"Error deleting temporary file '{temp_file_path}': {e_clean}")

# --- Chat Log and Session Endpoints ---

@router.get("/sessions/", response_model=List[SessionDetailModel], tags=["Chat History"])
async def get_list_of_chat_sessions(
    skip: int = Query(0, ge=0, description="Number of sessions to skip (for pagination)."),
    limit: int = Query(100, ge=1, le=200, description="Maximum number of sessions to return.")
):
    """
    Retrieves a list of chat sessions, ordered by the most recent interaction.
    Each session entry includes the session ID, the time of the last interaction,
    and the total number of messages in that session.
    """
    try:
        # Ensure MongoDB is connected (if not using Depends for every route)
        if mongo_storage.db is None:
            mongo_storage.connect_to_db()

        sessions_data = mongo_storage.list_chat_sessions(limit=limit, offset=skip)
        
        # FastAPI will automatically convert dicts to Pydantic models if keys match.
        # If explicit conversion is needed (e.g., for data transformation or validation):
        # return [SessionDetailModel(**session_dict) for session_dict in sessions_data]
        return sessions_data
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Database connection error: {e}")
    except Exception as e:
        print(f"Error listing chat sessions: {e}")
        # import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred while listing chat sessions: {str(e)}")


@router.get("/chat/{session_id}/history", response_model=List[ChatLogEntryModel], tags=["Chat History"])
async def get_session_chat_history(
    session_id: str = Path(..., description="The ID of the chat session to retrieve history for."),
    skip: int = Query(0, ge=0, description="Number of log entries to skip (for pagination)."),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of log entries to return.")
):
    """
    Retrieves the chat history for a specific session_id.
    Logs are returned sorted by timestamp in ascending order.
    """
    try:
        # Ensure MongoDB is connected
        if mongo_storage.db is None:
            mongo_storage.connect_to_db()
            
        logs_data = mongo_storage.get_chat_logs_for_session(session_id=session_id, limit=limit, offset=skip)
        
        if not logs_data and skip == 0: # Check if session_id might be invalid or just has no logs
            # Optionally, one could try to verify if the session_id has ever existed
            # by checking if list_chat_sessions with a filter for this session_id returns anything.
            # For now, an empty list is a valid response for a session with no (more) logs.
            # If we wanted to return 404 for truly non-existent sessions, more logic would be needed.
            print(f"No chat logs found for session '{session_id}' (limit: {limit}, skip: {skip}). Returning empty list.")
        
        # FastAPI will automatically convert dicts to Pydantic models.
        return logs_data
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Database connection error: {e}")
    except Exception as e:
        print(f"Error retrieving chat history for session '{session_id}': {e}")
        # import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred while retrieving chat history for session '{session_id}': {str(e)}")
