import uuid
from datetime import datetime
from pymongo import MongoClient, errors, ASCENDING, DESCENDING # Added ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.collection import Collection
from typing import List, Optional # Added List, Optional

from app.core.config import settings # Import settings

# Global variable to hold the database instance
db_client: MongoClient = None
db: Database = None

# Define collection names
DOCUMENT_COLLECTION = "documents"
CHAT_LOG_COLLECTION = "chat_logs"

def connect_to_db(uri: str = None, db_name: str = None) -> Database:
    """
    Establishes a connection to the MongoDB server and returns the database instance.
    Uses URI and database name from settings if not provided.
    """
    global db_client, db
    if db is not None:
        return db

    mongo_uri = uri or settings.MONGO_URI
    database_name = db_name or settings.MONGO_DB_NAME

    try:
        print(f"Attempting to connect to MongoDB at {mongo_uri} using database {database_name}...")
        db_client = MongoClient(mongo_uri)
        db_client.admin.command('ping')
        print("Successfully connected to MongoDB.")
        db = db_client[database_name]
        return db
    except errors.ConnectionFailure as e:
        print(f"Error connecting to MongoDB: {e}")
        raise ConnectionError(f"Could not connect to MongoDB at {mongo_uri}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during MongoDB connection: {e}")
        raise ConnectionError(f"An unexpected error occurred connecting to MongoDB: {e}")

def get_db_collection(collection_name: str) -> Collection: # Removed default for clarity
    """
    Ensures DB connection is established and returns the specified collection.
    """
    if db is None:
        connect_to_db() # Initialize connection using settings
    return db[collection_name]

# --- Document CRUD Functions (existing) ---
def add_document(content: str, metadata: dict) -> str | None:
    try:
        collection = get_db_collection(DOCUMENT_COLLECTION)
        doc_id = str(uuid.uuid4())
        current_time = datetime.utcnow()
        document_data = {
            "_id": doc_id,
            "doc_id": doc_id,
            "content": content,
            "metadata": metadata or {},
            "created_at": current_time,
            "updated_at": current_time,
        }
        if 'source' not in document_data['metadata']:
            document_data['metadata']['source'] = "unknown"
        result = collection.insert_one(document_data)
        print(f"Document added with ID: {result.inserted_id}")
        return doc_id
    except errors.PyMongoError as e:
        print(f"Error adding document: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while adding document: {e}")
        return None

def get_document(doc_id: str) -> dict | None:
    try:
        collection = get_db_collection(DOCUMENT_COLLECTION)
        document = collection.find_one({"doc_id": doc_id})
        if document:
            print(f"Document found with ID: {doc_id}")
        else:
            print(f"No document found with ID: {doc_id}")
        return document
    except errors.PyMongoError as e:
        print(f"Error retrieving document {doc_id}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while retrieving document {doc_id}: {e}")
        return None

def update_document(doc_id: str, content: str = None, metadata: dict = None) -> bool:
    if content is None and metadata is None:
        print(f"No update provided for document ID: {doc_id}.")
        return False
    try:
        collection = get_db_collection(DOCUMENT_COLLECTION)
        update_fields = {}
        if content is not None:
            update_fields["content"] = content
        if metadata is not None:
            update_fields["metadata"] = metadata
        update_fields["updated_at"] = datetime.utcnow()
        result = collection.update_one({"doc_id": doc_id}, {"$set": update_fields})
        if result.matched_count == 0:
            print(f"No document found with ID: {doc_id} to update.")
            return False
        if result.modified_count == 0 and result.matched_count > 0:
            print(f"Document {doc_id} found, but no changes were made.")
            return True
        print(f"Document updated successfully: {doc_id}")
        return True
    except errors.PyMongoError as e:
        print(f"Error updating document {doc_id}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while updating document {doc_id}: {e}")
        return False

def delete_document(doc_id: str) -> bool:
    try:
        collection = get_db_collection(DOCUMENT_COLLECTION)
        result = collection.delete_one({"doc_id": doc_id})
        if result.deleted_count > 0:
            print(f"Document deleted successfully: {doc_id}")
            return True
        else:
            print(f"No document found with ID: {doc_id} to delete.")
            return False
    except errors.PyMongoError as e:
        print(f"Error deleting document {doc_id}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while deleting document {doc_id}: {e}")
        return False

def close_db_connection():
    global db_client, db
    if db_client:
        db_client.close()
        db_client = None
        db = None
        print("MongoDB connection closed.")

# --- Chat Log Persistence Functions ---

def add_chat_log_entry(log_entry_data: dict) -> str | None:
    """
    Adds a new chat log entry to the database.
    Args:
        log_entry_data: A dictionary from ChatLogEntryModel.model_dump().
    Returns:
        The interaction_id of the entry, or None if insertion fails.
    """
    if not log_entry_data or "interaction_id" not in log_entry_data:
        print("Error: log_entry_data is missing or does not contain 'interaction_id'.")
        return None
    try:
        collection = get_db_collection(CHAT_LOG_COLLECTION)
        # Use interaction_id as MongoDB's _id for direct lookup if desired
        log_entry_data_db = log_entry_data.copy()
        log_entry_data_db["_id"] = log_entry_data_db["interaction_id"]
        
        result = collection.insert_one(log_entry_data_db)
        if result.inserted_id:
            print(f"Chat log entry added with interaction_id: {log_entry_data['interaction_id']}")
            return log_entry_data["interaction_id"]
        else:
            print(f"Failed to add chat log entry for {log_entry_data['interaction_id']} (no inserted_id).")
            return None
    except errors.DuplicateKeyError:
        # This can happen if _id (interaction_id) is not unique.
        # Should not happen if uuid4 is used for interaction_id.
        print(f"Error: Duplicate interaction_id '{log_entry_data['interaction_id']}'. Log entry not added.")
        return None
    except errors.PyMongoError as e:
        print(f"Error adding chat log entry (interaction_id: {log_entry_data.get('interaction_id')}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error adding chat log (interaction_id: {log_entry_data.get('interaction_id')}): {e}")
        return None

def get_chat_logs_for_session(session_id: str, limit: int = 100, offset: int = 0) -> List[dict]:
    """
    Retrieves chat logs for a specific session_id, sorted by timestamp.
    """
    try:
        collection = get_db_collection(CHAT_LOG_COLLECTION)
        # Exclude _id from MongoDB if it's redundant (since interaction_id is there)
        # Or, if _id is interaction_id, then it's fine.
        logs_cursor = collection.find({"session_id": session_id}, {"_id": 0})\
                                .sort("timestamp", ASCENDING)\
                                .skip(offset)\
                                .limit(limit)
        logs = list(logs_cursor)
        print(f"Retrieved {len(logs)} chat logs for session_id: {session_id}")
        return logs
    except errors.PyMongoError as e:
        print(f"Error retrieving chat logs for session_id {session_id}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error retrieving chat logs for session {session_id}: {e}")
        return []

def list_chat_sessions(limit: int = 100, offset: int = 0) -> List[dict]:
    """
    Lists chat sessions, showing the last interaction time and total messages per session.
    """
    try:
        collection = get_db_collection(CHAT_LOG_COLLECTION)
        pipeline = [
            {
                "$sort": {"timestamp": DESCENDING}
            },
            {
                "$group": {
                    "_id": "$session_id",
                    "last_interaction_time": {"$first": "$timestamp"},
                    "total_messages": {"$sum": 1} # Each log entry is one interaction (user+AI)
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "session_id": "$_id",
                    "last_interaction_time": "$last_interaction_time",
                    "total_messages": "$total_messages"
                }
            },
            {
                "$sort": {"last_interaction_time": DESCENDING}
            },
            {"$skip": offset},
            {"$limit": limit}
        ]
        sessions_cursor = collection.aggregate(pipeline)
        sessions = list(sessions_cursor)
        print(f"Retrieved {len(sessions)} chat session details.")
        return sessions
    except errors.PyMongoError as e:
        print(f"Error listing chat sessions: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error listing chat sessions: {e}")
        return []


if __name__ == "__main__":
    print("Running storage module example...")
    import time # For sleep between log entries

    try:
        db_instance = connect_to_db()
        if not db_instance:
            raise Exception("Failed to connect to the database. Exiting example.")

        print(f"\nDropping collection '{DOCUMENT_COLLECTION}' for a clean test run...")
        get_db_collection(DOCUMENT_COLLECTION).drop()
        print("Document collection dropped.")
        
        print(f"\nDropping collection '{CHAT_LOG_COLLECTION}' for a clean test run...")
        get_db_collection(CHAT_LOG_COLLECTION).drop()
        print("Chat log collection dropped.")

        # --- Test Document CRUD (shortened) ---
        print("\n--- Testing Document CRUD ---")
        doc_id1 = add_document(content="Doc 1 for main test.", metadata={"source": "main_test.txt"})
        if doc_id1: print(f"Added document: {doc_id1}")
        
        # --- Test Chat Log Functions ---
        print("\n--- Testing Chat Log Functions ---")
        
        log1_data = {
            "interaction_id": str(uuid.uuid4()), "session_id": "session_A", 
            "user_message": "Hello AI from A (1st)", "ai_response": "Hi user A (1st)", 
            "timestamp": datetime.utcnow(), "retrieved_doc_ids": ["doc1", "doc2"]
        }
        time.sleep(0.01) 
        log2_data = {
            "interaction_id": str(uuid.uuid4()), "session_id": "session_B", 
            "user_message": "Hello AI from B", "ai_response": "Hi user B", 
            "timestamp": datetime.utcnow(), "feedback": 1
        }
        time.sleep(0.01)
        log3_data = {
            "interaction_id": str(uuid.uuid4()), "session_id": "session_A", 
            "user_message": "How are you A (2nd)?", "ai_response": "I am fine A! (2nd)", 
            "timestamp": datetime.utcnow()
        }

        interaction_id1 = add_chat_log_entry(log1_data)
        interaction_id2 = add_chat_log_entry(log2_data)
        interaction_id3 = add_chat_log_entry(log3_data)

        if not (interaction_id1 and interaction_id2 and interaction_id3):
            print("Error adding one or more chat log entries. Aborting further chat log tests.")
        else:
            print("\n--- Getting Chat Logs for Session A (Expected: 2) ---")
            session_a_logs = get_chat_logs_for_session("session_A", limit=10)
            assert len(session_a_logs) == 2, f"Expected 2 logs for session_A, got {len(session_a_logs)}"
            for log in session_a_logs:
                print(f"  Log for session A: User: '{log['user_message']}', Timestamp: {log['timestamp']}")
            
            print("\n--- Listing Chat Sessions (Expected: 2) ---")
            chat_sessions = list_chat_sessions(limit=10)
            assert len(chat_sessions) == 2, f"Expected 2 active sessions, got {len(chat_sessions)}"
            print("Active chat sessions:")
            for sess in chat_sessions:
                print(f"  Session ID: {sess['session_id']}, Last Interaction: {sess['last_interaction_time']}, Total Messages: {sess['total_messages']}")
            
            session_a_detail = next((s for s in chat_sessions if s['session_id'] == 'session_A'), None)
            assert session_a_detail is not None
            assert session_a_detail['total_messages'] == 2
            assert session_a_detail['last_interaction_time'] == log3_data['timestamp'] # log3 is latest for session_A

    except ConnectionError as ce:
        print(f"Database connection failed: {ce}")
    except Exception as e:
        print(f"An error occurred in the example usage: {e}")
        import traceback
        traceback.print_exc()
    finally:
        close_db_connection()
        print("\nExample finished.")
