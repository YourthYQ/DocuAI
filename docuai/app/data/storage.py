import uuid
from datetime import datetime
from pymongo import MongoClient, errors
from pymongo.database import Database
from pymongo.collection import Collection

from app.core.config import settings # Import settings

# Global variable to hold the database instance
# This is a simple way to manage the connection; for more complex apps,
# you might use a dependency injection system or a more robust connection manager.
db_client: MongoClient = None
db: Database = None

# Define the name of the collection for documents
DOCUMENT_COLLECTION = "documents"

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
        # Ping the server to ensure connection
        db_client.admin.command('ping')
        print("Successfully connected to MongoDB.")
        db = db_client[database_name]
        return db
    except errors.ConnectionFailure as e:
        print(f"Error connecting to MongoDB: {e}")
        # In a real application, you might want to raise the exception
        # or implement a retry mechanism.
        raise ConnectionError(f"Could not connect to MongoDB at {mongo_uri}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during MongoDB connection: {e}")
        raise ConnectionError(f"An unexpected error occurred connecting to MongoDB: {e}")

def get_db_collection(collection_name: str = DOCUMENT_COLLECTION) -> Collection:
    """
    Ensures DB connection is established and returns the specified collection.
    """
    if db is None:
        connect_to_db() # Initialize connection using settings
    return db[collection_name]

def add_document(content: str, metadata: dict) -> str:
    """
    Adds a new document to the database.

    Args:
        content: The text content of the document.
        metadata: A dictionary containing metadata (e.g., source, filename).

    Returns:
        The unique ID (doc_id) of the newly added document.
    """
    try:
        collection = get_db_collection()
        doc_id = str(uuid.uuid4())
        current_time = datetime.utcnow()

        document_data = {
            "_id": doc_id,  # Use _id as the primary key for MongoDB
            "doc_id": doc_id,
            "content": content,
            "metadata": metadata or {},
            "created_at": current_time,
            "updated_at": current_time,
        }
        # Ensure 'source' and other common metadata fields are handled
        if 'source' not in document_data['metadata']:
            document_data['metadata']['source'] = "unknown"

        result = collection.insert_one(document_data)
        print(f"Document added with ID: {result.inserted_id}")
        return doc_id
    except errors.PyMongoError as e:
        print(f"Error adding document: {e}")
        # Consider logging the error and raising a custom exception
        return None
    except Exception as e:
        print(f"An unexpected error occurred while adding document: {e}")
        return None

def get_document(doc_id: str) -> dict | None:
    """
    Retrieves a document by its ID.

    Args:
        doc_id: The unique ID of the document.

    Returns:
        A dictionary representing the document, or None if not found.
    """
    try:
        collection = get_db_collection()
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
    """
    Updates the content or metadata of an existing document.

    Args:
        doc_id: The unique ID of the document to update.
        content: The new text content (optional).
        metadata: The new metadata (optional).

    Returns:
        True if the update was successful, False otherwise.
    """
    if content is None and metadata is None:
        print(f"No update provided for document ID: {doc_id}. Content and metadata are None.")
        return False

    try:
        collection = get_db_collection()
        update_fields = {}
        if content is not None:
            update_fields["content"] = content
        if metadata is not None:
            # Be careful with nested updates in MongoDB.
            # If you want to replace the whole metadata:
            # update_fields["metadata"] = metadata
            # If you want to update specific fields within metadata:
            # for key, value in metadata.items():
            #     update_fields[f"metadata.{key}"] = value
            # For simplicity, this example replaces the entire metadata object.
            update_fields["metadata"] = metadata

        update_fields["updated_at"] = datetime.utcnow()

        result = collection.update_one({"doc_id": doc_id}, {"$set": update_fields})

        if result.matched_count == 0:
            print(f"No document found with ID: {doc_id} to update.")
            return False
        if result.modified_count == 0 and result.matched_count > 0:
            print(f"Document {doc_id} found, but no changes were made (data might be the same).")
            # This can happen if the provided data is identical to the existing data.
            # Depending on requirements, this might still be considered a "success".
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
    """
    Deletes a document from the database.

    Args:
        doc_id: The unique ID of the document to delete.

    Returns:
        True if the deletion was successful, False otherwise.
    """
    try:
        collection = get_db_collection()
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
    """Closes the MongoDB connection if it's open."""
    global db_client
    if db_client:
        db_client.close()
        db_client = None
        global db
        db = None
        print("MongoDB connection closed.")

# Example Usage (can be run directly for testing, e.g., python -m app.data.storage)
if __name__ == "__main__":
    print("Running storage module example...")

    # Ensure you have a MongoDB instance running and accessible.
    # You might need to set MONGO_URI in your environment or a .env file.
    # Example: MONGO_URI="mongodb://localhost:27017/"
    #          MONGO_DB_NAME="docuai_test_db"

    try:
        # Connect to DB (uses settings from config.py)
        db_instance = connect_to_db()
        if not db_instance:
            raise Exception("Failed to connect to the database. Exiting example.")

        # Clean up collection before test
        print(f"Dropping collection '{DOCUMENT_COLLECTION}' for a clean test run...")
        get_db_collection().drop()
        print("Collection dropped.")


        # 1. Add a document
        print("\n--- Adding Document ---")
        test_metadata = {"source": "test_source.txt", "author": "Test User"}
        doc_id1 = add_document(content="This is the first test document.", metadata=test_metadata)
        if doc_id1:
            print(f"Added document with ID: {doc_id1}")
        else:
            print("Failed to add document.")
            close_db_connection()
            exit()

        doc_id2 = add_document(content="This is a second document about AI.", metadata={"source": "ai_paper.pdf"})
        if doc_id2:
            print(f"Added document with ID: {doc_id2}")

        # 2. Get a document
        print("\n--- Getting Document ---")
        retrieved_doc = get_document(doc_id1)
        if retrieved_doc:
            print(f"Retrieved document: {retrieved_doc}")
        else:
            print(f"Could not retrieve document {doc_id1}")

        non_existent_doc = get_document("non-existent-id")
        if non_existent_doc is None:
            print("Correctly handled non-existent document retrieval.")

        # 3. Update a document
        print("\n--- Updating Document ---")
        updated_content = "This is the updated content for the first document."
        updated_metadata = {"source": "test_source_v2.txt", "author": "Test Admin", "version": "2.0"}
        if update_document(doc_id1, content=updated_content, metadata=updated_metadata):
            print(f"Document {doc_id1} updated.")
            retrieved_doc_after_update = get_document(doc_id1)
            print(f"Retrieved after update: {retrieved_doc_after_update}")
        else:
            print(f"Failed to update document {doc_id1}")

        # Try updating only metadata
        if update_document(doc_id2, metadata={"status": "reviewed", "tags": ["ai", "research"]}):
             print(f"Document {doc_id2} metadata updated.")
             retrieved_doc2_after_update = get_document(doc_id2)
             print(f"Retrieved doc2 after metadata update: {retrieved_doc2_after_update}")

        # Try updating non-existent document
        if not update_document("non-existent-id", content="new content"):
            print("Correctly handled update for non-existent document.")


        # 4. Delete a document
        print("\n--- Deleting Document ---")
        if delete_document(doc_id1):
            print(f"Document {doc_id1} deleted.")
            if get_document(doc_id1) is None:
                print(f"Document {doc_id1} successfully verified as deleted.")
        else:
            print(f"Failed to delete document {doc_id1}")

        # Try deleting non-existent document
        if not delete_document("non-existent-id"):
            print("Correctly handled delete for non-existent document.")

        # Clean up the second document
        delete_document(doc_id2)
        print(f"Deleted second document {doc_id2} for cleanup.")

    except ConnectionError as ce:
        print(f"Database connection failed: {ce}")
    except Exception as e:
        print(f"An error occurred in the example usage: {e}")
    finally:
        # Close the connection
        close_db_connection()
        print("\nExample finished.")
