import pytest
from unittest.mock import MagicMock, patch # For more complex mocking if needed
from pymongo import errors as PyMongoErrors # To simulate PyMongo exceptions

# Module to be tested
from app.data import storage
from app.core.config import settings # For default MONGO_URI, DB_NAME if needed in tests

# Reset global db variables in storage module before and after each test
# to ensure test isolation, especially because connect_to_db modifies them.
@pytest.fixture(autouse=True)
def reset_storage_globals():
    # Before test: store original state if any, then reset
    original_db_client = storage.db_client
    original_db = storage.db
    storage.db_client = None
    storage.db = None
    yield
    # After test: restore original state
    storage.db_client = original_db_client
    storage.db = original_db


@pytest.fixture
def mock_mongo_client(mocker):
    """Fixture for a mocked MongoClient instance."""
    mock_client = mocker.MagicMock(spec=storage.MongoClient)
    # Mock the __getitem__ method to allow db_client[db_name] access
    mock_client.__getitem__.return_value = mocker.MagicMock(spec=storage.Database)
    return mock_client

@pytest.fixture
def mock_db(mocker):
    """Fixture for a mocked Database instance."""
    mock_database = mocker.MagicMock(spec=storage.Database)
    # Mock the __getitem__ method to allow db[collection_name] access
    mock_database.__getitem__.return_value = mocker.MagicMock(spec=storage.Collection)
    return mock_database

@pytest.fixture
def mock_collection(mocker):
    """Fixture for a mocked Collection instance."""
    return mocker.MagicMock(spec=storage.Collection)


def test_connect_to_db_success(mocker, mock_mongo_client, mock_db):
    """Test successful connection to MongoDB."""
    mocker.patch('app.data.storage.MongoClient', return_value=mock_mongo_client)
    # Simulate successful ping
    mock_mongo_client.admin.command.return_value = {"ok": 1}
    # Ensure __getitem__ on the client returns our mock_db
    mock_mongo_client.__getitem__.return_value = mock_db 

    db_instance = storage.connect_to_db(uri="mongodb://test:27017/", db_name="testdb")

    storage.MongoClient.assert_called_once_with("mongodb://test:27017/")
    mock_mongo_client.admin.command.assert_called_once_with('ping')
    assert db_instance is not None
    assert db_instance == mock_db # Should be the db instance from client[db_name]
    assert storage.db == mock_db # Global should be set


def test_connect_to_db_connection_failure(mocker):
    """Test connection failure to MongoDB."""
    # Simulate MongoClient raising ConnectionFailure
    mocker.patch('app.data.storage.MongoClient', side_effect=PyMongoErrors.ConnectionFailure("Connection failed"))

    with pytest.raises(ConnectionError, match="Could not connect to MongoDB"):
        storage.connect_to_db(uri="mongodb://fail:27017/", db_name="faildb")
    
    assert storage.db is None # Ensure global db is not set on failure


def test_connect_to_db_ping_failure(mocker, mock_mongo_client):
    """Test failure when pinging MongoDB."""
    mocker.patch('app.data.storage.MongoClient', return_value=mock_mongo_client)
    # Simulate ping command raising an exception
    mock_mongo_client.admin.command.side_effect = PyMongoErrors.OperationFailure("Ping failed")

    # The current implementation of connect_to_db does not specifically catch OperationFailure on ping
    # and re-raise as ConnectionError. It raises ConnectionError for ConnectionFailure or generic Exception.
    # This test will check if it raises ConnectionError for the generic Exception path.
    with pytest.raises(ConnectionError, match="An unexpected error occurred connecting to MongoDB"):
        storage.connect_to_db(uri="mongodb://pingfail:27017/", db_name="pingfaildb")

    storage.MongoClient.assert_called_once_with("mongodb://pingfail:27017/")
    mock_mongo_client.admin.command.assert_called_once_with('ping')
    assert storage.db is None


def test_get_db_collection(mocker, mock_db, mock_collection):
    """Test getting a collection from the database."""
    # Pre-set the global db to our mock_db
    storage.db = mock_db
    storage.db_client = MagicMock() # Needs to be not None for get_db_collection to skip connect
    
    # Configure mock_db to return mock_collection when a collection is accessed
    mock_db.__getitem__.return_value = mock_collection
    
    collection_instance = storage.get_db_collection("test_collection")
    
    mock_db.__getitem__.assert_called_once_with("test_collection")
    assert collection_instance == mock_collection


def test_get_db_collection_connects_if_db_none(mocker, mock_mongo_client, mock_db, mock_collection):
    """Test get_db_collection calls connect_to_db if storage.db is None."""
    assert storage.db is None # Ensure it's initially None due to reset_storage_globals

    mock_connect = mocker.patch('app.data.storage.connect_to_db', return_value=mock_db)
    mock_db.__getitem__.return_value = mock_collection # db['coll_name']

    collection_instance = storage.get_db_collection("my_docs")

    mock_connect.assert_called_once() # connect_to_db should have been called
    mock_db.__getitem__.assert_called_once_with("my_docs")
    assert collection_instance == mock_collection


def test_add_document_success(mocker, mock_collection):
    """Test successfully adding a document."""
    mocker.patch('app.data.storage.get_db_collection', return_value=mock_collection)
    mock_insert_result = MagicMock()
    mock_insert_result.inserted_id = "some_mongo_id" # Not directly used by add_document's return
    mock_collection.insert_one.return_value = mock_insert_result
    
    # Patch uuid.uuid4 to control doc_id
    fixed_uuid = "fixed_uuid_123"
    mocker.patch('app.data.storage.uuid.uuid4', return_value=fixed_uuid)

    content = "Test document content."
    metadata = {"source": "test.txt"}
    
    doc_id = storage.add_document(content, metadata)

    assert doc_id == fixed_uuid
    mock_collection.insert_one.assert_called_once()
    args, _ = mock_collection.insert_one.call_args
    inserted_doc = args[0]
    
    assert inserted_doc["_id"] == fixed_uuid
    assert inserted_doc["doc_id"] == fixed_uuid
    assert inserted_doc["content"] == content
    assert inserted_doc["metadata"]["source"] == "test.txt"
    assert "created_at" in inserted_doc
    assert "updated_at" in inserted_doc
    assert inserted_doc["created_at"] == inserted_doc["updated_at"]


def test_add_document_default_metadata(mocker, mock_collection):
    """Test adding a document with None metadata, ensuring defaults are applied."""
    mocker.patch('app.data.storage.get_db_collection', return_value=mock_collection)
    mock_collection.insert_one.return_value = MagicMock(inserted_id="mock_id")
    fixed_uuid = "fixed_uuid_456"
    mocker.patch('app.data.storage.uuid.uuid4', return_value=fixed_uuid)

    doc_id = storage.add_document(content="Content with no metadata.", metadata=None)
    
    assert doc_id == fixed_uuid
    args, _ = mock_collection.insert_one.call_args
    inserted_doc = args[0]
    assert inserted_doc["metadata"] == {"source": "unknown"} # Default source


def test_add_document_mongo_error(mocker, mock_collection):
    """Test error handling when insert_one fails."""
    mocker.patch('app.data.storage.get_db_collection', return_value=mock_collection)
    mock_collection.insert_one.side_effect = PyMongoErrors.PyMongoError("DB write error")

    doc_id = storage.add_document("Error content", {"source": "error.src"})
    
    assert doc_id is None


def test_get_document_found(mocker, mock_collection):
    """Test retrieving an existing document."""
    mocker.patch('app.data.storage.get_db_collection', return_value=mock_collection)
    sample_doc = {"doc_id": "existing_id", "content": "Found me!", "metadata": {}}
    mock_collection.find_one.return_value = sample_doc

    doc = storage.get_document("existing_id")

    mock_collection.find_one.assert_called_once_with({"doc_id": "existing_id"})
    assert doc == sample_doc


def test_get_document_not_found(mocker, mock_collection):
    """Test retrieving a non-existent document."""
    mocker.patch('app.data.storage.get_db_collection', return_value=mock_collection)
    mock_collection.find_one.return_value = None

    doc = storage.get_document("non_existent_id")

    mock_collection.find_one.assert_called_once_with({"doc_id": "non_existent_id"})
    assert doc is None


def test_get_document_mongo_error(mocker, mock_collection):
    """Test error handling when find_one fails."""
    mocker.patch('app.data.storage.get_db_collection', return_value=mock_collection)
    mock_collection.find_one.side_effect = PyMongoErrors.PyMongoError("DB read error")

    doc = storage.get_document("any_id")
    assert doc is None


def test_update_document_success(mocker, mock_collection):
    """Test successfully updating a document."""
    mocker.patch('app.data.storage.get_db_collection', return_value=mock_collection)
    mock_update_result = MagicMock()
    mock_update_result.matched_count = 1
    mock_update_result.modified_count = 1
    mock_collection.update_one.return_value = mock_update_result

    result = storage.update_document("doc_to_update", content="New content", metadata={"tag": "updated"})

    assert result is True
    mock_collection.update_one.assert_called_once()
    args, _ = mock_collection.update_one.call_args
    query_filter = args[0]
    update_payload = args[1]

    assert query_filter == {"doc_id": "doc_to_update"}
    assert "content" in update_payload["$set"]
    assert update_payload["$set"]["content"] == "New content"
    assert "metadata" in update_payload["$set"]
    assert update_payload["$set"]["metadata"] == {"tag": "updated"}
    assert "updated_at" in update_payload["$set"]


def test_update_document_no_changes_made(mocker, mock_collection):
    """Test update when document is matched but no fields are modified."""
    mocker.patch('app.data.storage.get_db_collection', return_value=mock_collection)
    mock_update_result = MagicMock()
    mock_update_result.matched_count = 1
    mock_update_result.modified_count = 0 # No actual change in DB
    mock_collection.update_one.return_value = mock_update_result
    
    result = storage.update_document("doc_id_no_change", content="Same old content")
    assert result is True # Current implementation returns True if matched_count > 0 and modified_count == 0


def test_update_document_not_found(mocker, mock_collection):
    """Test updating a non-existent document."""
    mocker.patch('app.data.storage.get_db_collection', return_value=mock_collection)
    mock_update_result = MagicMock()
    mock_update_result.matched_count = 0
    mock_update_result.modified_count = 0
    mock_collection.update_one.return_value = mock_update_result

    result = storage.update_document("non_existent_doc_id", content="Content for ghost")
    assert result is False


def test_update_document_no_content_or_metadata(mocker, mock_collection):
    """Test update_document when neither content nor metadata is provided."""
    mocker.patch('app.data.storage.get_db_collection', return_value=mock_collection)
    result = storage.update_document("any_id_here", content=None, metadata=None)
    assert result is False
    mock_collection.update_one.assert_not_called()


def test_update_document_mongo_error(mocker, mock_collection):
    """Test error handling when update_one fails."""
    mocker.patch('app.data.storage.get_db_collection', return_value=mock_collection)
    mock_collection.update_one.side_effect = PyMongoErrors.PyMongoError("DB update error")

    result = storage.update_document("any_id", content="Trying to update")
    assert result is False


def test_delete_document_success(mocker, mock_collection):
    """Test successfully deleting a document."""
    mocker.patch('app.data.storage.get_db_collection', return_value=mock_collection)
    mock_delete_result = MagicMock()
    mock_delete_result.deleted_count = 1
    mock_collection.delete_one.return_value = mock_delete_result

    result = storage.delete_document("doc_to_delete")

    assert result is True
    mock_collection.delete_one.assert_called_once_with({"doc_id": "doc_to_delete"})


def test_delete_document_not_found(mocker, mock_collection):
    """Test deleting a non-existent document."""
    mocker.patch('app.data.storage.get_db_collection', return_value=mock_collection)
    mock_delete_result = MagicMock()
    mock_delete_result.deleted_count = 0
    mock_collection.delete_one.return_value = mock_delete_result

    result = storage.delete_document("non_existent_for_delete")
    assert result is False


def test_delete_document_mongo_error(mocker, mock_collection):
    """Test error handling when delete_one fails."""
    mocker.patch('app.data.storage.get_db_collection', return_value=mock_collection)
    mock_collection.delete_one.side_effect = PyMongoErrors.PyMongoError("DB delete error")

    result = storage.delete_document("any_id_for_delete_error")
    assert result is False


def test_close_db_connection(mocker):
    """Test closing the database connection."""
    # First, simulate an open connection
    mock_client_instance = MagicMock()
    storage.db_client = mock_client_instance
    storage.db = MagicMock() # Needs to be not None to simulate open state

    storage.close_db_connection()

    mock_client_instance.close.assert_called_once()
    assert storage.db_client is None
    assert storage.db is None

    # Test closing when already closed (should not error)
    storage.db_client = None
    storage.db = None
    try:
        storage.close_db_connection() # Should run without error
    except Exception as e:
        pytest.fail(f"close_db_connection raised an exception when already closed: {e}")
    assert storage.db_client is None # Still None
    assert storage.db is None      # Still None
