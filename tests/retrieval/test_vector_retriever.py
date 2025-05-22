import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# Module to be tested
from app.retrieval import vector_retriever
from app.core.config import Settings # To mock settings if needed
from langchain_core.documents import Document as LangchainDocument
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore as LangchainPinecone
from pinecone import Pinecone, IndexDescription, IndexList, IndexModel # For specing mocks
from pinecone import ApiException as PineconeApiException


# --- Fixtures ---

@pytest.fixture(autouse=True)
def clear_vector_store_cache():
    """Automatically clear the vector store cache before each test."""
    vector_retriever._vector_store_cache.clear()
    yield
    vector_retriever._vector_store_cache.clear()

@pytest.fixture
def mock_settings(mocker):
    """Fixture to mock app.core.config.settings."""
    mock_settings_obj = Settings(
        OPENAI_API_KEY="fake_openai_key", # Must be non-None for lc_embeddings_model init
        PINECONE_API_KEY="fake_pinecone_key",
        PINECONE_ENVIRONMENT="fake-pinecone-env",
        PINECONE_INDEX_NAME="test-index"
    )
    mocker.patch('app.retrieval.vector_retriever.settings', mock_settings_obj)
    return mock_settings_obj

@pytest.fixture
def mock_lc_embeddings_model(mocker):
    """Mocks the global lc_embeddings_model instance."""
    mock_model = MagicMock(spec=OpenAIEmbeddings)
    mocker.patch('app.retrieval.vector_retriever.lc_embeddings_model', mock_model)
    return mock_model

@pytest.fixture
def mock_pinecone_admin_client(mocker):
    """Mocks the global pinecone_admin_client instance."""
    mock_client = MagicMock(spec=Pinecone)
    mocker.patch('app.retrieval.vector_retriever.pinecone_admin_client', mock_client)
    return mock_client

@pytest.fixture
def mock_langchain_pinecone_vs(mocker):
    """Mocks the LangchainPinecone class (PineconeVectorStore)."""
    mock_vs_instance = MagicMock(spec=LangchainPinecone)
    # Mock the class 'from_existing_index' method to return our instance
    mock_vs_class = mocker.patch('app.retrieval.vector_retriever.LangchainPinecone')
    mock_vs_class.from_existing_index.return_value = mock_vs_instance
    return mock_vs_instance # Return the instance for assertions

@pytest.fixture
def sample_langchain_document():
    """Returns a sample Langchain Document."""
    return LangchainDocument(page_content="Sample content", metadata={"doc_id": "sample_doc_1", "source": "test.txt"})


# --- Test Cases ---

def test_create_pinecone_index_if_not_exists_does_not_exist(mock_settings, mock_pinecone_admin_client):
    """Test creating an index when it does not exist."""
    # Simulate index not existing in list_indexes
    mock_pinecone_admin_client.list_indexes.return_value = IndexList(indexes=[])
    
    # Simulate describe_index initially raising NotFound then succeeding after creation
    # and index_description.status['ready'] being True
    mock_index_description_ready = MagicMock(spec=IndexDescription)
    mock_index_description_ready.status = {'ready': True}
    
    mock_pinecone_admin_client.describe_index.side_effect = [
        PineconeNotFoundException("Index not found initially"), # First call (or not called if create_index is fast)
        mock_index_description_ready # Subsequent calls after creation
    ]
    
    # Mock create_index
    mock_pinecone_admin_client.create_index.return_value = None # Does not return anything significant

    result = vector_retriever.create_pinecone_index_if_not_exists(
        index_name="new-test-index", dimension=1536, metric='cosine'
    )

    assert result is True
    mock_pinecone_admin_client.list_indexes.assert_called_once()
    mock_pinecone_admin_client.create_index.assert_called_once_with(
        name="new-test-index",
        dimension=1536,
        metric='cosine',
        spec={ "serverless": {"cloud": "aws", "region": mock_settings.PINECONE_ENVIRONMENT}}
    )
    # describe_index might be called multiple times in the loop
    assert mock_pinecone_admin_client.describe_index.call_count >= 1


def test_create_pinecone_index_if_not_exists_already_exists(mock_settings, mock_pinecone_admin_client):
    """Test creating an index when it already exists."""
    # Simulate index existing
    existing_index_model = IndexModel(name=mock_settings.PINECONE_INDEX_NAME, dimension=1536, metric='cosine', status={'ready': True}, host="host")
    mock_pinecone_admin_client.list_indexes.return_value = IndexList(indexes=[existing_index_model])

    result = vector_retriever.create_pinecone_index_if_not_exists(
        index_name=mock_settings.PINECONE_INDEX_NAME, dimension=1536
    )

    assert result is True
    mock_pinecone_admin_client.list_indexes.assert_called_once()
    mock_pinecone_admin_client.create_index.assert_not_called()


def test_create_pinecone_index_creation_fails(mock_settings, mock_pinecone_admin_client):
    """Test when index creation fails at API level."""
    mock_pinecone_admin_client.list_indexes.return_value = IndexList(indexes=[])
    mock_pinecone_admin_client.create_index.side_effect = PineconeApiException("Creation failed")

    result = vector_retriever.create_pinecone_index_if_not_exists("fail-index", 1536)
    assert result is False


def test_get_vector_store_success(mock_settings, mock_lc_embeddings_model, mock_langchain_pinecone_vs):
    """Test successfully getting a vector store instance."""
    # Mock create_pinecone_index_if_not_exists to always return True for this test
    with patch('app.retrieval.vector_retriever.create_pinecone_index_if_not_exists', return_value=True) as mock_create_index:
        vs_instance = vector_retriever.get_vector_store(index_name="my-test-index")

    assert vs_instance is not None
    assert vs_instance == mock_langchain_pinecone_vs # Check if it's the mocked instance
    mock_create_index.assert_called_once_with("my-test-index")
    # Check LangchainPinecone.from_existing_index was called correctly
    vector_retriever.LangchainPinecone.from_existing_index.assert_called_once_with(
        index_name="my-test-index",
        embedding=mock_lc_embeddings_model
    )


def test_get_vector_store_caching(mock_settings, mock_lc_embeddings_model, mock_langchain_pinecone_vs):
    """Test that vector store instances are cached."""
    with patch('app.retrieval.vector_retriever.create_pinecone_index_if_not_exists', return_value=True):
        vs1 = vector_retriever.get_vector_store(index_name="cache-test-index")
        vs2 = vector_retriever.get_vector_store(index_name="cache-test-index")

    assert vs1 == vs2
    # from_existing_index should only be called once due to caching
    vector_retriever.LangchainPinecone.from_existing_index.assert_called_once()


def test_get_vector_store_index_creation_fails(mock_settings, mock_lc_embeddings_model):
    """Test get_vector_store when underlying index creation fails."""
    with patch('app.retrieval.vector_retriever.create_pinecone_index_if_not_exists', return_value=False) as mock_create_index:
        vs_instance = vector_retriever.get_vector_store(index_name="non-creatable-index")
    
    assert vs_instance is None
    mock_create_index.assert_called_once_with("non-creatable-index")
    vector_retriever.LangchainPinecone.from_existing_index.assert_not_called()


def test_upsert_document_embedding_success(mocker, mock_langchain_pinecone_vs):
    """Test successful upsert of a document."""
    # Mock get_vector_store to return our specific mock_langchain_pinecone_vs instance
    mocker.patch('app.retrieval.vector_retriever.get_vector_store', return_value=mock_langchain_pinecone_vs)

    doc_id = "upsert_doc_1"
    content = "Content to upsert."
    metadata = {"source": "upsert_source.txt"}
    
    result = vector_retriever.upsert_document_embedding(doc_id, content, metadata)

    assert result is True
    mock_langchain_pinecone_vs.add_texts.assert_called_once()
    args, _ = mock_langchain_pinecone_vs.add_texts.call_args
    
    assert args[0] == [content] # texts
    assert args[1] == [doc_id]   # ids
    expected_metadata = {**metadata, "doc_id": doc_id, "original_content": content}
    assert args[2] == [expected_metadata] # metadatas


def test_upsert_document_embedding_vs_failure(mocker, mock_langchain_pinecone_vs):
    """Test upsert failure if vector store's add_texts fails."""
    mocker.patch('app.retrieval.vector_retriever.get_vector_store', return_value=mock_langchain_pinecone_vs)
    mock_langchain_pinecone_vs.add_texts.side_effect = Exception("Pinecone upsert failed")

    result = vector_retriever.upsert_document_embedding("doc_fail", "fail_content", {})
    assert result is False


def test_query_vector_store_success(mocker, mock_langchain_pinecone_vs, sample_langchain_document):
    """Test successful query of the vector store."""
    mocker.patch('app.retrieval.vector_retriever.get_vector_store', return_value=mock_langchain_pinecone_vs)
    
    # sample_langchain_document's metadata already has doc_id
    mock_results_with_scores = [(sample_langchain_document, 0.95)]
    mock_langchain_pinecone_vs.similarity_search_with_score.return_value = mock_results_with_scores

    query_text = "Find sample content"
    top_k = 1
    filter_criteria = {"source": "test.txt"}
    
    results = vector_retriever.query_vector_store(query_text, top_k, filter_criteria)

    assert results is not None
    assert len(results) == 1
    
    first_result = results[0]
    assert first_result["doc_id"] == sample_langchain_document.metadata["doc_id"]
    assert first_result["score"] == 0.95
    assert first_result["content"] == sample_langchain_document.page_content
    assert first_result["metadata"] == sample_langchain_document.metadata

    mock_langchain_pinecone_vs.similarity_search_with_score.assert_called_once_with(
        query=query_text,
        k=top_k,
        filter=filter_criteria
    )


def test_query_vector_store_vs_failure(mocker, mock_langchain_pinecone_vs):
    """Test query failure if vector store's search fails."""
    mocker.patch('app.retrieval.vector_retriever.get_vector_store', return_value=mock_langchain_pinecone_vs)
    mock_langchain_pinecone_vs.similarity_search_with_score.side_effect = Exception("Pinecone query failed")

    results = vector_retriever.query_vector_store("any query")
    assert results is None


def test_delete_pinecone_index_lc_success(mock_settings, mock_pinecone_admin_client):
    """Test successful deletion of an index."""
    # Simulate index existing
    existing_index_model = IndexModel(name=mock_settings.PINECONE_INDEX_NAME, dimension=1536, metric='cosine', status={'ready': True}, host="host")
    mock_pinecone_admin_client.list_indexes.side_effect = [
        IndexList(indexes=[existing_index_model]), # First call, index exists
        IndexList(indexes=[])                   # Second call, index is gone
    ]
    
    result = vector_retriever.delete_pinecone_index_lc(mock_settings.PINECONE_INDEX_NAME)

    assert result is True
    mock_pinecone_admin_client.delete_index.assert_called_once_with(mock_settings.PINECONE_INDEX_NAME)
    assert mock_pinecone_admin_client.list_indexes.call_count >= 2 # Called in loop


def test_delete_pinecone_index_lc_not_found(mock_settings, mock_pinecone_admin_client):
    """Test deleting an index that does not exist."""
    mock_pinecone_admin_client.list_indexes.return_value = IndexList(indexes=[]) # Index doesn't exist

    result = vector_retriever.delete_pinecone_index_lc("non-existent-index")

    assert result is True
    mock_pinecone_admin_client.delete_index.assert_not_called()


def test_delete_pinecone_index_lc_api_error(mock_settings, mock_pinecone_admin_client):
    """Test deletion failure due to Pinecone API error."""
    # Simulate index existing
    existing_index_model = IndexModel(name=mock_settings.PINECONE_INDEX_NAME, dimension=1536, metric='cosine', status={'ready': True}, host="host")
    mock_pinecone_admin_client.list_indexes.return_value = IndexList(indexes=[existing_index_model])
    mock_pinecone_admin_client.delete_index.side_effect = PineconeApiException("Deletion failed")

    result = vector_retriever.delete_pinecone_index_lc(mock_settings.PINECONE_INDEX_NAME)
    assert result is False

# Note: Global initializations of lc_embeddings_model and pinecone_admin_client
# are handled by patching them directly using mocker.patch in fixtures or tests.
# The mock_settings fixture ensures that when vector_retriever is imported by the test file,
# these global clients are initialized with non-None API keys if they are re-imported or
# their initialization code is re-run. However, patching them directly (as done with
# mock_lc_embeddings_model and mock_pinecone_admin_client fixtures) is more robust for isolation.
# The current setup uses these direct patches.
