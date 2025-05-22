import pytest
from unittest.mock import MagicMock, patch

# Modules and classes to test or use in tests
from app.llm import rag_chain
from app.core.config import Settings # To mock settings if needed
from langchain_core.documents import Document as LangchainDocument
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

# --- Fixtures ---

@pytest.fixture(autouse=True)
def mock_dependencies(mocker):
    """
    Automatically mock external dependencies and global variables within rag_chain.py
    for each test, ensuring a clean state.
    """
    # Mock settings if rag_chain.py uses it directly at module level beyond component init
    mock_settings_obj = Settings(
        OPENAI_API_KEY="fake_openai_key",
        DEFAULT_LLM_MODEL="gpt-test-model",
        PINECONE_API_KEY="fake_pinecone_key", # Needed by get_vector_store if called
        PINECONE_ENVIRONMENT="fake-env",
        PINECONE_INDEX_NAME="fake-index"
    )
    mocker.patch('app.llm.rag_chain.settings', mock_settings_obj)

    # Mock LLM instance
    mock_llm_instance = MagicMock(spec=ChatOpenAI)
    mock_llm_instance.invoke.return_value = AIMessage(content="Default mock LLM answer") # For StrOutputParser
    # If the chain expects a string directly from llm.invoke, then:
    # mock_llm_instance.invoke.return_value = "Default mock LLM answer"
    # StrOutputParser usually handles AIMessage from ChatModels.
    mocker.patch('app.llm.rag_chain.llm', mock_llm_instance)
    
    # Mock Retriever instance
    # The retriever in rag_chain.py is retriever = vector_store.as_retriever()
    # So, we need to mock get_vector_store and the chain of calls.
    mock_retriever_instance = MagicMock(spec=Runnable) # as_retriever returns a Runnable
    mock_retriever_instance.invoke.return_value = [ # Default: return some docs
        LangchainDocument(page_content="Doc 1 content", metadata={"doc_id": "doc1"}),
        LangchainDocument(page_content="Doc 2 content", metadata={"doc_id": "doc2"})
    ]
    
    mock_vector_store_instance = MagicMock()
    mock_vector_store_instance.as_retriever.return_value = mock_retriever_instance
    
    mocker.patch('app.llm.rag_chain.get_vector_store', return_value=mock_vector_store_instance)
    # This makes rag_chain.retriever the mock_retriever_instance
    # Re-initialize retriever in rag_chain module after patching get_vector_store
    if rag_chain.get_vector_store:
        vs = rag_chain.get_vector_store()
        if vs:
            rag_chain.retriever = vs.as_retriever(search_kwargs={'k': 5})
        else: # If get_vector_store was mocked to return None
             rag_chain.retriever = None # Or a mock, depending on test needs
    else: # If get_vector_store itself is None (e.g. due to earlier mock)
        rag_chain.retriever = mock_retriever_instance # Fallback, ensure it's a mock
    
    # We also need to ensure rag_chain.rag_chain_with_source is rebuilt with these mocks
    # This is tricky because it's constructed at module load time.
    # The best way is to re-trigger its construction within tests or a setup_method.
    # For now, individual tests might need to repatch or reconstruct the chain.
    # A simpler way: patch the final rag_chain_with_source directly in tests needing specific chain behavior.
    
    # Let's provide the individual mocks for direct use in tests
    # This fixture can return them if tests need to assert calls on these specific mocks
    return {
        "llm": mock_llm_instance,
        "retriever": mock_retriever_instance,
        "vector_store": mock_vector_store_instance
    }


@pytest.fixture
def sample_docs():
    """Provides a list of sample LangchainDocument objects."""
    return [
        LangchainDocument(page_content="First document about apples.", metadata={"doc_id": "fruit_doc_1", "source": "fruits.txt"}),
        LangchainDocument(page_content="Second document about bananas.", metadata={"doc_id": "fruit_doc_2", "source": "fruits.txt"}),
        LangchainDocument(page_content="A document about cars.", metadata={"doc_id": "auto_doc_1", "source": "vehicles.txt"}),
    ]

# --- Test Cases for format_docs ---

def test_format_docs_with_documents(sample_docs):
    """Test format_docs with a list of LangchainDocument objects."""
    formatted_string = rag_chain.format_docs(sample_docs)
    
    assert "Document ID: fruit_doc_1" in formatted_string
    assert "Content: First document about apples." in formatted_string
    assert "---" in formatted_string # Separator
    assert "Document ID: fruit_doc_2" in formatted_string
    assert "Content: Second document about bananas." in formatted_string
    assert "Document ID: auto_doc_1" in formatted_string
    assert "Content: A document about cars." in formatted_string
    
    # Check number of separators (n-1)
    assert formatted_string.count("---") == len(sample_docs) -1 if len(sample_docs) > 1 else 0


def test_format_docs_with_no_documents():
    """Test format_docs with an empty list."""
    formatted_string = rag_chain.format_docs([])
    assert formatted_string == "No relevant documents found."

def test_format_docs_with_one_document(sample_docs):
    """Test format_docs with a single document."""
    single_doc_list = [sample_docs[0]]
    formatted_string = rag_chain.format_docs(single_doc_list)
    
    assert "Document ID: fruit_doc_1" in formatted_string
    assert "Content: First document about apples." in formatted_string
    assert "---" not in formatted_string # No separator for a single doc

# --- Test Cases for RAG Chain Invocation ---

@patch('app.llm.rag_chain.RunnableParallel') # To inspect what's passed to it
@patch('app.llm.rag_chain.RunnablePassthrough')
@patch('app.llm.rag_chain.ChatPromptTemplate')
@patch('app.llm.rag_chain.ChatOpenAI') # Mock the class used for llm
@patch('app.llm.rag_chain.StrOutputParser')
@patch('app.llm.rag_chain.get_vector_store') # Mock the function that provides the vector store
def test_rag_chain_construction_and_invocation(
    mock_get_vector_store, mock_chatopenai, mock_prompt_template_class, 
    mock_runnable_passthrough_class, mock_runnable_parallel_class,
    mock_dependencies, sample_docs # mock_dependencies provides mocked llm and retriever instances
):
    """
    Test the RAG chain's construction and a successful invocation flow.
    This test is more about the wiring (LCEL) than individual component logic if mocks are too deep.
    Alternatively, test invoke_rag_chain and mock its internal dependencies (llm, retriever).
    """
    # --- Setup Mocks for RAG Chain Construction ---
    mock_retriever = mock_dependencies['retriever']
    mock_llm = mock_dependencies['llm']

    # Configure retriever mock
    mock_retriever.invoke.return_value = sample_docs
    
    # Configure LLM mock
    mock_llm.invoke.return_value = AIMessage(content="Specific LLM answer for this test")

    # Mock the from_template method of ChatPromptTemplate class
    mock_prompt_instance = MagicMock(spec=ChatPromptTemplate)
    mock_prompt_template_class.from_template.return_value = mock_prompt_instance
    
    # Mock StrOutputParser instance
    mock_parser_instance = MagicMock(spec=StrOutputParser)
    mock_parser_instance.invoke.return_value = "Specific LLM answer for this test" # Parser returns string
    # Patch StrOutputParser class to return the instance
    # Assuming StrOutputParser is instantiated as StrOutputParser() in rag_chain.py
    # If it's already mocked by mock_dependencies, this might not be needed or might need adjustment.
    # For this test, let's assume rag_chain.StrOutputParser() is called and we want to control its instance.
    with patch('app.llm.rag_chain.StrOutputParser', return_value=mock_parser_instance) as mock_str_output_parser_class:

        # --- Re-initialize the RAG chain with these specific mocks ---
        # This is important because the chain is defined at module level in rag_chain.py
        # We need to ensure it uses the mocks defined in this test.
        # One way is to re-run the chain construction logic from rag_chain.py within the test context.
        
        # Simplified re-construction for testing the flow:
        # This assumes the structure of rag_chain_with_source from rag_chain.py
        # This is a bit of a white-box test but necessary for LCEL.
        
        # 1. Mock the format_docs call within the chain
        # The chain uses: RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        # We can mock format_docs itself for this test to isolate chain logic.
        formatted_docs_output = "Formatted: " + "\n".join([d.page_content for d in sample_docs])
        mock_format_docs = patch('app.llm.rag_chain.format_docs', return_value=formatted_docs_output).start()

        # Rebuild the chain part by part as in rag_chain.py using mocks
        test_rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: mock_format_docs(x["context"])))
            | mock_prompt_instance
            | mock_llm
            | mock_parser_instance # Use the instance of StrOutputParser
        )
        test_rag_chain_with_source = RunnableParallel(
            {"context": mock_retriever, "question": RunnablePassthrough()}
        ).assign(answer=test_rag_chain_from_docs)

        # --- Invoke the chain ---
        query = "What are fruits?"
        # Temporarily replace the module's chain with our test-specific one
        original_chain = rag_chain.rag_chain_with_source
        rag_chain.rag_chain_with_source = test_rag_chain_with_source
        
        response = rag_chain.invoke_rag_chain(query)

        # Restore original chain
        rag_chain.rag_chain_with_source = original_chain
        mock_format_docs.stop()

    # --- Assertions ---
    assert response is not None
    assert response["question"] == query
    assert response["answer"] == "Specific LLM answer for this test"
    
    # Check context (it's transformed to dicts by invoke_rag_chain)
    assert len(response["context"]) == len(sample_docs)
    for i, doc_dict in enumerate(response["context"]):
        assert doc_dict["page_content"] == sample_docs[i].page_content
        assert doc_dict["metadata"] == sample_docs[i].metadata

    # Verify interactions
    mock_retriever.invoke.assert_called_once_with(query)
    mock_format_docs.assert_called_once_with(sample_docs) # format_docs was called with the retrieved docs
    
    # Prompt should be invoked with a dict containing 'context' and 'question'
    # The actual input to prompt.invoke depends on the RunnablePassthrough.assign output
    mock_prompt_instance.invoke.assert_called_once()
    prompt_invoke_args = mock_prompt_instance.invoke.call_args[0][0]
    assert prompt_invoke_args['context'] == formatted_docs_output
    assert prompt_invoke_args['question'] == query
    
    # LLM should be invoked with output from prompt
    mock_llm.invoke.assert_called_once_with(mock_prompt_instance.invoke.return_value)
    
    # Output parser should be invoked with output from LLM
    mock_parser_instance.invoke.assert_called_once_with(mock_llm.invoke.return_value)


def test_rag_chain_no_documents_found(mock_dependencies):
    """Test RAG chain when retriever finds no documents."""
    mock_retriever = mock_dependencies['retriever']
    mock_llm = mock_dependencies['llm']

    mock_retriever.invoke.return_value = [] # No documents found
    # LLM should respond based on "No relevant documents found." context
    # The prompt template in rag_chain.py instructs:
    # "If the context section states 'No relevant documents found.' ... state: 'The answer to your question is not found in the provided documents.'"
    expected_llm_answer_no_docs = "The answer to your question is not found in the provided documents."
    mock_llm.invoke.return_value = AIMessage(content=expected_llm_answer_no_docs) # Simulate LLM following instructions

    # Patch StrOutputParser's invoke to return the raw content for this test
    with patch.object(StrOutputParser, 'invoke', return_value=expected_llm_answer_no_docs) as mock_parser_invoke:
        # Rebuild the chain for this specific LLM mock behavior (if necessary, or ensure general mock_llm handles it)
        # For this test, we assume the globally mocked llm (via mock_dependencies) is used by the chain.
        # If rag_chain.py is structured to rebuild its chain or components on each call, this is fine.
        # If rag_chain_with_source is fixed at module load, we need to ensure it uses this test's mock_llm.
        # The mock_dependencies fixture patches rag_chain.llm, so it should be used.
        # We also need to ensure the StrOutputParser in the chain acts as expected.
        
        # To ensure the chain uses the specific LLM output for this test case:
        rag_chain.llm.invoke.return_value = AIMessage(content=expected_llm_answer_no_docs)
        # And ensure the parser returns this string
        rag_chain.StrOutputParser().invoke.return_value = expected_llm_answer_no_docs


        query = "Query for no docs"
        response = rag_chain.invoke_rag_chain(query)

    assert response is not None
    assert response["question"] == query
    assert response["answer"] == expected_llm_answer_no_docs
    assert response["context"] == [] # No documents
    
    mock_retriever.invoke.assert_called_once_with(query)
    
    # Verify the LLM was invoked. The input to LLM (after prompt) would contain "No relevant documents found."
    mock_llm.invoke.assert_called_once()
    # We can also check the input to the prompt template if needed by mocking/inspecting it.


def test_rag_chain_llm_error(mock_dependencies):
    """Test RAG chain when the LLM call raises an error."""
    mock_retriever = mock_dependencies['retriever']
    mock_llm = mock_dependencies['llm']

    mock_retriever.invoke.return_value = [LangchainDocument(page_content="Some content")]
    mock_llm.invoke.side_effect = Exception("LLM API error")

    query = "Query causing LLM error"
    response = rag_chain.invoke_rag_chain(query)

    assert response is not None
    assert response["question"] == query
    assert "Error processing your query: LLM API error" in response["answer"]
    # Context might still be there as retrieval succeeded
    assert len(response["context"]) == 1 


def test_rag_chain_retriever_error(mock_dependencies):
    """Test RAG chain when the retriever call raises an error."""
    mock_retriever = mock_dependencies['retriever']
    
    mock_retriever.invoke.side_effect = Exception("Retriever DB error")

    query = "Query causing retriever error"
    response = rag_chain.invoke_rag_chain(query)

    assert response is not None
    assert response["question"] == query
    assert "Error processing your query: Retriever DB error" in response["answer"]
    assert response["context"] == [] # No context as retrieval failed


def test_invoke_rag_chain_not_initialized():
    """Test invoke_rag_chain when the chain is not initialized."""
    original_chain = rag_chain.rag_chain_with_source
    rag_chain.rag_chain_with_source = None # Simulate chain not being initialized
    
    response = rag_chain.invoke_rag_chain("test query")
    
    assert response["answer"] == "Error: RAG chain is not initialized. Please check server logs."
    rag_chain.rag_chain_with_source = original_chain # Restore


def test_invoke_rag_chain_invalid_query():
    """Test invoke_rag_chain with invalid query types."""
    response_none_query = rag_chain.invoke_rag_chain(None)
    assert "Error: Invalid query provided." in response_none_query["answer"]

    response_empty_query = rag_chain.invoke_rag_chain("")
    assert "Error: Invalid query provided." in response_empty_query["answer"]

# Note: To run these tests, ensure PYTHONPATH is set up for pytest to find the 'app' module.
# E.g., from project root: PYTHONPATH=. pytest tests/llm/test_rag_chain.py
