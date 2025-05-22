from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document as LangchainDocument # Alias to avoid confusion if any other Document class is used

from app.core.config import settings
from app.retrieval.vector_retriever import get_vector_store # Assuming this is already LangChain compatible

# 1. Initialize Components
try:
    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model_name=settings.DEFAULT_LLM_MODEL, # Already added to Settings
        temperature=0.2
    )
    print(f"ChatOpenAI LLM initialized with model: {settings.DEFAULT_LLM_MODEL}")
except Exception as e:
    llm = None
    print(f"Error initializing ChatOpenAI LLM: {e}")

vector_store = None
retriever = None
if get_vector_store: # Check if function exists
    vector_store = get_vector_store() # From vector_retriever.py, uses default index
    if vector_store:
        retriever = vector_store.as_retriever(search_kwargs={'k': 5}) # Configure top_k
        print(f"Vector store retriever initialized with k=5. Default index: {getattr(vector_store, 'index_name', 'N/A')}")
    else:
        print("Failed to initialize vector store from vector_retriever. Retriever not set.")
else:
    print("get_vector_store function not found in vector_retriever. Retriever not set.")


# 2. Define Prompt Template
# This prompt is refined to guide the LLM based on whether context is found or not.
# The format_docs helper will explicitly state "No relevant documents found." if docs are empty.
template = """You are a helpful AI assistant. Your task is to answer the question concisely, based *solely* on the provided context documents.
- Do not use any external knowledge or information outside of the provided context. Your knowledge is limited to the documents given.
- If the context section states 'No relevant documents found.' or is empty, you *must* state: 'The answer to your question is not found in the provided documents.' Do not provide any other explanation or attempt to answer from general knowledge.
- If context documents are provided, base your answer only on the information within them. Be precise.

Context:
{context}

Question:
{question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)


# 3. Helper function format_docs
def format_docs(docs: list[LangchainDocument]) -> str:
    """
    Formats a list of LangChain Document objects into a single string for the prompt.
    """
    if not docs:
        return "No relevant documents found."
    
    return "\n---\n".join([
        f"Document ID: {doc.metadata.get('doc_id', 'N/A')}\nContent: {doc.page_content}" 
        for doc in docs
    ])

# 4. Construct the RAG Chain using LCEL
# Check if essential components are initialized before defining the chain
if llm and retriever and prompt:
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    # This chain takes a "question" string as input.
    # 1. Retrieves documents using the `retriever`.
    # 2. The original `question` is passed through.
    # 3. The retrieved `context` (list of Documents) and original `question` are passed to `rag_chain_from_docs`.
    # 4. `rag_chain_from_docs` formats the documents, then sends context+question to the prompt, then LLM, then parses output.
    # The final output of `rag_chain_with_source` will be a dictionary: {'context': [Docs...], 'question': '...', 'answer': '...'}.
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    print("RAG chain constructed successfully.")
else:
    rag_chain_with_source = None
    print("RAG chain construction failed due to uninitialized components (LLM, Retriever, or Prompt).")


# 5. Main Invocation Function
def invoke_rag_chain(query: str) -> dict | None:
    """
    Invokes the RAG chain with the user's query.
    Returns a dictionary containing the answer, original question, and retrieved context,
    or None if the chain is not available.
    """
    if not rag_chain_with_source:
        print("RAG chain is not available. Cannot invoke.")
        return {
            "question": query,
            "context": [],
            "answer": "Error: RAG chain is not initialized. Please check server logs."
        }
    if not query or not isinstance(query, str):
        print("Invalid query provided.")
        return {
            "question": query,
            "context": [],
            "answer": "Error: Invalid query provided."
        }

    print(f"Invoking RAG chain with query: '{query}'")
    try:
        response = rag_chain_with_source.invoke(query)
        # Ensure context docs are serializable if they aren't by default for some reason
        # (LangChain Documents are generally fine, but good to be mindful for API boundaries)
        if 'context' in response and isinstance(response['context'], list):
            response['context'] = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in response['context']
            ]
        return response
    except Exception as e:
        print(f"Error invoking RAG chain: {e}")
        # Consider specific error handling or re-raising
        return {
            "question": query,
            "context": [],
            "answer": f"Error processing your query: {e}"
        }

# 6. Example Usage
if __name__ == '__main__':
    print("\n--- Running RAG Chain Example ---")
    if not rag_chain_with_source:
        print("Cannot run RAG chain example because the chain is not initialized.")
        print("This might be due to missing API keys (OpenAI, Pinecone) or issues with Pinecone index setup.")
    else:
        sample_queries = [
            "What is LangChain?", # Should find lc_doc2
            "How does PineconeVectorStore integrate with LangChain?", # Should find lc_doc3
            "What is the capital of Mars?" # Should not find anything relevant
        ]

        # Ensure the test index used in vector_retriever.py's __main__ is set up
        # For this test, we assume the default index in settings is the one to use,
        # and it might have been populated by vector_retriever.py's __main__ example.
        # If not, results for specific questions might be empty.
        print(f"Note: This test relies on the default Pinecone index ('{settings.PINECONE_INDEX_NAME}') being populated,")
        print(f"e.g., by running the example in 'docuai/app/retrieval/vector_retriever.py' which uses '{settings.PINECONE_INDEX_NAME}-lc-test'.")
        print(f"Ensure your .env or settings point PINECONE_INDEX_NAME to a populated test index for meaningful results here.")


        for q in sample_queries:
            print(f"\n--- Querying: '{q}' ---")
            chain_response = invoke_rag_chain(q)
            if chain_response:
                print(f"  Question: {chain_response.get('question')}")
                print(f"  Retrieved Context Docs Count: {len(chain_response.get('context', []))}")
                for i, doc_dict in enumerate(chain_response.get('context', [])):
                    print(f"    Doc {i+1} Metadata: {doc_dict.get('metadata')}")
                    # print(f"    Doc {i+1} Content: {doc_dict.get('page_content', '')[:100]}...") # Optionally print content snippet
                print(f"  LLM Answer: {chain_response.get('answer')}")
            else:
                print("  Failed to get a response from the RAG chain.")

    print("\n--- RAG Chain Example Finished ---")
