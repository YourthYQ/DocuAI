# DocuAI

DocuAI is a sophisticated AI-powered document interaction and retrieval system. It leverages advanced Large Language Models (LLMs) and vector databases to provide intelligent and context-aware responses to user queries based on a given set of documents.

## Features

- **Document Upload and Processing**: Securely upload and process various document formats (PDF, DOCX, TXT).
- **Intelligent Search**: Perform semantic search across multiple documents to find relevant information.
- **Contextual Conversations**: Engage in interactive conversations with the AI, which understands the context of your documents.
- **Source Tracking**: Get references to the source documents for each piece of information provided.
- **Scalable Architecture**: Built with a modular and scalable design to handle growing data and user loads.
- **Document Ingestion**: Supports direct text input, and PDF/TXT file uploads for document processing and ingestion.
- **Chat Log Persistence**: Chat interactions are saved to MongoDB for long-term storage and audit. Session history can be retrieved via API.

## Tech Stack

- **Backend**: Python (FastAPI)
- **LLM Integration**: Langchain, OpenAI
- **Vector Database**: Pinecone (Used via LangChain's `PineconeVectorStore`)
- **Chat Session Memory**: Redis (via LangChain's `RedisChatMessageHistory`)
- **Chat Log Persistence**: MongoDB
- **Frontend**: (To be determined - e.g., React, Vue, or Streamlit for demos)
- **Containerization**: Docker, Docker Compose

## LLM Customization: Fine-tuning and Control

While DocuAI currently leverages Retrieval Augmented Generation (RAG) with general-purpose Large Language Models (LLMs) and sophisticated prompt engineering, further customization can be achieved through fine-tuning or other control mechanisms.

### 1. What is LLM Fine-tuning?

Fine-tuning is the process of adapting a pre-trained LLM to specific tasks, domains, or desired response styles. Instead of training a model from scratch (which requires vast resources), fine-tuning adjusts the weights of an existing model using a smaller, domain-specific dataset. This helps the LLM become more specialized and perform better on targeted tasks.

Common techniques include:
-   **Full Fine-tuning**: All parameters of the pre-trained model are updated during training. This can be resource-intensive.
-   **Parameter-Efficient Fine-tuning (PEFT)**: Only a small subset of the model's parameters are updated, or new, smaller modules (adapters) are added and trained. This significantly reduces computational and storage costs. Examples include Low-Rank Adaptation (LoRA), QLoRA, and prompt tuning.

### 2. When would Fine-tuning be beneficial for DocuAI?

Fine-tuning could be particularly beneficial for DocuAI in scenarios such as:
-   **Highly Specialized Vocabulary/Domains**: If the documents contain jargon, technical terms, or unique linguistic structures not well-represented in the base LLM's training data, fine-tuning can help the model better understand and utilize this specific language.
-   **Specific Response Style or Persona**: If DocuAI needs to consistently adhere to a very particular response style, tone, or persona (e.g., a legal assistant, a technical support bot with a specific level of formality) that is difficult to achieve reliably through prompting alone.
-   **Complex Domain-Specific Reasoning**: For tasks requiring intricate reasoning patterns unique to the document set that the base model struggles with even when provided context via RAG. For example, inferring relationships across multiple document sections in a way specific to a proprietary document format.
-   **Improved Data Efficiency**: Fine-tuning can sometimes lead to better performance with less context needing to be passed in the prompt, potentially reducing costs and latency.

### 3. What kind of dataset would be needed for DocuAI?

For fine-tuning DocuAI, especially in a RAG context, the dataset would typically consist of triples:
-   **Query**: The user's question or instruction.
-   **Context Documents**: The set of relevant document chunks retrieved by the RAG system in response to the query.
-   **Ideal Answer**: The desired response that a human expert would generate based *only* on the information available in the provided context documents. This is crucial to ensure the LLM learns to ground its answers in the provided sources.

The quality and quantity of this dataset are critical for successful fine-tuning.

### 4. General Steps for Fine-tuning (e.g., with LoRA)

A typical workflow for fine-tuning an LLM using a PEFT method like LoRA might involve:
1.  **Data Preparation**: Curating and formatting the (query, context_documents, ideal_answer) dataset into a structure suitable for training (e.g., JSONL format with prompts and completions).
2.  **Choosing a Base Model**: Selecting an appropriate pre-trained LLM (e.g., Llama 2, GPT-Neo, Falcon) that aligns with the project's needs and resources.
3.  **Setting up the Environment**: Installing necessary libraries such as Hugging Face `transformers`, `peft`, `datasets`, and deep learning frameworks like PyTorch or TensorFlow.
4.  **Training the LoRA Adapters**: Configuring the LoRA parameters (e.g., rank, alpha, target modules) and running the training process. The LLM's core weights are frozen, and only the adapter layers are trained.
5.  **Evaluation**: Assessing the fine-tuned model's performance on a held-out test set using relevant metrics (e.g., ROUGE, BLEU for text generation, or task-specific accuracy metrics) and human evaluation.
6.  **Integration**: Saving the trained LoRA adapters and integrating them into the DocuAI generation pipeline. This involves loading the base model and then applying the trained adapters to it before use.

### 5. Alternative Control Mechanisms in DocuAI

This DocuAI project primarily focuses on **Retrieval Augmented Generation (RAG)** and **advanced prompt engineering** as the core mechanisms for controlling LLM behavior and ensuring contextually relevant, factual responses.

Other potential control mechanisms that could be explored as future enhancements include:
-   **Query Transformation**: Modifying or expanding user queries before they are sent to the retrieval system or the LLM to improve relevance or clarity.
-   **Response Post-processing**: Analyzing and potentially modifying the LLM's output to filter out undesirable content, add citations more robustly, or ensure adherence to specific formatting rules.
-   **Guardrails / Moderation Layers**: Implementing checks to ensure responses meet safety guidelines or do not stray into restricted topics.

These methods, often combined, provide a powerful toolkit for guiding LLM behavior without necessarily requiring full fine-tuning.

## Getting Started

This guide will walk you through setting up and running the DocuAI application locally using Docker and Docker Compose.

### Prerequisites

Before you begin, ensure you have the following installed on your system:
-   **Git**: For cloning the repository.
-   **Docker**: For containerizing the application and its services.
-   **Docker Compose**: For orchestrating multi-container Docker applications (usually included with Docker Desktop).

### Setup Steps

1.  **Clone the Repository**:
    First, clone the DocuAI repository to your local machine. If you know the repository URL, replace `<your-repository-url>` with it.
    ```bash
    git clone <your-repository-url> 
    cd docuai 
    ```
    If you are working from a local copy already, navigate into the `docuai` project directory.

2.  **Configure Environment Variables**:
    The application requires API keys and other configurations to be set up in an environment file.
    -   Create a `.env` file in the project root directory by copying the `env.example` file:
        ```bash
        cp env.example .env
        ```
    -   Open the newly created `.env` file with a text editor and fill in your actual API keys and specific configurations:
        -   `OPENAI_API_KEY`: Your OpenAI API key. This is required for embedding generation and LLM responses.
        -   `PINECONE_API_KEY`: Your Pinecone API key. This is required for connecting to your Pinecone vector database.
        -   `PINECONE_ENVIRONMENT`: Your Pinecone index environment (e.g., `gcp-starter`, `us-west1-gcp`, `aws-starter`). You can find this in your Pinecone console under your index details. This is crucial for connecting to your Pinecone index.
        -   `PINECONE_INDEX_NAME`: (Optional) Default is `docuai-index`. The application will attempt to create this index with the correct dimensions (1536 for OpenAI's `text-embedding-ada-002`) if it doesn't exist during startup.
        -   `MONGO_URI`, `MONGO_DB_NAME`, `REDIS_HOST`, `REDIS_PORT`: These defaults are pre-configured for the Docker Compose setup (e.g., `MONGO_URI="mongodb://mongo:27017/docuai_db"`). Change these only if you are using external MongoDB or Redis instances not managed by this `docker-compose.yml`.
        -   `DEFAULT_LLM_MODEL`: (Optional) Default is `gpt-4`. You can specify other compatible OpenAI models like `gpt-3.5-turbo` if preferred.

3.  **Build and Run with Docker Compose**:
    Once your `.env` file is configured, you can build and run the application and its services using Docker Compose.
    ```bash
    docker-compose up --build -d
    ```
    -   The `--build` flag ensures that Docker images are built (or rebuilt if changes are detected).
    -   The `-d` flag runs the containers in detached mode (in the background). You can omit `-d` to see live logs from all services in your terminal.
    This command will:
    -   Build the Docker image for the DocuAI FastAPI application based on the `Dockerfile`.
    -   Pull official images for MongoDB and Redis.
    -   Start containers for the application, MongoDB, and Redis.
    -   Set up networking between the containers so they can communicate.
    -   Create named volumes for MongoDB and Redis to persist data across restarts.

4.  **Verify the Application**:
    After the containers are up and running (this might take a minute or two for the first build and for services to initialize), you can verify that the application is working:
    -   Access the API documentation (Swagger UI) in your browser:
        `http://localhost:8000/docs`
    -   You can also check the alternative API documentation (ReDoc):
        `http://localhost:8000/redoc`
    -   Check the health of the application and its connected services:
        `http://localhost:8000/api/v1/health`
        This endpoint should return a JSON response indicating the status of MongoDB, Redis, OpenAI, and Pinecone connections.

5.  **Using the API**:
    Refer to the "API Reference" section in this README for details on how to interact with the available endpoints, such as uploading documents and chatting with the AI.

6.  **Stopping the Application**:
    To stop the application and all related services:
    ```bash
    docker-compose down
    ```
    -   This command will stop and remove the containers defined in `docker-compose.yml`.
    -   To also remove the data volumes associated with MongoDB and Redis (for a completely clean restart, deleting all stored data), use:
        ```bash
        docker-compose down -v
        ```

## API Reference

This section details the main API endpoints available in DocuAI.

### Core Endpoints

-   **`POST /api/v1/documents/`**
    -   **Description**: Adds a new document directly from text content. The content is stored in MongoDB, and its embedding is upserted into Pinecone.
    -   **Request Body**: `DocumentInput` (JSON with `content: str` and optional `metadata: dict`).
    -   **Response**: `DocumentMinimalOutput` (JSON with `doc_id` and `message`).

-   **`POST /api/v1/documents/upload_file/`**
    -   **Description**: Uploads a document file (PDF or TXT) for processing. The file's text is extracted, split into chunks, and each chunk is embedded and stored in the vector store. Each chunk is linked to a main document ID.
    -   **Request Body**: `multipart/form-data` with a `file` field containing the PDF or TXT file.
    -   **Response**: `FileUploadResponse` (JSON with `filename`, `message`, `total_chunks_processed`, and `document_id` for the uploaded file).

-   **`GET /api/v1/documents/{doc_id}`**
    -   **Description**: Retrieves a document's content and metadata by its ID from MongoDB. (Note: This retrieves the original document if stored whole, not individual chunks from the vector store).
    -   **Path Parameters**: `doc_id: str`.
    -   **Response**: `DocumentOutput` (JSON with `doc_id`, `content`, `metadata`).

-   **`POST /api/v1/chat/`**
    -   **Description**: Handles a user's chat message. It retrieves relevant document chunks from the vector store, generates an AI response using the RAG chain, stores the interaction in Redis (for session memory) and MongoDB (for long-term logging).
    -   **Request Body**: `ChatMessageInput` (JSON with `session_id: str` and `user_message: str`).
    -   **Response**: `ChatMessageOutput` (JSON with `session_id`, `user_message`, `ai_response`, and `retrieved_docs` which includes content and metadata of retrieved chunks).

### Chat History Endpoints

-   **`GET /api/v1/sessions/`**
    -   **Description**: Lists previously recorded chat sessions, ordered by the most recent interaction.
    -   **Query Parameters**:
        -   `skip: int` (default 0): Number of sessions to skip for pagination.
        -   `limit: int` (default 100, max 200): Maximum number of sessions to return.
    -   **Response**: A list of `SessionDetailModel`, including `session_id`, `last_interaction_time`, and `total_messages` for each session.

-   **`GET /api/v1/chat/{session_id}/history`**
    -   **Description**: Retrieves the detailed chat history for a specific session, sorted by timestamp.
    -   **Path Parameters**: `session_id: str`.
    -   **Query Parameters**:
        -   `skip: int` (default 0): Number of log entries to skip.
        -   `limit: int` (default 100, max 1000): Maximum number of log entries to return.
    -   **Response**: A list of `ChatLogEntryModel`, including `interaction_id`, `user_message`, `ai_response`, `timestamp`, `retrieved_doc_ids`, and `feedback` for each interaction.

### Health Check

-   **`GET /api/v1/health`**
    -   **Description**: Performs a health check of the application and its connected services (MongoDB, Redis, OpenAI, Pinecone).
    -   **Response**: `HealthStatus` (JSON indicating the status of each component).


## Project Structure

```
docuai/
├── app/                  # Main application code
│   ├── api/              # API endpoints
│   ├── core/             # Core logic and settings
│   ├── data/             # Data models and schemas
│   ├── retrieval/        # Document retrieval and vector DB logic
│   ├── llm/              # LLM integration and prompting
│   └── services/         # Business logic services
├── tests/                # Unit and integration tests
├── docs/                 # Project documentation
├── scripts/              # Utility scripts
├── .gitignore            # Files to ignore in git
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image definition
└── docker-compose.yml    # Docker Compose setup
```

## Contributing

(Contribution guidelines will be added later)

## License

(License information will be added later)

## Deployment

This section provides instructions for deploying the DocuAI application using Docker and Docker Compose.

### Prerequisites

-   Docker installed on your system.
-   Docker Compose installed on your system (usually included with Docker Desktop).
-   Access to OpenAI and Pinecone API keys.

### 1. Environment Configuration (`.env` file)

Before running the application, create a `.env` file in the root of the project (i.e., in the `docuai/` directory, next to `docker-compose.yml`). This file will store your sensitive credentials and environment-specific configurations.

A sample `.env` file content:

```env
# .env

# OpenAI Configuration
OPENAI_API_KEY="your_openai_api_key_here"

# Pinecone Configuration
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_ENVIRONMENT="your_pinecone_environment_here" # e.g., "gcp-starter" or "us-west1-gcp"
PINECONE_INDEX_NAME="docuai-index" # Or your preferred index name

# MongoDB Configuration (for Docker Compose)
# These are defaults if you use the docker-compose setup as is.
# If using an external MongoDB, update MONGO_URI accordingly.
MONGO_URI="mongodb://mongo:27017/docuai_db"
MONGO_DB_NAME="docuai_db"

# Redis Configuration (for Docker Compose)
# These are defaults if you use the docker-compose setup as is.
# If using an external Redis, update REDIS_HOST and REDIS_PORT.
REDIS_HOST="redis"
REDIS_PORT="6379"
REDIS_CONVERSATION_DB="0"
CONVERSATION_TIMEOUT_SECONDS="3600"

# Application Settings (Optional Overrides)
# PROJECT_NAME="DocuAI"
# VERSION="0.1.0"
```

**Important**:
-   Replace placeholder values (like `your_openai_api_key_here`) with your actual credentials.
-   Ensure the `.env` file is listed in your `.gitignore` file to prevent committing sensitive information. (It is already in the provided `.gitignore`).

### 2. Building the Docker Image

You can build the Docker image for the application using the `Dockerfile` provided:

```bash
docker build -t docuai .
```

The `.` indicates that the build context is the current directory (which should be the `docuai` project root). The `-t docuai` tags the image with the name `docuai`.

### 3. Running with `docker-compose` (Recommended for Local Development & Testing)

The `docker-compose.yml` file is configured to run the application (`app` service) along with `mongo` and `redis` services.

**Steps:**

1.  **Ensure your `.env` file is created** as described above.
2.  **Build and start the services**:
    ```bash
    docker-compose up --build -d
    ```
    -   `--build`: Forces Docker Compose to rebuild the `app` image if it has changed.
    -   `-d`: Runs the containers in detached mode (in the background).
    You can also use `docker-compose up` to see logs in the foreground.

3.  **Accessing the API**:
    Once the services are running, the DocuAI API will be accessible at:
    -   **API Base URL**: `http://localhost:8000`
    -   **Health Check**: `http://localhost:8000/api/v1/health`
    -   **Interactive API Docs (Swagger UI)**: `http://localhost:8000/docs`
    -   **Alternative API Docs (ReDoc)**: `http://localhost:8000/redoc`

4.  **Stopping the services**:
    ```bash
    docker-compose down
    ```

### 4. Running a Standalone Docker Container (Production-like Testing)

You can also run the built Docker image as a standalone container. This is useful for testing the image itself but requires manually setting up MongoDB and Redis instances or providing connection strings to external instances.

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your_openai_api_key" \
  -e PINECONE_API_KEY="your_pinecone_api_key" \
  -e PINECONE_ENVIRONMENT="your_pinecone_environment" \
  -e PINECONE_INDEX_NAME="your_pinecone_index_name" \
  -e MONGO_URI="mongodb://your_mongo_host:27017/your_db_name" \
  -e MONGO_DB_NAME="your_db_name" \
  -e REDIS_HOST="your_redis_host" \
  -e REDIS_PORT="6379" \
  --name docuai_container \
  docuai
```

**Note**:
-   Replace placeholder values with your actual credentials and service URIs.
-   This command assumes MongoDB and Redis are accessible from where Docker is running. For local testing with this method, you might start MongoDB and Redis containers separately or use cloud-hosted instances.

### 5. General Cloud Deployment Guidance

For deploying DocuAI to a cloud environment, consider the following:

-   **Container Registry**:
    -   Push your built Docker image (`docuai`) to a container registry like:
        -   Docker Hub
        -   Amazon Elastic Container Registry (AWS ECR)
        -   Google Artifact Registry (GCP)
        -   Azure Container Registry (Azure)
    -   Example (Docker Hub):
        ```bash
        docker tag docuai yourusername/docuai:latest
        docker push yourusername/docuai:latest
        ```

-   **Cloud Platforms**:
    -   Most cloud platforms can pull your Docker image from a registry and run it. Popular choices include:
        -   **PaaS/Serverless Containers**: Render, Railway, AWS App Runner, Google Cloud Run. These often simplify deployment and scaling.
        -   **Orchestration Platforms**: AWS Elastic Kubernetes Service (EKS), Google Kubernetes Engine (GKE), Azure Kubernetes Service (AKS), or AWS Elastic Container Service (ECS) for more complex deployments.

-   **Managed Databases**:
    -   **MongoDB**: For production, it is highly recommended to use a managed MongoDB service like:
        -   MongoDB Atlas (official managed service)
        -   AWS DocumentDB (MongoDB-compatible)
        -   Azure Cosmos DB (with MongoDB API)
    -   Update the `MONGO_URI` and `MONGO_DB_NAME` environment variables in your deployment configuration to point to your managed MongoDB instance.
    -   **Redis**: Similarly, use a managed Redis service:
        -   Redis Enterprise Cloud
        -   AWS ElastiCache for Redis
        -   Google Cloud Memorystore for Redis
        -   Azure Cache for Redis
    -   Update `REDIS_HOST`, `REDIS_PORT`, and potentially `REDIS_PASSWORD` (if applicable) environment variables to point to your managed Redis instance.

-   **Environment Variables**: Securely manage your environment variables (API keys, database URIs) using the configuration management tools provided by your cloud platform (e.g., AWS Secrets Manager, Google Secret Manager, Azure Key Vault, or platform-specific environment variable settings).

-   **Production Uvicorn Workers**: For production, you might consider running Uvicorn with multiple workers instead of the `--reload` flag. This is often handled by process managers like Gunicorn running Uvicorn workers. Example `CMD` in `Dockerfile` for such a setup (though not implemented in the current Dockerfile for simplicity):
    `CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]`
    This would require adding `gunicorn` to `requirements.txt`. The current Dockerfile uses `uvicorn --reload`, which is suitable for development.
