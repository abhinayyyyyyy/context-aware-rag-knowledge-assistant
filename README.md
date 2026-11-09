# Context-Aware AI Knowledge Assistant (RAG)

An enterprise-grade, retrieval-augmented generation (RAG) knowledge assistant built with Streamlit, FAISS, PostgreSQL + pgvector, Ollama / Cloud LLM APIs, and object storage abstractions for local and cloud deployments.

---

## Architecture Overview

```
                          ┌───────────────────────────┐
                          │   Streamlit Web UI / CLI  │
                          └─────────────┬─────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│ Object Storage Layer │    │    RAG Pipeline      │    │  Persistence Layer   │
│  BaseObjectStorage   │    │  (Ingest/Embed/RAG)  │    │ (DatabaseService)    │
├──────────────────────┤    ├──────────────────────┤    ├──────────────────────┤
│ - LocalStorage       │    │ - sentence-transf.   │    │ - Documents meta     │
│ - S3Storage (AWS/    │    │ - FAISS / pgvector   │    │ - Conversations      │
│   MinIO/R2/Ceph)     │    │ - Ollama / Cloud LLM │    │ - Messages (JSONB)   │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘
```

---

## Features

- **Document Ingestion**: Multi-page PDF text extraction, whitespace normalization, and boundary-aware chunking.
- **Dual Vector Backends**: In-memory FAISS flat L2 vector store and relational PostgreSQL + pgvector.
- **Hybrid LLM Support**: Local Ollama (default `llama3.2:latest`) and OpenAI-compatible Cloud LLM providers (`gpt-4o-mini`, etc.).
- **Pluggable Object Storage**: Decoupled object storage with local filesystem and S3-compatible cloud storage providers.
- **Durable Relational Persistence**: Optional PostgreSQL tracking for document content hashes, conversations, and messages.
- **Container-Ready**: Multi-stage, non-root Docker configuration with container health checks.

---

## Configuration

Copy `.env.example` to `.env` to configure your environment:

```bash
cp .env.example .env
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `STORAGE_PROVIDER` | Object storage provider (`local` or `s3`) | `local` |
| `LOCAL_STORAGE_DIR` | Local storage directory when `STORAGE_PROVIDER=local` | `data/storage` |
| `S3_BUCKET_NAME` | S3 bucket name (required when `STORAGE_PROVIDER=s3`) | *(none)* |
| `S3_REGION_NAME` | AWS region name (e.g. `us-east-1`) | `us-east-1` |
| `S3_ENDPOINT_URL` | Custom S3 endpoint (for MinIO, Cloudflare R2, LocalStack) | *(none)* |
| `S3_ACCESS_KEY_ID` | AWS / S3 access key ID (optional if using IAM roles) | *(none)* |
| `S3_SECRET_ACCESS_KEY` | AWS / S3 secret access key (optional if using IAM roles) | *(none)* |
| `S3_PREFIX` | Optional object key prefix inside S3 bucket | *(none)* |
| `LLM_PROVIDER` | LLM engine (`ollama` or `cloud`) | `ollama` |
| `OLLAMA_MODEL` | Local Ollama model name | `llama3.2:latest` |
| `CLOUD_LLM_API_KEY` | Cloud LLM provider API key | *(none)* |
| `CLOUD_LLM_MODEL` | Cloud model name (e.g. `gpt-4o-mini`) | `gpt-4o-mini` |
| `VECTOR_STORE` | Vector store backend (`faiss` or `postgres`) | `faiss` |
| `PERSISTENCE_ENABLED` | Enable PostgreSQL state persistence (`true` / `false`) | `false` |
| `DATABASE_URL` | PostgreSQL connection URL | *(none)* |

---

## Local Development

### 1. Prerequisites
- Python 3.11 or 3.12
- [Ollama](https://ollama.ai) installed and running locally (`ollama run llama3.2:latest`)

### 2. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Docker Deployment

> [!WARNING]
> **Container Ephemerality Notice**: Files stored within the container filesystem are **not durable** across restarts or redeployments.
> - When using `STORAGE_PROVIDER=local`, mount a persistent host volume or Docker volume to `/app/data/storage` (e.g., `-v /path/to/host/storage:/app/data/storage`).
> - For production cloud deployments, use `STORAGE_PROVIDER=s3` backed by AWS S3, Cloudflare R2, or MinIO, and connect PostgreSQL using `DATABASE_URL` or an external managed database instance.

### Build the Image
```bash
docker build -t rag-knowledge-assistant:latest .
```

### Run Locally (Standalone Mode)
```bash
docker run -d \
  -p 8501:8501 \
  -e PORT=8501 \
  -e STORAGE_PROVIDER=local \
  -v $(pwd)/data/storage:/app/data/storage \
  --name rag-assistant \
  rag-knowledge-assistant:latest
```

### Run with Cloud Storage and Cloud LLM
```bash
docker run -d \
  -p 8501:8501 \
  -e STORAGE_PROVIDER=s3 \
  -e S3_BUCKET_NAME=my-rag-documents \
  -e S3_REGION_NAME=us-east-1 \
  -e AWS_ACCESS_KEY_ID="your-key-id" \
  -e AWS_SECRET_ACCESS_KEY="your-secret-key" \
  -e LLM_PROVIDER=cloud \
  -e CLOUD_LLM_API_KEY="your-openai-api-key" \
  --name rag-assistant \
  rag-knowledge-assistant:latest
```

---

## Testing

Run the automated test suite:

```bash
# Run all tests
python -m unittest discover -s . -p "test_*.py"

# Or run individual test suites
python test_storage.py
python test_persistence.py
```
