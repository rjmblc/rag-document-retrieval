# rag-document-retrieval

A Retrieval-Augmented Generation (RAG) application that ingests PDF documents, converts them into vector embeddings, stores them in a vector database, and retrieves context-aware responses using semantic search and Large Language Models (LLMs).

# Features

* PDF document ingestion
* Text chunking and preprocessing
* Vector embeddings generation
* ChromaDB vector storage
* Semantic similarity search
* Context-aware question answering
* LangChain-powered RAG pipeline

# Tech Stack

* Python
* LangChain
* ChromaDB
* Sentence Transformers
* HuggingFace Embeddings
* PyMuPDF
* OpenAI / LLM Integration

# Project Structure

```bash
rag-document-retrieval/
│
├── docs_dir/
├── vector_db/
├── rag_docs_ingestion.py
├── rag_retrieval_generation.py
├── requirements.txt
└── README.md
```

# Installation

## Clone Repository

```bash
git clone https://github.com/rjmblc/rag-document-retrieval.git
cd rag-document-retrieval
```

## Create Virtual Environment

```bash
python -m venv .venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

# Install Dependencies

```bash
pip install -r requirements.txt
```

# Add Documents

Place PDF files inside:

```bash
docs_dir/
```

# Run Document Ingestion

```bash
python rag_docs_ingestion.py
```

This performs:

* document loading
* chunking
* embeddings generation
* vector database creation

# Run Retrieval & Generation

```bash
python rag_retrieval_generation.py
```

Example Query:

```text
What is the Uric Acid result in the report?
```

# RAG Pipeline

```text
PDF Documents
      ↓
Document Loader
      ↓
Text Chunking
      ↓
Embeddings
      ↓
Vector Database
      ↓
Semantic Retrieval
      ↓
LLM Response Generation
```
# Future Improvements

* Streamlit chatbot UI
* FastAPI backend
* Multi-document retrieval
* Hybrid search
* Reranking pipelines
* Agentic RAG workflows

# Author

Rajmohan Balachandran

GitHub:
https://github.com/rjmblc

# License

This project is intended for learning and educational purposes.

