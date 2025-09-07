# 🏛️ LawBot - Legal AI Assistant

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)

> **An intelligent legal assistant powered by AI to answer questions based on Indian law documents using Retrieval-Augmented Generation (RAG)**

LawBot is a sophisticated RAG-based chatbot specifically designed to provide accurate legal information from Indian law documents. It combines the power of Mistral-7B language model with a comprehensive FAISS vector database to deliver precise, context-aware responses with proper source attribution.

## 📋 Table of Contents

- [🎯 Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🔧 Technical Stack](#-technical-stack)
- [📊 Performance Metrics](#-performance-metrics)
- [⚡ Quick Start](#-quick-start)
- [🚀 Installation](#-installation)
- [⚙️ Configuration](#️-configuration)
- [🎮 Usage](#-usage)
- [📚 Legal Documents](#-legal-documents)
- [🔍 How It Works](#-how-it-works)
- [📈 API Reference](#-api-reference)
- [🧪 Testing](#-testing)
- [🚀 Deployment](#-deployment)
- [🤝 Contributing](#-contributing)
- [📊 Project Status](#-project-status)
- [🗺️ Roadmap](#️-roadmap)
- [📄 License](#-license)

## 🎯 Key Features

### Core Capabilities
- **🔍 Context-Aware Responses**: Uses RAG to provide answers grounded in actual legal documents
- **📚 Comprehensive Knowledge Base**: Built on essential Indian law documents (Constitution, IPC, CrPC, CPC, Evidence Act)
- **💬 Interactive Chat Interface**: Clean, intuitive Streamlit-based web interface with session management
- **🎯 Source Attribution**: Every response includes relevant source document references with page numbers
- **🔒 Privacy-Focused**: Processes data locally; no external data storage or privacy concerns
- **⚡ Fast Retrieval**: Optimized FAISS vector search for sub-second document retrieval

### Technical Features
- **🤖 Advanced NLP**: Powered by Mistral-7B-Instruct model via Hugging Face
- **🗃️ Vector Database**: FAISS-based efficient document retrieval with 384-dimensional embeddings
- **🔄 Session Management**: Maintains conversation context across interactions
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices
- **🛡️ Robust Error Handling**: Graceful degradation with comprehensive error management
- **💾 Smart Caching**: Streamlit caching for optimal performance

## 🏗️ System Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Streamlit Web Interface]
        USER[End User]
    end
    
    subgraph "Application Layer"
        STREAMLIT[Streamlit App Controller]
        SESSION[Session Management]
        CACHE[Streamlit Cache]
    end
    
    subgraph "Business Logic Layer"
        QA_CHAIN[RetrievalQA Chain]
        PROMPT_ENGINE[Prompt Template Engine]
        RESPONSE_FORMATTER[Response Formatter]
    end
    
    subgraph "AI/ML Layer"
        LLM[Mistral-7B-Instruct Model]
        EMBEDDINGS[HuggingFace Embeddings]
        RETRIEVER[Document Retriever]
    end
    
    subgraph "Data Layer"
        VECTOR_STORE[(FAISS Vector Store)]
        LEGAL_DOCS[(Legal Documents)]
        INDEX[Vector Index]
    end
    
    subgraph "External Services"
        HF_HUB[Hugging Face Hub]
    end
    
    USER --> UI
    UI --> STREAMLIT
    STREAMLIT --> QA_CHAIN
    QA_CHAIN --> RETRIEVER
    RETRIEVER --> VECTOR_STORE
    QA_CHAIN --> LLM
    LLM --> HF_HUB
    
    style USER fill:#e1f5fe
    style QA_CHAIN fill:#fff3e0
    style LLM fill:#e8f5e8
    style VECTOR_STORE fill:#fce4ec
```

### RAG Pipeline Flow

```mermaid
sequenceDiagram
    participant User
    participant StreamlitUI
    participant QAChain
    participant VectorStore
    participant LLM
    participant HuggingFace
    
    User->>StreamlitUI: Submit Legal Query
    StreamlitUI->>QAChain: Process Query Request
    QAChain->>VectorStore: Search Similar Documents (k=3)
    VectorStore->>VectorStore: Compute Query Embeddings
    VectorStore->>QAChain: Return Relevant Documents
    QAChain->>QAChain: Assemble Context + Query
    QAChain->>LLM: Send Structured Prompt
    LLM->>HuggingFace: API Call to Mistral-7B
    HuggingFace->>LLM: Generated Response
    LLM->>QAChain: Answer with Reasoning
    QAChain->>QAChain: Format with Source Attribution
    QAChain->>StreamlitUI: Complete Response
    StreamlitUI->>User: Display Answer + Sources
```

## 🔧 Technical Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Frontend** | Streamlit | 1.28+ | Web interface and user interaction |
| **Language Model** | Mistral-7B-Instruct-v0.3 | Latest | Response generation and legal reasoning |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Latest | Document and query vectorization |
| **Vector DB** | FAISS | 1.7.4+ | Efficient similarity search (384-dim) |
| **RAG Framework** | LangChain | 0.1+ | Pipeline orchestration and management |
| **ML Platform** | Hugging Face | Latest | Model hosting and inference API |
| **Backend** | Python | 3.8+ | Core application logic |
| **Environment** | Pipenv | Latest | Dependency management |

### Architecture Patterns

- **RAG (Retrieval-Augmented Generation)**: Combines parametric and non-parametric knowledge
- **Microservices Architecture**: Modular components with clear separation of concerns
- **Event-Driven Processing**: Reactive UI updates based on processing pipeline state
- **Caching Strategy**: Multi-level caching for performance optimization

## 📊 Performance Metrics

### Current Performance

| Metric | Current Value | Target | Status |
|--------|---------------|--------|--------|
| **Query Response Time** | 3-8 seconds | <5 seconds | ⚠️ Optimizing |
| **Vector Search Time** | <100ms | <50ms | ✅ Good |
| **Model Loading Time** | 30-60 seconds | <30 seconds | ⚠️ Improving |
| **Memory Usage** | 2-4GB | <2GB | ⚠️ High |
| **Document Retrieval Accuracy** | ~85% | >90% | ✅ Good |
| **Concurrent Users** | 1-5 | 10+ | 🔄 Scaling |

### Resource Requirements

```yaml
Minimum System Requirements:
  RAM: 8GB (16GB recommended)
  Storage: 5GB free space
  CPU: 4 cores (8 cores recommended)
  Internet: Required for model download
  
Production Requirements:
  RAM: 16GB+
  Storage: 20GB+ SSD
  CPU: 8+ cores
  Network: High-speed internet for API calls
```

## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/01rahulsingh86/LawBot---Your-Legal-Assistant.git
cd LawBot---Your-Legal-Assistant

# Set up environment
export HF_TOKEN=your_hugging_face_token_here

# Install and run
pipenv install
pipenv run streamlit run lowbot.py

# Access at http://localhost:8501
```

## 🚀 Installation

### Method 1: Pipenv (Recommended)

```bash
# Prerequisites
pip install pipenv

# Install dependencies
pipenv install

# Activate environment
pipenv shell

# Verify installation
python -c "import streamlit; print('✅ Installation successful')"
```

### Method 2: Virtual Environment

```bash
# Create virtual environment
python -m venv lawbot_env
source lawbot_env/bin/activate  # Windows: lawbot_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Method 3: Docker

```dockerfile
# Dockerfile included in repository
docker build -t lawbot .
docker run -p 8501:8501 -e HF_TOKEN=your_token lawbot
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```env
# Required - Get from https://huggingface.co/settings/tokens
HF_TOKEN=your_hugging_face_token_here
HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token_here

# Optional Configuration
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE_PATH=vector_store/db_faiss
MAX_NEW_TOKENS=512
TEMPERATURE=0.5
SEARCH_K=3
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Advanced Configuration

```yaml
# config.yaml (optional)
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
  temperature: 0.5
  max_tokens: 512
  
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
  
vector_store:
  path: "vector_store/db_faiss"
  search_k: 3
  similarity_threshold: 0.7
  
documents:
  chunk_size: 1000
  chunk_overlap