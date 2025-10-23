# AI-Powered Legal Search Engine

A standalone Streamlit application that enables legal professionals to upload, search, and query Indonesian legal documents using semantic search and RAG capabilities.

## 🚀 Features

- **📄 Document Upload**: Upload Indonesian legal documents (PDF/TXT) up to 50MB
- **🔍 Semantic Search**: Natural language search using vector embeddings
- **🤖 AI-Powered Q&A**: Get cited answers using Groq LLM
- **⚖️ Legal Structure Recognition**: Automatic extraction of UU, Bab, Pasal, Ayat hierarchy
- **📊 Performance**: <2 second search responses, <30 second document processing
- **🔐 Privacy**: Local-first architecture with no external data storage

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Vector Database**: FAISS (CPU)
- **Metadata Database**: SQLite
- **Embeddings**: Cohere embed-v4
- **LLM**: Groq (Mixtral/Llama3)
- **Document Parsing**: PyMuPDF
- **Language**: Python 3.11+

## 📋 Prerequisites

- Python 3.11 or higher
- Cohere API key
- Groq API key
- 4GB RAM minimum (8GB recommended)
- 2GB available disk space

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "AI-Powered Legal Search Engine"
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
# Add your Cohere and Groq API keys
```

### 5. Initialize Application

```bash
# Run setup script
python scripts/setup.py

# Or manually:
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

## 📚 Usage Guide

### Upload Documents

1. Open the application in your browser
2. Go to the "Upload Documents" tab
3. Select PDF or TXT files containing Indonesian legal text
4. Wait for processing to complete
5. Verify documents appear in the processed documents list

### Search Documents

1. Go to the "Search" tab
2. Enter your legal question in natural language
3. Use filters to narrow results by UU Title or Chapter
4. Review search results with similarity scores
5. Click "Get AI-Powered Answer" for comprehensive responses

### AI-Powered Q&A

1. After performing a search, click "Get AI-Powered Answer"
2. The AI will generate a comprehensive answer using retrieved context
3. All information includes proper legal citations
4. Review the response and cited sources

## 📁 Project Structure

```
src/
├── app.py                    # Main Streamlit application
├── models/                   # Data models
│   ├── document.py          # LegalDocument entity
│   ├── chunk.py             # LegalChunk entity
│   └── search.py            # Search entities
├── services/                # Business logic
│   ├── database.py          # SQLite operations
│   ├── vector_store.py      # FAISS vector operations
│   ├── embedding.py         # Cohere integration
│   ├── search_service.py    # Semantic search
│   └── rag_service.py       # RAG and LLM integration
├── ui/                      # User interface
│   ├── components/          # Reusable UI components
│   └── pages/               # Application pages
└── utils/                   # Utilities
    ├── config.py            # Configuration management
    └── validation.py        # Input validation

tests/                       # Test suite
scripts/                     # Utility scripts
requirements.txt             # Dependencies
.env.example                 # Environment template
README.md                    # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# API Keys (required)
COHERE_API_KEY=your_cohere_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional: Override defaults
DATABASE_PATH=src/data/.legal_db.sqlite
FAISS_INDEX_PATH=src/data/.faiss_index
UPLOAD_DIR=src/data/uploads
```

### Performance Settings

Adjust performance settings in the Settings tab or via environment variables:

- `EMBEDDING_BATCH_SIZE`: Number of texts processed per API call (default: 100)
- `MAX_SEARCH_RESULTS`: Maximum search results returned (default: 20)
- `SEARCH_TIMEOUT_MS`: Search timeout in milliseconds (default: 5000)
- `LLM_TIMEOUT_MS`: LLM response timeout in milliseconds (default: 30000)

## 🧪 Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
pytest tests/unit/
pytest tests/integration/
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html
```

## 📊 Performance

### Target Performance Metrics

- **Document Processing**: <30 seconds per 50MB file
- **Semantic Search**: <2 seconds (P90 latency)
- **RAG Generation**: <2 seconds total
- **Memory Usage**: <2GB typical
- **Index Size**: <1GB for 100+ documents

### Optimization Tips

- Use appropriate batch sizes for embedding generation
- Limit concurrent document processing
- Regularly backup vector index and database
- Monitor memory usage with large document collections

## 🔒 Security & Privacy

- ✅ All documents stored locally on user machine
- ✅ No document content transmitted to external services (except API calls)
- ✅ API keys stored securely in environment variables
- ✅ No external logging of document content
- ✅ Local processing ensures data privacy

## 🚨 Troubleshooting

### Common Issues

**"ModuleNotFoundError"**:
```bash
# Ensure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

**API Key Errors**:
1. Verify API keys are correct in `.env` file
2. Check API key status on provider dashboards
3. Ensure sufficient credits/quota available

**Performance Issues**:
1. Check network connection to API services
2. Reduce embedding batch size
3. Close other applications using memory
4. Restart application to reload index

**Database/Index Errors**:
```bash
# Reset data files
rm src/data/.legal_db.sqlite
rm src/data/.faiss_index
python scripts/setup.py
```

## 📖 API Documentation

### Supported Document Types

- **PDF**: Legal documents, statutes, regulations
- **TXT**: Plain text legal documents
- **Maximum Size**: 50MB per file
- **Supported Languages**: Indonesian, English

### Search Features

- **Semantic Search**: Context-based, not keyword-based
- **Legal Filtering**: Filter by UU Title and Chapter
- **Similarity Scoring**: Cosine similarity with threshold
- **Citation Extraction**: Automatic legal citation formatting

### RAG Capabilities

- **Context Assembly**: Top-5 search results used as context
- **Citation Generation**: Automatic citation extraction and formatting
- **Answer Quality**: Grounded in provided documents only
- **Response Time**: <2 seconds for typical queries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and ensure they pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Cohere** - Embedding API for semantic search
- **Groq** - Fast LLM inference for RAG
- **FAISS** - Vector similarity search
- **Streamlit** - Web application framework

## 📞 Support

For issues, questions, or support:

1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Include error messages and system information

---

**Built with ❤️ for legal professionals in Indonesia**