# AssistDoc

[![GitHub License](https://img.shields.io/github/license/YsnHdn/AssistDoc)](https://github.com/YsnHdn/AssistDoc/blob/main/LICENSE)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.30.0-FF4B4B)

**AssistDoc** is an intelligent document assistant that uses Retrieval-Augmented Generation (RAG) to help you interact with your documents. It allows you to ask questions, generate summaries, and extract structured information from your PDF, DOCX, and TXT files.

![AssistDoc Screenshot](https://via.placeholder.com/800x450?text=AssistDoc+Screenshot)

## 🌟 Features

- **Document Processing**: Upload and automatically index documents (PDF, DOCX, TXT)
- **Persistent Storage**: Documents remain available between sessions
- **Question Answering**: Ask specific questions about your documents
- **Document Summarization**: Generate customizable summaries with different styles and lengths
- **Information Extraction**: Extract structured information in JSON, table, or text format
- **Multiple LLM Support**: Use GitHub Inference API or Hugging Face models
- **Vector Search**: Semantic search powered by FAISS/ChromaDB

## 📋 Requirements

- Python 3.8+
- Required Python packages (see `requirements.txt`)
- GitHub Personal Access Token (for GitHub Inference API) or Hugging Face API key

## 🚀 Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YsnHdn/AssistDoc.git
   cd AssistDoc
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a configuration file:
   - Copy `config.py.example` to `config.py`
   - Add your API keys to `config.py`

5. Run the application:
   ```bash
   streamlit run app.py
   ```

6. Open your browser at `http://localhost:8501`

### Configuration

Create a `config.py` file with the following structure:

```python
"""
Configuration for the AssistDoc application.
This file contains API keys and other sensitive configurations.
DO NOT INCLUDE THIS FILE IN YOUR GIT REPOSITORY.
"""

# API keys for different LLM providers
API_KEYS = {
    "github_inference": "your_github_token_here",
    "openai": "",
    "anthropic": "",
    "huggingface": ""
}

# Base URLs for APIs (optional)
API_BASE_URLS = {
    "github_inference": "https://models.inference.ai.azure.com",
    "openai": None,
    "anthropic": None,
    "azure_openai": None
}

# Other configurations
DEFAULT_PROVIDER = "github_inference"
DEFAULT_MODEL = "gpt-4o"
```

## 📁 Project Structure

```
AssistDoc/
├── app.py                 # Main application entry point
├── requirements.txt       # Project dependencies
├── README.md              # Documentation
├── .gitignore             # Git ignore file
│
├── data/                  # Data storage directory
│   ├── vector_store/      # Vector database
│   └── uploaded_files/    # Stored uploaded documents
│
├── src/                   # Main source code
│   ├── document_processor/ # Document processing modules
│   │   ├── parser.py      # Text extraction from different formats
│   │   ├── chunker.py     # Document chunking 
│   │   └── embedder.py    # Embedding generation
│   │
│   ├── vector_db/         # Vector database management
│   │   ├── store.py       # FAISS/ChromaDB interface
│   │   └── retriever.py   # Context retrieval
│   │
│   └── llm/               # LLM interaction modules
│       ├── models.py      # Model configuration
│       ├── prompts.py     # Prompt templates
│       └── chain.py       # Processing chains
│
└── ui/                    # User interface
    ├── pages/             # Application pages
    │   ├── home.py        # Home page
    │   ├── qa.py          # Question-answering page
    │   ├── summary.py     # Summary page
    │   └── extraction.py  # Information extraction page
    └── components/        # Reusable components
        ├── sidebar.py     # Sidebar component
        ├── document_uploader.py # Document management
        └── visualization.py # Visualization components
```

## 💡 Usage

### Document Upload and Indexing

1. Upload your documents using the sidebar
2. Documents are automatically processed and indexed
3. Your documents remain available between sessions

### Question-Answering

1. Navigate to the Q&A page
2. Select a document from the dropdown
3. Ask a question about the document
4. Review the answer with highlighted source references

### Summarization

1. Navigate to the Summary page
2. Select a document to summarize
3. Choose summary length and style
4. Generate the summary
5. Download the summary as a markdown file

### Information Extraction

1. Navigate to the Extraction page
2. Select a document to analyze
3. Choose predefined extraction types or create a custom one
4. Select output format (JSON, Table, Text)
5. Extract and review the information

## 🔧 Advanced Configuration

### Vector Store Configuration

You can configure the vector store by modifying the `create_default_retriever` function in `src/vector_db/retriever.py`:

```python
def create_default_retriever(
    store_path: str,
    embedder_model: str = "all-MiniLM-L6-v2",
    store_type: str = "faiss",
    top_k: int = 5
) -> DocumentRetriever:
    # ...
```

### LLM Configuration

You can add or modify LLM configurations in `src/llm/models.py`:

```python
DEFAULT_CONFIGS = {
    "github-gpt-4o": LLMConfig(
        provider=LLMProvider.GITHUB_INFERENCE,
        model_name="gpt-4o",
        temperature=0.7,
        max_tokens=1000,
        api_base="https://models.inference.ai.azure.com"
    ),
    # ...
}
```

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Projects

- [LangChain](https://github.com/langchain-ai/langchain) - Framework for developing applications powered by language models
- [FAISS](https://github.com/facebookresearch/faiss) - Library for efficient similarity search
- [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers) - Generate embeddings for sentences and documents

## 🙏 Acknowledgements

- The Streamlit team for their amazing framework
- The creators of Sentence-Transformers, FAISS, and ChromaDB
- The open-source community for making RAG applications accessible

---

Created with ❤️ by [Yassine Handaine](https://github.com/YsnHdn)