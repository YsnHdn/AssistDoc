from setuptools import setup, find_packages

setup(
    name="assistdoc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Interface utilisateur
        "streamlit>=1.29.0",
        
        # Traitement des documents
        "PyPDF2>=3.0.0",
        "python-docx>=0.8.11",
        "pdfminer.six>=20221105",
        "python-pptx>=0.6.21",
        "docx2txt>=0.8",
        
        # Gestion des embeddings et bases vectorielles
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "chromadb>=0.4.18",
        
        # LLM et chaînes de traitement - Version Hugging Face
        "langchain>=0.0.335",
        "langchain-community>=0.0.10",
        "transformers>=4.34.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "torch>=2.0.0",
        "optimum>=1.12.0",
        
        # Pour les modèles spécifiques comme Deepseek
        "huggingface-hub>=0.19.0",
        "einops>=0.7.0",
        
        # Utilitaires
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "nltk>=3.8.1",
    ],
    author="Yassine HANDANE",
    author_email="y.handane@gmail.com",
    description="AssistDoc - Assistant intelligent d'analyse de documents",
)