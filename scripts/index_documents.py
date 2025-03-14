# index_documents.py

import sys
import os
from pathlib import Path
import numpy as np
import nltk

# Ajouter le répertoire racine du projet au chemin d'importation
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent  # Remontez jusqu'à la racine du projet
sys.path.append(str(project_root))

import argparse
from src.document_processor.parser import DocumentParser
from src.document_processor.chunker import DocumentChunker
from src.document_processor.embedder import DocumentEmbedder
from src.vector_db.store import create_vector_store

def ensure_nltk_resources():
    """
    Vérifie et télécharge les ressources NLTK nécessaires.
    """
    try:
        # Essayer d'accéder à punkt pour vérifier s'il est déjà téléchargé
        nltk.data.find('tokenizers/punkt')
        print("Ressources NLTK 'punkt' déjà disponibles")
    except LookupError:
        print("Téléchargement des ressources NLTK 'punkt'...")
        nltk.download('punkt')
        print("Ressources NLTK téléchargées avec succès")
    
    # Vérifier aussi punkt_tab si nécessaire pour le chunking sémantique
    try:
        nltk.data.find('tokenizers/punkt_tab/english')
        print("Ressources NLTK 'punkt_tab' déjà disponibles")
    except LookupError:
        print("Téléchargement des ressources NLTK supplémentaires...")
        nltk.download('punkt_tab')
        print("Ressources NLTK supplémentaires téléchargées avec succès")

def index_documents(docs_dir, store_path, chunk_strategy="paragraph", chunk_size=1000, chunk_overlap=200):
    """
    Indexe tous les documents d'un répertoire dans une base vectorielle.
    
    Args:
        docs_dir: Répertoire contenant les documents à indexer
        store_path: Chemin où stocker la base vectorielle
        chunk_strategy: Stratégie de chunking
        chunk_size: Taille des chunks
        chunk_overlap: Chevauchement entre chunks
    """
    print(f"Indexation des documents de {docs_dir} vers {store_path}")
    
    # S'assurer que les ressources NLTK sont disponibles
    ensure_nltk_resources()
    
    # Initialiser les composants
    parser = DocumentParser()
    chunker = DocumentChunker()
    embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")
    
    # Déterminer la dimension des embeddings
    dimension = embedder.embedding_dim
    
    # Créer le vector store
    vector_store = create_vector_store(
        store_type="faiss",
        dimension=dimension,
        store_path=store_path
    )
    
    # Parcourir tous les documents du répertoire
    docs_path = Path(docs_dir)
    for file_path in docs_path.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.txt']:
            try:
                print(f"Traitement de {file_path}")
                
                # 1. Parser le document
                document = parser.parse(file_path)
                
                # 2. Chunker le document
                chunks = chunker.chunk_document(
                    document,
                    strategy=chunk_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # 3. Générer les embeddings
                chunks_with_embeddings = embedder.embed_document_chunks(chunks)
                
                # 4. Ajouter à la base vectorielle
                vectors = [chunk['embedding'] for chunk in chunks_with_embeddings]
                metadata = [{k: v for k, v in chunk.items() if k != 'embedding'} for chunk in chunks_with_embeddings]
                
                # Convertir la liste de vecteurs en tableau NumPy
                vectors_array = np.array(vectors, dtype=np.float32)
                vector_store.add(vectors_array, metadata)
                
                print(f"  => {len(chunks)} chunks ajoutés")
                
            except Exception as e:
                print(f"Erreur lors du traitement de {file_path}: {str(e)}")
    
    # Sauvegarder la base vectorielle
    vector_store.save()
    print(f"Base vectorielle sauvegardée dans {store_path}")
    print(f"Nombre total de vecteurs: {vector_store.get_store_size()}")

def main():
    parser = argparse.ArgumentParser(description="Indexer des documents pour AssistDoc")
    parser.add_argument("--docs-dir", type=str, required=True, 
                      help="Répertoire contenant les documents à indexer")
    parser.add_argument("--store-path", type=str, default="./data/vector_store",
                      help="Chemin où stocker la base vectorielle")
    parser.add_argument("--chunk-strategy", type=str, default="paragraph",
                      choices=["paragraph", "sentence", "fixed", "semantic"],
                      help="Stratégie de chunking")
    parser.add_argument("--chunk-size", type=int, default=1000,
                      help="Taille des chunks")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                      help="Chevauchement entre chunks")
    
    args = parser.parse_args()
    
    index_documents(
        args.docs_dir,
        args.store_path,
        args.chunk_strategy,
        args.chunk_size,
        args.chunk_overlap
    )

if __name__ == "__main__":
    main()