"""
Script de test pour le module vector_db.
Ce script teste les fonctionnalités de stockage vectoriel et de récupération.
"""

import sys
import os
from pathlib import Path
import argparse
import time
import numpy as np

# Ajouter le répertoire racine du projet au chemin d'importation
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # Remontez jusqu'à la racine du projet
sys.path.append(str(project_root))

# Imports
from src.document_processor.parser import DocumentParser
from src.document_processor.chunker import DocumentChunker
from src.document_processor.embedder import DocumentEmbedder
from src.vector_db.store import create_vector_store
from src.vector_db.retriever import DocumentRetriever

def test_vector_db(file_path, store_type="faiss", store_path=None, chunk_strategy="paragraph"):
    """
    Teste le pipeline complet incluant le traitement de document et la recherche vectorielle.
    
    Args:
        file_path: Chemin vers le fichier à traiter
        store_type: Type de base de données vectorielle ("faiss" ou "chroma")
        store_path: Chemin pour stocker l'index (optionnel)
        chunk_strategy: Stratégie de chunking à utiliser
    """
    print(f"\n{'='*80}\n")
    print(f"TEST DU MODULE VECTOR_DB\n")
    print(f"Fichier: {file_path}")
    print(f"Type de store: {store_type}")
    print(f"Stockage: {'Persistant' if store_path else 'En mémoire'}")
    print(f"Stratégie de chunking: {chunk_strategy}")
    print(f"\n{'='*80}\n")
    
    # 1. Traitement du document
    print("1. TRAITEMENT DU DOCUMENT")
    print("-" * 40)
    
    # Initialiser les composants
    parser = DocumentParser()
    chunker = DocumentChunker()
    embedder = DocumentEmbedder()
    
    # Parser le document
    start_time = time.time()
    document = parser.parse(file_path)
    print(f"Document parsé: {document.get('file_name')}")
    
    # Chunking
    chunks = chunker.chunk_document(document, strategy=chunk_strategy)
    print(f"Document découpé en {len(chunks)} chunks")
    
    # Générer des embeddings
    chunks_with_embeddings = embedder.embed_document_chunks(chunks)
    processing_time = time.time() - start_time
    print(f"Traitement terminé en {processing_time:.2f} secondes")
    
    # 2. Stockage des embeddings
    print("\n2. STOCKAGE DES EMBEDDINGS")
    print("-" * 40)
    
    # Créer le vector store
    dimension = embedder.embedding_dim
    if store_path:
        os.makedirs(store_path, exist_ok=True)
    
    vector_store = create_vector_store(
        store_type=store_type,
        dimension=dimension,
        store_path=store_path
    )
    
    # Extraire les vecteurs et métadonnées
    vectors = np.array([chunk.get("embedding") for chunk in chunks_with_embeddings])
    metadata = []
    for chunk in chunks_with_embeddings:
        # Créer une copie du chunk sans l'embedding (trop volumineux pour les métadonnées)
        meta = {k: v for k, v in chunk.items() if k != "embedding"}
        metadata.append(meta)
    
    # Ajouter au store
    start_time = time.time()
    ids = vector_store.add(vectors, metadata)
    storage_time = time.time() - start_time
    
    print(f"Embeddings stockés dans {store_type}")
    print(f"Nombre de vecteurs: {vector_store.get_store_size()}")
    print(f"Stockage terminé en {storage_time:.2f} secondes")
    
    # Sauvegarder si un chemin est spécifié
    if store_path:
        vector_store.save()
        print(f"Index sauvegardé à {store_path}")
    
    # 3. Test de récupération
    print("\n3. TEST DE RÉCUPÉRATION")
    print("-" * 40)
    
    # Créer le retriever
    retriever = DocumentRetriever(
        vector_store=vector_store,
        embedder=embedder,
        top_k=3
    )
    
    # Définir quelques requêtes de test
    if len(chunks) > 0:
        # Utiliser des mots du premier chunk comme requête simple
        words = chunks[0].get("text", "").split()
        query1 = " ".join(words[:5]) if len(words) > 5 else chunks[0].get("text", "")[:50]
        
        # Requête plus complexe
        query2 = f"Que dit le document à propos de {words[0] if words else 'ce sujet'}?"
    else:
        query1 = "Information"
        query2 = "Que contient ce document?"
    
    # Test 1 : Requête simple
    print(f"\nRequête 1: '{query1}'")
    start_time = time.time()
    results1 = retriever.retrieve(query1)
    query_time = time.time() - start_time
    
    print(f"Récupération terminée en {query_time*1000:.2f} ms")
    print(f"Nombre de résultats: {len(results1)}")
    
    if results1:
        print("\nTop résultat:")
        top_result = results1[0]
        print(f"Score: {top_result.get('score'):.4f}")
        print(f"Texte: {top_result.get('text', '')[:150]}...")
    
    # Test 2 : Requête plus complexe
    print(f"\nRequête 2: '{query2}'")
    start_time = time.time()
    results2 = retriever.retrieve(query2)
    query_time = time.time() - start_time
    
    print(f"Récupération terminée en {query_time*1000:.2f} ms")
    print(f"Nombre de résultats: {len(results2)}")
    
    if results2:
        print("\nTop résultat:")
        top_result = results2[0]
        print(f"Score: {top_result.get('score'):.4f}")
        print(f"Texte: {top_result.get('text', '')[:150]}...")
    
    # Test 3 : Recherche avec mots-clés
    if len(words) > 3:
        keywords = [words[0], words[2], words[3]]
        print(f"\nRequête avec mots-clés: '{query1}' + {keywords}")
        start_time = time.time()
        results3 = retriever.retrieve_with_keywords(query1, keywords)
        query_time = time.time() - start_time
        
        print(f"Récupération terminée en {query_time*1000:.2f} ms")
        print(f"Nombre de résultats: {len(results3)}")
        
        if results3:
            print("\nTop résultat:")
            top_result = results3[0]
            print(f"Score: {top_result.get('score'):.4f}")
            print(f"Correspondances de mots-clés: {top_result.get('keyword_matches', 0)}")
            print(f"Texte: {top_result.get('text', '')[:150]}...")
    
    # Résumé
    print(f"\n{'='*80}\n")
    print("RÉSUMÉ DU TEST VECTOR_DB")
    print(f"Type de store: {store_type}")
    print(f"Dimension des embeddings: {dimension}")
    print(f"Nombre de chunks: {len(chunks)}")
    print(f"Vecteurs stockés: {vector_store.get_store_size()}")
    
    if store_path:
        print(f"Index sauvegardé à: {store_path}")
    print(f"\n{'='*80}\n")
    
    print("Test du module vector_db terminé avec succès!")
    
    return vector_store, retriever

if __name__ == "__main__":
    # Configurer les arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Test du module vector_db")
    parser.add_argument("file_path", type=str, help="Chemin vers le fichier à traiter")
    parser.add_argument("--store-type", type=str, default="faiss", choices=["faiss", "chroma"],
                        help="Type de base de données vectorielle à utiliser")
    parser.add_argument("--store-path", type=str, default=None,
                        help="Chemin pour stocker l'index (optionnel)")
    parser.add_argument("--strategy", type=str, default="paragraph", 
                        choices=["paragraph", "sentence", "fixed", "semantic"],
                        help="Stratégie de chunking à utiliser")
    
    args = parser.parse_args()
    
    test_vector_db(
        args.file_path,
        store_type=args.store_type,
        store_path=args.store_path,
        chunk_strategy=args.strategy
    )