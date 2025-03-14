"""
Script de test pour les modules de traitement de documents:
parser, chunker et embedder.
Ce script effectue un test complet du pipeline de traitement des documents.
"""

import sys
import os
from pathlib import Path
import argparse
import time

# Ajouter le répertoire racine du projet au chemin d'importation
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # Remontez jusqu'à la racine du projet
sys.path.append(str(project_root))

# Maintenant les imports devraient fonctionner
from src.document_processor.parser import DocumentParser
from src.document_processor.chunker import DocumentChunker
from src.document_processor.embedder import DocumentEmbedder

def test_document_processing(file_path, chunk_strategy="paragraph", chunk_size=1000, chunk_overlap=200):
    """
    Teste le pipeline complet de traitement de documents.
    
    Args:
        file_path: Chemin vers le fichier à traiter
        chunk_strategy: Stratégie de chunking à utiliser
        chunk_size: Taille des chunks (pour les stratégies appropriées)
        chunk_overlap: Chevauchement entre les chunks
    """
    print(f"\n{'='*80}\n")
    print(f"TEST DU PIPELINE DE TRAITEMENT DE DOCUMENTS\n")
    print(f"Fichier: {file_path}")
    print(f"Stratégie de chunking: {chunk_strategy}")
    print(f"Taille de chunk: {chunk_size}, Chevauchement: {chunk_overlap}")
    print(f"\n{'='*80}\n")
    
    # 1. Parsing du document
    print("1. PARSING DU DOCUMENT")
    print("-" * 40)
    
    parser = DocumentParser()
    start_time = time.time()
    
    try:
        document = parser.parse(file_path)
        parsing_time = time.time() - start_time
        print(f"Parsing réussi en {parsing_time:.2f} secondes")
        print(f"Type de document: {document.get('file_type')}")
        
        if 'num_pages' in document:
            print(f"Nombre de pages: {document.get('num_pages')}")
        elif 'paragraphs' in document:
            print(f"Nombre de paragraphes: {len(document.get('paragraphs', []))}")
        elif 'lines' in document:
            print(f"Nombre de lignes: {len(document.get('lines', []))}")
        
        print(f"Taille du texte: {len(document.get('full_text', ''))} caractères")
        
        # Afficher un extrait du texte
        text_preview = document.get('full_text', '')[:500]
        print(f"\nExtrait du texte:\n{text_preview}...\n")
        
    except Exception as e:
        print(f"Erreur lors du parsing: {str(e)}")
        return
    
    # 2. Chunking du document
    print("\n2. CHUNKING DU DOCUMENT")
    print("-" * 40)
    
    chunker = DocumentChunker()
    start_time = time.time()
    
    try:
        chunks = chunker.chunk_document(
            document,
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunking_time = time.time() - start_time
        print(f"Chunking réussi en {chunking_time:.2f} secondes")
        print(f"Nombre de chunks créés: {len(chunks)}")
        
        # Afficher des statistiques sur les chunks
        chunk_sizes = [len(chunk.get('text', '')) for chunk in chunks]
        if chunk_sizes:
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)
            print(f"Taille moyenne des chunks: {avg_size:.1f} caractères")
            print(f"Taille min/max des chunks: {min_size}/{max_size} caractères")
        
        # Afficher un exemple de chunk
        if chunks:
            print(f"\nExemple de chunk (1/{len(chunks)}):")
            example_chunk = chunks[0]
            chunk_text = example_chunk.get('text', '')[:300]
            print(f"ID: {example_chunk.get('chunk_id')}")
            print(f"Index: {example_chunk.get('chunk_index')}")
            if 'page' in example_chunk:
                print(f"Page: {example_chunk.get('page')}")
            print(f"Texte: {chunk_text}...\n")
        
    except Exception as e:
        print(f"Erreur lors du chunking: {str(e)}")
        return
    
    # 3. Génération d'embeddings
    print("\n3. GÉNÉRATION D'EMBEDDINGS")
    print("-" * 40)
    
    embedder = DocumentEmbedder()
    start_time = time.time()
    
    try:
        # Afficher les informations sur le modèle d'embedding
        model_info = embedder.get_model_info()
        print(f"Modèle d'embedding: {model_info.get('model_name')}")
        print(f"Dimension des embeddings: {model_info.get('embedding_dimension')}")
        print(f"Dispositif utilisé: {model_info.get('device')}")
        
        # Générer les embeddings pour les chunks
        chunks_with_embeddings = embedder.embed_document_chunks(chunks)
        embedding_time = time.time() - start_time
        print(f"\nGénération d'embeddings réussie en {embedding_time:.2f} secondes")
        
        # Vérifier que les embeddings ont été ajoutés correctement
        if chunks_with_embeddings:
            first_embedding = chunks_with_embeddings[0].get('embedding')
            if first_embedding is not None:
                print(f"Dimension de l'embedding: {len(first_embedding)}")
                
                # Afficher quelques valeurs de l'embedding (juste un aperçu)
                print(f"Aperçu de l'embedding: {first_embedding[:5]}...")
            else:
                print("Avertissement: Pas d'embedding trouvé dans le premier chunk")
        
        # Calculer la similarité entre les deux premiers chunks si disponibles
        if len(chunks_with_embeddings) >= 2:
            import numpy as np
            
            def cosine_similarity(v1, v2):
                return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
            emb1 = chunks_with_embeddings[0].get('embedding')
            emb2 = chunks_with_embeddings[1].get('embedding')
            
            if emb1 is not None and emb2 is not None:
                similarity = cosine_similarity(emb1, emb2)
                print(f"\nSimilarité cosinus entre les deux premiers chunks: {similarity:.4f}")
                
    except Exception as e:
        print(f"Erreur lors de la génération d'embeddings: {str(e)}")
        return
    
    # Résumé du pipeline complet
    print(f"\n{'='*80}\n")
    print("RÉSUMÉ DU PIPELINE DE TRAITEMENT")
    print(f"Fichier: {Path(file_path).name}")
    print(f"Temps total de traitement: {parsing_time + chunking_time + embedding_time:.2f} secondes")
    print(f"  - Parsing: {parsing_time:.2f} secondes")
    print(f"  - Chunking: {chunking_time:.2f} secondes")
    print(f"  - Embedding: {embedding_time:.2f} secondes")
    print(f"Nombre de chunks: {len(chunks)}")
    print(f"Dimension des embeddings: {model_info.get('embedding_dimension')}")
    print(f"\n{'='*80}\n")
    
    print("Test du pipeline de traitement terminé avec succès!")

if __name__ == "__main__":
    # Configurer les arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Test du pipeline de traitement de documents")
    parser.add_argument("file_path", type=str, help="Chemin vers le fichier à traiter")
    parser.add_argument("--strategy", type=str, default="paragraph", 
                        choices=["paragraph", "sentence", "fixed", "semantic"],
                        help="Stratégie de chunking à utiliser")
    parser.add_argument("--chunk-size", type=int, default=1000, 
                        help="Taille maximale des chunks (en caractères)")
    parser.add_argument("--chunk-overlap", type=int, default=200, 
                        help="Chevauchement entre les chunks adjacents (en caractères)")
    
    args = parser.parse_args()
    
    test_document_processing(
        args.file_path,
        chunk_strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )