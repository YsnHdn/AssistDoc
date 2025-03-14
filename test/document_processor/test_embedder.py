"""
Script de test pour le module embedder.py.
Vérifie que l'embedder fonctionne correctement avec différents types de textes.
"""

import sys
import os
import numpy as np
from pathlib import Path
import time

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules du projet
sys.path.append(str(Path(__file__).parent.parent))

from src.document_processor.embedder import DocumentEmbedder

def test_embedder():
    """Test des fonctionnalités de base de l'embedder."""
    
    print("\n=== Test du module DocumentEmbedder ===\n")
    
    # Création de l'instance avec le modèle par défaut (plus léger et rapide)
    print("Initialisation du modèle d'embedding...")
    embedder = DocumentEmbedder()
    
    # Affichage des informations du modèle
    model_info = embedder.get_model_info()
    print(f"\nInformations sur le modèle:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Exemples de textes pour tester
    texts = [
        "Ceci est un exemple de texte pour tester la génération d'embeddings.",
        "Un deuxième exemple avec un contenu différent.",
        "Les embeddings similaires devraient être proches dans l'espace vectoriel.",
        "Ceci est très similaire au premier exemple pour tester les embeddings."
    ]
    
    # Test de la génération d'embeddings
    print("\nGénération d'embeddings pour des textes de test...")
    start_time = time.time()
    embeddings = embedder.embed_texts(texts)
    elapsed_time = time.time() - start_time
    
    print(f"Embeddings générés en {elapsed_time:.2f} secondes")
    print(f"Dimensions des embeddings: {embeddings.shape}")
    
    # Vérification de la similitude (en calculant les distances cosinus)
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    print("\nTest de similarité cosinus entre les textes:")
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"Similarité entre texte {i+1} et texte {j+1}: {sim:.4f}")
    
    # Test avec des chunks de document
    print("\nTest avec des chunks de document...")
    chunks = [
        {"id": 1, "text": texts[0], "page": 1, "source": "test.pdf"},
        {"id": 2, "text": texts[1], "page": 1, "source": "test.pdf"},
        {"id": 3, "text": texts[2], "page": 2, "source": "test.pdf"},
        {"id": 4, "text": texts[3], "page": 2, "source": "test.pdf"}
    ]
    
    chunks_with_embeddings = embedder.embed_document_chunks(chunks)
    
    # Vérification que les embeddings sont bien ajoutés
    print(f"\nVérification des embeddings dans les chunks:")
    print(f"  Type de l'embedding du premier chunk: {type(chunks_with_embeddings[0].get('embedding'))}")
    print(f"  Dimension de l'embedding du premier chunk: {chunks_with_embeddings[0].get('embedding').shape}")
    
    print("\n=== Test terminé avec succès ===")

if __name__ == "__main__":
    test_embedder()