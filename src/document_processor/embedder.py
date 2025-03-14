"""
Module pour la génération d'embeddings à partir de chunks de texte.
Utilise les modèles de Sentence Transformers pour créer des représentations 
vectorielles des textes.
"""

import os
import numpy as np
from typing import List, Union, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """
    Classe pour générer des embeddings à partir de chunks de texte en utilisant
    des modèles de Sentence Transformers.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2", 
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialise l'embedder avec un modèle spécifique.
        
        Args:
            model_name: Nom du modèle Sentence Transformers à utiliser
                        (default: "all-MiniLM-L6-v2")
            device: Dispositif sur lequel exécuter le modèle ('cpu', 'cuda', etc.)
            cache_dir: Répertoire pour stocker les modèles téléchargés
        """
        try:
            logger.info(f"Initialisation de l'embedder avec le modèle {model_name}")
            self.model_name = model_name
            
            # Déterminer le device automatiquement si non spécifié
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Charger le modèle
            self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Modèle chargé avec succès. Dimension des embeddings: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'embedder: {str(e)}")
            raise

    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = True) -> np.ndarray:
        """
        Génère des embeddings pour une liste de textes.
        
        Args:
            texts: Liste de textes à transformer en embeddings
            batch_size: Taille du batch pour le traitement
            show_progress_bar: Afficher la barre de progression pendant le traitement
            
        Returns:
            Tableau numpy contenant les embeddings (shape: [n_texts, embedding_dim])
        """
        if not texts:
            logger.warning("Liste de textes vide fournie à embed_texts")
            return np.array([])
        
        try:
            logger.info(f"Génération d'embeddings pour {len(texts)} textes")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )
            logger.info(f"Embeddings générés avec succès. Shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des embeddings: {str(e)}")
            raise

    def embed_document_chunks(self, 
                             chunks: List[Dict[str, Any]], 
                             text_key: str = "text",
                             batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Génère des embeddings pour une liste de chunks de document et les ajoute
        aux dictionnaires de chunks.
        
        Args:
            chunks: Liste de dictionnaires représentant les chunks de document
            text_key: Clé dans le dictionnaire qui contient le texte à encoder
            batch_size: Taille du batch pour le traitement
            
        Returns:
            Liste des chunks avec les embeddings ajoutés sous la clé 'embedding'
        """
        if not chunks:
            logger.warning("Liste de chunks vide fournie à embed_document_chunks")
            return []
        
        try:
            # Extraire les textes des chunks
            texts = [chunk.get(text_key, "") for chunk in chunks]
            
            # Vérifier les textes vides
            for i, text in enumerate(texts):
                if not text:
                    logger.warning(f"Texte vide détecté au chunk {i}")
            
            # Générer les embeddings
            embeddings = self.embed_texts(texts, batch_size=batch_size, show_progress_bar=True)
            
            # Ajouter les embeddings aux chunks
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk["embedding"] = embeddings[i]
            
            logger.info(f"Embeddings ajoutés à {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout d'embeddings aux chunks: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Renvoie des informations sur le modèle d'embedding utilisé.
        
        Returns:
            Dictionnaire contenant les informations du modèle
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "model_max_sequence_length": self.model.get_max_seq_length(),
            "device": self.model.device
        }