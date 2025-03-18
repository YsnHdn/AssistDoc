"""
Module pour récupérer les chunks de documents les plus pertinents 
en fonction d'une requête textuelle.
Mise à jour pour supporter l'isolation des données par utilisateur
tout en gardant la compatibilité avec le code existant.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from pathlib import Path

from ..document_processor.embedder import DocumentEmbedder
from .store import VectorStore, create_vector_store

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentRetriever:
    """
    Classe pour récupérer les chunks de documents les plus pertinents
    en fonction d'une requête textuelle.
    """
    
    def __init__(self, 
                 vector_store: VectorStore,
                 embedder: DocumentEmbedder,
                 top_k: int = 5,
                 min_score: float = 0.0):
        """
        Initialise le récupérateur de documents.
        
        Args:
            vector_store: Instance de VectorStore contenant les embeddings
            embedder: Instance de DocumentEmbedder pour encoder les requêtes
            top_k: Nombre maximum de résultats à retourner (par défaut: 5)
            min_score: Score minimum de similarité pour inclure un résultat (par défaut: 0.0)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score
        
        logger.info(f"DocumentRetriever initialisé avec top_k={top_k}, min_score={min_score}")
    
    def retrieve(self, 
                query: str, 
                top_k: Optional[int] = None,
                min_score: Optional[float] = None,
                filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Récupère les chunks de documents les plus pertinents pour la requête.
        
        Args:
            query: Requête textuelle
            top_k: Nombre maximum de résultats à retourner (remplace la valeur par défaut)
            min_score: Score minimum de similarité (remplace la valeur par défaut)
            filter_metadata: Filtrer les résultats selon certains critères de métadonnées
            
        Returns:
            Liste de dictionnaires contenant les chunks et leurs scores
        """
        if not query.strip():
            logger.warning("Requête vide fournie à retrieve()")
            return []
        
        # Utiliser les valeurs spécifiées ou les valeurs par défaut
        top_k = top_k if top_k is not None else self.top_k
        min_score = min_score if min_score is not None else self.min_score
        
        # Encoder la requête
        logger.info(f"Encodage de la requête: '{query}'")
        query_embedding = self.embedder.embed_texts([query])[0]
        
        # Rechercher dans la base vectorielle
        logger.info(f"Recherche des {top_k} documents les plus pertinents")
        ids, scores, metadatas = self.vector_store.search(query_embedding, k=top_k)
        
        # Filtrer selon le score minimum et les critères de métadonnées
        results = []
        for i, (doc_id, score, metadata) in enumerate(zip(ids, scores, metadatas)):
            # Vérifier le score minimum
            if score < min_score:
                continue
                
            # Vérifier les critères de filtrage des métadonnées
            if filter_metadata:
                match = True
                for key, value in filter_metadata.items():
                    if key not in metadata or metadata[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Ajouter le résultat
            result = {
                "id": doc_id,
                "score": score,
                "metadata": metadata,
                "text": metadata.get("text", ""),  # Le texte est généralement stocké dans les métadonnées
                "rank": i + 1
            }
            results.append(result)
        
        # Retrier les résultats selon les scores ajustés
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limiter au nombre demandé
        if top_k:
            results = results[:top_k]
        
        # Mettre à jour les rangs
        for i, result in enumerate(results):
            result["rank"] = i + 1
        
        logger.info(f"Récupéré {len(results)} documents pertinents")
        return results
    
    def retrieve_and_rerank(self,
                          query: str,
                          top_k_first_pass: int = 20,
                          top_k_final: Optional[int] = None,
                          reranking_function: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Récupère les documents en deux passes: d'abord une recherche large,
        puis un reclassement avec une fonction plus sophistiquée.
        
        Args:
            query: Requête textuelle
            top_k_first_pass: Nombre de documents à récupérer lors de la première passe
            top_k_final: Nombre final de documents à retourner
            reranking_function: Fonction de reclassement qui prend (query, results) et retourne results reclassés
            
        Returns:
            Liste de dictionnaires contenant les chunks reclassés
        """
        # Valeur finale de top_k
        top_k_final = top_k_final if top_k_final is not None else self.top_k
        
        # Première passe: récupérer un ensemble plus large de documents
        initial_results = self.retrieve(query, top_k=top_k_first_pass, min_score=0.0)
        
        if not initial_results:
            return []
        
        # Si aucune fonction de reclassement n'est fournie, simplement limiter les résultats
        if reranking_function is None:
            return initial_results[:top_k_final]
        
        # Deuxième passe: reclasser les résultats
        logger.info(f"Reclassement de {len(initial_results)} résultats")
        reranked_results = reranking_function(query, initial_results)
        
        # Limiter au nombre final demandé
        reranked_results = reranked_results[:top_k_final]
        
        # Mettre à jour les rangs
        for i, result in enumerate(reranked_results):
            result["rank"] = i + 1
        
        logger.info(f"Retourné {len(reranked_results)} documents après reclassement")
        return reranked_results


class UserAwareDocumentRetriever:
    """
    Classe pour récupérer les chunks de documents les plus pertinents
    en fonction d'une requête textuelle, avec isolation par utilisateur.
    """
    
    def __init__(self, 
                 vector_store: VectorStore,
                 embedder: DocumentEmbedder,
                 user_id: str,
                 top_k: int = 5,
                 min_score: float = 0.0):
        """
        Initialise le récupérateur de documents avec isolation par utilisateur.
        
        Args:
            vector_store: Instance de VectorStore contenant les embeddings
            embedder: Instance de DocumentEmbedder pour encoder les requêtes
            user_id: Identifiant unique de l'utilisateur pour filtrer les résultats
            top_k: Nombre maximum de résultats à retourner (par défaut: 5)
            min_score: Score minimum de similarité pour inclure un résultat (par défaut: 0.0)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.user_id = user_id
        self.top_k = top_k
        self.min_score = min_score
        
        logger.info(f"UserAwareDocumentRetriever initialisé pour l'utilisateur {user_id[:8]}... avec top_k={top_k}, min_score={min_score}")
    
    def retrieve(self, 
                query: str, 
                top_k: Optional[int] = None,
                min_score: Optional[float] = None,
                filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Récupère les chunks de documents les plus pertinents pour la requête,
        en filtrant automatiquement par utilisateur.
        
        Args:
            query: Requête textuelle
            top_k: Nombre maximum de résultats à retourner (remplace la valeur par défaut)
            min_score: Score minimum de similarité (remplace la valeur par défaut)
            filter_metadata: Filtrer les résultats selon certains critères de métadonnées
            
        Returns:
            Liste de dictionnaires contenant les chunks et leurs scores
        """
        if not query.strip():
            logger.warning("Requête vide fournie à retrieve()")
            return []
        
        # Utiliser les valeurs spécifiées ou les valeurs par défaut
        top_k = top_k if top_k is not None else self.top_k
        min_score = min_score if min_score is not None else self.min_score
        
        # S'assurer que le filtre inclut l'ID utilisateur
        if filter_metadata is None:
            filter_metadata = {}
        
        # Ajouter le filtre utilisateur
        user_filter = {"user_id": self.user_id}
        
        # Combiner avec les filtres existants, en donnant priorité aux filtres existants
        # mais en s'assurant que l'utilisateur ne peut pas contourner le filtre utilisateur
        combined_filter = {**user_filter, **filter_metadata}
        
        # Encoder la requête
        logger.info(f"Encodage de la requête: '{query}'")
        query_embedding = self.embedder.embed_texts([query])[0]
        
        # Rechercher dans la base vectorielle
        logger.info(f"Recherche des {top_k} documents les plus pertinents pour l'utilisateur {self.user_id[:8]}...")
        ids, scores, metadatas = self.vector_store.search(query_embedding, k=top_k*2)  # Récupérer plus de résultats pour le filtrage
        
        # Filtrer selon le score minimum et les critères de métadonnées
        results = []
        for i, (doc_id, score, metadata) in enumerate(zip(ids, scores, metadatas)):
            # Vérifier le score minimum
            if score < min_score:
                continue
                
            # Vérifier les critères de filtrage des métadonnées, y compris l'utilisateur
            match = True
            for key, value in combined_filter.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break
            if not match:
                continue
            
            # Ajouter le résultat
            result = {
                "id": doc_id,
                "score": score,
                "metadata": metadata,
                "text": metadata.get("text", ""),  # Le texte est généralement stocké dans les métadonnées
                "rank": i + 1
            }
            results.append(result)
        
        # Retrier les résultats selon les scores
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limiter au nombre demandé
        if top_k:
            results = results[:top_k]
        
        # Mettre à jour les rangs
        for i, result in enumerate(results):
            result["rank"] = i + 1
        
        logger.info(f"Récupéré {len(results)} documents pertinents pour l'utilisateur {self.user_id[:8]}...")
        return results
    
    # Les autres méthodes comme retrieve_and_rerank, retrieve_multi_query, etc. 
    # peuvent être adaptées de la même manière en ajoutant le filtre utilisateur


def create_default_retriever(
    store_path: str,
    embedder_model: str = "all-MiniLM-L6-v2",
    store_type: str = "faiss",
    top_k: int = 5
) -> DocumentRetriever:
    """
    Fonction utilitaire pour créer un DocumentRetriever avec des paramètres par défaut.
    
    Args:
        store_path: Chemin vers l'index vectoriel
        embedder_model: Nom du modèle d'embedding à utiliser
        store_type: Type de base de données vectorielle ("faiss" ou "chroma")
        top_k: Nombre par défaut de résultats à retourner
        
    Returns:
        Instance de DocumentRetriever configurée
    """
    from ..document_processor.embedder import DocumentEmbedder
    from .store import create_vector_store
    
    # Créer l'embedder
    embedder = DocumentEmbedder(model_name=embedder_model)
    dimension = embedder.embedding_dim
    
    # Créer le vector store
    vector_store = create_vector_store(
        store_type=store_type,
        dimension=dimension,
        store_path=store_path
    )
    
    # Créer et retourner le retriever
    return DocumentRetriever(
        vector_store=vector_store,
        embedder=embedder,
        top_k=top_k
    )


def create_user_aware_retriever(
    user_id: str,
    store_path: str,
    embedder_model: str = "all-MiniLM-L6-v2",
    store_type: str = "faiss",
    top_k: int = 5
) -> UserAwareDocumentRetriever:
    """
    Fonction utilitaire pour créer un DocumentRetriever spécifique à un utilisateur.
    
    Args:
        user_id: Identifiant unique de l'utilisateur
        store_path: Chemin vers l'index vectoriel
        embedder_model: Nom du modèle d'embedding à utiliser
        store_type: Type de base de données vectorielle ("faiss" ou "chroma")
        top_k: Nombre par défaut de résultats à retourner
        
    Returns:
        Instance de UserAwareDocumentRetriever configurée
    """
    from ..document_processor.embedder import DocumentEmbedder
    from .store import create_vector_store
    
    # Créer l'embedder
    embedder = DocumentEmbedder(model_name=embedder_model)
    dimension = embedder.embedding_dim
    
    # Créer le vector store
    vector_store = create_vector_store(
        store_type=store_type,
        dimension=dimension,
        store_path=store_path
    )
    
    # Créer et retourner le retriever spécifique à l'utilisateur
    return UserAwareDocumentRetriever(
        vector_store=vector_store,
        embedder=embedder,
        user_id=user_id,
        top_k=top_k
    )