"""
Module pour stocker et indexer des embeddings dans une base de données vectorielle.
Supporte FAISS et ChromaDB comme backends de stockage.
"""

import os
import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

# Imports conditionnels pour les backends
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# À la place de l'import actuel de ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    # Capture à la fois les ImportError et RuntimeError (SQLite)
    CHROMA_AVAILABLE = False
    import logging
    logging.warning(f"ChromaDB n'est pas disponible: {str(e)}")

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Classe abstraite pour stocker et rechercher des embeddings vectoriels.
    """
    
    def __init__(self, dimension: int, store_path: Optional[str] = None):
        """
        Initialise la base de données vectorielle.
        
        Args:
            dimension: Dimension des vecteurs d'embedding
            store_path: Chemin pour la persistance de l'index (optionnel)
        """
        self.dimension = dimension
        self.store_path = Path(store_path) if store_path else None
        
    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> List[str]:
        """
        Ajoute des vecteurs d'embedding à la base de données avec leurs métadonnées associées.
        
        Args:
            vectors: Tableau NumPy de vecteurs d'embedding (shape: [n_vectors, dimension])
            metadata: Liste de dictionnaires contenant les métadonnées associées à chaque vecteur
            
        Returns:
            Liste des IDs générés pour les vecteurs ajoutés
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        Recherche les k vecteurs les plus proches du vecteur de requête.
        
        Args:
            query_vector: Vecteur de requête (shape: [dimension])
            k: Nombre de résultats à retourner
            
        Returns:
            Tuple contenant:
                - Liste des IDs des vecteurs trouvés
                - Liste des scores de similarité
                - Liste des métadonnées associées
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    def delete(self, ids: List[str]) -> None:
        """
        Supprime des vecteurs de la base de données.
        
        Args:
            ids: Liste des IDs des vecteurs à supprimer
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    def save(self) -> None:
        """
        Persiste l'index sur disque.
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    def load(self) -> None:
        """
        Charge l'index depuis le disque.
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    def get_store_size(self) -> int:
        """
        Retourne le nombre de vecteurs stockés dans la base de données.
        
        Returns:
            Nombre de vecteurs stockés
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")


class FAISSVectorStore(VectorStore):
    """
    Implémentation de VectorStore utilisant FAISS comme backend.
    """
    
    def __init__(self, 
                 dimension: int, 
                 store_path: Optional[str] = None,
                 index_type: str = "Flat",
                 metric: str = "cosine"):
        """
        Initialise une base de données vectorielle FAISS.
        
        Args:
            dimension: Dimension des vecteurs d'embedding
            store_path: Chemin pour la persistance de l'index (optionnel)
            index_type: Type d'index FAISS à utiliser (Flat, HNSW, IVF, etc.)
            metric: Métrique de similarité (cosine, l2, inner_product)
        """
        super().__init__(dimension, store_path)
        
        if not FAISS_AVAILABLE:
            raise ImportError("Le package FAISS n'est pas installé. Installez-le avec 'pip install faiss-cpu' ou 'pip install faiss-gpu'.")
        
        self.index_type = index_type
        self.metric = metric
        self.index = None
        self.id_to_index = {}  # Mapping des IDs aux indices FAISS
        self.index_to_id = {}  # Mapping inverse
        self.metadata = {}     # Stockage des métadonnées
        self.next_id = 0       # Compteur pour générer des IDs uniques
        
        # Initialiser l'index FAISS
        self._create_index()
        
        # Charger l'index existant si un chemin est spécifié et qu'il existe
        if self.store_path and self.store_path.exists():
            self.load()
    
    def _create_index(self) -> None:
        """
        Crée l'index FAISS en fonction du type et de la métrique spécifiés.
        """
        # Configurer la métrique
        if self.metric == "cosine":
            # Pour la similarité cosinus, nous normalisons les vecteurs
            self.normalize_vectors = True
            metric_param = faiss.METRIC_INNER_PRODUCT
        elif self.metric == "l2":
            self.normalize_vectors = False
            metric_param = faiss.METRIC_L2
        elif self.metric == "inner_product":
            self.normalize_vectors = False
            metric_param = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Métrique non reconnue: {self.metric}")
        
        # Créer l'index en fonction du type
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.dimension) if metric_param == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "HNSW":
            # HNSW est plus rapide pour les grandes bases de données mais plus lent à construire
            m = 16  # Nombre de connexions par noeud
            self.index = faiss.IndexHNSWFlat(self.dimension, m, metric_param)
        elif self.index_type.startswith("IVF"):
            # IVF est bon pour les bases de données de taille moyenne
            try:
                nlist = int(self.index_type.split("IVF")[1])
            except:
                nlist = 100  # Valeur par défaut
            
            # On a besoin d'une base de données pour créer les centroids
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, metric_param)
            # Necessaire d'appeler cette méthode avant d'ajouter des vecteurs
            self.index.train(np.zeros((1, self.dimension), dtype=np.float32))
        else:
            raise ValueError(f"Type d'index non reconnu: {self.index_type}")
        
        logger.info(f"Index FAISS créé: {self.index_type} avec métrique {self.metric}")
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalise les vecteurs pour la similarité cosinus.
        
        Args:
            vectors: Vecteurs à normaliser
            
        Returns:
            Vecteurs normalisés
        """
        if not self.normalize_vectors:
            return vectors
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Éviter la division par zéro
        norms[norms == 0] = 1.0
        return vectors / norms
    
    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> List[str]:
        """
        Ajoute des vecteurs à l'index FAISS.
        
        Args:
            vectors: Tableau NumPy de vecteurs (shape: [n_vectors, dimension])
            metadata: Liste de dictionnaires contenant les métadonnées
            
        Returns:
            Liste des IDs générés
        """
        if len(vectors) != len(metadata):
            raise ValueError("Le nombre de vecteurs doit correspondre au nombre d'objets metadata")
        
        if len(vectors) == 0:
            return []
        
        # Normaliser les vecteurs si nécessaire (pour cosine)
        normalized_vectors = self._normalize(vectors.astype(np.float32))
        
        # Générer des IDs uniques
        ids = [f"vec_{self.next_id + i}" for i in range(len(vectors))]
        
        # Mettre à jour les mappings
        for i, vec_id in enumerate(ids):
            idx = self.next_id + i
            self.id_to_index[vec_id] = idx
            self.index_to_id[idx] = vec_id
            self.metadata[vec_id] = metadata[i]
        
        # Ajouter les vecteurs à l'index
        self.index.add(normalized_vectors)
        
        # Mettre à jour le compteur
        self.next_id += len(vectors)
        
        logger.info(f"Ajouté {len(vectors)} vecteurs à l'index FAISS")
        return ids
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        Recherche les k vecteurs les plus proches du vecteur de requête.
        
        Args:
            query_vector: Vecteur de requête (shape: [dimension])
            k: Nombre de résultats à retourner
            
        Returns:
            Tuple contenant:
                - Liste des IDs des vecteurs trouvés
                - Liste des scores de similarité
                - Liste des métadonnées associées
        """
        if self.get_store_size() == 0:
            logger.warning("L'index est vide, impossible d'effectuer une recherche")
            return [], [], []
        
        # Normaliser le vecteur de requête si nécessaire
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        normalized_query = self._normalize(query_vector)
        
        # Ajuster k si nécessaire
        k = min(k, self.get_store_size())
        
        # Effectuer la recherche
        scores, indices = self.index.search(normalized_query, k)
        
        # Convertir les indices en IDs et récupérer les métadonnées
        scores = scores[0].tolist()  # Convertir en liste simple
        indices = indices[0].tolist()
        
        ids = [self.index_to_id.get(idx) for idx in indices if idx in self.index_to_id]
        meta = [self.metadata.get(vec_id, {}) for vec_id in ids]
        
        return ids, scores, meta
    
    def delete(self, ids: List[str]) -> None:
        """
        Supprime des vecteurs de l'index.
        
        Args:
            ids: Liste des IDs des vecteurs à supprimer
        """
        # FAISS ne supporte pas la suppression directe pour tous les types d'index
        # Une approche est de recréer l'index sans les vecteurs supprimés
        indices_to_remove = [self.id_to_index[vec_id] for vec_id in ids if vec_id in self.id_to_index]
        
        if not indices_to_remove:
            return
        
        # Pour les index qui supportent la suppression
        if hasattr(self.index, 'remove_ids'):
            # Convertir les indices en un tableau d'IDs FAISS
            faiss_ids = np.array(indices_to_remove, dtype=np.int64)
            self.index.remove_ids(faiss_ids)
            
            # Mettre à jour les mappings
            for vec_id in ids:
                if vec_id in self.id_to_index:
                    idx = self.id_to_index[vec_id]
                    del self.index_to_id[idx]
                    del self.id_to_index[vec_id]
                    del self.metadata[vec_id]
            
            logger.info(f"Supprimé {len(indices_to_remove)} vecteurs de l'index FAISS")
        else:
            logger.warning(f"La suppression n'est pas supportée pour le type d'index {self.index_type}. Recréation de l'index requise.")
            # Ici, implémenter la recréation de l'index si nécessaire
    
    def save(self) -> None:
        """
        Persiste l'index sur disque.
        """
        if not self.store_path:
            logger.warning("Aucun chemin de stockage spécifié, impossible de sauvegarder l'index")
            return
        
        # Créer le répertoire parent si nécessaire
        os.makedirs(self.store_path.parent, exist_ok=True)
        
        # Sauvegarder l'index FAISS
        index_path = self.store_path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        
        # Sauvegarder les métadonnées et mappings
        metadata_path = self.store_path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id,
                'metadata': self.metadata,
                'next_id': self.next_id,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'metric': self.metric
            }, f)
        
        logger.info(f"Index FAISS sauvegardé à {self.store_path}")
    
    def load(self) -> None:
        """
        Charge l'index depuis le disque.
        """
        if not self.store_path:
            logger.warning("Aucun chemin de stockage spécifié, impossible de charger l'index")
            return
        
        index_path = self.store_path / "faiss_index.bin"
        metadata_path = self.store_path / "metadata.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            logger.warning(f"Les fichiers d'index n'existent pas à {self.store_path}")
            return
        
        try:
            # Charger l'index FAISS
            self.index = faiss.read_index(str(index_path))
            
            # Charger les métadonnées et mappings
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.id_to_index = data['id_to_index']
                self.index_to_id = data['index_to_id']
                self.metadata = data['metadata']
                self.next_id = data['next_id']
                self.dimension = data['dimension']
                self.index_type = data['index_type']
                self.metric = data['metric']
                self.normalize_vectors = (self.metric == "cosine")
            
            logger.info(f"Index FAISS chargé depuis {self.store_path}")
            logger.info(f"L'index contient {self.get_store_size()} vecteurs")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'index: {str(e)}")
            # Réinitialiser l'index
            self._create_index()
    
    def get_store_size(self) -> int:
        """
        Retourne le nombre de vecteurs stockés dans l'index.
        
        Returns:
            Nombre de vecteurs stockés
        """
        return self.index.ntotal


class ChromaVectorStore(VectorStore):
    """
    Implémentation de VectorStore utilisant ChromaDB comme backend.
    """
    
    def __init__(self, 
                 dimension: int, 
                 store_path: Optional[str] = None,
                 collection_name: str = "document_embeddings",
                 distance_func: str = "cosine"):
        """
        Initialise une base de données vectorielle ChromaDB.
        
        Args:
            dimension: Dimension des vecteurs d'embedding
            store_path: Chemin pour la persistance de l'index (optionnel)
            collection_name: Nom de la collection ChromaDB
            distance_func: Fonction de distance (cosine, l2, ip)
        """
        super().__init__(dimension, store_path)
        
        if not CHROMA_AVAILABLE:
            raise ImportError("Le package ChromaDB n'est pas installé. Installez-le avec 'pip install chromadb'.")
        
        self.collection_name = collection_name
        self.distance_func = distance_func
        
        # Initialiser ChromaDB
        persist_directory = str(self.store_path) if self.store_path else None
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        
        # Créer ou récupérer la collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=None  # Nous fournissons directement les embeddings
            )
            logger.info(f"Collection ChromaDB existante récupérée: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=None,
                metadata={"dimension": dimension, "distance_func": distance_func}
            )
            logger.info(f"Nouvelle collection ChromaDB créée: {collection_name}")
    
    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> List[str]:
        """
        Ajoute des vecteurs à la collection ChromaDB.
        
        Args:
            vectors: Tableau NumPy de vecteurs (shape: [n_vectors, dimension])
            metadata: Liste de dictionnaires contenant les métadonnées
            
        Returns:
            Liste des IDs générés
        """
        if len(vectors) != len(metadata):
            raise ValueError("Le nombre de vecteurs doit correspondre au nombre d'objets metadata")
        
        if len(vectors) == 0:
            return []
        
        # Générer des IDs uniques
        ids = [f"vec_{uuid.uuid4()}" for _ in range(len(vectors))]
        
        # Convertir les vecteurs en liste pour ChromaDB
        embeddings = vectors.tolist()
        
        # Ajouter les vecteurs à la collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata
        )
        
        if self.store_path:
            self.client.persist()
        
        logger.info(f"Ajouté {len(vectors)} vecteurs à la collection ChromaDB")
        return ids
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        Recherche les k vecteurs les plus proches du vecteur de requête.
        
        Args:
            query_vector: Vecteur de requête (shape: [dimension])
            k: Nombre de résultats à retourner
            
        Returns:
            Tuple contenant:
                - Liste des IDs des vecteurs trouvés
                - Liste des scores de similarité
                - Liste des métadonnées associées
        """
        if self.get_store_size() == 0:
            logger.warning("La collection est vide, impossible d'effectuer une recherche")
            return [], [], []
        
        # Convertir le vecteur de requête en liste pour ChromaDB
        query_embedding = query_vector.tolist()
        
        # Effectuer la recherche
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.get_store_size())
        )
        
        # Extraire les résultats
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        # Convertir les distances en scores de similarité
        if self.distance_func == "cosine" or self.distance_func == "l2":
            # Pour cosine et l2, les petites distances sont meilleures,
            # donc on convertit en score de similarité en inversant
            scores = [1 - d for d in distances]
        else:  # Pour "ip" (inner product), les grandes valeurs sont meilleures
            scores = distances
        
        return ids, scores, metadatas
    
    def delete(self, ids: List[str]) -> None:
        """
        Supprime des vecteurs de la collection.
        
        Args:
            ids: Liste des IDs des vecteurs à supprimer
        """
        if not ids:
            return
        
        self.collection.delete(ids=ids)
        
        if self.store_path:
            self.client.persist()
        
        logger.info(f"Supprimé {len(ids)} vecteurs de la collection ChromaDB")
    
    def save(self) -> None:
        """
        Persiste la collection sur disque.
        """
        if not self.store_path:
            logger.warning("Aucun chemin de stockage spécifié, impossible de sauvegarder la collection")
            return
        
        self.client.persist()
        logger.info(f"Collection ChromaDB sauvegardée à {self.store_path}")
    
    def load(self) -> None:
        """
        Charge la collection depuis le disque.
        """
        # ChromaDB charge automatiquement les données au démarrage
        # si un persist_directory est spécifié
        pass
    
    def get_store_size(self) -> int:
        """
        Retourne le nombre de vecteurs stockés dans la collection.
        
        Returns:
            Nombre de vecteurs stockés
        """
        return self.collection.count()


def create_vector_store(
    store_type: str,
    dimension: int,
    store_path: Optional[str] = None,
    **kwargs
) -> VectorStore:
    """
    Fonction utilitaire pour créer une instance de VectorStore du type spécifié.
    
    Args:
        store_type: Type de base de données vectorielle ("faiss" ou "chroma")
        dimension: Dimension des vecteurs d'embedding
        store_path: Chemin pour la persistance
        **kwargs: Arguments spécifiques au type de store
        
    Returns:
        Instance de VectorStore
    """
    # Si ChromaDB n'est pas disponible ou explicitement demandé FAISS
    if not CHROMA_AVAILABLE or store_type.lower() == "faiss":
        return FAISSVectorStore(dimension, store_path, **kwargs)
    elif store_type.lower() == "chroma":
        return ChromaVectorStore(dimension, store_path, **kwargs)
    else:
        raise ValueError(f"Type de vector store non reconnu: {store_type}")