"""
Module amélioré pour la gestion des documents avec persistance et isolation par utilisateur.
Gère le stockage des fichiers originaux et maintient un registre des documents séparé pour chaque utilisateur.
"""

import os
import json
import shutil
import hashlib
import tempfile
from pathlib import Path
import streamlit as st
import time
import uuid

# Import des modules de traitement de documents
from src.document_processor.parser import DocumentParser
from src.document_processor.chunker import DocumentChunker
from src.document_processor.embedder import DocumentEmbedder
from src.vector_db.store import create_vector_store

# Constantes pour les chemins de stockage
DATA_DIR = Path("data")
USERS_DIR = DATA_DIR / "users"

def get_user_id():
    """
    Récupère ou génère un identifiant utilisateur unique.
    Stocke l'ID dans la session pour persistance.
    
    Returns:
        Identifiant utilisateur unique
    """
    if "user_id" not in st.session_state:
        # Générer un nouvel ID utilisateur unique
        st.session_state.user_id = str(uuid.uuid4())
    
    return st.session_state.user_id

def get_user_data_path(user_id):
    """
    Obtient le chemin du répertoire de données pour un utilisateur spécifique.
    
    Args:
        user_id: Identifiant unique de l'utilisateur
        
    Returns:
        Chemin vers le répertoire de données de l'utilisateur
    """
    return USERS_DIR / user_id

def ensure_user_directories(user_id):
    """
    Crée les répertoires nécessaires pour un utilisateur spécifique.
    
    Args:
        user_id: Identifiant unique de l'utilisateur
    """
    DATA_DIR.mkdir(exist_ok=True)
    USERS_DIR.mkdir(exist_ok=True)
    
    user_dir = get_user_data_path(user_id)
    user_dir.mkdir(exist_ok=True)
    (user_dir / "uploaded_files").mkdir(exist_ok=True)
    (user_dir / "vector_store").mkdir(exist_ok=True)

def get_user_registry_path(user_id):
    """
    Obtient le chemin du fichier de registre pour un utilisateur spécifique.
    
    Args:
        user_id: Identifiant unique de l'utilisateur
        
    Returns:
        Chemin vers le fichier de registre des documents de l'utilisateur
    """
    return get_user_data_path(user_id) / "documents_registry.json"

def generate_file_hash(file_content):
    """
    Génère un hash unique pour un fichier.
    
    Args:
        file_content: Contenu binaire du fichier
        
    Returns:
        Hash SHA-256 du fichier
    """
    return hashlib.sha256(file_content).hexdigest()

def save_uploaded_file(uploaded_file, user_id):
    """
    Sauvegarde un fichier téléchargé sur le disque dans le répertoire de l'utilisateur.
    
    Args:
        uploaded_file: Objet fichier de Streamlit
        user_id: Identifiant unique de l'utilisateur
        
    Returns:
        Chemin vers le fichier sauvegardé et son hash
    """
    # Lire le contenu du fichier
    file_content = uploaded_file.getbuffer()
    
    # Générer un hash pour le fichier
    file_hash = generate_file_hash(file_content)
    
    # Créer un nom de fichier unique basé sur le nom original et le hash
    original_filename = Path(uploaded_file.name)
    unique_filename = f"{original_filename.stem}_{file_hash[:8]}{original_filename.suffix}"
    
    # Chemin complet pour sauvegarder le fichier dans l'espace utilisateur
    user_files_dir = get_user_data_path(user_id) / "uploaded_files"
    save_path = user_files_dir / unique_filename
    
    # Sauvegarder le fichier
    with open(save_path, "wb") as f:
        f.write(file_content)
    
    return str(save_path), file_hash

def save_documents_registry(user_id):
    """
    Sauvegarde le registre des documents dans un fichier JSON spécifique à l'utilisateur.
    
    Args:
        user_id: Identifiant unique de l'utilisateur
    """
    if "documents" in st.session_state:
        # Créer une liste avec les informations essentielles des documents
        documents_info = []
        for doc in st.session_state.documents:
            # Ne pas stocker le texte complet pour économiser de l'espace
            doc_info = {
                "file_name": doc.get("file_name", ""),
                "file_type": doc.get("file_type", ""),
                "file_path": doc.get("file_path", ""),
                "storage_path": doc.get("storage_path", ""),
                "file_hash": doc.get("file_hash", ""),
                "num_pages": doc.get("num_pages", 1),
                "metadata": doc.get("metadata", {}),
                "timestamp": doc.get("timestamp", time.time())
            }
            documents_info.append(doc_info)
        
        # Sauvegarder dans un fichier JSON spécifique à l'utilisateur
        registry_path = get_user_registry_path(user_id)
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(documents_info, f, ensure_ascii=False, indent=2)

def load_documents_registry(user_id):
    """
    Charge le registre des documents depuis un fichier spécifique à l'utilisateur.
    Vérifie également que les fichiers référencés existent toujours.
    
    Args:
        user_id: Identifiant unique de l'utilisateur
        
    Returns:
        True si le chargement est réussi, False sinon
    """
    try:
        registry_path = get_user_registry_path(user_id)
        if not registry_path.exists():
            return False
        
        with open(registry_path, "r", encoding="utf-8") as f:
            documents_info = json.load(f)
        
        # Vérifier l'existence des fichiers et nettoyer le registre
        valid_documents = []
        for doc in documents_info:
            storage_path = doc.get("storage_path", "")
            if storage_path and Path(storage_path).exists():
                valid_documents.append(doc)
            else:
                st.warning(f"Le fichier {doc.get('file_name', '')} n'a pas été trouvé sur le serveur et a été retiré du registre.")
        
        # Mettre à jour le registre
        if "documents" not in st.session_state:
            st.session_state.documents = []
        
        # Charger les documents valides
        st.session_state.documents = valid_documents
        
        # Marquer la base vectorielle comme initialisée si des documents valides existent
        if valid_documents:
            st.session_state.vector_store_initialized = True
            return True
        else:
            return False
            
    except (json.JSONDecodeError, Exception) as e:
        st.error(f"Erreur lors du chargement du registre des documents: {str(e)}")
        # Réinitialiser le registre en cas d'erreur
        if "documents" not in st.session_state:
            st.session_state.documents = []
        return False

def delete_document(doc_index, user_id):
    """
    Supprime un document du registre et du disque pour un utilisateur spécifique.
    
    Args:
        doc_index: Index du document dans la liste st.session_state.documents
        user_id: Identifiant unique de l'utilisateur
        
    Returns:
        True si la suppression est réussie, False sinon
    """
    try:
        if "documents" not in st.session_state or doc_index >= len(st.session_state.documents):
            return False
        
        # Récupérer les informations du document
        doc = st.session_state.documents[doc_index]
        storage_path = doc.get("storage_path", "")
        
        # Supprimer le fichier s'il existe
        if storage_path and Path(storage_path).exists():
            try:
                os.remove(storage_path)
            except:
                st.warning(f"Impossible de supprimer le fichier {storage_path}")
        
        # Supprimer du registre
        st.session_state.documents.pop(doc_index)
        
        # Enregistrer le registre mis à jour
        save_documents_registry(user_id)
        
        # Réinitialiser la base vectorielle si aucun document ne reste
        if not st.session_state.documents:
            st.session_state.vector_store_initialized = False
        
        return True
        
    except Exception as e:
        st.error(f"Erreur lors de la suppression du document: {str(e)}")
        return False

def clear_all_documents(user_id):
    """
    Supprime tous les documents du registre et du disque pour un utilisateur spécifique.
    
    Args:
        user_id: Identifiant unique de l'utilisateur
        
    Returns:
        True si la suppression est réussie, False sinon
    """
    try:
        if "documents" not in st.session_state:
            return True
        
        # Supprimer chaque fichier
        for doc in st.session_state.documents:
            storage_path = doc.get("storage_path", "")
            if storage_path and Path(storage_path).exists():
                try:
                    os.remove(storage_path)
                except:
                    pass
        
        # Vider le registre
        st.session_state.documents = []
        
        # Mettre à jour le fichier JSON
        save_documents_registry(user_id)
        
        # Réinitialiser l'état de la base vectorielle
        st.session_state.vector_store_initialized = False
        
        return True
        
    except Exception as e:
        st.error(f"Erreur lors de la suppression de tous les documents: {str(e)}")
        return False

def show_document_uploader():
    """
    Affiche le composant d'upload de documents et gère l'indexation automatique,
    avec isolation des données par utilisateur.
    """
    # Obtenir l'ID utilisateur
    user_id = get_user_id()
    
    # Afficher l'ID de session utilisateur (peut être caché en production)
    st.caption(f"ID de session: {user_id[:8]}...")
    
    # S'assurer que les répertoires nécessaires existent pour cet utilisateur
    ensure_user_directories(user_id)
    
    # Essayer de charger le registre des documents pour cet utilisateur
    if "documents" not in st.session_state:
        load_documents_registry(user_id)
    
    # Uploader de fichiers
    uploaded_files = st.file_uploader(
        "Charger des documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Formats supportés: PDF, DOCX, TXT"
    )
    
    # Si des fichiers ont été uploadés
    if uploaded_files:
        # Vérifier si les documents ont changé depuis la dernière fois
        current_filenames = [f.name for f in uploaded_files]
        
        # Stocker dans session_state si c'est la première fois
        if "last_uploaded_files" not in st.session_state:
            st.session_state.last_uploaded_files = []
            
        # Si de nouveaux fichiers ont été téléchargés ou des fichiers ont été supprimés
        if set(current_filenames) != set(st.session_state.last_uploaded_files):
            # Mettre à jour la liste des derniers fichiers téléchargés
            st.session_state.last_uploaded_files = current_filenames
            
            # Traiter et indexer les documents automatiquement
            with st.spinner("Traitement en cours..."):
                process_documents(uploaded_files, user_id)
    
    # Afficher les options de gestion des documents
    if "documents" in st.session_state and st.session_state.documents:
        with st.expander("Gestion des documents", expanded=False):
            # Liste des documents avec option de suppression
            st.write("Documents chargés:")
            
            # Utiliser un conteneur pour mieux contrôler la mise en page
            for i, doc in enumerate(st.session_state.documents):
                # Créer un conteneur pour chaque document
                with st.container():
                    # Utiliser des colonnes avec un ratio plus adapté (80% nom, 20% bouton)
                    cols = st.columns([4, 1])
                    
                    with cols[0]:
                        # Tronquer le nom s'il est trop long
                        doc_name = doc.get('file_name', 'Document')
                        if len(doc_name) > 30:
                            display_name = doc_name[:27] + "..."
                        else:
                            display_name = doc_name
                        
                        # Afficher le nom du document avec tooltip pour voir le nom complet
                        st.markdown(f"**{i+1}.** {display_name}")
                    
                    with cols[1]:
                        # Bouton de suppression avec clé unique
                        delete_btn = st.button("🗑️", key=f"delete_doc_{i}", help=f"Supprimer {doc_name}")
                        if delete_btn:
                            if delete_document(i, user_id):
                                st.success(f"Document supprimé avec succès")
                                st.rerun()
                    
                    # Ligne de séparation entre les documents
                    if i < len(st.session_state.documents) - 1:
                        st.markdown("---")
            
            # Bouton pour supprimer tous les documents
            if st.button("Supprimer tous les documents", type="secondary"):
                if clear_all_documents(user_id):
                    st.success("Tous les documents ont été supprimés")
                    st.rerun()
            
def process_documents(uploaded_files, user_id):
    """
    Traite et indexe les documents uploadés par un utilisateur spécifique.
    
    Args:
        uploaded_files: Liste des fichiers uploadés via st.file_uploader
        user_id: Identifiant unique de l'utilisateur
    """
    try:
        # Initialiser les composants de traitement
        parser = DocumentParser()
        chunker = DocumentChunker()
        embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")
        
        # Initialiser le vector store spécifique à l'utilisateur
        user_vector_store_path = get_user_data_path(user_id) / "vector_store"
        store = create_vector_store(
            store_type="faiss",
            dimension=embedder.embedding_dim,
            store_path=str(user_vector_store_path)
        )
        
        # Traiter chaque fichier
        progress_bar = st.progress(0)
        parsed_documents = []
        
        for i, file in enumerate(uploaded_files):
            # Mise à jour de la barre de progression
            progress_value = (i / len(uploaded_files)) * 0.3
            progress_bar.progress(progress_value)
            
            # Vérifier si le document est déjà dans le registre
            if "documents" in st.session_state:
                # Lire le contenu du fichier pour le hachage
                file_content = file.getbuffer()
                file_hash = generate_file_hash(file_content)
                
                # Vérifier si un document avec le même hash existe déjà
                if any(doc.get("file_hash") == file_hash for doc in st.session_state.documents):
                    st.info(f"Le document {file.name} est déjà indexé.")
                    continue
                
                # Réinitialiser le curseur de fichier après avoir calculé le hash
                file.seek(0)
            
            # Sauvegarder le fichier dans le répertoire de l'utilisateur
            storage_path, file_hash = save_uploaded_file(file, user_id)
            
            # Sauvegarder temporairement le fichier pour le parser
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
                tmp_file.write(file.getbuffer())
                file_path = tmp_file.name
            
            # Parser le document
            try:
                parsed_doc = parser.parse(file_path)
                parsed_doc["file_name"] = file.name  # Conserver le nom original
                parsed_doc["storage_path"] = storage_path  # Chemin de stockage permanent
                parsed_doc["file_hash"] = file_hash  # Hash du fichier pour la détection des doublons
                parsed_doc["timestamp"] = time.time()  # Horodatage
                parsed_doc["user_id"] = user_id  # Ajouter l'ID utilisateur pour le traçage
                parsed_documents.append(parsed_doc)
                
                # Nettoyer le fichier temporaire
                os.unlink(file_path)
            except Exception as e:
                st.error(f"Erreur lors du parsing de {file.name}: {str(e)}")
                continue
        
        # Chunking et embedding
        all_chunks = []
        for i, doc in enumerate(parsed_documents):
            # Mise à jour de la barre de progression
            progress_value = 0.3 + (i / len(parsed_documents)) * 0.3
            progress_bar.progress(progress_value)
            
            # Découper le document en chunks
            chunks = chunker.chunk_document(
                doc,
                strategy="semantic",
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Ajouter les métadonnées du document dans les chunks
            for chunk in chunks:
                chunk["document_id"] = doc["file_name"]
                chunk["file_name"] = doc["file_name"]
                chunk["storage_path"] = doc["storage_path"]
                chunk["file_hash"] = doc["file_hash"]
                chunk["user_id"] = user_id  # Ajouter l'ID utilisateur dans les chunks
            
            all_chunks.extend(chunks)
        
        # Générer les embeddings
        progress_bar.progress(0.6)
        
        # Assurez-vous que les vecteurs sont bien dans un tableau NumPy
        import numpy as np
        
        chunks_with_embeddings = embedder.embed_document_chunks(all_chunks)
        
        # Extraire les vecteurs et métadonnées
        vectors = np.array([chunk["embedding"] for chunk in chunks_with_embeddings], dtype=np.float32)
        metadatas = []
        
        for chunk in chunks_with_embeddings:
            # Créer une copie des métadonnées sans l'embedding (trop volumineux)
            metadata = {k: v for k, v in chunk.items() if k != "embedding"}
            metadatas.append(metadata)
        
        # Ajouter à la base vectorielle
        if vectors.size > 0:  # Vérifier que nous avons des vecteurs à ajouter
            store.add(vectors, metadatas)
            
            # Sauvegarder la base vectorielle
            store.save()
        
        # Finaliser
        progress_bar.progress(1.0)
        
        # Mettre à jour l'état de la session
        if "documents" not in st.session_state:
            st.session_state.documents = []
        
        # Ajouter les nouveaux documents au registre
        st.session_state.documents.extend(parsed_documents)
        
        # Sauvegarder le registre des documents
        save_documents_registry(user_id)
        
        # Marquer la base vectorielle comme initialisée
        st.session_state.vector_store_initialized = True
        
        # Afficher un message de succès
        st.success(f"{len(parsed_documents)} nouveau(x) document(s) indexé(s) avec succès!")
        
    except Exception as e:
        import traceback
        st.error(f"Erreur lors du traitement des documents: {str(e)}")
        st.error(traceback.format_exc())

def get_supported_formats():
    """
    Récupère les formats de documents supportés.
    
    Returns:
        Liste des extensions supportées
    """
    try:
        parser = DocumentParser()
        formats = parser.get_supported_formats()
        # Aplatir la liste des extensions
        extensions = []
        for format_exts in formats.values():
            extensions.extend(format_exts)
        return extensions
    except:
        # Valeurs par défaut si l'importation échoue
        return [".pdf", ".docx", ".txt"]