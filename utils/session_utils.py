"""
Utilitaires pour la gestion des sessions utilisateur.
"""

import streamlit as st
import uuid
from pathlib import Path

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