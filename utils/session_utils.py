"""
Utilitaires pour la gestion des sessions utilisateur.
Fournit des fonctions communes pour l'isolation des données par utilisateur.
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

def get_user_registry_path(user_id):
    """
    Obtient le chemin du fichier de registre pour un utilisateur spécifique.
    
    Args:
        user_id: Identifiant unique de l'utilisateur
        
    Returns:
        Chemin vers le fichier de registre des documents de l'utilisateur
    """
    return get_user_data_path(user_id) / "documents_registry.json"

"""
Fonction de récupération des clés API adaptée pour utiliser les clés stockées en session.
Cette fonction est à insérer dans chaque fichier de page (qa.py, summary.py, extraction.py).
"""

def get_api_key_and_base(provider):
    """
    Récupère la clé API et l'URL de base pour le fournisseur spécifié.
    Priorise la clé API stockée en session, puis les secrets Streamlit,
    puis le fichier config.py local.
    
    Args:
        provider: Fournisseur LLM (github_inference ou huggingface)
        
    Returns:
        Tuple contenant (api_key, api_base)
    """
    # Initialiser api_key et api_base avec des valeurs par défaut
    api_key = None
    api_base = None
    
    # 1. Priorité 1: Utiliser la clé entrée par l'utilisateur dans la session
    if provider == "github_inference" and "github_api_key" in st.session_state and st.session_state.github_api_key:
        api_key = st.session_state.github_api_key
        api_base = "https://models.inference.ai.azure.com"
        return api_key, api_base
    
    if provider == "huggingface" and "huggingface_api_key" in st.session_state and st.session_state.huggingface_api_key:
        api_key = st.session_state.huggingface_api_key
        return api_key, api_base
    
    # 2. Priorité 2: Utiliser les secrets Streamlit
    try:
        if hasattr(st, "secrets") and "api_keys" in st.secrets:
            api_key = st.secrets["api_keys"].get(provider)
            api_base = st.secrets.get("api_base_urls", {}).get(provider)
            
            if api_key:
                return api_key, api_base
    except Exception as e:
        st.warning(f"Erreur lors de l'accès aux secrets Streamlit: {str(e)}")
    
    # 3. Priorité 3: Utiliser config.py local
    try:
        from config import API_KEYS, API_BASE_URLS
        api_key = API_KEYS.get(provider)
        api_base = API_BASE_URLS.get(provider)
        
        if api_key:
            return api_key, api_base
    except ImportError:
        # Valeurs par défaut si le fichier n'existe pas
        if provider == "github_inference":
            api_base = "https://models.inference.ai.azure.com"
    except Exception as e:
        st.warning(f"Erreur lors de la lecture de config.py: {str(e)}")
    
    # Retourner les valeurs (qui pourraient être None)
    return api_key, api_base