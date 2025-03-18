"""
Utilitaires pour la gestion des clés API et des services LLM.
"""

import streamlit as st

def get_api_credentials(provider):
    """
    Récupère les informations d'authentification API pour le fournisseur LLM spécifié.
    Priorité: secrets Streamlit > config.py local
    
    Args:
        provider: Le fournisseur LLM ('github_inference' ou 'huggingface')
        
    Returns:
        Tuple (api_key, api_base) contenant la clé API et l'URL de base
    """
    api_key = None
    api_base = "https://models.inference.ai.azure.com" if provider == "github_inference" else None
    
    # 1. Vérifier les secrets Streamlit
    if hasattr(st, "secrets") and "api_keys" in st.secrets:
        api_key = st.secrets["api_keys"].get(provider)
        base_url = st.secrets.get("api_base_urls", {}).get(provider)
        if base_url:
            api_base = base_url
    
    # 2. Si pas de secrets ou pas de clé dans les secrets, essayer config.py local
    if not api_key:
        try:
            from config import API_KEYS, API_BASE_URLS
            api_key = API_KEYS.get(provider)
            if provider in API_BASE_URLS and API_BASE_URLS[provider]:
                api_base = API_BASE_URLS[provider]
        except ImportError:
            pass
    
    return api_key, api_base