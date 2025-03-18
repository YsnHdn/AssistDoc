"""
Composant pour la barre latérale de l'application.
Gère la navigation et les paramètres globaux.
"""

import streamlit as st
import os
from pathlib import Path

# Import des composants avec le nouveau module de gestion des documents
from ui.components.document_uploader import show_document_uploader, clear_all_documents

def create_sidebar(change_page_callback):
    """
    Crée la barre latérale de l'application.
    
    Args:
        change_page_callback: Fonction pour changer de page
    """
    with st.sidebar:
        # Logo et titre
        st.image("https://via.placeholder.com/150x80?text=AssistDoc", width=150)
        
        st.title("📚 AssistDoc")
        st.caption("Assistant intelligent pour vos documents")
        
        # Séparateur
        st.divider()
        
        # Navigation
        st.subheader("Navigation")
        
        # Boutons pour naviguer entre les pages
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Accueil", use_container_width=True):
                change_page_callback("home")
            if st.button("💬 Q&A", use_container_width=True):
                change_page_callback("qa")
        with col2:
            if st.button("📝 Résumé", use_container_width=True):
                change_page_callback("summary")
            if st.button("🔍 Extraction", use_container_width=True):
                change_page_callback("extraction")
        
        # Séparateur
        st.divider()
        
        # Uploader de documents
        st.subheader("Documents")
        show_document_uploader()
        
        # Statut des documents - géré maintenant directement dans show_document_uploader
        
        # Séparateur
        st.divider()
        
        # Configuration du modèle LLM
        st.subheader("Configuration")
        
        # Sélection du fournisseur LLM
        provider_options = {
            "github_inference": "GitHub Inference API",
            "huggingface": "Hugging Face"
        }
        
        selected_provider = st.selectbox(
            "Fournisseur LLM",
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            index=list(provider_options.keys()).index(st.session_state.llm_provider)
        )
        
        # Mettre à jour le fournisseur LLM dans la session
        if selected_provider != st.session_state.llm_provider:
            st.session_state.llm_provider = selected_provider
        
        # Options de modèle en fonction du fournisseur
        model_options = {
            "github_inference": ["gpt-4o", "gpt-4o-mini","DeepSeek-V3", "Phi-4-mini-instruct","Phi-3.5-mini-instruct","Llama-3.3-70B-Instruct"],
            "huggingface": ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-chat-hf", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
        }
        
        # Sélection du modèle
        selected_model = st.selectbox(
            "Modèle",
            options=model_options[selected_provider],
            index=0 if st.session_state.llm_model not in model_options[selected_provider] else model_options[selected_provider].index(st.session_state.llm_model)
        )
        
        # Mettre à jour le modèle LLM dans la session
        if selected_model != st.session_state.llm_model:
            st.session_state.llm_model = selected_model
        
        # Afficher le statut de connexion
        try:
            # Vérifier si les secrets sont disponibles sur Streamlit Cloud
            if hasattr(st, "secrets") and "api_keys" in st.secrets:
                provider_key = st.secrets["api_keys"].get(selected_provider)
                if provider_key:
                    st.success(f"Connecté à {provider_options[selected_provider]} via secrets Streamlit")
                else:
                    if selected_provider == "huggingface":
                        st.info("Vous pouvez utiliser un modèle local Hugging Face sans clé API")
                    else:
                        st.warning(f"Token {selected_provider} non configuré dans les secrets Streamlit")
            else:
                # Si pas de secrets, essayer d'utiliser config.py local
                try:
                    from config import API_KEYS
                    
                    if selected_provider == "github_inference" and API_KEYS.get("github_inference"):
                        st.success("Connecté à GitHub Inference API via config.py")
                    elif selected_provider == "huggingface" and API_KEYS.get("huggingface"):
                        st.success("Connecté à Hugging Face via config.py")
                    else:
                        if selected_provider == "huggingface":
                            st.info("Vous pouvez utiliser un modèle local Hugging Face sans clé API")
                        else:
                            st.warning(f"Token {selected_provider} non configuré ou vide dans config.py")
                except ImportError:
                    if selected_provider == "huggingface":
                        st.info("Vous pouvez utiliser un modèle local Hugging Face sans clé API")
                    else:
                        st.warning("Fichier config.py non trouvé")
                        st.info("Créez un fichier config.py avec vos clés API ou utilisez un modèle local Hugging Face")
                
                # Afficher l'exemple de config.py uniquement si aucun secret n'est disponible
                if not (hasattr(st, "secrets") and "api_keys" in st.secrets):
                    # Créer un exemple de fichier config.py
                    with st.expander("Exemple de config.py"):
                        st.code('''
"""
Configuration de l'application AssistDoc.
Ce fichier contient les clés d'API et autres configurations sensibles.
NE PAS INCLURE CE FICHIER DANS LE DÉPÔT GIT.
"""

# Clés d'API pour les différents fournisseurs LLM
API_KEYS = {
    "github_inference": "votre_token_github_ici",
    "openai": "",
    "anthropic": "",
    "huggingface": ""
}

# URLs de base pour les API (optionnel)
API_BASE_URLS = {
    "github_inference": "https://models.inference.ai.azure.com",
    "openai": None,
    "anthropic": None,
    "azure_openai": None
}

# Autres configurations
DEFAULT_PROVIDER = "github_inference"
DEFAULT_MODEL = "gpt-4o"
''', language="python")
        except Exception as e:
            st.error(f"Erreur lors de la vérification des clés API: {str(e)}")
        
        # Bouton de réinitialisation de la base de données
        with st.expander("Paramètres avancés", expanded=False):
            if st.button("Réinitialiser l'application", type="secondary"):
                if clear_all_documents():
                    # Réinitialiser tous les états de session
                    for key in list(st.session_state.keys()):
                        if key != "current_page":  # Conserver la page actuelle
                            del st.session_state[key]
                    
                    # Réinitialiser les valeurs par défaut
                    st.session_state.documents = []
                    st.session_state.vector_store_initialized = False
                    st.session_state.chat_history = []
                    st.session_state.llm_provider = "github_inference"
                    st.session_state.llm_model = "gpt-4o"
                    
                    st.success("Application réinitialisée avec succès")
                    st.rerun()
        
        # Footer
        st.divider()
        st.caption("© 2025 AssistDoc • v1.0.0")