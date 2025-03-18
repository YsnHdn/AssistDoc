"""
Composant pour la barre latérale de l'application.
Gère la navigation et les paramètres globaux.
"""

import streamlit as st
import os
from pathlib import Path

# Import des composants avec le nouveau module de gestion des documents
from ui.components.document_uploader import show_document_uploader, clear_all_documents
from utils.session_utils import get_user_id

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
        
        # Afficher l'identifiant de session utilisateur (tronqué pour la sécurité)
        user_id = get_user_id()
        st.info(f"Session privée: {user_id[:8]}...")
        
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
        st.subheader("Mes Documents")
        show_document_uploader()
        
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
                        st.info("Contactez l'administrateur pour configurer l'accès à l'API")
            else:
                # Si pas de secrets, vérifier le développement local avec config.py
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
                    st.warning("Mode développement: fichier config.py non trouvé")
                    st.info("En mode production, les clés API sont gérées par l'administrateur")
        except Exception as e:
            st.error(f"Erreur lors de la vérification des clés API: {str(e)}")
        
        # Bouton de réinitialisation de la base de données pour cet utilisateur
        with st.expander("Paramètres avancés", expanded=False):
            if st.button("Réinitialiser ma session", type="secondary"):
                # Nettoyer les documents de l'utilisateur actuel uniquement
                if clear_all_documents(user_id):
                    # Réinitialiser les états de session
                    for key in list(st.session_state.keys()):
                        if key not in ["current_page", "user_id"]:
                            del st.session_state[key]
                    
                    # Réinitialiser les valeurs par défaut
                    st.session_state.documents = []
                    st.session_state.vector_store_initialized = False
                    
                    # Nettoyer l'historique de chat de cet utilisateur
                    if "document_chats" in st.session_state:
                        user_chats = {k: v for k, v in st.session_state.document_chats.items() 
                                   if not k.startswith(f"{user_id}_")}
                        st.session_state.document_chats = user_chats
                    else:
                        st.session_state.document_chats = {}
                    
                    # Réinitialiser le modèle par défaut
                    st.session_state.llm_provider = "github_inference"
                    st.session_state.llm_model = "gpt-4o"
                    
                    st.success("Votre session a été réinitialisée avec succès")
                    st.rerun()
        
        # Footer
        st.divider()
        st.caption("© 2025 AssistDoc • v1.0.0")