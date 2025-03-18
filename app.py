"""
AssistDoc - Application Streamlit pour l'assistance documentaire basée sur RAG.
Point d'entrée principal avec support d'isolation par utilisateur.
"""

import streamlit as st
import os
import time
from pathlib import Path

# Import des modules de l'application
from ui.pages.home import show_home_page
from ui.pages.qa import show_qa_page
from ui.pages.summary import show_summary_page
from ui.pages.extraction import show_extraction_page
from ui.components.sidebar import create_sidebar

# CSS personnalisé pour améliorer l'apparence
custom_css = """
<style>
.main-title {
    font-size: 2.5rem;
    color: #2E6AC0;
    margin-bottom: 1rem;
}
.stButton button {
    background-color: #4C86D0;
    color: white;
}
.stButton button:hover {
    background-color: #2E6AC0;
}
</style>
"""

def change_page(page_name):
    """
    Change la page actuelle dans la session.
    
    Args:
        page_name: Nom de la page à afficher
    """
    st.session_state.current_page = page_name
    # Forcer la réexécution du script
    st.rerun()

def main():
    """
    Fonction principale de l'application.
    """
    # Configuration de la page Streamlit
    st.set_page_config(
        page_title="AssistDoc - Assistant documentaire",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Injecter le CSS personnalisé
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # Initialiser les variables de session si nécessaire
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    
    if "documents" not in st.session_state:
        st.session_state.documents = []
    
    if "vector_store_initialized" not in st.session_state:
        st.session_state.vector_store_initialized = False
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Configuration par défaut du modèle LLM
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "github_inference"
    
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "gpt-4o"
    
    # S'assurer que l'ID utilisateur existe (géré par document_uploader.py)
    if "user_id" not in st.session_state:
        from ui.components.document_uploader import get_user_id
        st.session_state.user_id = get_user_id()
    
    # Créer la barre latérale
    create_sidebar(change_page)
    
    # Afficher la page appropriée
    if st.session_state.current_page == "home":
        show_home_page(change_page)
    elif st.session_state.current_page == "qa":
        show_qa_page()
    elif st.session_state.current_page == "summary":
        show_summary_page()
    elif st.session_state.current_page == "extraction":
        show_extraction_page()

if __name__ == "__main__":
    main()