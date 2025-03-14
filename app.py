"""
Point d'entrée principal de l'application AssistDoc.
Ce fichier initialise l'application Streamlit et gère la navigation entre les pages.
"""

import streamlit as st
import os
from pathlib import Path

# Import des modules de l'application
from ui.pages.home import show_home_page
from ui.pages.qa import show_qa_page
from ui.pages.summary import show_summary_page
from ui.pages.extraction import show_extraction_page
from ui.components.sidebar import create_sidebar
from ui.components.document_uploader import load_documents_registry  # Importer la fonction de chargement du registre

# Configuration de la page Streamlit
st.set_page_config(
    page_title="AssistDoc - RAG pour vos documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialisation des variables de session si elles n'existent pas déjà
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "documents" not in st.session_state:
    st.session_state.documents = []
if "vector_store_initialized" not in st.session_state:
    st.session_state.vector_store_initialized = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "github_inference"
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-4o"

# Fonction pour changer de page
def change_page(page):
    st.session_state.current_page = page

# CSS pour le style de l'application
def load_css():
    css = """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #f0f2f6;
        }
        .assistant-message {
            background-color: #e6f3ff;
        }
        .message-content {
            margin-top: 0.5rem;
        }
        .source-info {
            font-size: 0.8rem;
            color: #555;
            margin-top: 0.5rem;
        }
        .stButton button {
            width: 100%;
        }
        /* Personnalisation de la barre latérale */
        [data-testid=stSidebar] {
            background-color: #f8f9fa;
        }
        /* Personnalisation des onglets */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            background-color: #f0f2f6;
        }
        .stTabs [aria-selected="true"] {
            background-color: #e6f3ff;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def main():
    load_css()
    
    # Vérifier si c'est la première fois que nous chargeons les documents dans cette session
    if "documents_loaded_from_registry" not in st.session_state:
        # Charger le registre des documents au démarrage de l'application
        success = load_documents_registry()
        
        # Afficher le message uniquement lors du premier chargement
        #if success and "documents" in st.session_state and st.session_state.documents:
        #    st.sidebar.success(f"{len(st.session_state.documents)} document(s) chargé(s) depuis le registre")
        
        # Marquer que les documents ont été chargés
        st.session_state.documents_loaded_from_registry = True
    
    # Afficher la barre latérale
    create_sidebar(change_page)
    
    # Sélection de la page à afficher
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