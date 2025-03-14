"""
Page d'accueil de l'application AssistDoc.
Présente l'application et affiche l'état actuel.
"""

import streamlit as st
import os
from pathlib import Path

def show_home_page(change_page_callback):
    """
    Affiche la page d'accueil.
    
    Args:
        change_page_callback: Fonction pour naviguer entre les pages
    """
    # Titre de la page
    st.markdown("<h1 class='main-title'>Bienvenue sur AssistDoc 📚</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        AssistDoc est votre assistant intelligent pour interagir avec vos documents.
        Basé sur la technique RAG (Retrieval-Augmented Generation), il vous permet de :
        
        * **Poser des questions** sur vos documents et obtenir des réponses précises
        * **Générer des résumés** adaptés à vos besoins
        * **Extraire des informations structurées** de vos documents
        """
    )
    
    # État actuel de l'application
    st.subheader("État actuel")
    
    # Créer deux colonnes pour afficher les statuts
    col1, col2 = st.columns(2)
    
    with col1:
        # Vérifier si des documents sont chargés
        if "documents" in st.session_state and st.session_state.documents:
            num_docs = len(st.session_state.documents)
            st.success(f"{num_docs} document(s) chargé(s) et prêt(s) à être utilisé(s)")
            
            # Afficher la liste des documents
            with st.expander("Voir les documents", expanded=False):
                for i, doc in enumerate(st.session_state.documents):
                    st.write(f"{i+1}. {doc.get('file_name', 'Document')}")
        else:
            st.warning("Aucun document chargé")
            st.markdown(
                """
                Commencez par charger vos documents via le panneau latéral.
                AssistDoc supporte les formats PDF, DOCX et TXT.
                """
            )
    
    with col2:
        # Vérifier si la base vectorielle est initialisée
        if st.session_state.vector_store_initialized:
            st.success("Base vectorielle initialisée et prête à être utilisée")
        else:
            st.warning("Base vectorielle non initialisée")
            st.markdown(
                """
                La base vectorielle sera automatiquement créée
                lorsque vous chargerez vos premiers documents.
                """
            )
    
    # Statistiques
    if "documents" in st.session_state and st.session_state.documents:
        st.subheader("Statistiques")
        
        # Créer trois colonnes pour les métriques
        metrics_cols = st.columns(3)
        
        with metrics_cols[0]:
            # Nombre total de documents
            st.metric("Documents", len(st.session_state.documents))
        
        with metrics_cols[1]:
            # Nombre total de pages (pour les PDFs)
            total_pages = sum(
                doc.get("num_pages", 1) 
                for doc in st.session_state.documents 
                if "num_pages" in doc
            )
            st.metric("Pages", total_pages)
        
        with metrics_cols[2]:
            # Estimation du nombre total de tokens
            est_tokens = sum(
                len(doc.get("full_text", "")) // 4  # ~4 caractères par token
                for doc in st.session_state.documents
            )
            st.metric("Est. Tokens", f"{est_tokens:,}")
    
    # Accès rapide
    st.subheader("Accès rapide")
    
    # Créer trois colonnes pour les boutons d'accès rapide
    btn_cols = st.columns(3)
    
    with btn_cols[0]:
        qa_btn = st.button("💬 Poser des questions", use_container_width=True)
        if qa_btn:
            change_page_callback("qa")
    
    with btn_cols[1]:
        summary_btn = st.button("📝 Générer un résumé", use_container_width=True)
        if summary_btn:
            change_page_callback("summary")
    
    with btn_cols[2]:
        extract_btn = st.button("🔍 Extraire des informations", use_container_width=True)
        if extract_btn:
            change_page_callback("extraction")
    
    # Informations sur l'application
    st.subheader("À propos d'AssistDoc")
    
    # Deux colonnes pour les informations
    info_cols = st.columns(2)
    
    with info_cols[0]:
        st.markdown(
            """
            **Comment ça marche ?**
            
            1. **Chargez** vos documents dans l'application
            2. Les documents sont **analysés et indexés** automatiquement
            3. **Interagissez** avec vos documents via les différentes fonctionnalités
            4. Recevez des **réponses précises** avec les sources correspondantes
            """
        )
    
    with info_cols[1]:
        st.markdown(
            """
            **Technologies utilisées :**
            
            * **RAG** (Retrieval-Augmented Generation)
            * **LLM** (Large Language Models) via GitHub Inference API/Hugging Face
            * **Embeddings vectoriels** pour la recherche sémantique
            * **Streamlit** pour l'interface utilisateur
            """
        )