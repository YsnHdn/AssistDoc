"""
Page de questions-réponses de l'application AssistDoc.
Permet de poser des questions sur les documents et affiche les réponses avec sources.
"""

import streamlit as st
import time
import os
from pathlib import Path

# Import des modules de l'application
from ui.components.visualization import display_chat_message, display_model_info
from src.vector_db.retriever import create_default_retriever
from src.llm.models import LLMConfig, LLMProvider, create_llm

def detect_streamlit_cloud():
    """Détecte si l'application s'exécute sur Streamlit Cloud"""
    return os.path.exists("/mount/src")

def show_qa_page():
    """
    Affiche la page de questions-réponses avec un chat dédié par document.
    """
    # Titre de la page
    st.markdown("<h1 class='main-title'>Questions & Réponses 💬</h1>", unsafe_allow_html=True)
    
    # Vérifier si des documents sont chargés
    if not st.session_state.get("documents", []) or not st.session_state.get("vector_store_initialized", False):
        st.warning("Aucun document chargé. Veuillez d'abord charger et indexer des documents dans la barre latérale.")
        return
    
    # Initialiser les historiques de chat par document s'ils n'existent pas
    if "document_chats" not in st.session_state:
        st.session_state.document_chats = {}
    
    # Sélection du document pour la conversation
    doc_options = [doc.get("file_name", f"Document {i+1}") for i, doc in enumerate(st.session_state.documents)]
    
    selected_doc = st.selectbox(
        "Sélectionnez un document pour discuter",
        options=doc_options
    )
    
    # Obtenir l'index du document sélectionné
    selected_doc_index = doc_options.index(selected_doc)
    selected_doc_id = st.session_state.documents[selected_doc_index].get("file_name")
    
    # Créer un identifiant unique pour chaque document
    if selected_doc_id not in st.session_state.document_chats:
        st.session_state.document_chats[selected_doc_id] = []
    
    # Afficher la session de chat pour le document sélectionné
    st.subheader(f"Conversation sur: {selected_doc}")
    
    # Informations sur le modèle utilisé
    display_model_info()
    
    # Section de chat pour le document sélectionné
    with st.container():
        # Afficher l'historique de chat pour ce document
        for message in st.session_state.document_chats[selected_doc_id]:
            display_chat_message(
                message["content"],
                is_user=message["role"] == "user"
            )
        
        # Zone de saisie pour la question
        user_question = st.chat_input(f"Posez une question sur {selected_doc}...")
        
        # Si l'utilisateur a posé une question
        if user_question:
            # Ajouter la question à l'historique du document actuel
            st.session_state.document_chats[selected_doc_id].append({
                "role": "user",
                "content": user_question
            })
            
            # Afficher la question (mise à jour immédiate de l'interface)
            display_chat_message(user_question, is_user=True)
            
            # Traiter la question et générer une réponse
            with st.spinner(f"Recherche de la réponse dans {selected_doc}..."):
                answer, sources = process_question(user_question, selected_doc_id, selected_doc_index)
            
            # Créer le message de réponse avec les sources
            response_message = {
                "role": "assistant",
                "content": answer,
                "sources": sources
            }
            
            # Ajouter la réponse à l'historique du document actuel
            st.session_state.document_chats[selected_doc_id].append({
                "role": "assistant",
                "content": response_message
            })
            
            # Afficher la réponse
            display_chat_message(response_message)
    
    # Boutons d'action
    col1, col2 = st.columns(2)
    
    with col1:
        # Bouton pour effacer la conversation actuelle
        if st.session_state.document_chats.get(selected_doc_id) and st.button("Effacer cette conversation", type="secondary"):
            st.session_state.document_chats[selected_doc_id] = []
            st.rerun()
    
    with col2:
        # Bouton pour effacer toutes les conversations
        if any(st.session_state.document_chats.values()) and st.button("Effacer toutes les conversations", type="secondary"):
            st.session_state.document_chats = {key: [] for key in st.session_state.document_chats.keys()}
            st.rerun()

def process_question(question, doc_id, doc_index):
    """
    Traite une question et génère une réponse en utilisant le système RAG,
    en limitant la recherche au document spécifié.
    
    Args:
        question: Question posée par l'utilisateur
        doc_id: Identifiant du document
        doc_index: Index du document dans la liste des documents
        
    Returns:
        Tuple contenant (réponse, sources)
    """
    try:
        # Récupérer les paramètres de configuration LLM
        provider = st.session_state.llm_provider
        model = st.session_state.llm_model
        
        # Importer les clés API depuis le fichier de configuration
        try:
            from config import API_KEYS, API_BASE_URLS
            api_key = API_KEYS.get(provider)
            api_base = API_BASE_URLS.get(provider)
        except ImportError:
            # Utiliser des valeurs par défaut si le fichier n'existe pas
            api_key = None
            api_base = "https://models.inference.ai.azure.com" if provider == "github_inference" else None
            st.warning("Fichier config.py non trouvé. Les API nécessitant une authentification pourraient ne pas fonctionner.")
        
        # Créer la configuration LLM
        config = LLMConfig(
            provider=provider,
            model_name=model,
            api_key=api_key,
            api_base=api_base,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Si c'est Hugging Face, ajouter des paramètres spécifiques
        if provider == "huggingface":
            config.extra_params["use_api"] = True
        
        # Créer le LLM
        llm = create_llm(config)
        
        # Créer le retriever - Toujours utiliser FAISS sur Streamlit Cloud
        vector_store_path = "data/vector_store"
        store_type = "faiss"  # Toujours utiliser FAISS pour garantir la compatibilité
        retriever = create_default_retriever(
            store_path=vector_store_path,
            embedder_model="all-MiniLM-L6-v2",
            store_type=store_type,
            top_k=5
        )
        
        # APPROCHE AMÉLIORÉE : Récupérer les chunks en deux étapes
        # 1. Essayer de récupérer des chunks pertinents pour la question spécifique
        question_chunks = retriever.retrieve(question, top_k=10)
        
        # Filtrer pour ne garder que les chunks du document sélectionné
        filtered_chunks = [chunk for chunk in question_chunks if chunk.get("file_name") == doc_id]
        
        # 2. Si pas assez de chunks pertinents trouvés, récupérer des chunks généraux du document
        if len(filtered_chunks) < 2:
            # Requête générale sur le document
            doc_chunks = retriever.retrieve(
                f"contenu important de {doc_id}", 
                top_k=10
            )
            
            # Filtrer pour ce document spécifique
            doc_chunks = [chunk for chunk in doc_chunks if chunk.get("file_name") == doc_id]
            
            # Si toujours pas de chunks, essayer d'accéder au texte brut du document
            if not doc_chunks and doc_index is not None:
                # Récupérer le document complet
                doc = st.session_state.documents[doc_index]
                if "full_text" in doc:
                    # Diviser le texte en petits morceaux
                    text = doc["full_text"]
                    chunks = []
                    chunk_size = 1000  # Taille approximative de chunk
                    
                    for i in range(0, len(text), chunk_size):
                        end = min(i + chunk_size, len(text))
                        chunk_text = text[i:end]
                        chunks.append({
                            "text": chunk_text,
                            "file_name": doc_id,
                            "score": 0.5  # Score arbitraire
                        })
                    
                    # Utiliser ces chunks comme fallback
                    if chunks:
                        doc_chunks = chunks[:5]  # Limiter à 5 chunks
            
            # Ajouter les chunks du document à ceux déjà trouvés
            filtered_chunks.extend(doc_chunks)
            
            # Éliminer les doublons potentiels
            seen_texts = set()
            unique_chunks = []
            for chunk in filtered_chunks:
                text = chunk.get("text", "")
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_chunks.append(chunk)
            
            filtered_chunks = unique_chunks[:5]  # Limiter à 5 chunks
        
        # S'il n'y a toujours pas de chunks, indiquer qu'il n'y a pas d'informations
        if not filtered_chunks:
            return f"Je n'ai pas trouvé d'informations suffisantes dans le document {doc_id}. Ce document pourrait être vide ou mal indexé.", []
        
        # Convertir les chunks en texte formaté
        context_text = ""
        for i, chunk in enumerate(filtered_chunks):
            context_text += f"[EXTRAIT {i+1}]\n{chunk.get('text', '')}\n\n"
        
        # Système prompt explicite
        system_prompt = """Tu es un assistant intelligent qui répond aux questions sur des documents.
        Réponds en te basant sur les extraits fournis, même si la réponse n'est pas explicite.
        Si la question est générale (comme "de quoi parle ce document"), fais une synthèse des extraits.
        N'invente pas d'informations qui ne seraient pas dans les extraits."""
        
        # Prompt utilisateur direct
        user_prompt = f"""Voici des extraits du document "{doc_id}":

{context_text}

Question: {question}

Réponds à cette question en te basant sur les extraits ci-dessus. Si la question est générale (comme "de quoi parle ce document"), fais une synthèse des informations présentes dans les extraits."""
        
        # Générer directement la réponse
        response = llm.generate(user_prompt, system_prompt=system_prompt)
        
        # Créer les sources
        sources = []
        for chunk in filtered_chunks[:5]:  # Limiter à 5 sources pour l'affichage
            source = {
                "text": chunk.get("text", ""),
                "file_name": chunk.get("file_name", "Document inconnu"),
                "score": chunk.get("score", 0)
            }
            sources.append(source)
        
        return response.content, sources
        
    except Exception as e:
        import traceback
        error_text = f"Désolé, une erreur s'est produite: {str(e)}\n\n{traceback.format_exc()}"
        st.error(error_text)
        return f"Désolé, une erreur s'est produite: {str(e)}", []