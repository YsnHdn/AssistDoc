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
        
        # Initialiser api_key et api_base avec des valeurs par défaut
        api_key = None
        api_base = None
        
        # Importer les clés API depuis le fichier de configuration
        try:
            # Vérifier si nous sommes sur Streamlit Cloud (les secrets sont accessibles)
            if hasattr(st, "secrets") and "api_keys" in st.secrets:
                api_key = st.secrets["api_keys"].get(provider)
                api_base = st.secrets.get("api_base_urls", {}).get(provider)
            
                # Si pas de token dans les secrets pour ce provider, afficher un avertissement
                if not api_key:
                    st.warning(f"Aucun token {provider} configuré dans les secrets Streamlit.")
            else:
                # Si pas de secrets, essayer d'utiliser config.py local
                try:
                    from config import API_KEYS, API_BASE_URLS
                    api_key = API_KEYS.get(provider)
                    api_base = API_BASE_URLS.get(provider)
                except ImportError:
                    # Valeurs par défaut si le fichier n'existe pas
                    api_key = None
                    api_base = "https://models.inference.ai.azure.com" if provider == "github_inference" else None
                    st.warning("Fichier config.py non trouvé. Les API nécessitant une authentification pourraient ne pas fonctionner.")
        except Exception as e:
            st.warning(f"Erreur lors de la récupération des clés API: {str(e)}")
            # En dernier recours, utiliser des valeurs par défaut
            api_key = None
            api_base = "https://models.inference.ai.azure.com" if provider == "github_inference" else None
        
        # Vérifier si GitHub Inference est choisi sans clé API
        if provider == "github_inference" and not api_key:
            return "Erreur: Aucune clé API GitHub Inference trouvée. Veuillez configurer une clé API dans config.py ou utiliser un autre fournisseur LLM comme Hugging Face.", []
        
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
        
        # Créer le retriever
        vector_store_path = "data/vector_store"
        retriever = create_default_retriever(
            store_path=vector_store_path,
            embedder_model="all-MiniLM-L6-v2",
            store_type="faiss",
            top_k=5
        )
        
        # APPROCHE AMÉLIORÉE AVEC MEILLEUR LOGGING
        st.write(f"Recherche pour le document: {doc_id}")
        
        # Obtenir tous les chunks du document d'abord, sans filtrage par requête
        all_doc_chunks = []
        if doc_index is not None:
            # Récupérer le document complet
            doc = st.session_state.documents[doc_index]
            
            # 1. Essayer d'abord de récupérer des chunks depuis la base vectorielle
            try:
                # Récupérer tous les chunks avec une requête générique
                doc_chunks = retriever.retrieve(
                    f"contenu de {doc_id}", 
                    top_k=20
                )
                
                # Filtre amélioré - comparaison plus souple du nom de fichier
                filtered_chunks = []
                for chunk in doc_chunks:
                    chunk_file = chunk.get("file_name", "").lower()
                    # Vérifier si le nom du fichier est contenu dans l'autre ou vice versa
                    if doc_id.lower() in chunk_file or chunk_file in doc_id.lower():
                        filtered_chunks.append(chunk)
                
                all_doc_chunks.extend(filtered_chunks)
            except Exception as e:
                st.warning(f"Erreur lors de la récupération des chunks: {str(e)}")
        
        # 2. Si pas assez de chunks trouvés, chercher avec la question spécifique
        if len(all_doc_chunks) < 2:
            try:
                question_chunks = retriever.retrieve(question, top_k=15)
                
                # Filtre amélioré pour la question aussi
                filtered_question_chunks = []
                for chunk in question_chunks:
                    chunk_file = chunk.get("file_name", "").lower()
                    if doc_id.lower() in chunk_file or chunk_file in doc_id.lower():
                        filtered_question_chunks.append(chunk)
                
                all_doc_chunks.extend(filtered_question_chunks)
                
                # Éliminer les doublons
                seen_texts = set()
                unique_chunks = []
                for chunk in all_doc_chunks:
                    text = chunk.get("text", "")
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        unique_chunks.append(chunk)
                
                all_doc_chunks = unique_chunks
            except Exception as e:
                st.warning(f"Erreur lors de la recherche par question: {str(e)}")
        
        # 3. En dernier recours, utiliser le texte brut du document
        if len(all_doc_chunks) < 2 and doc_index is not None:
            doc = st.session_state.documents[doc_index]
            if "full_text" in doc and doc["full_text"]:
                st.info("Utilisation du texte brut du document comme fallback")
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
                
                # Limiter à un nombre raisonnable de chunks
                all_doc_chunks.extend(chunks[:8])
        
        # S'il n'y a toujours pas de chunks, indiquer qu'il n'y a pas d'informations
        if not all_doc_chunks:
            return f"Je n'ai pas trouvé d'informations suffisantes dans le document {doc_id}. Ce document pourrait être vide ou mal indexé.", []
        
        # Limiter le nombre total de chunks pour le prompt
        filtered_chunks = all_doc_chunks[:8]  # Limiter à 8 chunks maximum
        
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