"""
Composants de visualisation pour l'application.
Contient des fonctions pour afficher les messages du chat, les sources, etc.
"""

import streamlit as st
import json
from pathlib import Path
import pandas as pd
import re

def display_chat_message(message, is_user=False):
    """
    Affiche un message dans l'interface de chat.
    
    Args:
        message: Contenu du message ou dictionnaire contenant le message et les sources
        is_user: True si c'est un message utilisateur, False sinon
    """
    if isinstance(message, dict):
        content = message.get("content", "")
        sources = message.get("sources", [])
    else:
        content = message
        sources = []
    
    message_type = "user" if is_user else "assistant"
    bg_color = "#f0f2f6" if is_user else "#e6f3ff"
    
    # Container du message
    with st.container():
        # En-tête du message avec l'avatar
        col1, col2 = st.columns([1, 11])
        with col1:
            avatar = "👤" if is_user else "🤖"
            st.markdown(f"<h3 style='margin: 0; font-size: 1.5rem;'>{avatar}</h3>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{'Vous' if is_user else 'AssistDoc'}**")
        
        # Contenu du message
        st.markdown(
            f"<div style='background-color: {bg_color}; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>{content}</div>", 
            unsafe_allow_html=True
        )
        
        # Afficher les sources si disponibles
        if sources and not is_user:
            with st.expander("Sources", expanded=False):
                for i, source in enumerate(sources):
                    st.markdown(f"**Source {i+1}:** {source.get('text', '')[:150]}...")
                    st.caption(f"Document: {source.get('file_name', 'Inconnu')} | Score: {source.get('score', 0):.2f}")
                    st.divider()

def display_model_info():
    """
    Affiche les informations sur le modèle LLM actuellement utilisé.
    """
    model = st.session_state.llm_model
    provider = st.session_state.llm_provider
    
    provider_name = {
        "github_inference": "GitHub Inference API",
        "huggingface": "Hugging Face"
    }.get(provider, provider)
    
    st.caption(f"Modèle: {model} via {provider_name}")

def display_extraction_results(extraction_data, format_type="json"):
    """
    Affiche les résultats d'extraction d'informations.
    
    Args:
        extraction_data: Données extraites
        format_type: Format d'affichage (json, table, text)
    """
    if not extraction_data:
        st.warning("Aucune donnée extraite")
        return
    
    # Convertir en dictionnaire si c'est une chaîne JSON
    if isinstance(extraction_data, str):
        try:
            if extraction_data.strip().startswith("{"):
                # Extraire uniquement la partie JSON si la réponse contient du texte explicatif
                json_text = extract_json_from_text(extraction_data)
                extraction_data = json.loads(json_text)
            else:
                st.text(extraction_data)
                return
        except json.JSONDecodeError:
            st.text(extraction_data)
            return
    
    # Afficher selon le format demandé
    if format_type.lower() == "json":
        st.json(extraction_data)
    elif format_type.lower() == "table":
        # Créer un DataFrame bien formaté à partir des données d'extraction
        if isinstance(extraction_data, dict):
            # Créer un DataFrame avec une seule ligne
            data = {k: [v] for k, v in extraction_data.items()}
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        elif isinstance(extraction_data, list):
            # Si c'est une liste d'objets
            df = pd.DataFrame(extraction_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.write(extraction_data)
    else:
        # Format texte par défaut
        st.write(extraction_data)

def extract_json_from_text(text):
    """
    Extrait la partie JSON d'un texte qui peut contenir d'autres informations.
    
    Args:
        text: Texte contenant potentiellement du JSON
        
    Returns:
        Texte JSON extrait
    """
    # Chercher du texte entre accolades
    json_match = re.search(r'({[\s\S]*})', text)
    if json_match:
        return json_match.group(1)
    return text

def display_summary(summary_text, metadata=None):
    """
    Affiche un résumé généré avec des métadonnées.
    
    Args:
        summary_text: Texte du résumé
        metadata: Métadonnées sur le résumé (longueur, style, etc.)
    """
    st.markdown("## Résumé")
    
    # Afficher les métadonnées si disponibles
    if metadata:
        cols = st.columns(3)
        with cols[0]:
            st.metric("Longueur", metadata.get("length", "N/A"))
        with cols[1]:
            st.metric("Documents", metadata.get("num_docs", "N/A"))
        with cols[2]:
            st.metric("Style", metadata.get("style", "Standard"))
    
    # Afficher le résumé dans un conteneur stylisé
    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; 
        border-left: 5px solid #4b9afa; margin: 20px 0;">
        {summary_text}
        </div>
        """, 
        unsafe_allow_html=True
    )