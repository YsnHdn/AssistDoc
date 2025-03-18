"""
Page de résumé de l'application AssistDoc.
Permet de générer des résumés des documents avec différentes options.
Mise à jour pour supporter l'isolation des données par utilisateur.
"""

import streamlit as st
import time
import os
from pathlib import Path
import io

# Import des modules de l'application
from ui.components.visualization import display_summary, display_model_info
from src.vector_db.retriever import create_user_aware_retriever
from src.llm.models import LLMConfig, LLMProvider, create_llm
from utils.session_utils import get_user_id, get_user_data_path

def detect_streamlit_cloud():
    """Détecte si l'application s'exécute sur Streamlit Cloud"""
    return os.path.exists("/mount/src")

def show_summary_page():
    """
    Affiche la page de génération de résumé avec isolation par utilisateur.
    """
    # Titre de la page
    st.markdown("<h1 class='main-title'>Résumé de Documents 📝</h1>", unsafe_allow_html=True)
    
    # Obtenir l'ID utilisateur
    user_id = get_user_id()
    
    # Vérifier si des documents sont chargés pour cet utilisateur
    if not st.session_state.get("documents", []) or not st.session_state.get("vector_store_initialized", False):
        st.warning("Aucun document chargé. Veuillez d'abord charger et indexer des documents dans la barre latérale.")
        return
    
    # Informations sur le modèle utilisé
    display_model_info()
    
    # Panneau de configuration du résumé
    with st.expander("Options de résumé", expanded=True):
        # Sélection du document à résumer
        doc_options = [doc.get("file_name", f"Document {i+1}") for i, doc in enumerate(st.session_state.documents)]
        
        selected_doc = st.selectbox(
            "Document à résumer",
            options=doc_options
        )
        
        # Obtenir l'index du document sélectionné
        selected_doc_index = doc_options.index(selected_doc)
        selected_doc_id = st.session_state.documents[selected_doc_index].get("file_name")
        
        # Longueur du résumé
        col1, col2 = st.columns(2)
        
        with col1:
            length_options = {
                "court": "Court (environ 150 mots)",
                "moyen": "Moyen (environ 300 mots)",
                "long": "Long (environ 500 mots)",
                "personnalisé": "Personnalisé"
            }
            
            summary_length = st.selectbox(
                "Longueur du résumé",
                options=list(length_options.keys()),
                format_func=lambda x: length_options[x],
                index=1  # Option par défaut: moyen
            )
            
            # Si longueur personnalisée
            if summary_length == "personnalisé":
                custom_length = st.number_input("Nombre de mots", min_value=50, max_value=1000, value=300, step=50)
                summary_length_value = f"environ {custom_length} mots"
            else:
                # Convertir l'option en valeur utilisable
                summary_length_map = {
                    "court": "environ 150 mots",
                    "moyen": "environ 300 mots",
                    "long": "environ 500 mots"
                }
                summary_length_value = summary_length_map.get(summary_length, "environ 300 mots")
        
        with col2:
            # Style du résumé
            style_options = {
                "informatif": "Informatif (factuel)",
                "analytique": "Analytique (critique)",
                "simplifié": "Simplifié (vulgarisé)",
                "bullet_points": "Points clés (liste)"
            }
            
            summary_style = st.selectbox(
                "Style du résumé",
                options=list(style_options.keys()),
                format_func=lambda x: style_options[x],
                index=0  # Option par défaut: informatif
            )
        
        # Sélection des documents à résumer (si plusieurs sont chargés)
        if len(st.session_state.documents) > 1:
            multi_doc = st.checkbox("Inclure d'autres documents dans le résumé", value=False)
            
            if multi_doc:
                additional_docs = st.multiselect(
                    "Documents additionnels",
                    options=[doc for doc in doc_options if doc != selected_doc],
                    default=[],
                    help="Sélectionnez des documents supplémentaires à inclure dans le résumé"
                )
                
                # Ajouter le document principal aux documents sélectionnés
                selected_docs = [selected_doc] + additional_docs
                
                # Filtrer les documents sélectionnés
                selected_doc_indices = [i for i, name in enumerate(doc_options) if name in selected_docs]
            else:
                # Si un seul document est sélectionné
                selected_doc_indices = [selected_doc_index]
        else:
            # Si un seul document est chargé, le sélectionner automatiquement
            selected_doc_indices = [0]
    
    # Bouton pour générer le résumé
    if st.button("Générer le résumé", type="primary", use_container_width=True):
        # Vérifier qu'au moins un document est sélectionné
        if not selected_doc_indices:
            st.error("Veuillez sélectionner au moins un document à résumer")
            return
        
        # Générer le résumé
        with st.spinner("Génération du résumé en cours..."):
            summary_text, metadata = generate_summary(
                summary_length_value,
                summary_style,
                selected_doc_indices,
                user_id
            )
        
        # Afficher le résumé
        if summary_text:
            display_summary(summary_text, metadata)
            
            # Permettre le téléchargement du résumé
            offer_download(summary_text, metadata)

def generate_summary(length, style, doc_indices, user_id):
    """
    Génère un résumé des documents sélectionnés pour un utilisateur spécifique.
    
    Args:
        length: Longueur souhaitée pour le résumé
        style: Style souhaité pour le résumé
        doc_indices: Indices des documents à résumer
        user_id: Identifiant unique de l'utilisateur
        
    Returns:
        Tuple contenant (texte du résumé, métadonnées)
    """
    try:
        # Récupérer les paramètres de configuration LLM
        provider = st.session_state.llm_provider
        model = st.session_state.llm_model
        
        # Récupérer les informations d'authentification
        from utils.api_helpers import get_api_credentials
        api_key, api_base = get_api_credentials(provider)

        # Vérifier si GitHub Inference est choisi sans clé API
        if provider == "github_inference" and not api_key:
            return "Erreur: API GitHub Inference non configurée. Contactez l'administrateur ou choisissez un modèle Hugging Face.", []

        # Créer la configuration LLM
        config = LLMConfig(
            provider=provider,
            model_name=model,
            api_key=api_key,
            api_base=api_base,
            temperature=0.7,
            max_tokens=1500  # Augmenter pour les résumés
        )
        
        # Si c'est Hugging Face, ajouter des paramètres spécifiques
        if provider == "huggingface":
            config.extra_params["use_api"] = True
        
        # Créer le LLM
        llm = create_llm(config)
        
        # Créer le retriever spécifique à l'utilisateur
        user_vector_store_path = str(get_user_data_path(user_id) / "vector_store")
        store_type = "faiss"  # Utiliser FAISS pour assurer la compatibilité
        retriever = create_user_aware_retriever(
            user_id=user_id,
            store_path=user_vector_store_path,
            embedder_model="all-MiniLM-L6-v2",
            store_type=store_type,
            top_k=10
        )
        
        # Obtenir les noms des documents sélectionnés
        selected_doc_names = [st.session_state.documents[i].get("file_name") for i in doc_indices]
        
        # Cartographier les styles aux textes descriptifs
        style_map = {
            "informatif": "informatif et factuel",
            "analytique": "analytique avec des insights",
            "simplifié": "simplifié et accessible",
            "bullet_points": "sous forme de liste à puces des points clés"
        }
        style_description = style_map.get(style, "informatif et factuel")
        
        # Récupérer les chunks pertinents spécifiques à l'utilisateur
        all_chunks = []
        for doc_name in selected_doc_names:
            doc_chunks = retriever.retrieve(
                query=f"Contenu important de {doc_name}", 
                top_k=20,
                filter_metadata={"file_name": doc_name}  # Filtre explicite pour ce document
            )
            
            all_chunks.extend(doc_chunks)
        
        # Extraire le texte de tous les chunks
        document_text = ""
        for i, chunk in enumerate(all_chunks):
            document_text += f"--- Extrait {i+1} ---\n"
            document_text += chunk.get("text", "") + "\n\n"
        
        # Système prompt très explicite
        system_prompt = f"""Tu es un assistant spécialisé dans la création de résumés.
        Tu vas recevoir des extraits d'un ou plusieurs documents, et tu dois en faire un résumé.
        Le résumé doit être {style_description} et faire {length}.
        Ne demande pas plus d'informations et ne mentionne pas que tu résumes des extraits.
        Produis directement un résumé cohérent et bien structuré."""
        
        # Prompt utilisateur direct
        user_prompt = f"""Voici des extraits de document(s) à résumer:

{document_text}

Crée un résumé {style_description} de {length}. Le résumé doit être clair, cohérent et couvrir les points principaux des extraits fournis."""
        
        # Générer directement la réponse avec le LLM
        llm_response = llm.generate(user_prompt, system_prompt=system_prompt)
        
        # Créer les métadonnées pour l'affichage
        metadata = {
            "length": length.split()[-2] if "environ" in length else length,
            "style": style,
            "num_docs": len(doc_indices),
            "doc_names": ", ".join(selected_doc_names),
            "sources": all_chunks[:5]  # Limiter le nombre de sources pour l'affichage
        }
        
        return llm_response.content, metadata
        
    except Exception as e:
        import traceback
        error_text = f"Erreur lors de la génération du résumé: {str(e)}\n\n{traceback.format_exc()}"
        st.error(error_text)
        return f"Erreur: {str(e)}", {}

def offer_download(summary_text, metadata):
    """
    Offre des options pour télécharger le résumé.
    
    Args:
        summary_text: Texte du résumé
        metadata: Métadonnées du résumé
    """
    # Préparer le contenu du fichier
    buffer = io.StringIO()
    
    # Ajouter un en-tête
    buffer.write(f"# Résumé généré par AssistDoc\n\n")
    buffer.write(f"Style: {metadata.get('style', 'Standard')}\n")
    buffer.write(f"Longueur: {metadata.get('length', 'N/A')}\n")
    buffer.write(f"Documents: {metadata.get('num_docs', 'N/A')}\n\n")
    
    # Ajouter le résumé
    buffer.write("## Résumé\n\n")
    buffer.write(summary_text)
    
    # Ajouter les sources si disponibles
    if "sources" in metadata and metadata["sources"]:
        buffer.write("\n\n## Sources\n\n")
        for i, source in enumerate(metadata["sources"], 1):
            doc_name = source.get("file_name", "Document inconnu")
            text_snippet = source.get("text", "")[:100] + "..."
            buffer.write(f"{i}. **{doc_name}**: {text_snippet}\n\n")
    
    # Ajouter un pied de page
    buffer.write("\n---\n")
    buffer.write("Généré avec AssistDoc - Assistant intelligent pour vos documents")
    
    # Proposer le téléchargement
    st.download_button(
        label="Télécharger le résumé",
        data=buffer.getvalue(),
        file_name="resume_assistdoc.md",
        mime="text/markdown",
    )
