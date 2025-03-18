"""
Page d'extraction d'informations de l'application AssistDoc.
Permet d'extraire des informations structurées des documents.
"""

import streamlit as st
import time
import json
import os
import re
from pathlib import Path
import io

# Import des modules de l'application
from ui.components.visualization import display_extraction_results, display_model_info
from src.vector_db.retriever import create_default_retriever
from src.llm.models import LLMConfig, LLMProvider, create_llm

def detect_streamlit_cloud():
    """Détecte si l'application s'exécute sur Streamlit Cloud"""
    return os.path.exists("/mount/src")

def show_extraction_page():
    """
    Affiche la page d'extraction d'informations.
    """
    # Titre de la page
    st.markdown("<h1 class='main-title'>Extraction d'Informations 🔍</h1>", unsafe_allow_html=True)
    
    # Vérifier si des documents sont chargés
    if not st.session_state.get("documents", []) or not st.session_state.get("vector_store_initialized", False):
        st.warning("Aucun document chargé. Veuillez d'abord charger et indexer des documents dans la barre latérale.")
        return
    
    # Informations sur le modèle utilisé
    display_model_info()
    
    # Panneau de configuration de l'extraction
    with st.expander("Options d'extraction", expanded=True):
        # Sélection du document à analyser
        doc_options = [doc.get("file_name", f"Document {i+1}") for i, doc in enumerate(st.session_state.documents)]
        
        selected_doc = st.selectbox(
            "Document à analyser",
            options=doc_options
        )
        
        # Obtenir l'index du document sélectionné
        selected_doc_index = doc_options.index(selected_doc)
        selected_doc_id = st.session_state.documents[selected_doc_index].get("file_name")
        
        # Éléments à extraire
        st.subheader("Éléments à extraire")
        
        # Types d'extraction prédéfinis
        extraction_types = {
            "custom": "Personnalisé",
            "metadata": "Métadonnées du document",
            "contacts": "Informations de contact",
            "dates": "Dates et événements",
            "entities": "Entités nommées",
            "key_concepts": "Concepts clés"
        }
        
        extraction_type = st.selectbox(
            "Type d'extraction",
            options=list(extraction_types.keys()),
            format_func=lambda x: extraction_types[x],
            index=1  # Option par défaut: métadonnées
        )
        
        # Définir les éléments à extraire selon le type sélectionné
        predefined_items = {
            "metadata": ["titre", "auteur", "date", "organisation", "mots_clés", "sujet_principal"],
            "contacts": ["noms", "emails", "téléphones", "adresses", "organisations", "rôles"],
            "dates": ["dates_mentionnées", "événements", "échéances", "périodes_importantes"],
            "entities": ["personnes", "organisations", "lieux", "produits", "technologies"],
            "key_concepts": ["thèmes_principaux", "concepts_clés", "terminologie", "idées_principales", "conclusions"]
        }
        
        # Si type personnalisé, permettre à l'utilisateur de définir ses propres éléments
        if extraction_type == "custom":
            custom_items = st.text_area(
                "Éléments à extraire (un par ligne)",
                value="titre\nauteur\ndate\nmots_clés",
                height=150,
                help="Entrez un élément à extraire par ligne"
            )
            
            # Convertir en liste
            items_to_extract = [item.strip() for item in custom_items.split('\n') if item.strip()]
        else:
            # Utiliser les éléments prédéfinis
            items_to_extract = predefined_items.get(extraction_type, [])
            
            # Afficher les éléments qui seront extraits
            st.write("Éléments qui seront extraits:")
            for item in items_to_extract:
                st.write(f"- {item}")
        
        # Format de sortie
        col1, col2 = st.columns(2)
        
        with col1:
            format_options = ["JSON", "Tableau", "Texte"]
            output_format = st.selectbox("Format de sortie", options=format_options, index=0)
        
        with col2:
            # Option pour inclure d'autres documents
            if len(st.session_state.documents) > 1:
                multi_doc = st.checkbox("Inclure d'autres documents", value=False)
                
                if multi_doc:
                    additional_docs = st.multiselect(
                        "Documents additionnels",
                        options=[doc for doc in doc_options if doc != selected_doc],
                        default=[],
                        help="Sélectionnez des documents supplémentaires à analyser"
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
    
    # Bouton pour lancer l'extraction
    if st.button("Extraire les informations", type="primary", use_container_width=True):
        # Vérifier qu'au moins un document est sélectionné
        if not selected_doc_indices:
            st.error("Veuillez sélectionner au moins un document à analyser")
            return
        
        # Vérifier qu'au moins un élément est à extraire
        if not items_to_extract:
            st.error("Veuillez spécifier au moins un élément à extraire")
            return
        
        # Lancer l'extraction
        with st.spinner("Extraction des informations en cours..."):
            extraction_result = extract_information(
                items_to_extract,
                output_format,
                selected_doc_indices
            )
        
        # Afficher les résultats
        if extraction_result:
            st.subheader("Résultats de l'extraction")
            
            # Format d'affichage
            display_format = "json" if output_format == "JSON" else "table" if output_format == "Tableau" else "text"
            display_extraction_results(extraction_result, display_format)
            
            # Permettre le téléchargement des résultats
            offer_download(extraction_result, output_format, items_to_extract)

def extract_information(items_to_extract, output_format, doc_indices):
    """
    Extrait des informations des documents sélectionnés.
    
    Args:
        items_to_extract: Liste des éléments à extraire
        output_format: Format de sortie souhaité
        doc_indices: Indices des documents à analyser
        
    Returns:
        Résultat de l'extraction (structure JSON ou texte)
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
            return "Erreur: Aucune clé API GitHub Inference trouvée. Veuillez configurer une clé API ou utiliser un autre fournisseur LLM comme Hugging Face."
        
        # Créer la configuration LLM
        config = LLMConfig(
            provider=provider,
            model_name=model,
            api_key=api_key,
            api_base=api_base,
            temperature=0.3,  # Réduire la température pour l'extraction (plus précis)
            max_tokens=1500
        )
        
        # Si c'est Hugging Face, ajouter des paramètres spécifiques
        if provider == "huggingface":
            config.extra_params["use_api"] = True
        
        # Créer le LLM
        llm = create_llm(config)
        
        # Créer le retriever - Toujours utiliser FAISS sur Streamlit Cloud
        vector_store_path = "data/vector_store"
        store_type = "faiss"  # Utiliser FAISS pour assurer la compatibilité
        retriever = create_default_retriever(
            store_path=vector_store_path,
            embedder_model="all-MiniLM-L6-v2",
            store_type=store_type,
            top_k=8
        )
        
        # Obtenir les noms des documents sélectionnés
        selected_doc_names = [st.session_state.documents[i].get("file_name") for i in doc_indices]
        
        # APPROCHE DIRECTE: récupérer les chunks, les filtrer, puis générer la réponse
        all_chunks = []
        
        # Pour chaque document, récupérer des chunks pertinents
        for doc_name in selected_doc_names:
            # Récupérer tous les chunks disponibles pour ce document
            chunks = retriever.retrieve(
                query=f"informations importantes sur {doc_name}",
                top_k=20
            )
            
            # Filtrer pour ce document spécifique
            doc_chunks = [chunk for chunk in chunks if chunk.get("file_name") == doc_name]
            
            # Si aucun chunk trouvé, essayer d'accéder au texte brut du document
            if not doc_chunks:
                for i, doc in enumerate(st.session_state.documents):
                    if doc.get("file_name") == doc_name and "full_text" in doc:
                        # Diviser le texte en chunks
                        text = doc["full_text"]
                        chunk_size = 1000
                        
                        for j in range(0, len(text), chunk_size):
                            end = min(j + chunk_size, len(text))
                            doc_chunks.append({
                                "text": text[j:end],
                                "file_name": doc_name,
                                "score": 0.5
                            })
                        
                        # Limiter le nombre de chunks
                        if doc_chunks:
                            doc_chunks = doc_chunks[:5]
                            break
            
            all_chunks.extend(doc_chunks)
        
        # Limiter le nombre total de chunks
        all_chunks = all_chunks[:15]
        
        # Formater les chunks en texte
        context_text = ""
        for i, chunk in enumerate(all_chunks):
            doc_name = chunk.get("file_name", "Document inconnu")
            context_text += f"[EXTRAIT {i+1} de {doc_name}]\n{chunk.get('text', '')}\n\n"
        
        # Formater les éléments à extraire
        items_str = "\n".join([f"- {item}" for item in items_to_extract])
        
        # Système prompt très directif
        system_prompt = f"""Tu es un assistant spécialisé dans l'extraction d'informations précises à partir de documents.
        Tu dois extraire UNIQUEMENT les informations demandées dans le format spécifié.
        Si une information n'est pas trouvée, indique-le par "Non trouvé" ou "N/A"."""
        
        # Construire le prompt utilisateur
        user_prompt = f"""Extrais les informations suivantes des documents fournis:

{items_str}

EXTRAITS DES DOCUMENTS:
{context_text}

Format de sortie: {output_format}
"""
        
        if output_format == "JSON":
            user_prompt += """
Réponds UNIQUEMENT avec un objet JSON valide contenant les informations extraites.
Exemple de format attendu:
{
  "titre": "Titre du document",
  "auteur": "Nom de l'auteur",
  "date": "Date trouvée",
  ...
}
"""
        elif output_format == "Tableau":
            user_prompt += """
Réponds avec les informations formatées de manière structurée pour un tableau.
Chaque élément doit avoir une valeur claire, même si c'est "Non trouvé" quand l'information n'est pas disponible.
Je convertis ta réponse en format tabulaire, donc la structure doit être nette et précise.
"""
        
        # Générer la réponse
        response = llm.generate(user_prompt, system_prompt=system_prompt)
        
        # Traiter la réponse selon le format demandé
        content = response.content
        
        # Si format JSON, essayer d'extraire et parser le JSON
        if output_format == "JSON":
            # Chercher un objet JSON valide dans la réponse
            import re
            
            # Chercher un texte entre accolades
            json_match = re.search(r'({[\s\S]*})', content)
            if json_match:
                json_str = json_match.group(1)
                try:
                    # Tenter de parser le JSON
                    parsed_json = json.loads(json_str)
                    return parsed_json
                except json.JSONDecodeError:
                    pass
        
        # Si format Tableau et pas déjà JSON, essayer de le convertir en structure tabulaire
        elif output_format == "Tableau":
            try:
                import re
                # D'abord essayer de voir si le texte contient du JSON
                json_match = re.search(r'({[\s\S]*})', content)
                if json_match:
                    try:
                        parsed_json = json.loads(json_match.group(1))
                        return parsed_json
                    except:
                        pass
                
                # Sinon, essayer de parser le texte en format clé-valeur
                result_dict = {}
                
                # Chercher les paires clé-valeur (format "clé: valeur" ou "clé - valeur")
                lines = content.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('|') or line.startswith('-'):  # Ignorer les lignes de séparation
                        continue
                    
                    # Essayer différents formats de séparation clé-valeur
                    separators = [':', '-', '=']
                    for sep in separators:
                        if sep in line:
                            parts = line.split(sep, 1)
                            if len(parts) == 2:
                                key = parts[0].strip().rstrip(':.-').strip()
                                value = parts[1].strip()
                                result_dict[key] = value
                                break
                
                # Si des données ont été extraites, retourner le dictionnaire
                if result_dict:
                    return result_dict
                
                # Dernière tentative: essayer de parser comme tableau markdown ou CSV
                if "|" in content:
                    import pandas as pd
                    import io
                    
                    # Nettoyer le contenu pour le format markdown
                    lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
                    
                    # Trouver les lignes qui semblent être un en-tête et des données
                    header_line = None
                    data_lines = []
                    
                    for i, line in enumerate(lines):
                        if "|" in line:
                            if not header_line:
                                header_line = line
                            else:
                                # Ignorer les lignes de séparation (avec des tirets)
                                if not all(c == '-' or c == '|' or c.isspace() for c in line):
                                    data_lines.append(line)
                    
                    if header_line and data_lines:
                        # Parser l'en-tête
                        headers = [h.strip() for h in header_line.split('|')]
                        headers = [h for h in headers if h]  # Supprimer les cellules vides
                        
                        # Parser les données
                        data = {}
                        for header in headers:
                            data[header] = []
                        
                        for line in data_lines:
                            cells = [c.strip() for c in line.split('|')]
                            cells = [c for c in cells if c]  # Supprimer les cellules vides
                            
                            for i, cell in enumerate(cells):
                                if i < len(headers):
                                    data[headers[i]].append(cell)
                        
                        # Vérifier qu'on a des données
                        if any(data.values()):
                            # Convertir en dictionnaire simple si une seule ligne
                            if all(len(v) == 1 for v in data.values()):
                                return {k: v[0] for k, v in data.items()}
                            else:
                                return data
            except Exception as e:
                st.warning(f"Erreur lors de la conversion en tableau: {str(e)}")
        
        # Retourner le texte brut si le parsing a échoué
        return content
        
    except Exception as e:
        import traceback
        st.error(f"Erreur lors de l'extraction des informations: {str(e)}")
        st.error(traceback.format_exc())
        return f"Erreur lors de l'extraction: {str(e)}"

def offer_download(extraction_result, format_type, items_extracted):
    """
    Offre des options pour télécharger les résultats d'extraction.
    
    Args:
        extraction_result: Résultat de l'extraction
        format_type: Format souhaité pour le téléchargement
        items_extracted: Liste des éléments extraits
    """
    try:
        # Préparer le contenu du fichier selon le format
        if format_type == "JSON":
            # Si c'est déjà un dictionnaire
            if isinstance(extraction_result, dict):
                content = json.dumps(extraction_result, indent=2, ensure_ascii=False)
                mime_type = "application/json"
                file_extension = "json"
            else:
                # Essayer de convertir en JSON
                try:
                    from ui.components.visualization import extract_json_from_text
                    json_text = extract_json_from_text(extraction_result)
                    content = json_text
                    mime_type = "application/json"
                    file_extension = "json"
                except:
                    content = extraction_result
                    mime_type = "text/plain"
                    file_extension = "txt"
        elif format_type == "Tableau":
            # Créer un CSV propre à partir des données
            if isinstance(extraction_result, dict):
                # En-tête CSV
                content = ",".join([f'"{k}"' for k in extraction_result.keys()]) + "\n"
                
                # Ligne de données
                values = []
                for v in extraction_result.values():
                    # Gérer les valeurs complexes
                    if isinstance(v, (list, dict)):
                        v = json.dumps(v, ensure_ascii=False)
                    # Échapper les guillemets
                    if isinstance(v, str):
                        v = v.replace('"', '""')
                    values.append(f'"{v}"')
                
                content += ",".join(values)
                mime_type = "text/csv"
                file_extension = "csv"
            else:
                content = extraction_result
                mime_type = "text/plain"
                file_extension = "txt"
        else:
            # Format texte par défaut
            content = extraction_result
            mime_type = "text/plain"
            file_extension = "txt"
        
        # Proposer le téléchargement
        st.download_button(
            label=f"Télécharger ({file_extension.upper()})",
            data=content,
            file_name=f"extraction_assistdoc.{file_extension}",
            mime=mime_type,
        )
    except Exception as e:
        st.error(f"Erreur lors de la préparation du téléchargement: {str(e)}")