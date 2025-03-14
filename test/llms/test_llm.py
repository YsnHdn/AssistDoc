"""
Script de test pour le module LLM d'AssistDoc.
Ce script effectue des tests des différentes chaînes de traitement RAG
avec GitHub Inference et Hugging Face.
"""

import sys
import os
from pathlib import Path
import argparse
import time

# Ajouter le répertoire racine du projet au chemin d'importation
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # Remontez jusqu'à la racine du projet
sys.path.append(str(project_root))

# Essayer d'importer les configurations
try:
    from config import API_KEYS, API_BASE_URLS, DEFAULT_PROVIDER, DEFAULT_MODEL
except ImportError:
    # Configuration par défaut si le fichier n'existe pas
    API_KEYS = {
        "github_inference": "",
        "openai": "",
        "anthropic": "",
        "huggingface": ""
    }
    API_BASE_URLS = {
        "github_inference": "https://models.inference.ai.azure.com",
        "openai": None,
        "anthropic": None,
        "azure_openai": None
    }
    DEFAULT_PROVIDER = "github_inference"
    DEFAULT_MODEL = "gpt-4o"

# Imports du module LLM
from src.llm.models import (
    LLMConfig, 
    LLMProvider, 
    create_llm, 
    get_default_config,
    GitHubInferenceAPILLM,
    HuggingFaceLLM
)
from src.llm.prompts import (
    PromptTemplate, 
    PromptTemplateRegistry, 
    DEFAULT_TEMPLATES
)
from src.llm.chain import (
    RAGChain,
    QAChain, 
    SummaryChain, 
    ExtractionChain, 
    create_rag_chain
)

# Imports des autres modules nécessaires
from src.vector_db.retriever import DocumentRetriever, create_default_retriever
from src.document_processor.embedder import DocumentEmbedder

def test_llm_models(model_name=DEFAULT_MODEL, api_key=None, api_base=None, provider=DEFAULT_PROVIDER, use_api=False):
    """
    Teste l'initialisation et l'utilisation d'un modèle LLM.
    
    Args:
        model_name: Nom du modèle à utiliser
        api_key: Clé API (optionnelle)
        api_base: URL de base de l'API (pour GitHub/Azure)
        provider: Fournisseur LLM (si spécifié)
        use_api: Utiliser l'API HuggingFace au lieu du modèle local
    """
    print(f"\n{'='*80}\n")
    print(f"TEST DU MODÈLE LLM: {provider}/{model_name}\n")
    print(f"{'='*80}\n")
    
    try:
        # Créer la configuration
        provider_enum = LLMProvider(provider.lower())
        
        config_params = {
            "provider": provider_enum,
            "model_name": model_name,
            "api_key": api_key,
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        # Ajouter l'URL de base si disponible
        if api_base:
            config_params["api_base"] = api_base
        
        # Ajouter les paramètres spécifiques à HuggingFace
        if provider_enum == LLMProvider.HUGGINGFACE:
            config_params["use_api"] = use_api
        
        config = LLMConfig(**config_params)
        
        print("Configuration du modèle:")
        print(f"  Provider: {config.provider}")
        print(f"  Modèle: {config.model_name}")
        print(f"  API Base: {config.api_base}")
        print(f"  Température: {config.temperature}")
        print(f"  Max tokens: {config.max_tokens}")
        if provider_enum == LLMProvider.HUGGINGFACE:
            print(f"  Utilisation API: {use_api}")
        
        # Initialiser le modèle
        print("\nInitialisation du modèle...")
        llm = create_llm(config)
        print(f"Modèle initialisé: {type(llm).__name__}")
        
        # Tester une génération simple
        prompt = "Qu'est-ce qu'une base de données vectorielle et comment est-elle utilisée dans un système RAG?"
        
        print(f"\nPrompt: {prompt}")
        start_time = time.time()
        response = llm.generate(prompt)
        elapsed = time.time() - start_time
        
        print(f"\nRéponse générée en {elapsed:.2f} secondes:")
        print(f"Tokens: {response.tokens_input} (entrée) + {response.tokens_output} (sortie) = {response.total_tokens} (total)")
        print("\nContenu de la réponse:")
        print("-" * 80)
        print(response.content)
        print("-" * 80)
        
        return llm
        
    except Exception as e:
        print(f"Erreur lors du test du modèle LLM: {str(e)}")
        raise

def test_prompt_templates():
    """
    Teste l'utilisation des templates de prompts.
    """
    print(f"\n{'='*80}\n")
    print(f"TEST DES TEMPLATES DE PROMPTS\n")
    print(f"{'='*80}\n")
    
    try:
        # Créer un registre avec les templates par défaut
        registry = PromptTemplateRegistry()
        for name, template in DEFAULT_TEMPLATES.items():
            registry.register(name, template)
        
        print(f"Templates disponibles: {registry.list_templates()}")
        
        # Tester un template QA
        print("\nTest du template QA:")
        qa_template = registry.get("qa")
        
        # Exemple de contexte et de question
        context = [
            {"text": "Le RAG (Retrieval-Augmented Generation) est une technique qui combine la récupération d'informations avec la génération de texte par les grands modèles de langage."},
            {"text": "Dans un système RAG, on utilise généralement une base de données vectorielle pour stocker et rechercher efficacement des chunks de documents pertinents pour une requête donnée."}
        ]
        
        question = "Comment fonctionne le RAG?"
        
        # Formatter le prompt
        prompt = qa_template.format(
            context=context,
            question=question,
            language="français"
        )
        
        print("\nPrompt formaté:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)
        
        # Tester la création d'un template personnalisé
        print("\nCréation d'un template personnalisé:")
        custom_template = PromptTemplate(
            template="""
            Analysez le texte suivant et répondez à la question.
            
            TEXTE:
            {{ text }}
            
            QUESTION: {{ question }}
            
            ANALYSE:
            """,
            template_type="custom_analysis"
        )
        
        # Formatter le template personnalisé
        custom_prompt = custom_template.format(
            text="Les bases de données vectorielles permettent de stocker et rechercher efficacement des vecteurs de haute dimension.",
            question="Pourquoi utiliser une base de données vectorielle?"
        )
        
        print("\nPrompt personnalisé formaté:")
        print("-" * 80)
        print(custom_prompt)
        print("-" * 80)
        
        return registry
        
    except Exception as e:
        print(f"Erreur lors du test des templates: {str(e)}")
        raise

def test_rag_chain(llm, vector_store_path, query="Comment fonctionne le RAG?"):
    """
    Teste une chaîne RAG complète.
    
    Args:
        llm: Instance de BaseLLM à utiliser
        vector_store_path: Chemin vers la base vectorielle
        query: Requête de test
    """
    print(f"\n{'='*80}\n")
    print(f"TEST DE LA CHAÎNE RAG\n")
    print(f"{'='*80}\n")
    
    try:
        # Créer le retriever
        print("Initialisation du retriever...")
        retriever = create_default_retriever(
            store_path=vector_store_path,
            embedder_model="all-MiniLM-L6-v2",
            store_type="faiss",
            top_k=3
        )
        
        print(f"Retriever initialisé: {type(retriever).__name__}")
        
        # Créer la chaîne QA
        print("\nCréation d'une chaîne QA...")
        qa_chain = QAChain(
            llm=llm,
            retriever=retriever,
            max_chunks=3,
            language="français"
        )
        
        print("Chaîne QA créée")
        
        # Exécuter la chaîne QA
        print(f"\nExécution de la chaîne QA avec la requête: '{query}'")
        start_time = time.time()
        result = qa_chain.run(query)
        elapsed = time.time() - start_time
        
        print(f"\nRésultat généré en {elapsed:.2f} secondes:")
        print(f"Nombre de chunks récupérés: {len(result.retrieved_chunks)}")
        print(f"Tokens: {result.llm_response.total_tokens} (total)")
        
        # Afficher les chunks récupérés
        print("\nChunks récupérés:")
        for i, chunk in enumerate(result.retrieved_chunks):
            print(f"\n[Chunk {i+1}] Score: {chunk.get('score', 'N/A'):.4f}")
            text = chunk.get('text', '')
            print(f"{text[:150]}..." if len(text) > 150 else text)
        
        # Afficher la réponse
        print("\nRéponse:")
        print("-" * 80)
        print(result.content)
        print("-" * 80)
        
        # Tester la chaîne de résumé
        print("\nCréation d'une chaîne de résumé...")
        summary_chain = SummaryChain(
            llm=llm,
            retriever=retriever,
            max_chunks=5,
            summary_length="environ 100 mots",
            language="français"
        )
        
        print("Chaîne de résumé créée")
        
        # Exécuter la chaîne de résumé
        print("\nExécution de la chaîne de résumé...")
        result_summary = summary_chain.run()
        
        print("\nRésumé:")
        print("-" * 80)
        print(result_summary.content)
        print("-" * 80)
        
        # Tester la chaîne d'extraction
        print("\nCréation d'une chaîne d'extraction...")
        extraction_chain = ExtractionChain(
            llm=llm,
            retriever=retriever,
            items_to_extract=["technologies", "concepts_clés", "avantages", "applications"],
            output_format="JSON",
            language="français"
        )
        
        print("Chaîne d'extraction créée")
        
        # Exécuter la chaîne d'extraction
        print("\nExécution de la chaîne d'extraction...")
        result_extraction = extraction_chain.run()
        
        print("\nInformations extraites:")
        print("-" * 80)
        print(result_extraction.content)
        print("-" * 80)
        
        # Vérifier si le parsing JSON a fonctionné
        if "parsed_result" in result_extraction.metadata:
            print("\nRésultat parsé:")
            print(result_extraction.metadata["parsed_result"])
        
        return qa_chain, summary_chain, extraction_chain
        
    except Exception as e:
        print(f"Erreur lors du test de la chaîne RAG: {str(e)}")
        raise

def test_llm_only(llm, query="Qu'est-ce que le RAG et quels sont ses avantages?"):
    """
    Teste uniquement la fonctionnalité LLM sans retriever.
    Utile pour les tests rapides ou quand la base vectorielle n'est pas disponible.
    """
    print(f"\n{'='*80}\n")
    print(f"TEST DU LLM UNIQUEMENT\n")
    print(f"{'='*80}\n")
    
    print(f"Requête: {query}")
    
    start_time = time.time()
    response = llm.generate(query)
    elapsed = time.time() - start_time
    
    print(f"\nRéponse générée en {elapsed:.2f} secondes:")
    print(f"Tokens: {response.total_tokens} (total)")
    
    print("\nContenu de la réponse:")
    print("-" * 80)
    print(response.content)
    print("-" * 80)
    
    return response

def main():
    """
    Fonction principale qui exécute les tests.
    """
    # Configurer les arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Test du module LLM d'AssistDoc")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                        help=f"Modèle à utiliser (défaut: {DEFAULT_MODEL})")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Clé API (si non définie dans config.py)")
    parser.add_argument("--api-base", type=str, default=None,
                        help="URL de base de l'API (si non définie dans config.py)")
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER,
                       choices=["github_inference", "huggingface"],
                       help=f"Fournisseur LLM à utiliser (défaut: {DEFAULT_PROVIDER})")
    parser.add_argument("--vector-store", type=str, default="data/vector_store",
                        help="Chemin vers la base vectorielle")
    parser.add_argument("--query", type=str, default="Comment fonctionne le RAG?",
                        help="Requête de test")
    parser.add_argument("--llm-only", action="store_true",
                        help="Tester uniquement le LLM sans chaîne RAG")
    parser.add_argument("--use-api", action="store_true",
                        help="Utiliser l'API HuggingFace au lieu du modèle local (pertinent uniquement pour HuggingFace)")
    parser.add_argument("--open-model", action="store_true",
                        help="Utiliser un modèle ouvert Hugging Face (TinyLlama-1.1B-Chat-v1.0)")
    
    args = parser.parse_args()
    
    # Récupérer les variables d'environnement ou la configuration
    api_key = args.api_key
    if api_key is None:
        api_key = API_KEYS.get(args.provider.lower())
    
    api_base = args.api_base
    if api_base is None:
        api_base = API_BASE_URLS.get(args.provider.lower())
    
    # Utiliser un modèle ouvert HuggingFace si demandé
    if args.open_model and args.provider == "huggingface":
        args.model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Utilisation du modèle ouvert: {args.model}")
    
    # Exécuter les tests
    try:
        # Test des modèles LLM
        llm = test_llm_models(args.model, api_key, api_base, args.provider, args.use_api)
        
        # Test des templates de prompts
        registry = test_prompt_templates()
        
        # Test du LLM uniquement ou de la chaîne RAG complète
        if args.llm_only:
            test_llm_only(llm, args.query)
        else:
            if not os.path.exists(args.vector_store):
                print(f"\nATTENTION: La base vectorielle '{args.vector_store}' n'existe pas.")
                print("Vous pouvez créer une base vectorielle avec le script index_documents.py")
                print("ou utiliser --llm-only pour tester seulement le LLM.")
                return 1
            
            chains = test_rag_chain(llm, args.vector_store, args.query)
        
        print(f"\n{'='*80}\n")
        print("TOUS LES TESTS ONT RÉUSSI !")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n{'='*80}\n")
        print(f"ERREUR LORS DES TESTS: {str(e)}")
        print(f"{'='*80}\n")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())