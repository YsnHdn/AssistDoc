"""
Module pour la création et l'exécution de chaînes de traitement LLM complètes.
Implémente diverses chaînes pour le RAG (Retrieval-Augmented Generation) comme
la question-réponse, le résumé, l'extraction d'informations, etc.
Compatible avec les modèles GitHub Inference et Hugging Face.
"""

import time
import logging
import json
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from enum import Enum
from dataclasses import dataclass
import re

# Imports des autres modules de l'application
from .models import BaseLLM, LLMResponse, create_llm, LLMConfig
from .prompts import PromptTemplate, PromptTemplateRegistry, DEFAULT_TEMPLATES
from ..vector_db.retriever import DocumentRetriever

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChainResult:
    """
    Classe pour encapsuler le résultat d'une chaîne de traitement RAG.
    """
    llm_response: LLMResponse
    retrieved_chunks: List[Dict[str, Any]]
    query: str
    total_time: float
    metadata: Dict[str, Any]
    
    @property
    def content(self) -> str:
        """
        Récupère le contenu de la réponse LLM.
        
        Returns:
            Texte de la réponse
        """
        return self.llm_response.content
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit le résultat en dictionnaire.
        
        Returns:
            Représentation sous forme de dictionnaire
        """
        return {
            "content": self.content,
            "query": self.query,
            "llm_response": self.llm_response.to_dict(),
            "retrieved_chunks": [
                {k: v for k, v in chunk.items() if k != 'embedding'} 
                for chunk in self.retrieved_chunks
            ],
            "total_time": self.total_time,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        """
        Représentation sous forme de chaîne de caractères.
        
        Returns:
            Représentation du résultat
        """
        chunks_info = f"{len(self.retrieved_chunks)} chunks"
        return (f"ChainResult(query='{self.query[:30]}...', "
                f"response_length={len(self.content)}, "
                f"chunks={chunks_info}, "
                f"time={self.total_time:.2f}s)")


class RAGChain:
    """
    Chaîne de traitement RAG de base.
    """
    
    def __init__(self,
                 llm: BaseLLM,
                 retriever: DocumentRetriever,
                 prompt_template: Union[str, PromptTemplate],
                 prompt_registry: Optional[PromptTemplateRegistry] = None,
                 system_prompt: Optional[str] = None,
                 max_chunks: int = 5,
                 language: str = "français",
                 preprocessing_func: Optional[Callable[[str], str]] = None,
                 postprocessing_func: Optional[Callable[[str], str]] = None):
        """
        Initialise une chaîne RAG.
        
        Args:
            llm: Modèle de langage à utiliser
            retriever: Module pour la récupération des chunks pertinents
            prompt_template: Template de prompt ou nom du template dans le registre
            prompt_registry: Registre de templates (optionnel)
            system_prompt: Instructions système à envoyer au LLM (optionnel)
            max_chunks: Nombre maximum de chunks à récupérer
            language: Langue pour la réponse (français par défaut)
            preprocessing_func: Fonction pour prétraiter la requête
            postprocessing_func: Fonction pour post-traiter la réponse
        """
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.max_chunks = max_chunks
        self.language = language
        self.preprocessing_func = preprocessing_func
        self.postprocessing_func = postprocessing_func
        
        # Initialiser le registre de templates si nécessaire
        if prompt_registry is None:
            self.prompt_registry = PromptTemplateRegistry()
            # Ajouter les templates par défaut
            for name, template in DEFAULT_TEMPLATES.items():
                self.prompt_registry.register(name, template)
        else:
            self.prompt_registry = prompt_registry
        
        # Configurer le template de prompt
        if isinstance(prompt_template, str):
            if prompt_template in self.prompt_registry.templates:
                self.prompt_template = self.prompt_registry.get(prompt_template)
            else:
                # Si c'est un template littéral et non un nom de template
                self.prompt_template = PromptTemplate(prompt_template)
        else:
            self.prompt_template = prompt_template
        
        logger.info(f"RAGChain initialisée avec template '{self.prompt_template.template_type}'")
    
    def run(self, 
            query: str, 
            **kwargs) -> ChainResult:
        """
        Exécute la chaîne RAG complète.
        
        Args:
            query: Requête ou question de l'utilisateur
            **kwargs: Paramètres additionnels pour le template
            
        Returns:
            Résultat de la chaîne
        """
        start_time = time.time()
        metadata = {"chain_type": self.__class__.__name__}
        
        # 1. Prétraitement de la requête
        processed_query = self._preprocess_query(query)
        
        # 2. Récupération des chunks pertinents
        chunks = self._retrieve_chunks(processed_query)
        metadata["num_chunks"] = len(chunks)
        
        # 3. Formatage du prompt avec le template
        prompt = self._format_prompt(processed_query, chunks, **kwargs)
        
        # 4. Génération de la réponse avec le LLM
        llm_response = self._generate_response(prompt)
        
        # 5. Post-traitement de la réponse
        if self.postprocessing_func:
            llm_response.content = self.postprocessing_func(llm_response.content)
        
        # Calcul du temps total
        total_time = time.time() - start_time
        
        # Création du résultat
        result = ChainResult(
            llm_response=llm_response,
            retrieved_chunks=chunks,
            query=query,
            total_time=total_time,
            metadata=metadata
        )
        
        logger.info(f"RAGChain exécutée en {total_time:.2f}s: {result}")
        return result
    
    def _preprocess_query(self, query: str) -> str:
        """
        Prétraite la requête utilisateur.
        
        Args:
            query: Requête originale
            
        Returns:
            Requête prétraitée
        """
        if self.preprocessing_func:
            return self.preprocessing_func(query)
        return query
    
    def _retrieve_chunks(self, query: str) -> List[Dict[str, Any]]:
        """
        Récupère les chunks pertinents pour la requête.
        
        Args:
            query: Requête prétraitée
            
        Returns:
            Liste des chunks pertinents
        """
        return self.retriever.retrieve(query, top_k=self.max_chunks)
    
    def _format_prompt(self, 
                      query: str, 
                      chunks: List[Dict[str, Any]], 
                      **kwargs) -> str:
        """
        Formate le prompt en utilisant le template et les chunks récupérés.
        
        Args:
            query: Requête prétraitée
            chunks: Chunks récupérés
            **kwargs: Variables additionnelles pour le template
            
        Returns:
            Prompt formaté
        """
        template_vars = {
            "question": query,
            "query": query,
            "context": chunks,
            "language": self.language,
            **kwargs
        }
        
        return self.prompt_template.format(**template_vars)
    
    def _generate_response(self, prompt: str) -> LLMResponse:
        """
        Génère une réponse en utilisant le LLM.
        
        Args:
            prompt: Prompt formaté
            
        Returns:
            Réponse du LLM
        """
        return self.llm.generate(prompt, system_prompt=self.system_prompt)


class QAChain(RAGChain):
    """
    Chaîne spécialisée pour les questions-réponses.
    """
    
    def __init__(self,
                 llm: BaseLLM,
                 retriever: DocumentRetriever,
                 prompt_template: Optional[Union[str, PromptTemplate]] = None,
                 **kwargs):
        """
        Initialise une chaîne de question-réponse.
        
        Args:
            llm: Modèle de langage
            retriever: Récupérateur de documents
            prompt_template: Template de prompt (utilise 'qa' par défaut)
            **kwargs: Arguments additionnels pour RAGChain
        """
        # Utiliser le template 'qa' par défaut si non spécifié
        if prompt_template is None:
            prompt_template = "qa"
        
        # Système prompt par défaut pour QA
        system_prompt = kwargs.pop("system_prompt", "Vous êtes un assistant spécialisé dans la réponse à des questions basées uniquement sur le contexte fourni. Soyez précis et direct.")
        
        super().__init__(
            llm=llm,
            retriever=retriever,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            **kwargs
        )


class SummaryChain(RAGChain):
    """
    Chaîne spécialisée pour la génération de résumés.
    """
    
    def __init__(self,
                 llm: BaseLLM,
                 retriever: DocumentRetriever,
                 prompt_template: Optional[Union[str, PromptTemplate]] = None,
                 summary_length: str = "environ 250 mots",
                 summary_style: str = "informatif et neutre",
                 **kwargs):
        """
        Initialise une chaîne de résumé.
        
        Args:
            llm: Modèle de langage
            retriever: Récupérateur de documents
            prompt_template: Template de prompt (utilise 'summary' par défaut)
            summary_length: Longueur souhaitée pour le résumé
            summary_style: Style souhaité pour le résumé
            **kwargs: Arguments additionnels pour RAGChain
        """
        # Utiliser le template 'summary' par défaut si non spécifié
        if prompt_template is None:
            prompt_template = "summary"
        
        # Système prompt par défaut pour les résumés
        system_prompt = kwargs.pop("system_prompt", "Vous êtes un assistant spécialisé dans la création de résumés clairs et concis. Concentrez-vous sur les points essentiels.")
        
        super().__init__(
            llm=llm,
            retriever=retriever,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            **kwargs
        )
        
        self.summary_length = summary_length
        self.summary_style = summary_style
    
    def run(self, query: str = "Résumez ce document", **kwargs) -> ChainResult:
        """
        Exécute la chaîne de résumé.
        
        Args:
            query: Requête (utilisée principalement pour la récupération)
            **kwargs: Paramètres additionnels
            
        Returns:
            Résultat de la chaîne
        """
        # Ajouter les paramètres spécifiques au résumé
        return super().run(
            query=query,
            length=kwargs.get("length", self.summary_length),
            style=kwargs.get("style", self.summary_style),
            **kwargs
        )
    
    def _retrieve_chunks(self, query: str) -> List[Dict[str, Any]]:
        """
        Pour les résumés, récupère généralement plus de chunks
        et les trie par ordre (si possible).
        
        Args:
            query: Requête
            
        Returns:
            Liste de chunks
        """
        # Récupérer un nombre plus important de chunks pour un résumé
        chunks = self.retriever.retrieve(query, top_k=self.max_chunks * 2)
        
        # Essayer de trier par ordre logique (page, chunk_index, etc.) si possible
        try:
            # D'abord par page si disponible
            if all('page' in chunk for chunk in chunks):
                chunks = sorted(chunks, key=lambda x: x.get('page', 0))
            # Ensuite par chunk_index si disponible
            elif all('chunk_index' in chunk for chunk in chunks):
                chunks = sorted(chunks, key=lambda x: x.get('chunk_index', 0))
        except Exception as e:
            logger.warning(f"Impossible de trier les chunks par ordre: {str(e)}")
        
        # Limiter au nombre maximum de chunks
        return chunks[:self.max_chunks]


class ExtractionChain(RAGChain):
    """
    Chaîne spécialisée pour l'extraction d'informations structurées.
    """
    
    def __init__(self,
                 llm: BaseLLM,
                 retriever: DocumentRetriever,
                 prompt_template: Optional[Union[str, PromptTemplate]] = None,
                 items_to_extract: List[str] = None,
                 output_format: str = "JSON",
                 **kwargs):
        """
        Initialise une chaîne d'extraction d'informations.
        
        Args:
            llm: Modèle de langage
            retriever: Récupérateur de documents
            prompt_template: Template de prompt (utilise 'extraction' par défaut)
            items_to_extract: Liste des informations à extraire
            output_format: Format de sortie (JSON, YAML, etc.)
            **kwargs: Arguments additionnels pour RAGChain
        """
        # Utiliser le template 'extraction' par défaut si non spécifié
        if prompt_template is None:
            prompt_template = "extraction"
        
        # Système prompt par défaut pour l'extraction
        system_prompt = kwargs.pop("system_prompt", "Vous êtes un assistant spécialisé dans l'extraction précise d'informations. Suivez strictement le format demandé.")
        
        # Liste par défaut des éléments à extraire si non spécifiée
        self.items_to_extract = items_to_extract or ["titre", "auteur", "date", "mots-clés"]
        self.output_format = output_format
        
        super().__init__(
            llm=llm,
            retriever=retriever,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            **kwargs
        )
    
    def run(self, query: str = "Extrayez les informations demandées", **kwargs) -> ChainResult:
        """
        Exécute la chaîne d'extraction.
        
        Args:
            query: Requête (utilisée principalement pour la récupération)
            **kwargs: Paramètres additionnels
            
        Returns:
            Résultat de la chaîne
        """
        # Extraire ou utiliser les paramètres spécifiques
        items = kwargs.pop("items_to_extract", self.items_to_extract)
        format_type = kwargs.pop("format", self.output_format)
        
        # Exécuter la chaîne avec les paramètres spécifiques
        result = super().run(
            query=query,
            items_to_extract=items,
            format=format_type,
            **kwargs
        )
        
        # Essayer de parser la réponse si elle est au format JSON
        if format_type.upper() == "JSON":
            try:
                # Extraire uniquement la partie JSON si la réponse contient du texte explicatif
                json_text = self._extract_json_from_text(result.content)
                parsed_json = json.loads(json_text)
                # Ajouter le résultat parsé aux métadonnées
                result.metadata["parsed_result"] = parsed_json
            except json.JSONDecodeError as e:
                logger.warning(f"Impossible de parser la réponse JSON: {str(e)}")
        
        return result
    
    def _extract_json_from_text(self, text: str) -> str:
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


# Fonction utilitaire pour créer une chaîne RAG complète
def create_rag_chain(
    chain_type: str,
    llm_config: Union[LLMConfig, str],
    retriever: DocumentRetriever,
    prompt_registry: Optional[PromptTemplateRegistry] = None,
    **kwargs
) -> RAGChain:
    """
    Crée une chaîne RAG complète avec la configuration spécifiée.
    
    Args:
        chain_type: Type de chaîne ('qa', 'summary', 'extraction')
        llm_config: Configuration du LLM ou nom d'une config prédéfinie
        retriever: Récupérateur de documents
        prompt_registry: Registre de templates (optionnel)
        **kwargs: Paramètres spécifiques à la chaîne
        
    Returns:
        Instance de RAGChain configurée
    """
    # Obtenir la configuration LLM
    if isinstance(llm_config, str):
        from .models import get_default_config
        config = get_default_config(llm_config)
    else:
        config = llm_config
    
    # Créer le LLM
    from .models import create_llm
    llm = create_llm(config)
    
    # Créer la chaîne appropriée
    if chain_type.lower() == 'qa':
        return QAChain(llm=llm, retriever=retriever, prompt_registry=prompt_registry, **kwargs)
    elif chain_type.lower() == 'summary':
        return SummaryChain(llm=llm, retriever=retriever, prompt_registry=prompt_registry, **kwargs)
    elif chain_type.lower() == 'extraction':
        return ExtractionChain(llm=llm, retriever=retriever, prompt_registry=prompt_registry, **kwargs)
    else:
        # Chaîne RAG générique
        return RAGChain(llm=llm, retriever=retriever, prompt_registry=prompt_registry, **kwargs)