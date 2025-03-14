"""
Module pour l'intégration des modèles de langage (LLM) dans l'application.
Limité aux modèles GitHub Inference et Hugging Face.
"""

# Import des classes et fonctions principales
from .models import (
    LLMConfig, 
    LLMProvider, 
    LLMResponse,
    BaseLLM,
    GitHubInferenceAPILLM,
    HuggingFaceLLM,
    create_llm,
    get_default_config,
    DEFAULT_CONFIGS
)

from .prompts import (
    PromptTemplate,
    PromptTemplateRegistry,
    DEFAULT_TEMPLATES
)

from .chain import (
    RAGChain,
    QAChain,
    SummaryChain,
    ExtractionChain,
    create_rag_chain
)

# Exposer les classes principales
__all__ = [
    'LLMConfig', 'LLMProvider', 'LLMResponse',
    'BaseLLM', 'GitHubInferenceAPILLM', 'HuggingFaceLLM', 
    'create_llm', 'get_default_config', 'DEFAULT_CONFIGS',
    'PromptTemplate', 'PromptTemplateRegistry', 'DEFAULT_TEMPLATES',
    'RAGChain', 'QAChain', 'SummaryChain', 'ExtractionChain', 'create_rag_chain'
]