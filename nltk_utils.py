"""
Utilitaires pour gérer les problèmes potentiels avec NLTK.
"""

import re
import nltk
import logging

logger = logging.getLogger(__name__)

def simple_sentence_tokenize(text):
    """
    Fonction simple de tokenisation en phrases qui ne dépend pas de ressources externes.
    
    Args:
        text: Texte à découper en phrases
        
    Returns:
        Liste de phrases
    """
    # Expression régulière pour découper sur les ponctuations de fin de phrase
    # suivies d'un espace ou d'une nouvelle ligne
    if not text:
        return []
    
    # Nettoyer le texte
    text = text.replace('\n', ' ').strip()
    
    # Découper sur les points, points d'exclamation, points d'interrogation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filtrer les phrases vides
    return [s.strip() for s in sentences if s.strip()]

def safe_sent_tokenize(text, language='french'):
    """
    Version sécurisée de nltk.sent_tokenize qui utilise un fallback en cas d'erreur.
    
    Args:
        text: Texte à découper en phrases
        language: Langue du texte (par défaut: français)
        
    Returns:
        Liste de phrases
    """
    if not text:
        return []
    
    try:
        # Essayer d'utiliser NLTK
        return nltk.sent_tokenize(text, language=language)
    except Exception as e:
        logger.warning(f"Erreur lors de l'utilisation de nltk.sent_tokenize: {str(e)}")
        logger.info("Utilisation de la méthode de tokenisation simple comme fallback")
        return simple_sentence_tokenize(text)

# Monkeypatch pour nltk.data.find
try:
    original_find = nltk.data.find
    
    def patched_find(resource_name):
        """
        Version patchée de nltk.data.find qui redirige punkt_tab vers punkt.
        """
        if 'punkt_tab' in resource_name:
            # Rediriger vers punkt
            modified_resource = resource_name.replace('punkt_tab', 'punkt')
            return original_find(modified_resource)
        return original_find(resource_name)
    
    # Appliquer le patch
    nltk.data.find = patched_find
    logger.info("Patch appliqué à nltk.data.find pour rediriger punkt_tab vers punkt")
except Exception as e:
    logger.warning(f"Impossible d'appliquer le patch à nltk.data.find: {str(e)}")