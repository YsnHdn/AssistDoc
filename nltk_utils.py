"""
Utilitaires pour gérer NLTK sur Streamlit Cloud.
"""

import os
import nltk
from functools import wraps

# Sauvegarde de la fonction originale
original_find = nltk.data.find

# Fonction de remplacement qui accepte les mêmes arguments que l'original
@wraps(original_find)
def patched_find(resource_name, paths=None):
    """
    Version corrigée de nltk.data.find qui fonctionne avec Streamlit Cloud.
    Accepte le paramètre paths correctement.
    
    Args:
        resource_name: Nom de la ressource NLTK
        paths: Chemins alternatifs où chercher (optionnel)
        
    Returns:
        Chemin vers la ressource
    """
    try:
        return original_find(resource_name, paths)
    except LookupError:
        # Télécharger automatiquement la ressource manquante
        nltk.download(resource_name.split('/')[-1])
        # Réessayer après le téléchargement
        return original_find(resource_name, paths)

# Remplacer la fonction originale par notre version
nltk.data.find = patched_find