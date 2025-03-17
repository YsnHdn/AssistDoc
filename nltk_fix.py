import os
import nltk
import shutil
import logging

def ensure_punkt_resources():
    """Assure que les ressources NLTK punkt et punkt_tab sont disponibles."""
    logger = logging.getLogger("nltk_fix")
    
    # Télécharger punkt
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK punkt déjà disponible")
    except LookupError:
        logger.info("Téléchargement de NLTK punkt...")
        nltk.download('punkt')
        logger.info("NLTK punkt téléchargé")
    
    # Gérer punkt_tab
    try:
        nltk.data.find('tokenizers/punkt_tab')
        logger.info("NLTK punkt_tab déjà disponible")
    except LookupError:
        logger.info("punkt_tab non trouvé, création d'une solution de contournement...")
        
        # Trouver le chemin de punkt
        try:
            punkt_path = nltk.data.find('tokenizers/punkt')
            logger.info(f"Chemin de punkt trouvé: {punkt_path}")
            
            # Extraire le répertoire parent
            punkt_dir = os.path.dirname(punkt_path)
            
            # Chemin pour punkt_tab
            punkt_tab_dir = os.path.join(os.path.dirname(punkt_dir), 'punkt_tab')
            
            # Créer le répertoire punkt_tab s'il n'existe pas
            os.makedirs(punkt_tab_dir, exist_ok=True)
            
            # Copier les fichiers de punkt vers punkt_tab
            for item in os.listdir(punkt_dir):
                src = os.path.join(punkt_dir, item)
                dst = os.path.join(punkt_tab_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            
            logger.info(f"punkt_tab créé avec succès à {punkt_tab_dir}")
        except Exception as e:
            logger.error(f"Erreur lors de la création de punkt_tab: {str(e)}")