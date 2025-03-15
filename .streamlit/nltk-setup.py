import nltk
import os

# Créer le répertoire de données NLTK s'il n'existe pas
os.makedirs('/home/appuser/nltk_data', exist_ok=True)

# Télécharger les ressources nécessaires
nltk.download('punkt', download_dir='/home/appuser/nltk_data')