"""
Module pour l'extraction de texte à partir de différents formats de documents.
Supporte les formats PDF, DOCX, TXT sans dépendance à textract.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Import des bibliothèques pour différents formats de documents
import PyPDF2
import docx
import docx2txt
from pdfminer.high_level import extract_text as pdf_extract_text

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentParser:
    """
    Classe pour extraire le texte de différents formats de documents
    """
    
    def __init__(self):
        """Initialise le parser de documents"""
        logger.info("Initialisation du DocumentParser")
        
        # Formats supportés et leurs extensions
        self.supported_formats = {
            'pdf': ['.pdf'],
            'word': ['.docx', '.doc'],
            'text': ['.txt'],
        }
        
        # Liste de toutes les extensions supportées
        self.supported_extensions = []
        for ext_list in self.supported_formats.values():
            self.supported_extensions.extend(ext_list)
            
        logger.info(f"Formats supportés: {self.supported_extensions}")
    
    def parse(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse un document et extrait son texte et ses métadonnées
        
        Args:
            file_path: Chemin vers le fichier à parser
            
        Returns:
            Dictionnaire contenant le texte extrait et les métadonnées
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Le fichier n'existe pas: {file_path}")
            raise FileNotFoundError(f"Le fichier n'existe pas: {file_path}")
        
        # Vérifier si le format est supporté
        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            logger.error(f"Format de fichier non supporté: {extension}")
            raise ValueError(f"Format de fichier non supporté: {extension}")
        
        # Déterminer la méthode de parsing en fonction de l'extension
        if extension in self.supported_formats['pdf']:
            return self._parse_pdf(file_path)
        elif extension in self.supported_formats['word']:
            return self._parse_docx(file_path)
        elif extension in self.supported_formats['text']:
            return self._parse_txt(file_path)
        else:
            # Ce cas ne devrait jamais arriver si la vérification ci-dessus est correcte
            logger.error(f"Format non géré: {extension}")
            raise ValueError(f"Format non géré: {extension}")
    
    def _parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse un fichier PDF en utilisant PyPDF2 et pdfminer.six
        
        Args:
            file_path: Chemin vers le fichier PDF
            
        Returns:
            Dictionnaire contenant le texte extrait et les métadonnées du PDF
        """
        logger.info(f"Parsing du fichier PDF: {file_path}")
        
        try:
            # Utiliser PyPDF2 pour les métadonnées
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Extraction des métadonnées
                metadata = reader.metadata
                num_pages = len(reader.pages)
            
            # Extraction page par page avec PyPDF2
            pages = []
            full_text = ""
            
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    pages.append({
                        'page_num': i + 1,
                        'text': text
                    })
                    full_text += text + "\n\n"
            
            # Si le texte extrait est faible, essayer pdfminer
            if len(full_text.strip()) < num_pages * 50:
                try:
                    full_text = pdf_extract_text(str(file_path))
                    logger.info("Utilisation de pdfminer.six pour l'extraction du texte")
                except Exception as e:
                    logger.warning(f"Échec de l'extraction avec pdfminer: {str(e)}")
            
            result = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_type': 'pdf',
                'num_pages': num_pages,
                'metadata': {
                    'title': metadata.get('/Title', ''),
                    'author': metadata.get('/Author', ''),
                    'creator': metadata.get('/Creator', ''),
                    'producer': metadata.get('/Producer', ''),
                    'creation_date': metadata.get('/CreationDate', ''),
                },
                'full_text': full_text,
                'pages': pages
            }
            
            logger.info(f"PDF parsé avec succès: {num_pages} pages")
            return result
                
        except Exception as e:
            logger.error(f"Erreur lors du parsing du PDF: {str(e)}")
            raise
    
    def _parse_docx(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse un fichier DOCX en utilisant python-docx et docx2txt
        
        Args:
            file_path: Chemin vers le fichier DOCX
            
        Returns:
            Dictionnaire contenant le texte extrait et les métadonnées du DOCX
        """
        logger.info(f"Parsing du fichier DOCX: {file_path}")
        
        try:
            # Utiliser python-docx pour les métadonnées et la structure
            doc = docx.Document(file_path)
            
            # Extraction du texte paragraphe par paragraphe
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():  # Ignorer les paragraphes vides
                    paragraphs.append(para.text)
            
            # Extraction du texte avec docx2txt
            full_text = docx2txt.process(str(file_path))
            
            # Extraction des métadonnées
            core_properties = doc.core_properties
            
            result = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_type': 'docx',
                'metadata': {
                    'title': core_properties.title or '',
                    'author': core_properties.author or '',
                    'created': str(core_properties.created) if core_properties.created else '',
                    'modified': str(core_properties.modified) if core_properties.modified else '',
                    'last_modified_by': core_properties.last_modified_by or '',
                },
                'full_text': full_text,
                'paragraphs': paragraphs
            }
            
            logger.info(f"DOCX parsé avec succès: {len(paragraphs)} paragraphes")
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du parsing du DOCX: {str(e)}")
            raise
    
    def _parse_txt(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse un fichier texte
        
        Args:
            file_path: Chemin vers le fichier texte
            
        Returns:
            Dictionnaire contenant le texte extrait
        """
        logger.info(f"Parsing du fichier texte: {file_path}")
        
        try:
            # Essayer différents encodages
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    break  # Si on arrive ici, l'encodage a fonctionné
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                logger.warning("Aucun encodage n'a fonctionné, utilisation de l'encodage par défaut avec erreurs ignorées")
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
            
            # Diviser le contenu en lignes
            lines = content.splitlines()
            
            result = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_type': 'txt',
                'metadata': {
                    'size': os.path.getsize(file_path),
                    'modified': str(os.path.getmtime(file_path)),
                },
                'full_text': content,
                'lines': lines
            }
            
            logger.info(f"Fichier texte parsé avec succès: {len(lines)} lignes")
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du parsing du fichier texte: {str(e)}")
            raise
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Renvoie la liste des formats supportés
        
        Returns:
            Dictionnaire des formats supportés et leurs extensions
        """
        return self.supported_formats