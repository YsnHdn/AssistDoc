"""
Module pour découper les documents en chunks plus petits.
Propose différentes stratégies de découpage: par paragraphes, par phrases,
par taille fixe, ou par segmentation sémantique.
"""

import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Callable, Union
import nltk
from nltk.tokenize import sent_tokenize

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Classe pour découper des documents en chunks plus petits
    selon différentes stratégies.
    """
    
    def __init__(self):
        """Initialise le chunker de documents"""
        logger.info("Initialisation du DocumentChunker")
        
        # S'assurer que les ressources NLTK nécessaires sont disponibles
        try:
            nltk.data.find('tokenizers/punkt')
            logger.info("Ressources NLTK 'punkt' déjà disponibles")
        except LookupError:
            logger.info("Téléchargement des ressources NLTK 'punkt'...")
            nltk.download('punkt')
            logger.info("Ressources NLTK téléchargées avec succès")
    
    def chunk_document(self,
                       document: Dict[str, Any],
                       strategy: str = 'paragraph',
                       chunk_size: int = 1000,
                       chunk_overlap: int = 200,
                       min_chunk_length: int = 50) -> List[Dict[str, Any]]:
        """
        Découpe un document en chunks selon la stratégie spécifiée
        
        Args:
            document: Document parsé à découper (output du DocumentParser)
            strategy: Stratégie de découpage ('paragraph', 'sentence', 'fixed', 'semantic')
            chunk_size: Taille maximale des chunks (en caractères)
            chunk_overlap: Chevauchement entre les chunks adjacents (en caractères)
            min_chunk_length: Longueur minimale d'un chunk (en caractères)
            
        Returns:
            Liste de dictionnaires représentant les chunks du document
        """
        logger.info(f"Découpage du document avec stratégie: {strategy}")
        
        if not document:
            logger.warning("Document vide fourni pour le chunking")
            return []
        
        # Sélectionner la fonction de chunking appropriée
        if strategy == 'paragraph':
            chunks = self._chunk_by_paragraph(document, min_chunk_length)
        elif strategy == 'sentence':
            chunks = self._chunk_by_sentence(document, min_chunk_length)
        elif strategy == 'fixed':
            chunks = self._chunk_by_fixed_size(document, chunk_size, chunk_overlap, min_chunk_length)
        elif strategy == 'semantic':
            chunks = self._chunk_by_semantic(document, chunk_size, chunk_overlap, min_chunk_length)
        else:
            logger.error(f"Stratégie de chunking non reconnue: {strategy}")
            raise ValueError(f"Stratégie de chunking non reconnue: {strategy}")
        
        # Ajouter des metadonnées aux chunks
        for i, chunk in enumerate(chunks):
            # Générer un ID unique pour chaque chunk
            chunk_id = str(uuid.uuid4())
            
            # Ajouter des métadonnées communes à tous les chunks
            chunk.update({
                'chunk_id': chunk_id,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'file_name': document.get('file_name', ''),
                'file_path': document.get('file_path', ''),
                'file_type': document.get('file_type', ''),
                'strategy': strategy
            })
        
        logger.info(f"Document découpé en {len(chunks)} chunks")
        return chunks
    
    def _chunk_by_paragraph(self, document: Dict[str, Any], min_length: int) -> List[Dict[str, Any]]:
        """
        Découpe le document en paragraphes
        
        Args:
            document: Document parsé
            min_length: Longueur minimale d'un paragraphe à considérer
            
        Returns:
            Liste de chunks basés sur les paragraphes
        """
        chunks = []
        
        # Pour les documents PDF
        if document.get('file_type') == 'pdf':
            for page in document.get('pages', []):
                page_num = page.get('page_num', 0)
                text = page.get('text', '')
                
                # Diviser en paragraphes
                paragraphs = re.split(r'\n\s*\n', text)
                
                for para_index, para in enumerate(paragraphs):
                    para = para.strip()
                    if len(para) >= min_length:
                        chunks.append({
                            'text': para,
                            'page': page_num,
                            'paragraph_index': para_index,
                        })
        
        # Pour les documents DOCX
        elif document.get('file_type') == 'docx':
            paragraphs = document.get('paragraphs', [])
            for para_index, para in enumerate(paragraphs):
                if len(para) >= min_length:
                    chunks.append({
                        'text': para,
                        'paragraph_index': para_index,
                    })
        
        # Pour les fichiers texte
        elif document.get('file_type') == 'txt':
            text = document.get('full_text', '')
            paragraphs = re.split(r'\n\s*\n', text)
            
            for para_index, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) >= min_length:
                    chunks.append({
                        'text': para,
                        'paragraph_index': para_index,
                    })
        
        # Si aucun chunk n'a été créé, utiliser le texte complet
        if not chunks and document.get('full_text'):
            chunks.append({
                'text': document.get('full_text'),
                'paragraph_index': 0,
            })
            
        return chunks
    
    def _chunk_by_sentence(self, document: Dict[str, Any], min_length: int) -> List[Dict[str, Any]]:
        """
        Découpe le document en phrases
        
        Args:
            document: Document parsé
            min_length: Longueur minimale d'une phrase à considérer
            
        Returns:
            Liste de chunks basés sur les phrases
        """
        chunks = []
        
        # Récupérer le texte complet
        full_text = document.get('full_text', '')
        
        # Tokeniser en phrases
        sentences = sent_tokenize(full_text)
        
        # Créer un chunk pour chaque phrase suffisamment longue
        for sent_index, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) >= min_length:
                # Déterminer la page ou le paragraphe pour le contexte si disponible
                context = {}
                
                # Pour les PDF, essayer de déterminer la page de la phrase
                if document.get('file_type') == 'pdf':
                    for page in document.get('pages', []):
                        if sentence in page.get('text', ''):
                            context['page'] = page.get('page_num', 0)
                            break
                
                chunks.append({
                    'text': sentence,
                    'sentence_index': sent_index,
                    **context
                })
        
        return chunks
    
    def _chunk_by_fixed_size(self, 
                            document: Dict[str, Any], 
                            chunk_size: int, 
                            chunk_overlap: int,
                            min_length: int) -> List[Dict[str, Any]]:
        """
        Découpe le document en chunks de taille fixe avec chevauchement
        
        Args:
            document: Document parsé
            chunk_size: Taille maximale des chunks (en caractères)
            chunk_overlap: Chevauchement entre les chunks adjacents (en caractères)
            min_length: Longueur minimale d'un chunk à considérer
            
        Returns:
            Liste de chunks de taille fixe
        """
        chunks = []
        
        # Récupérer le texte complet
        full_text = document.get('full_text', '')
        
        # Vérifier que le chevauchement est inférieur à la taille du chunk
        if chunk_overlap >= chunk_size:
            logger.warning(f"Le chevauchement ({chunk_overlap}) est >= à la taille du chunk ({chunk_size}). Ajustement.")
            chunk_overlap = chunk_size // 2
        
        # Calculer le pas entre les débuts de chunks
        step = chunk_size - chunk_overlap
        
        # Création des chunks
        for i in range(0, len(full_text), step):
            chunk_text = full_text[i:i + chunk_size]
            
            # Ne garder que les chunks suffisamment longs
            if len(chunk_text) >= min_length:
                # Déterminer la page ou le paragraphe pour le contexte si disponible
                context = {}
                
                # Pour les PDF, essayer de déterminer la page du chunk
                if document.get('file_type') == 'pdf':
                    start_pos = i
                    end_pos = i + chunk_size
                    
                    # Trouver à quelle(s) page(s) appartient ce chunk
                    pages_covered = set()
                    current_pos = 0
                    
                    for page in document.get('pages', []):
                        page_text = page.get('text', '')
                        page_start = current_pos
                        page_end = current_pos + len(page_text)
                        
                        # Si le chunk chevauche cette page
                        if not (end_pos <= page_start or start_pos >= page_end):
                            pages_covered.add(page.get('page_num', 0))
                        
                        current_pos = page_end
                    
                    if pages_covered:
                        context['pages'] = list(pages_covered)
                        context['page'] = min(pages_covered)  # Page principale
                
                chunks.append({
                    'text': chunk_text,
                    'start_char': i,
                    'end_char': i + len(chunk_text),
                    **context
                })
        
        return chunks
    
    def _chunk_by_semantic(self, 
                          document: Dict[str, Any], 
                          chunk_size: int, 
                          chunk_overlap: int,
                          min_length: int) -> List[Dict[str, Any]]:
        """
        Découpe le document en chunks sémantiquement cohérents.
        Cette méthode combine le chunking par phrase et par taille fixe.
        
        Args:
            document: Document parsé
            chunk_size: Taille maximale des chunks (en caractères)
            chunk_overlap: Chevauchement entre les chunks adjacents (en caractères)
            min_length: Longueur minimale d'un chunk à considérer
            
        Returns:
            Liste de chunks sémantiquement cohérents
        """
        # D'abord, nous divisons le document en phrases
        sentence_chunks = self._chunk_by_sentence(document, 0)  # min_length = 0 car on va regrouper ensuite
        
        # Si pas de phrases trouvées, revenir à la méthode par taille fixe
        if not sentence_chunks:
            logger.warning("Pas de phrases trouvées, utilisation du chunking par taille fixe")
            return self._chunk_by_fixed_size(document, chunk_size, chunk_overlap, min_length)
        
        # Ensuite, regrouper les phrases en chunks de taille maximale chunk_size
        chunks = []
        current_chunk = {
            'text': '',
            'sentences': [],
            'sentence_indices': []
        }
        
        for sentence in sentence_chunks:
            sentence_text = sentence.get('text', '')
            
            # Si l'ajout de cette phrase dépasse la taille maximale et que le chunk n'est pas vide
            if len(current_chunk['text']) + len(sentence_text) > chunk_size and current_chunk['text']:
                # Sauvegarder le chunk actuel s'il est assez long
                if len(current_chunk['text']) >= min_length:
                    chunks.append(current_chunk)
                
                # Créer un nouveau chunk avec chevauchement (dernières phrases du chunk précédent)
                overlap_size = 0
                overlap_sentences = []
                overlap_indices = []
                
                # Ajouter des phrases du chunk précédent jusqu'à atteindre le chevauchement souhaité
                for i in range(len(current_chunk['sentences']) - 1, -1, -1):
                    if overlap_size >= chunk_overlap:
                        break
                    
                    overlap_sentences.insert(0, current_chunk['sentences'][i])
                    overlap_indices.insert(0, current_chunk['sentence_indices'][i])
                    overlap_size += len(current_chunk['sentences'][i])
                
                # Initialiser le nouveau chunk avec les phrases chevauchantes
                current_chunk = {
                    'text': ''.join(overlap_sentences),
                    'sentences': overlap_sentences.copy(),
                    'sentence_indices': overlap_indices.copy()
                }
            
            # Ajouter la phrase au chunk actuel
            current_chunk['text'] += sentence_text
            current_chunk['sentences'].append(sentence_text)
            current_chunk['sentence_indices'].append(sentence.get('sentence_index', -1))
        
        # Ajouter le dernier chunk s'il est assez long
        if current_chunk['text'] and len(current_chunk['text']) >= min_length:
            chunks.append(current_chunk)
        
        # Finaliser les chunks avec des métadonnées additionnelles
        result_chunks = []
        for chunk in chunks:
            # Déterminer la page ou le paragraphe pour le contexte si disponible
            context = {}
            
            # Pour les PDF, essayer de déterminer la page du chunk
            if document.get('file_type') == 'pdf':
                pages_covered = set()
                
                for sentence_index in chunk['sentence_indices']:
                    if 0 <= sentence_index < len(sentence_chunks):
                        if 'page' in sentence_chunks[sentence_index]:
                            pages_covered.add(sentence_chunks[sentence_index]['page'])
                
                if pages_covered:
                    context['pages'] = list(pages_covered)
                    context['page'] = min(pages_covered)  # Page principale
            
            result_chunks.append({
                'text': chunk['text'],
                'num_sentences': len(chunk['sentences']),
                'start_sentence': chunk['sentence_indices'][0] if chunk['sentence_indices'] else -1,
                'end_sentence': chunk['sentence_indices'][-1] if chunk['sentence_indices'] else -1,
                **context
            })
        
        return result_chunks