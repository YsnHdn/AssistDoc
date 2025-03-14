"""
Module pour la gestion des templates de prompts pour différentes tâches LLM.
Utilise Jinja2 pour créer des templates modulaires et personnalisables pour
la question-réponse, le résumé, l'extraction d'information, etc.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pathlib import Path
import json

# Import de Jinja2 pour les templates
try:
    import jinja2
    JINJA_AVAILABLE = True
except ImportError:
    JINJA_AVAILABLE = False

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptTemplate:
    """
    Classe pour gérer les templates de prompts avec Jinja2.
    """
    
    def __init__(self, 
                 template: str,
                 template_type: str = "generic",
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialise un template de prompt.
        
        Args:
            template: Texte du template au format Jinja2
            template_type: Type de template (qa, summary, extraction, etc.)
            metadata: Métadonnées associées au template
        """
        if not JINJA_AVAILABLE:
            raise ImportError("Le package 'jinja2' n'est pas installé. "
                             "Installez-le avec 'pip install jinja2'.")
        
        self.template_text = template
        self.template_type = template_type
        self.metadata = metadata or {}
        
        # Créer l'environnement Jinja et compiler le template
        self.env = jinja2.Environment(
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        self.template = self.env.from_string(template)
        
        logger.info(f"Template de prompt '{template_type}' initialisé")
    
    def format(self, **kwargs) -> str:
        """
        Remplit le template avec les variables fournies.
        
        Args:
            **kwargs: Variables à insérer dans le template
            
        Returns:
            Prompt formaté
        """
        try:
            prompt = self.template.render(**kwargs)
            return prompt
        except Exception as e:
            logger.error(f"Erreur lors du formatage du template: {str(e)}")
            logger.error(f"Variables fournies: {kwargs}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit le template en dictionnaire.
        
        Returns:
            Dictionnaire représentant le template
        """
        return {
            "template_text": self.template_text,
            "template_type": self.template_type,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """
        Crée un template à partir d'un dictionnaire.
        
        Args:
            data: Dictionnaire contenant les données du template
            
        Returns:
            Instance de PromptTemplate
        """
        return cls(
            template=data["template_text"],
            template_type=data.get("template_type", "generic"),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "PromptTemplate":
        """
        Charge un template depuis un fichier.
        
        Args:
            file_path: Chemin vers le fichier de template
            
        Returns:
            Instance de PromptTemplate
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Le fichier de template n'existe pas: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Essayer de détecter les métadonnées au format JSON en tête de fichier
        metadata = {}
        template_type = file_path.stem  # Nom du fichier sans extension
        
        # Rechercher les métadonnées entre --- (format YAML/JSON frontmatter)
        frontmatter_match = re.match(r'---\s+(.*?)\s+---\s+(.*)', content, re.DOTALL)
        if frontmatter_match:
            try:
                metadata_str = frontmatter_match.group(1)
                metadata = json.loads(metadata_str)
                content = frontmatter_match.group(2)
                
                # Extraire le type du template s'il est spécifié
                if "type" in metadata:
                    template_type = metadata.pop("type")
            except json.JSONDecodeError:
                # Ignorer les erreurs de parsing JSON
                logger.warning(f"Erreur lors du parsing des métadonnées du template: {file_path}")
        
        return cls(
            template=content,
            template_type=template_type,
            metadata=metadata
        )
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Sauvegarde le template dans un fichier.
        
        Args:
            file_path: Chemin où sauvegarder le template
        """
        file_path = Path(file_path)
        
        # Créer le répertoire parent si nécessaire
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Préparer le contenu avec les métadonnées
        if self.metadata:
            metadata_str = json.dumps(self.metadata, indent=2)
            content = f"---\n{metadata_str}\n---\n\n{self.template_text}"
        else:
            content = self.template_text
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Template sauvegardé dans: {file_path}")


class PromptTemplateRegistry:
    """
    Registre pour gérer plusieurs templates de prompts.
    """
    
    def __init__(self, templates_dir: Optional[Union[str, Path]] = None):
        """
        Initialise le registre de templates.
        
        Args:
            templates_dir: Répertoire contenant les fichiers de templates à charger
        """
        self.templates: Dict[str, PromptTemplate] = {}
        
        # Charger les templates depuis le répertoire si spécifié
        if templates_dir:
            self.load_templates_from_directory(templates_dir)
    
    def register(self, name: str, template: PromptTemplate) -> None:
        """
        Enregistre un template dans le registre.
        
        Args:
            name: Nom du template
            template: Instance de PromptTemplate
        """
        self.templates[name] = template
        logger.info(f"Template '{name}' enregistré dans le registre")
    
    def get(self, name: str) -> PromptTemplate:
        """
        Récupère un template par son nom.
        
        Args:
            name: Nom du template
            
        Returns:
            Template correspondant
            
        Raises:
            KeyError: Si le template n'existe pas
        """
        if name not in self.templates:
            raise KeyError(f"Template non trouvé: {name}")
        
        return self.templates[name]
    
    def format(self, name: str, **kwargs) -> str:
        """
        Formate un template avec les variables fournies.
        
        Args:
            name: Nom du template
            **kwargs: Variables à insérer dans le template
            
        Returns:
            Prompt formaté
        """
        template = self.get(name)
        return template.format(**kwargs)
    
    def list_templates(self) -> List[str]:
        """
        Liste tous les templates disponibles.
        
        Returns:
            Liste des noms de templates
        """
        return list(self.templates.keys())
    
    def load_templates_from_directory(self, directory: Union[str, Path]) -> None:
        """
        Charge tous les templates depuis un répertoire.
        
        Args:
            directory: Chemin vers le répertoire contenant les templates
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Le répertoire de templates n'existe pas: {directory}")
            return
        
        # Charger tous les fichiers avec extensions .txt, .j2, .tmpl, .prompt
        extensions = ['.txt', '.j2', '.tmpl', '.prompt']
        for ext in extensions:
            for file_path in directory.glob(f"*{ext}"):
                try:
                    template = PromptTemplate.from_file(file_path)
                    template_name = file_path.stem
                    self.register(template_name, template)
                except Exception as e:
                    logger.error(f"Erreur lors du chargement du template {file_path}: {str(e)}")
        
        logger.info(f"Chargé {len(self.templates)} templates depuis {directory}")
    
    def save_templates_to_directory(self, directory: Union[str, Path]) -> None:
        """
        Sauvegarde tous les templates dans un répertoire.
        
        Args:
            directory: Chemin vers le répertoire où sauvegarder les templates
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for name, template in self.templates.items():
            file_path = directory / f"{name}.tmpl"
            template.save_to_file(file_path)
        
        logger.info(f"Sauvegardé {len(self.templates)} templates dans {directory}")


# Templates prédéfinis pour les tâches courantes
QA_TEMPLATE = """
Vous êtes un assistant intelligent qui aide à répondre aux questions en utilisant uniquement les informations fournies dans le contexte ci-dessous.

CONTEXTE:
{% for chunk in context %}
---
{{ chunk.text }}
---
{% endfor %}

INSTRUCTIONS:
- Utilisez UNIQUEMENT les informations du CONTEXTE pour répondre
- Si le CONTEXTE ne contient pas l'information, dites clairement "Je ne trouve pas cette information dans le contexte fourni"
- Ne fabriquez pas d'informations ou de sources
- Citez les portions pertinentes du contexte dans votre réponse
- Répondez en {{ language }}

QUESTION: {{ question }}

RÉPONSE:
"""

SUMMARY_TEMPLATE = """
Vous êtes un assistant spécialisé dans la création de résumés pertinents et concis.

DOCUMENT À RÉSUMER:
{% for chunk in context %}
{{ chunk.text }}
{% endfor %}

INSTRUCTIONS:
- Créez un résumé clair et concis du document ci-dessus
- Longueur du résumé: {{ length|default('environ 250 mots') }}
- Style: {{ style|default('informatif et neutre') }}
- Mettez en évidence les points clés, les arguments principaux et les conclusions
- Préservez le ton et le point de vue du document original
- N'introduisez pas d'informations qui ne seraient pas dans le document
- Rédigez en {{ language }}

RÉSUMÉ:
"""

EXTRACTION_TEMPLATE = """
Vous êtes un assistant spécialisé dans l'extraction précise d'informations structurées à partir de documents.

DOCUMENT:
{% for chunk in context %}
{{ chunk.text }}
{% endfor %}

INSTRUCTIONS:
- Extrayez les informations demandées du document ci-dessus
- Format de sortie: {{ format|default('JSON') }}
- Éléments à extraire:
{% for item in items_to_extract %}
  - {{ item }}{% endfor %}
- Si une information n'est pas présente, indiquez-le par "Non spécifié" ou "N/A"
- Ne faites pas de suppositions sur des informations manquantes
- Soyez précis et fidèle au texte d'origine
- Répondez en {{ language }}

INFORMATIONS EXTRAITES:
"""

MULTI_DOCUMENT_TEMPLATE = """
Vous êtes un assistant spécialisé dans l'analyse et la comparaison de plusieurs documents.

DOCUMENTS:
{% for doc in documents %}
DOCUMENT {{ loop.index }} - {{ doc.title }}:
{{ doc.text }}

{% endfor %}

INSTRUCTIONS:
- {{ instruction }}
- Identifiez les similitudes et différences entre les documents
- Citez des parties spécifiques des documents pour appuyer votre analyse
- Structurez votre réponse de manière claire et organisée
- Répondez en {{ language }}

ANALYSE:
"""

EVALUATION_TEMPLATE = """
Vous êtes un évaluateur impartial chargé d'analyser la qualité et la pertinence d'une réponse à une question.

QUESTION POSÉE:
{{ question }}

CONTEXTE FOURNI:
{% for chunk in context %}
---
{{ chunk.text }}
---
{% endfor %}

RÉPONSE À ÉVALUER:
{{ response }}

INSTRUCTIONS D'ÉVALUATION:
- Évaluez si la réponse est basée uniquement sur les informations du contexte
- Vérifiez l'exactitude et la pertinence de la réponse par rapport à la question
- Identifiez toute information fabricée ou non présente dans le contexte
- Évaluez la clarté et la structure de la réponse
- Notez la réponse sur une échelle de 1 à 10 et justifiez votre notation
- Répondez en {{ language }}

ÉVALUATION:
"""

CUSTOM_INSTRUCTION_TEMPLATE = """
Vous êtes un assistant intelligent qui aide les utilisateurs avec leurs documents.

CONTEXTE:
{% for chunk in context %}
---
{{ chunk.text }}
---
{% endfor %}

INSTRUCTIONS PERSONNALISÉES:
{{ custom_instruction }}

{% if additional_context %}
INFORMATIONS SUPPLÉMENTAIRES:
{{ additional_context }}
{% endif %}

Répondez en {{ language }}

RÉPONSE:
"""

# Initialiser le registre avec les templates prédéfinis
DEFAULT_TEMPLATES = {
    "qa": PromptTemplate(QA_TEMPLATE, "qa", {"description": "Template pour questions-réponses"}),
    "summary": PromptTemplate(SUMMARY_TEMPLATE, "summary", {"description": "Template pour résumés"}),
    "extraction": PromptTemplate(EXTRACTION_TEMPLATE, "extraction", {"description": "Template pour extraction d'information"}),
    "multi_document": PromptTemplate(MULTI_DOCUMENT_TEMPLATE, "multi_document", {"description": "Template pour analyse multi-documents"}),
    "evaluation": PromptTemplate(EVALUATION_TEMPLATE, "evaluation", {"description": "Template pour évaluation de réponses"}),
    "custom": PromptTemplate(CUSTOM_INSTRUCTION_TEMPLATE, "custom", {"description": "Template avec instructions personnalisées"})
}