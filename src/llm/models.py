"""
Module pour la configuration et la gestion des modèles de langage (LLMs).
Fournit une interface unifiée pour GitHub Inference API et Hugging Face.
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
import requests
from abc import ABC, abstractmethod

# Imports conditionnels pour les différents fournisseurs
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Énumération des fournisseurs de LLM supportés."""
    GITHUB_INFERENCE = "github_inference"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

class LLMConfig:
    """Configuration pour les modèles de langage."""
    
    def __init__(self,
                 provider: Union[LLMProvider, str],
                 model_name: str,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 top_p: float = 1.0,
                 timeout: int = 60,
                 max_retries: int = 3,
                 retry_delay: int = 2,
                 **kwargs):
        """
        Initialise la configuration du LLM.
        
        Args:
            provider: Fournisseur du LLM (GitHub Inference ou Hugging Face)
            model_name: Nom du modèle spécifique à utiliser
            api_key: Clé API pour accéder au LLM (peut aussi être définie dans les variables d'environnement)
            api_base: URL de base pour l'API (pour les API auto-hébergées)
            temperature: Niveau de créativité du modèle (0.0 à 1.0)
            max_tokens: Nombre maximum de tokens dans la réponse
            top_p: Échantillonnage du modèle (nucleus sampling)
            timeout: Délai d'attente maximum pour l'API (en secondes)
            max_retries: Nombre maximal de tentatives en cas d'échec
            retry_delay: Délai entre les tentatives (en secondes)
            **kwargs: Paramètres additionnels spécifiques au fournisseur
        """
        self.provider = provider if isinstance(provider, LLMProvider) else LLMProvider(provider.lower())
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_params = kwargs
        
        # Validation et chargement des API keys depuis les variables d'environnement si nécessaire
        self._validate_and_load_api_keys()
        
        logger.info(f"LLMConfig initialisée pour {self.provider.value}:{self.model_name}")
    
    def _validate_and_load_api_keys(self):
        """Valide et charge les clés API depuis les variables d'environnement si nécessaire."""
        if self.provider == LLMProvider.GITHUB_INFERENCE:
            env_key = os.environ.get("GITHUB_TOKEN")
            if not self.api_key and not env_key:
                logger.warning("Aucune clé API GitHub fournie et GITHUB_TOKEN non définie")
            elif not self.api_key:
                self.api_key = env_key
                
            if not self.api_base:
                self.api_base = "https://models.inference.ai.azure.com"
        
        # HuggingFace peut fonctionner sans API key pour les modèles locaux
        elif self.provider == LLMProvider.HUGGINGFACE:
            if not self.api_key:
                self.api_key = os.environ.get("HUGGINGFACE_API_KEY")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la configuration en dictionnaire.
        
        Returns:
            Dictionnaire contenant la configuration
        """
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            **self.extra_params
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        """
        Crée une instance de LLMConfig à partir d'un dictionnaire.
        
        Args:
            config_dict: Dictionnaire contenant la configuration
            
        Returns:
            Instance de LLMConfig
        """
        provider = config_dict.pop("provider")
        model_name = config_dict.pop("model_name")
        
        return cls(
            provider=provider,
            model_name=model_name,
            **config_dict
        )
    
    def clone(self, **kwargs) -> "LLMConfig":
        """
        Crée une copie de cette configuration avec les modifications spécifiées.
        
        Args:
            **kwargs: Paramètres à modifier dans la nouvelle configuration
            
        Returns:
            Nouvelle instance de LLMConfig avec les modifications
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        
        return LLMConfig.from_dict(config_dict)


class LLMResponse:
    """
    Encapsule la réponse d'un LLM avec des métadonnées utiles.
    """
    
    def __init__(self,
                 content: str,
                 provider: str,
                 model: str,
                 tokens_input: int,
                 tokens_output: int,
                 latency: float,
                 raw_response: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialise une réponse de LLM.
        
        Args:
            content: Texte généré par le LLM
            provider: Fournisseur du LLM
            model: Nom du modèle utilisé
            tokens_input: Nombre de tokens dans le prompt
            tokens_output: Nombre de tokens dans la réponse
            latency: Temps d'exécution (en secondes)
            raw_response: Réponse brute complète de l'API
            metadata: Métadonnées additionnelles
        """
        self.content = content
        self.provider = provider
        self.model = model
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self.total_tokens = tokens_input + tokens_output
        self.latency = latency
        self.raw_response = raw_response or {}
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la réponse en dictionnaire.
        
        Returns:
            Dictionnaire contenant les informations de la réponse
        """
        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "total_tokens": self.total_tokens,
            "latency": self.latency,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        """
        Représentation sous forme de chaîne de caractères.
        
        Returns:
            Représentation de la réponse
        """
        return (f"LLMResponse(provider={self.provider}, model={self.model}, "
                f"tokens={self.total_tokens}, latency={self.latency:.2f}s)")


class BaseLLM(ABC):
    """
    Classe abstraite pour tous les modèles de langage.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialise le modèle de langage.
        
        Args:
            config: Configuration du LLM
        """
        self.config = config
        self._validate_dependencies()
        self._setup()
        logger.info(f"Modèle LLM initialisé: {self.config.provider.value}:{self.config.model_name}")
    
    @abstractmethod
    def _validate_dependencies(self):
        """Vérifie que les dépendances nécessaires sont disponibles."""
        pass
    
    @abstractmethod
    def _setup(self):
        """Configure le client LLM."""
        pass
    
    @abstractmethod
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None) -> LLMResponse:
        """
        Génère du texte à partir d'un prompt.
        
        Args:
            prompt: Le prompt utilisateur principal
            system_prompt: Instructions système (pour les modèles qui le supportent)
            max_tokens: Limite de tokens pour la réponse
            temperature: Niveau de créativité (override la config)
            
        Returns:
            Réponse du LLM encapsulée
        """
        pass
    
    def _handle_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Gère les retries pour les appels API.
        
        Args:
            func: Fonction à exécuter
            *args: Arguments positionnels
            **kwargs: Arguments nommés
            
        Returns:
            Résultat de la fonction
        """
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                sleep_time = self.config.retry_delay * (2 ** attempt)  # Backoff exponentiel
                logger.warning(f"Tentative {attempt+1}/{self.config.max_retries} a échoué: {str(e)}. "
                              f"Nouvelle tentative dans {sleep_time}s...")
                time.sleep(sleep_time)
        
        # Si on arrive ici, toutes les tentatives ont échoué
        logger.error(f"Echec après {self.config.max_retries} tentatives. Dernière erreur: {str(last_error)}")
        raise last_error
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estime approximativement le nombre de tokens dans un texte.
        Cette méthode est une estimation grossière et peut varier selon le modèle.
        
        Args:
            text: Texte à évaluer
            
        Returns:
            Estimation du nombre de tokens
        """
        # Estimation simple: ~4 caractères par token pour l'anglais/français
        # Cette estimation est imprécise mais rapide
        return len(text) // 4


class GitHubInferenceAPILLM(BaseLLM):
    """
    Implémentation pour les modèles accessibles via l'API GitHub Inference
    """
    
    def _validate_dependencies(self):
        """Vérifie que les dépendances nécessaires sont disponibles."""
        # Besoin de la version 1.0+ de l'API OpenAI
        try:
            import importlib.metadata
            openai_version = importlib.metadata.version("openai")
            major_version = int(openai_version.split('.')[0])
            if major_version < 1:
                raise ImportError("Ce module nécessite openai>=1.0.0. Installez-le avec 'pip install --upgrade openai'")
        except (ImportError, ValueError):
            raise ImportError("Ce module nécessite openai>=1.0.0. Installez-le avec 'pip install --upgrade openai'")
    
    def _setup(self):
        """Configure le client GitHub Inference API."""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.config.api_base or "https://models.inference.ai.azure.com",
                api_key=self.config.api_key
            )
        except (ImportError, AttributeError):
            raise ImportError("Erreur lors de l'initialisation du client OpenAI v1.0+")
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None) -> LLMResponse:
        """
        Génère du texte avec un modèle via l'API GitHub Inference.
        
        Args:
            prompt: Le prompt utilisateur principal
            system_prompt: Instructions système
            max_tokens: Limite de tokens pour la réponse
            temperature: Niveau de créativité
            
        Returns:
            Réponse du modèle encapsulée
        """
        start_time = time.time()
        
        # Préparer les messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Préparer les paramètres
        try:
            max_tokens_value = max_tokens if max_tokens is not None else self.config.max_tokens
            temp_value = temperature if temperature is not None else self.config.temperature
            
            # Utiliser la nouvelle API OpenAI v1.0+
            response = self._handle_retry(
                self.client.chat.completions.create,
                model=self.config.model_name,
                messages=messages,
                temperature=temp_value,
                max_tokens=max_tokens_value,
                top_p=self.config.top_p
            )
            
            # Extraire le contenu
            content = response.choices[0].message.content
            
            # Obtenir les informations de token si disponibles
            try:
                tokens_input = response.usage.prompt_tokens
                tokens_output = response.usage.completion_tokens
            except AttributeError:
                # Estimer si non disponible
                tokens_input = self.estimate_tokens(prompt)
                if system_prompt:
                    tokens_input += self.estimate_tokens(system_prompt)
                tokens_output = self.estimate_tokens(content)
            
            # Calculer la latence
            latency = time.time() - start_time
            
            return LLMResponse(
                content=content,
                provider="github_inference",
                model=self.config.model_name,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                latency=latency,
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API GitHub Inference: {str(e)}")
            raise


class HuggingFaceLLM(BaseLLM):
    """
    Implémentation pour les modèles Hugging Face (locaux ou via l'API Inference)
    """
    
    def _validate_dependencies(self):
        """Vérifie que les dépendances pour Hugging Face sont disponibles."""
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("Le package 'transformers' n'est pas installé. "
                             "Installez-le avec 'pip install transformers'.")
    
    def _setup(self):
        """Configure le modèle Hugging Face."""
        self.use_api = self.config.extra_params.get("use_api", False)
        self.api_url = self.config.extra_params.get(
            "api_url", 
            f"https://api-inference.huggingface.co/models/{self.config.model_name}"
        )
        
        # Si on n'utilise pas l'API, charger le modèle localement
        if not self.use_api:
            try:
                logger.info(f"Chargement du modèle local: {self.config.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    device_map="auto"  # Utiliser CUDA si disponible
                )
                self.generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer
                )
                logger.info(f"Modèle local chargé avec succès: {self.config.model_name}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle local: {str(e)}")
                raise
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None) -> LLMResponse:
        """
        Génère du texte avec un modèle Hugging Face.
        
        Args:
            prompt: Le prompt utilisateur principal
            system_prompt: Instructions système (ajoutées au début du prompt)
            max_tokens: Limite de tokens pour la réponse
            temperature: Niveau de créativité
            
        Returns:
            Réponse du modèle Hugging Face encapsulée
        """
        start_time = time.time()
        
        # Combinaison du system prompt et du prompt utilisateur si nécessaire
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n"
        full_prompt += prompt
        
        # Utiliser l'API Inference ou le modèle local
        if self.use_api:
            return self._generate_via_api(
                full_prompt, 
                max_tokens,
                temperature
            )
        else:
            return self._generate_locally(
                full_prompt,
                max_tokens,
                temperature
            )
    
    def _generate_via_api(self, 
                         prompt: str, 
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None) -> LLMResponse:
        """
        Génère du texte via l'API Hugging Face Inference.
        
        Args:
            prompt: Le prompt complet
            max_tokens: Limite de tokens pour la réponse
            temperature: Niveau de créativité
            
        Returns:
            Réponse du modèle encapsulée
        """
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "top_p": self.config.top_p,
            }
        }
        
        if max_tokens is not None:
            payload["parameters"]["max_new_tokens"] = max_tokens
        elif self.config.max_tokens is not None:
            payload["parameters"]["max_new_tokens"] = self.config.max_tokens
        
        try:
            # Appel à l'API avec retry
            response = self._handle_retry(
                requests.post, 
                self.api_url, 
                headers=headers, 
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # L'API peut retourner différentes structures selon le modèle
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    content = result[0]["generated_text"]
                else:
                    content = result[0]
            else:
                content = result.get("generated_text", str(result))
            
            # Retirer le prompt de la réponse si présent
            if content.startswith(prompt):
                content = content[len(prompt):].strip()
            
            # Estimation des tokens
            prompt_tokens = self.estimate_tokens(prompt)
            output_tokens = self.estimate_tokens(content)
            
            # Calcul de la latence
            latency = time.time() - start_time
            
            return LLMResponse(
                content=content,
                provider=self.config.provider.value,
                model=self.config.model_name,
                tokens_input=prompt_tokens,
                tokens_output=output_tokens,
                latency=latency,
                raw_response=result
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API Hugging Face: {str(e)}")
            raise
    
    def _generate_locally(self, 
                         prompt: str, 
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None) -> LLMResponse:
        """
        Génère du texte en utilisant le modèle local.
        
        Args:
            prompt: Le prompt complet
            max_tokens: Limite de tokens pour la réponse
            temperature: Niveau de créativité
            
        Returns:
            Réponse du modèle encapsulée
        """
        start_time = time.time()
        
        try:
            # Préparer les paramètres
            temp = temperature if temperature is not None else self.config.temperature
            max_new_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
            if max_new_tokens is None:
                max_new_tokens = 512  # Valeur par défaut raisonnable
            
            # Compter les tokens d'entrée
            input_tokens = len(self.tokenizer.encode(prompt))
            
            # Générer le texte
            outputs = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_p=self.config.top_p,
                num_return_sequences=1
            )
            
            # Extraire le contenu généré
            generated_text = outputs[0]['generated_text']
            
            # Retirer le prompt de la réponse
            if generated_text.startswith(prompt):
                content = generated_text[len(prompt):].strip()
            else:
                content = generated_text
            
            # Compter les tokens de sortie
            output_tokens = len(self.tokenizer.encode(content))
            
            # Calcul de la latence
            latency = time.time() - start_time
            
            return LLMResponse(
                content=content,
                provider=self.config.provider.value,
                model=self.config.model_name,
                tokens_input=input_tokens,
                tokens_output=output_tokens,
                latency=latency,
                raw_response={"generated_text": generated_text}
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération avec le modèle local: {str(e)}")
            raise


def create_llm(config: LLMConfig) -> BaseLLM:
    """
    Crée une instance de LLM en fonction de la configuration fournie.
    
    Args:
        config: Configuration du LLM
        
    Returns:
        Instance de BaseLLM appropriée
    """
    if config.provider == LLMProvider.GITHUB_INFERENCE:
        return GitHubInferenceAPILLM(config)
    elif config.provider == LLMProvider.HUGGINGFACE:
        return HuggingFaceLLM(config)
    else:
        raise ValueError(f"Fournisseur LLM non supporté: {config.provider}")


# Configurations prédéfinies pour les modèles courants
DEFAULT_CONFIGS = {
    "github-gpt-4o": LLMConfig(
        provider=LLMProvider.GITHUB_INFERENCE,
        model_name="gpt-4o",
        temperature=0.7,
        max_tokens=1000,
        api_base="https://models.inference.ai.azure.com"
    ),
    "github-gpt-4": LLMConfig(
        provider=LLMProvider.GITHUB_INFERENCE,
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        api_base="https://models.inference.ai.azure.com"
    ),
    "mistral-7b": LLMConfig(
        provider=LLMProvider.HUGGINGFACE,
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.7,
        max_tokens=1000,
        use_api=False
    ),
    "llama-2-7b": LLMConfig(
        provider=LLMProvider.HUGGINGFACE,
        model_name="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.7,
        max_tokens=1000,
        use_api=False
    ),
    "hf-mistral-api": LLMConfig(
        provider=LLMProvider.HUGGINGFACE,
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.7,
        max_tokens=1000,
        use_api=True
    )
}


def get_default_config(model_key: str) -> LLMConfig:
    """
    Récupère une configuration prédéfinie pour un modèle.
    
    Args:
        model_key: Clé du modèle dans DEFAULT_CONFIGS
        
    Returns:
        Instance de LLMConfig
    """
    if model_key not in DEFAULT_CONFIGS:
        raise ValueError(f"Configuration prédéfinie non trouvée pour: {model_key}. "
                        f"Options disponibles: {list(DEFAULT_CONFIGS.keys())}")
    
    return DEFAULT_CONFIGS[model_key]