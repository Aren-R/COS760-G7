from typing import Dict, List, Optional
from models import TranslationModel
import torch
from sacremoses import MosesTokenizer, MosesDetokenizer
import json
import os
from pathlib import Path

class TranslationPipeline:
    """Pipeline for handling translations with caching and batching"""
    
    def __init__(self, models: Dict[str, TranslationModel], cache_dir: str = "results/translations"):
        self.models = models
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = MosesTokenizer()
        self.detokenizer = MosesDetokenizer()
        
    def _get_cache_path(self, model_name: str, source_lang: str, target_lang: str) -> Path:
        """Get the path for the cache file"""
        return self.cache_dir / f"{model_name}_{source_lang}_{target_lang}.json"
    
    def _load_cache(self, cache_path: Path) -> Dict[str, str]:
        """Load translations from cache"""
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self, cache_path: Path, cache: Dict[str, str]):
        """Save translations to cache"""
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    
    def _tokenize_batch(self, texts: List[str], lang: str) -> List[str]:
        """Tokenize a batch of texts using Moses tokenizer"""
        return [self.tokenizer.tokenize(text, return_str=True) for text in texts]
    
    def _detokenize_batch(self, texts: List[str], lang: str) -> List[str]:
        """Detokenize a batch of texts using Moses detokenizer"""
        return [self.detokenizer.detokenize(text.split()) for text in texts]
    
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        model_name: str,
        batch_size: int = 32,
        use_cache: bool = True
    ) -> List[str]:
        """
        Translate a batch of texts using the specified model
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            model_name: Name of the model to use
            batch_size: Size of batches for translation
            use_cache: Whether to use translation cache
        
        Returns:
            List of translated texts
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        cache = {}
        
        if use_cache:
            cache_path = self._get_cache_path(model_name, source_lang, target_lang)
            cache = self._load_cache(cache_path)
        
        # Tokenize input texts
        tokenized_texts = self._tokenize_batch(texts, source_lang)
        
        # Process in batches
        translations = []
        for i in range(0, len(tokenized_texts), batch_size):
            batch = tokenized_texts[i:i + batch_size]
            
            # Check cache for each text in batch
            batch_translations = []
            texts_to_translate = []
            indices_to_translate = []
            
            for j, text in enumerate(batch):
                if text in cache:
                    batch_translations.append(cache[text])
                else:
                    texts_to_translate.append(text)
                    indices_to_translate.append(j)
            
            # Translate texts not in cache
            if texts_to_translate:
                new_translations = model.translate(texts_to_translate, source_lang, target_lang)
                
                # Update cache
                if use_cache:
                    for text, translation in zip(texts_to_translate, new_translations):
                        cache[text] = translation
                
                # Insert translations at correct positions
                for idx, translation in zip(indices_to_translate, new_translations):
                    batch_translations.insert(idx, translation)
            
            translations.extend(batch_translations)
        
        # Save updated cache
        if use_cache:
            self._save_cache(cache_path, cache)
        
        # Detokenize translations
        return self._detokenize_batch(translations, target_lang)
    
    def translate_dataset(
        self,
        dataset: Dict[str, List[str]],
        source_lang: str,
        target_lang: str,
        model_name: str,
        batch_size: int = 32,
        use_cache: bool = True
    ) -> Dict[str, List[str]]:
        """
        Translate an entire dataset
        
        Args:
            dataset: Dictionary mapping language codes to lists of texts
            source_lang: Source language code
            target_lang: Target language code
            model_name: Name of the model to use
            batch_size: Size of batches for translation
            use_cache: Whether to use translation cache
        
        Returns:
            Dictionary mapping language codes to lists of translated texts
        """
        if source_lang not in dataset:
            raise ValueError(f"Source language {source_lang} not found in dataset")
        
        translations = self.translate_batch(
            dataset[source_lang],
            source_lang,
            target_lang,
            model_name,
            batch_size,
            use_cache
        )
        
        return {target_lang: translations} 