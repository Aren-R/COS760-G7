from abc import ABC, abstractmethod
from typing import List, Dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch

OPUS_MT_LANG_MAP = {
    'hau': 'ha',
    'nso': 'nso',
    'tso': 'ts',
    'zul': 'zu',
    'en': 'en'
}

MADLAD_LANG_MAP = {
    'hau': 'ha',
    'nso': 'nso',
    'zul': 'zu',
    'en': 'en',
    'tso': 'ts'
}

class TranslationModel(ABC):
    """Base class for translation models"""
    
    @abstractmethod
    def translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate a list of texts from source language to target language"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model"""
        pass

class NLLBModel(TranslationModel):
    """NLLB model wrapper"""
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
    def translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        src_lang_code = f"{source_lang}_Latn"
        tgt_lang_code = f"{target_lang}_Latn"
        
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang_code),
            max_length=512
        )
        
        translations = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translations
    
    def get_model_name(self) -> str:
        return "NLLB"

class OPUSMTModel(TranslationModel):
    """OPUS-MT model wrapper"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}  # Cache for loaded models
    
    def _get_model_name(self, source_lang: str, target_lang: str) -> str:
        """Get the appropriate OPUS-MT model name for the language pair"""
        src_code = OPUS_MT_LANG_MAP.get(source_lang, source_lang)
        tgt_code = OPUS_MT_LANG_MAP.get(target_lang, target_lang)
        if (tgt_code == "zu"):
            return f"Helsinki-NLP/opus-mt-{src_code}-mul"
        return f"Helsinki-NLP/opus-mt-{src_code}-{tgt_code}"
    
    def _load_model(self, source_lang: str, target_lang: str):
        """Load the appropriate model for the language pair"""
        src_code = OPUS_MT_LANG_MAP.get(source_lang, source_lang)
        tgt_code = OPUS_MT_LANG_MAP.get(target_lang, target_lang)

        model_name = self._get_model_name(src_code, tgt_code)
        if model_name not in self.models:
            try:
                self.models[model_name] = {
                    'tokenizer': AutoTokenizer.from_pretrained(model_name),
                    'model': AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
                }
            except Exception as e:
                raise ValueError(f"Could not load OPUS-MT model for {source_lang}-{target_lang}: {str(e)}")
        return self.models[model_name]
    
    def translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        model_data = self._load_model(source_lang, target_lang)
        tokenizer = model_data['tokenizer']
        model = model_data['model']
        
        if target_lang == 'zul':
            tagged_texts = [f">>zul<< {text}" for text in texts]
            inputs = tokenizer(tagged_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        else:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        translated_tokens = model.generate(
            **inputs,
            early_stopping=True
        )
        
        # Decode translations
        translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translations
    
    def get_model_name(self) -> str:
        return "OPUS-MT"

class MADLADModel(TranslationModel):
    """MADLAD-400 model wrapper for multilingual translation"""

    def __init__(self, model_name: str = "google/madlad400-3b-mt"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        src_code = MADLAD_LANG_MAP.get(source_lang, source_lang)
        tgt_code = MADLAD_LANG_MAP.get(target_lang, target_lang)

        prompts = [f"<2{tgt_code}> {text}" for text in texts]

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        translated_tokens = self.model.generate(
            **inputs
        )

        translations = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translations

    def get_model_name(self) -> str:
        return "MADLAD-400"


def initialize_models() -> Dict[str, TranslationModel]:
    """Initialize all translation models"""
    return {
        "nllb": NLLBModel(),
        "opus-mt": OPUSMTModel(),
        "madlad": MADLADModel()
    } 