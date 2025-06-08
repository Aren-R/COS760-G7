from abc import ABC, abstractmethod
from typing import List, Dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Language code mapping for OPUS-MT
OPUS_MT_LANG_MAP = {
    'hau': 'ha',  # Hausa
    'nso': 'nso', # Northern Sotho
    'tso': 'ts',  # Xitsonga
    'zul': 'zu',  # isiZulu
    'en': 'en'   # English
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
        
        # Tokenize with source language
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Generate translations
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang_code),
            max_length=512
        )
        
        # Decode translations
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
        # Convert language codes to OPUS-MT format
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
        # Load the appropriate model for this language pair
        model_data = self._load_model(source_lang, target_lang)
        tokenizer = model_data['tokenizer']
        model = model_data['model']
        
        if target_lang == 'zul':
            tagged_texts = [f">>zul<< {text}" for text in texts]
            inputs = tokenizer(tagged_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        else:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Generate translations with proper max_length and early_stopping
        translated_tokens = model.generate(
            **inputs,
            early_stopping=True
        )
        
        # Decode translations
        translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translations
    
    def get_model_name(self) -> str:
        return "OPUS-MT"

# class M2M100Model(TranslationModel):
#     """M2M-100 model wrapper"""
    
#     def __init__(self, model_name: str = "facebook/m2m100_418M"):
#         self.model_name = model_name
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
    
#     def translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
#         # Set source and target languages
#         self.tokenizer.src_lang = source_lang
#         self.tokenizer.tgt_lang = target_lang
        
#         # Tokenize
#         inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
#         # Generate translations
#         translated_tokens = self.model.generate(**inputs, max_length=512)
        
#         # Decode translations
#         translations = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
#         return translations
    
#     def get_model_name(self) -> str:
#         return "M2M-100"

def initialize_models() -> Dict[str, TranslationModel]:
    """Initialize all translation models"""
    return {
        "nllb": NLLBModel(),
        "opus-mt": OPUSMTModel(),
        # "m2m100": M2M100Model()
    } 