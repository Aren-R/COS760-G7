from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm import tqdm

class TranslationModel:
    def __init__(self, model_name: str):
        """Initialize translation model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        self.model_name = model_name

    def translate(self, texts: list, target_lang: str) -> list:
        """Translate a list of texts to target language"""
        translations = []
        for text in tqdm(texts, desc=f"Translating to {target_lang}", unit="text"):
            # Handle different model types
            if "nllb" in self.model_name:
                # NLLB uses specific language codes
                lang_map = {
                    "hau": "hau_Latn",
                    "nso": "nso_Latn",
                    "tso": "tso_Latn",
                    "zul": "zul_Latn"
                }
                lang_code = target_lang.split('_')[0]
                nllb_lang = lang_map[lang_code]
                # Set source and target languages for NLLB
                self.tokenizer.src_lang = "eng_Latn"
                self.tokenizer.tgt_lang = nllb_lang
                inputs = self.tokenizer(text, return_tensors="pt", padding=True)
                outputs = self.model.generate(**inputs)
            elif "opus-mt" in self.model_name:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True)
                outputs = self.model.generate(**inputs)
            elif "m2m100" in self.model_name:
                self.tokenizer.src_lang = "en"
                self.tokenizer.tgt_lang = target_lang
                inputs = self.tokenizer(text, return_tensors="pt", padding=True)
                outputs = self.model.generate(**inputs)
            else:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True)
                outputs = self.model.generate(**inputs)
                
            translation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            translations.append(translation)
        return translations

class ModelFactory:
    @staticmethod
    def get_model(model_type: str) -> TranslationModel:
        """Get translation model by type"""
        model_map = {
            "nllb": "facebook/nllb-200-distilled-600M",
            "opus-mt": "Helsinki-NLP/opus-mt-en-af",
            "m2m100": "facebook/m2m100_418M"
        }
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")
        return TranslationModel(model_map[model_type]) 