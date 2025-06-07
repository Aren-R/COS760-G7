from typing import List
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer
)
from tqdm import tqdm

class MTModel:
    def __init__(self, model_name: str, device: str):
        """
        initializes a machine translation model for a given model name and device
        """
        self.model_name = model_name
        self.device = device
        self.setup_model()
        
    def setup_model(self):
        """
        loads the appropriate model and tokenizer based on the model name
        """
        if "nllb" in self.model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        elif "opus" in self.model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{self.model_name.split('-')[-1]}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-en-{self.model_name.split('-')[-1]}")
        elif "m2m" in self.model_name.lower():
            self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
            self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        
        self.model.to(self.device)
        
    def translate(self, texts: List[str], target_lang: str) -> List[str]:
        """
        translates a list of texts to the target language
        """
        translations = []
        
        # set target language for M2M100
        if "m2m" in self.model_name.lower():
            self.tokenizer.src_lang = "en"
            self.tokenizer.tgt_lang = target_lang
        
        for text in tqdm(texts, desc=f"Translating with {self.model_name}"):
            # tokenize and prepare input for the model
            if "nllb" in self.model_name.lower():
                inputs = self.tokenizer(text, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # nllb uses special language codes
                lang_code = {
                    'hau': 'hau_Latn',
                    'nso': 'nso_Latn',
                    'tso': 'tso_Latn',
                    'zul': 'zul_Latn'
                }[target_lang]
                outputs = self.model.generate(**inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[lang_code])
            elif "opus" in self.model_name.lower():
                inputs = self.tokenizer(text, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model.generate(**inputs)
            else:  # M2M100
                inputs = self.tokenizer(text, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model.generate(**inputs)
            
            # decode the model output to get the translation
            translation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            translations.append(translation)
            
        return translations 