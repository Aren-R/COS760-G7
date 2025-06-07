from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm import tqdm

class TranslationModel:
    def __init__(self, model_name: str):
        """Initialize translation model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Enable GPU acceleration if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def translate_batch(self, texts: list, target_lang: str, batch_size: int = 32) -> list:
        """Translate a batch of texts to target language with optimized performance"""
        all_translations = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Translating to {target_lang}", unit="batch"):
            batch_texts = texts[i:i + batch_size]
            
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
                nllb_lang = lang_map.get(lang_code, target_lang)
                # Set source and target languages for NLLB
                self.tokenizer.src_lang = "eng_Latn"
                self.tokenizer.tgt_lang = nllb_lang
            elif "m2m100" in self.model_name:
                self.tokenizer.src_lang = "en"
                self.tokenizer.tgt_lang = target_lang
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate translations with optimized parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode batch translations
            batch_translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_translations.extend(batch_translations)
        
        return all_translations

    def translate(self, texts: list, target_lang: str) -> list:
        """Translate a list of texts to target language (legacy method - uses batch processing internally)"""
        return self.translate_batch(texts, target_lang, batch_size=8)

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