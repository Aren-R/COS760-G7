import logging
import nltk
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from comet.models import download_model, load_from_checkpoint
from bert_score import score
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

class Metrics:
    def __init__(self):
        """Initialize evaluation metrics"""
        self.comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        self.smoothing = SmoothingFunction().method1
        logging.info("Initialized evaluation metrics")

    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU score for a single sentence pair"""
        try:
            reference_tokens = nltk.word_tokenize(reference.lower())
            hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
            return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=self.smoothing)
        except Exception as e:
            logging.error(f"Error calculating BLEU score: {str(e)}")
            return 0.0

    def calculate_comet(self, source: str, reference: str, hypothesis: str) -> float:
        """Calculate COMET score for a single sentence pair"""
        try:
            data = [{"src": source, "mt": hypothesis, "ref": reference}]
            gpu_count = 1 if torch.cuda.is_available() else 0
            model_output = self.comet_model.predict(data, batch_size=8, gpus=gpu_count)
            return model_output.scores[0]
        except Exception as e:
            logging.error(f"Error calculating COMET score: {str(e)}")
            return 0.0

    def calculate_bert_score(self, reference: str, hypothesis: str) -> float:
        """Calculate BERTScore for a single sentence pair"""
        try:
            P, R, F1 = score([hypothesis], [reference], lang="en", verbose=False)
            return F1.mean().item()
        except Exception as e:
            logging.error(f"Error calculating BERTScore: {str(e)}")
            return 0.0

    def evaluate_batch(self, sources: list, hypotheses: list, 
                      original_references: list, corrected_references: list) -> dict:
        """Evaluate a batch of translations using all metrics"""
        results = {
            'original': {'bleu': [], 'comet': [], 'bert_score': []},
            'corrected': {'bleu': [], 'comet': [], 'bert_score': []}
        }
        
        for src, hyp, orig_ref, corr_ref in zip(sources, hypotheses, original_references, corrected_references):
            # Original reference scores
            results['original']['bleu'].append(self.calculate_bleu(orig_ref, hyp))
            results['original']['comet'].append(self.calculate_comet(src, orig_ref, hyp))
            results['original']['bert_score'].append(self.calculate_bert_score(orig_ref, hyp))
            
            # Corrected reference scores
            results['corrected']['bleu'].append(self.calculate_bleu(corr_ref, hyp))
            results['corrected']['comet'].append(self.calculate_comet(src, corr_ref, hyp))
            results['corrected']['bert_score'].append(self.calculate_bert_score(corr_ref, hyp))
        
        # Calculate averages
        for ref_type in ['original', 'corrected']:
            for metric in ['bleu', 'comet', 'bert_score']:
                results[ref_type][metric] = np.mean(results[ref_type][metric])
        
        return results 