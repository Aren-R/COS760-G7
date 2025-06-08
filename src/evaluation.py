from typing import Dict, List, Optional
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from comet import download_model, load_from_checkpoint
from bert_score import score as bert_score
import torch
import json
from pathlib import Path

class TranslationEvaluator:
    """Main class for evaluating machine translations using multiple metrics."""
    
    def __init__(self, results_dir: str = "results/evaluations"):
        """Initialize the evaluator with necessary components."""
        self.metrics = {
            'bleu': self._calculate_bleu,
            'comet': self._calculate_comet,
            'bertscore': self._calculate_bertscore
        }
        # Create results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Initialize COMET model
        try:
            model_path = download_model("Unbabel/wmt22-comet-da")
            self.comet_model = load_from_checkpoint(model_path)
        except Exception as e:
            print(f"Warning: Could not initialize COMET model: {str(e)}")
            self.comet_model = None
    
    def evaluate_translations(
        self,
        translations: List[str],
        references: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate translations against references using specified metrics.
        
        Args:
            translations: List of translated texts
            references: List of reference texts
            metrics: List of metrics to use (default: all available metrics)
            
        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = list(self.metrics.keys())
        
        scores = {}
        for metric in metrics:
            if metric in self.metrics:
                try:
                    scores[metric] = self.metrics[metric](translations, references)
                except Exception as e:
                    print(f"Warning: Error calculating {metric} score: {str(e)}")
                    scores[metric] = None
        
        return scores
    
    def _calculate_bleu(self, translations: List[str], references: List[str]) -> float:
        """
        Calculate BLEU score for translations.
        
        Args:
            translations: List of translated texts
            references: List of reference texts
            
        Returns:
            Average BLEU score across all translations
        """
        if len(translations) != len(references):
            raise ValueError("Number of translations must match number of references")
        
        smoothing = SmoothingFunction().method1
        scores = []
        
        for translation, reference in zip(translations, references):
            translation_tokens = nltk.word_tokenize(translation.lower())
            reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in [reference]]
            
            score = sentence_bleu(
                reference_tokens,
                translation_tokens,
                smoothing_function=smoothing
            )
            scores.append(score)
        
        return float(np.mean(scores))
    
    def _calculate_comet(self, translations: List[str], references: List[str]) -> float:
        """
        Calculate COMET score for translations.
        
        Args:
            translations: List of translated texts
            references: List of reference texts
            
        Returns:
            Average COMET score across all translations
        """
        if self.comet_model is None:
            raise ValueError("COMET model not initialized")
        
        if len(translations) != len(references):
            raise ValueError("Number of translations must match number of references")
        
        # Prepare data for COMET
        data = [
            {
                "src": "This is a placeholder source text",  # COMET requires source text
                "mt": translation,
                "ref": reference
            }
            for translation, reference in zip(translations, references)
        ]
        
        # Get model predictions
        model_output = self.comet_model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
        return float(np.mean(model_output.scores))
    
    def _calculate_bertscore(self, translations: List[str], references: List[str]) -> float:
        """
        Calculate BERTScore for translations.
        
        Args:
            translations: List of translated texts
            references: List of reference texts
            
        Returns:
            Average BERTScore F1 across all translations
        """
        if len(translations) != len(references):
            raise ValueError("Number of translations must match number of references")
        
        # Calculate BERTScore
        P, R, F1 = bert_score(
            translations,
            references,
            lang="en",  # Using English as default, adjust if needed
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        return float(F1.mean().item())
    
    def compare_rankings(
        self,
        original_scores: Dict[str, Dict[str, float]],
        corrected_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare rankings between original and corrected reference scores.
        
        Args:
            original_scores: Dictionary of scores using original references
            corrected_scores: Dictionary of scores using corrected references
            
        Returns:
            Dictionary containing ranking comparison metrics
        """
        # TODO: Implement ranking comparison
        pass

    def save_evaluation_results(
        self,
        model_name: str,
        target_lang: str,
        original_scores: Dict[str, float],
        corrected_scores: Dict[str, float],
        translations: List[str],
        original_refs: List[str],
        corrected_refs: List[str]
    ):
        """
        Save evaluation results in a structured format.
        
        Args:
            model_name: Name of the translation model
            target_lang: Target language code
            original_scores: Scores using original references
            corrected_scores: Scores using corrected references
            translations: List of translations
            original_refs: List of original references
            corrected_refs: List of corrected references
        """
        # Create model-specific directory
        model_dir = self.results_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Create language-specific directory
        lang_dir = model_dir / target_lang
        lang_dir.mkdir(exist_ok=True)
        
        # Save scores
        scores = {
            "model": model_name,
            "target_language": target_lang,
            "metrics": {
                "original": original_scores,
                "corrected": corrected_scores
            }
        }
        
        with open(lang_dir / "scores.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        
        # Save translations and references
        translations_data = {
            "model": model_name,
            "target_language": target_lang,
            "data": [
                {
                    "translation": trans,
                    "original_reference": orig_ref,
                    "corrected_reference": corr_ref
                }
                for trans, orig_ref, corr_ref in zip(translations, original_refs, corrected_refs)
            ]
        }
        
        with open(lang_dir / "translations.json", "w", encoding="utf-8") as f:
            json.dump(translations_data, f, ensure_ascii=False, indent=2)
