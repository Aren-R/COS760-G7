from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from comet import download_model, load_from_checkpoint
from bert_score import score as bert_score
import torch
import json
from pathlib import Path
from scipy import stats

class TranslationEvaluator:
    """Main class for evaluating machine translations using multiple metrics."""
    
    def __init__(self, results_dir: str = "results/evaluations"):
        """Initialize the evaluator with necessary components."""
        self.metrics = {
            'bleu': self._calculate_bleu,
            'comet': self._calculate_comet,
            'bertscore': self._calculate_bertscore
        }

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
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
    ) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Evaluate translations against references using specified metrics.
        
        Args:
            translations: List of translated texts
            references: List of reference texts
            metrics: List of metrics to use (default: all available metrics)
            
        Returns:
            Dictionary containing both mean scores and individual scores for each metric
        """
        if metrics is None:
            metrics = list(self.metrics.keys())
        
        scores = {}
        for metric in metrics:
            if metric in self.metrics:
                try:
                    mean_score, individual_scores = self.metrics[metric](translations, references)
                    scores[metric] = {
                        'mean': mean_score,
                        'individual': individual_scores
                    }
                except Exception as e:
                    print(f"Warning: Error calculating {metric} score: {str(e)}")
                    scores[metric] = {'mean': None, 'individual': None}
        
        return scores
    
    def _calculate_bleu(self, translations: List[str], references: List[str]) -> Tuple[float, List[float]]:
        """
        Calculate BLEU score for translations.
        
        Args:
            translations: List of translated texts
            references: List of reference texts
            
        Returns:
            Tuple of (mean BLEU score, list of individual BLEU scores)
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
            scores.append(float(score))
        
        return float(np.mean(scores)), scores
        
    def _calculate_comet(self, translations: List[str], references: List[str]) -> Tuple[float, List[float]]:
        """
        Calculate COMET score for translations.
        
        Args:
            translations: List of translated texts
            references: List of reference texts
            
        Returns:
            Tuple of (mean COMET score, list of individual COMET scores)
        """
        if self.comet_model is None:
            raise ValueError("COMET model not initialized")
        
        if len(translations) != len(references):
            raise ValueError("Number of translations must match number of references")
        
        data = [
            {
                "src": "This is a placeholder source text",  # COMET requires source text
                "mt": translation,
                "ref": reference
            }
            for translation, reference in zip(translations, references)
        ]
        
        model_output = self.comet_model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
        scores = [float(score) for score in model_output.scores]
        return float(np.mean(scores)), scores
    
    def _calculate_bertscore(self, translations: List[str], references: List[str]) -> Tuple[float, List[float]]:
        """
        Calculate BERTScore for translations.
        
        Args:
            translations: List of translated texts
            references: List of reference texts
            
        Returns:
            Tuple of (mean BERTScore F1, list of individual BERTScore F1 scores)
        """
        if len(translations) != len(references):
            raise ValueError("Number of translations must match number of references")
        
        P, R, F1 = bert_score(
            translations,
            references,
            lang="en",  # Using English as default, adjust if needed
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        scores = [float(score) for score in F1]
        return float(F1.mean().item()), scores
    

    def calculate_rank_correlations(
        self,
        original_scores: Dict[str, Dict[str, Union[float, List[float]]]],
        corrected_scores: Dict[str, Dict[str, Union[float, List[float]]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate Spearman's rank correlation coefficient between original and corrected scores.
        
        Args:
            original_scores: Dictionary containing scores using original references
            corrected_scores: Dictionary containing scores using corrected references
            
        Returns:
            Dictionary containing correlation coefficients for each metric
        """
        correlations = {}
        
        for metric in original_scores.keys():
            if metric in corrected_scores:
                orig_individual = original_scores[metric]['individual']
                corr_individual = corrected_scores[metric]['individual']
                
                if orig_individual is not None and corr_individual is not None:
                    try:
                        correlation, p_value = stats.spearmanr(orig_individual, corr_individual)
                        correlations[metric] = {
                            'correlation': float(correlation),
                            'p_value': float(p_value)
                        }
                    except Exception as e:
                        print(f"Warning: Error calculating correlation for {metric}: {str(e)}")
                        correlations[metric] = {
                            'correlation': None,
                            'p_value': None
                        }
        
        return correlations

    def save_evaluation_results(
        self,
        model_name: str,
        target_lang: str,
        original_scores: Dict[str, Dict[str, Union[float, List[float]]]],
        corrected_scores: Dict[str, Dict[str, Union[float, List[float]]]],
        translations: List[str],
        original_refs: List[str],
        corrected_refs: List[str]
    ):
        """
        Save evaluation results in a structured format.
        
        Args:
            model_name: Name of the translation model
            target_lang: Target language code
            original_scores: Scores using original references (containing both mean and individual scores)
            corrected_scores: Scores using corrected references (containing both mean and individual scores)
            translations: List of translations
            original_refs: List of original references
            corrected_refs: List of corrected references
        """
        model_dir = self.results_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        lang_dir = model_dir / target_lang
        lang_dir.mkdir(exist_ok=True)
        
        correlations = self.calculate_rank_correlations(original_scores, corrected_scores)
        
        original_mean_scores = {metric: {'mean': scores['mean']} for metric, scores in original_scores.items()}
        corrected_mean_scores = {metric: {'mean': scores['mean']} for metric, scores in corrected_scores.items()}
        
        deltas = {}
        for metric in original_scores.keys():
            if metric in corrected_scores:
                orig_mean = original_scores[metric]['mean']
                corr_mean = corrected_scores[metric]['mean']
                if orig_mean is not None and corr_mean is not None:
                    deltas[metric] = {
                        'mean_delta': corr_mean - orig_mean,
                    }
        
        topic_translations = get_translations_by_topic(translations, original_refs, corrected_refs)
        
        topic_scores = {}
        for topic, topic_data in topic_translations.items():
            if len(topic_data) <= 15:
                print(f"Skipping topic '{topic}' due to insufficient translations ({len(topic_data)} samples)")
                continue
                
            topic_trans, topic_orig_refs, topic_corr_refs = zip(*topic_data)
            
            topic_original_scores = self.evaluate_translations(
                translations=list(topic_trans),
                references=list(topic_orig_refs),
                metrics=list(self.metrics.keys())
            )
            
            topic_corrected_scores = self.evaluate_translations(
                translations=list(topic_trans),
                references=list(topic_corr_refs),
                metrics=list(self.metrics.keys())
            )
            
            topic_correlations = self.calculate_rank_correlations(
                topic_original_scores,
                topic_corrected_scores
            )
            
            topic_scores[topic] = {
                'original': {metric: {'mean': scores['mean']} for metric, scores in topic_original_scores.items()},
                'corrected': {metric: {'mean': scores['mean']} for metric, scores in topic_corrected_scores.items()},
                'correlations': topic_correlations
            }

            print(topic_scores)
        
        scores = {
            "model": model_name,
            "target_language": target_lang,
            "metrics": {
                "original": original_mean_scores,
                "corrected": corrected_mean_scores,
                "deltas": deltas,
                "correlations": correlations
            },
            "topic_analysis": topic_scores
        }
        
        with open(lang_dir / "scores.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        
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

def get_translations_by_topic(
    translations: List[str],
    original_refs: List[str],
    corrected_refs: List[str],
    metadata_path: str = "data/metadataDevTest.json"
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Organize translations and references by topic.
    
    Args:
        translations: List of translated texts
        original_refs: List of original reference texts
        corrected_refs: List of corrected reference texts
        metadata_path: Path to the metadata JSON file
        
    Returns:
        Dictionary mapping topics to lists of (translation, original_ref, corrected_ref) tuples
    """
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    topic_translations = {}
    
    if not (len(translations) == len(original_refs) == len(corrected_refs) == len(metadata)):
        raise ValueError("All input lists must have the same length")
    
    for trans, orig_ref, corr_ref, meta in zip(translations, original_refs, corrected_refs, metadata):
        topic = meta.get('topic', 'unknown')
        if topic is None:
            topic = 'unknown'
            
        topic = topic.lower().strip()
        
        if topic not in topic_translations:
            topic_translations[topic] = []
            
        topic_translations[topic].append((trans, orig_ref, corr_ref))
    
    return topic_translations
