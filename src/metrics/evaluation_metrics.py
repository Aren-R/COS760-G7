from typing import List, Dict
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from comet import download_model, load_from_checkpoint
from bert_score import score
import torch

class EvaluationMetrics:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        sets up the evaluation metrics and loads the necessary models
        """
        self.device = device
        self.setup_metrics()
        
    def setup_metrics(self):
        """
        downloads and loads the COMET and BERTScore models
        """
        # download and set up the COMET model
        model_path = download_model("Unbabel/wmt22-comet-da")
        self.comet_model = load_from_checkpoint(model_path)
        
        # set the BERTScore model type
        self.bert_score_model = "microsoft/deberta-xlarge-mnli"

    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """
        computes the BLEU score for a single sentence pair
        """
        reference_tokens = reference.split()
        hypothesis_tokens = hypothesis.split()
        smoothing = SmoothingFunction().method1
        return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing)

    def calculate_comet(self, source: str, reference: str, hypothesis: str) -> float:
        """
        computes the COMET score for a single translation
        """
        data = [{"src": source, "mt": hypothesis, "ref": reference}]
        model_output = self.comet_model.predict(data, batch_size=8, gpus=1 if self.device == "cuda" else 0)
        return model_output.scores[0]

    def calculate_bert_score(self, reference: str, hypothesis: str) -> float:
        """
        computes the BERTScore for a single translation
        """
        P, R, F1 = score([hypothesis], [reference], model_type=self.bert_score_model)
        return F1.mean().item()

    def evaluate_batch(
        self,
        sources: List[str],
        hypotheses: List[str],
        original_references: List[str],
        corrected_references: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        evaluates a batch of translations using all metrics and returns the results
        """
        results = {
            'original': {'bleu': [], 'comet': [], 'bert_score': []},
            'corrected': {'bleu': [], 'comet': [], 'bert_score': []}
        }
        
        for src, hyp, orig_ref, corr_ref in zip(
            sources, hypotheses, original_references, corrected_references
        ):
            # compute BLEU scores for both original and corrected references
            results['original']['bleu'].append(self.calculate_bleu(orig_ref, hyp))
            results['corrected']['bleu'].append(self.calculate_bleu(corr_ref, hyp))
            
            # compute COMET scores for both original and corrected references
            results['original']['comet'].append(self.calculate_comet(src, orig_ref, hyp))
            results['corrected']['comet'].append(self.calculate_comet(src, corr_ref, hyp))
            
            # compute BERTScore for both original and corrected references
            results['original']['bert_score'].append(self.calculate_bert_score(orig_ref, hyp))
            results['corrected']['bert_score'].append(self.calculate_bert_score(corr_ref, hyp))
        
        # calculate means and differences for each metric
        final_results = {}
        for dataset in ['original', 'corrected']:
            final_results[dataset] = {
                metric: np.mean(scores) for metric, scores in results[dataset].items()
            }
        
        # calculate the difference between corrected and original scores
        final_results['differences'] = {
            metric: final_results['corrected'][metric] - final_results['original'][metric]
            for metric in ['bleu', 'comet', 'bert_score']
        }
        
        return final_results 