import logging
import pandas as pd
from tqdm import tqdm
import torch
import json

from models.translation_models import MTModel
from metrics.evaluation_metrics import EvaluationMetrics
from metrics.analysis import AnalysisMetrics
from utils.data_loader import DataLoader

# set up logging for the evaluation process
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MTEvaluator:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        initializes the MT evaluator with all necessary models and metrics
        """
        self.device = device
        self.setup_components()
        
    def setup_components(self):
        """
        loads all models, metrics, and data utilities needed for evaluation
        """
        # set up metrics and analysis tools
        self.metrics = EvaluationMetrics(self.device)
        self.analysis = AnalysisMetrics()
        
        # load all translation models
        self.models = {
            'nllb': MTModel('nllb', self.device),
            'opus-mt': MTModel('opus-mt', self.device),
            'm2m100': MTModel('m2m100', self.device)
        }
        
        # initialize the data loader
        self.data_loader = DataLoader()

def main():
    # create the evaluator instance
    evaluator = MTEvaluator()
    
    # list of languages to evaluate
    languages = ['hau', 'nso', 'tso', 'zul']
    
    # these dictionaries will store all results for later analysis
    all_results = {}
    all_correlations = {}
    all_domain_analysis = {}
    
    for lang in languages:
        logger.info(f"\nEvaluating {lang}...")
        
        try:
            # try to load both the original and corrected datasets
            original_dataset, corrected_dataset = evaluator.data_loader.load_flores_data(lang)
            
            # get the test splits for both datasets
            original_test = original_dataset['test']
            corrected_test = corrected_dataset['test']
            
            # grab the source texts for translation
            source_texts = original_test['translation']
            print(source_texts)
            
            # set up structures to store metric scores for correlation analysis
            metric_scores = {
                'original': {'bleu': {}, 'comet': {}, 'bert_score': {}},
                'corrected': {'bleu': {}, 'comet': {}, 'bert_score': {}}
            }
            
            # set up structures to store scores for domain analysis
            domain_scores = {
                'original': {model: [] for model in evaluator.models.keys()},
                'corrected': {model: [] for model in evaluator.models.keys()}
            }
            
            # loop through each model and evaluate
            model_results = {}
            for model_name, model in evaluator.models.items():
                logger.info(f"\nEvaluating {model_name}...")
                
                # translate the source texts
                translations = model.translate(source_texts, lang)
                
                # evaluate the translations using all metrics
                scores = evaluator.metrics.evaluate_batch(
                    sources=source_texts,
                    hypotheses=translations,
                    original_references=original_test['translation'],
                    corrected_references=corrected_test['translation']
                )
                
                # store the average metric scores for correlation analysis
                for metric in ['bleu', 'comet', 'bert_score']:
                    metric_scores['original'][metric][model_name] = scores['original'][metric]
                    metric_scores['corrected'][metric][model_name] = scores['corrected'][metric]
                
                # store BLEU scores for domain analysis
                domain_scores['original'][model_name] = scores['original']['bleu']
                domain_scores['corrected'][model_name] = scores['corrected']['bleu']
                
                model_results[model_name] = scores
            
            # calculate Spearman correlations for each metric
            correlations = {}
            for metric in ['bleu', 'comet', 'bert_score']:
                correlation, p_value = evaluator.analysis.calculate_spearman_correlation(
                    metric_scores['original'][metric],
                    metric_scores['corrected'][metric]
                )
                correlations[metric] = {
                    'correlation': correlation,
                    'p_value': p_value
                }
            
            # analyze the impact of corrections across different domains
            domain_analysis = evaluator.analysis.analyze_domain_impact(
                texts=source_texts,
                original_scores=domain_scores['original'],
                corrected_scores=domain_scores['corrected']
            )
            
            # store all results for this language
            all_results[lang] = model_results
            all_correlations[lang] = correlations
            all_domain_analysis[lang] = domain_analysis
            
        except Exception as e:
            logger.error(f"Error processing {lang}: {str(e)}")
            continue
    
    # save the main evaluation results to a CSV file
    results_df = pd.DataFrame(all_results).T
    results_df.to_csv('evaluation_results.csv')
    logger.info("basic evaluation results saved to evaluation_results.csv")
    
    # save the correlation analysis to a JSON file
    with open('correlation_analysis.json', 'w') as f:
        json.dump(all_correlations, f, indent=2)
    logger.info("correlation analysis saved to correlation_analysis.json")
    
    # save the domain analysis to a JSON file
    with open('domain_analysis.json', 'w') as f:
        json.dump(all_domain_analysis, f, indent=2)
    logger.info("domain analysis saved to domain_analysis.json")

if __name__ == "__main__":
    main() 