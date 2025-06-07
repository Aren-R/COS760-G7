import logging
import os
import json
import pandas as pd
from typing import List, Dict
import huggingface_hub
from tqdm import tqdm

from models import ModelFactory
from metrics import Metrics
from domain_analyzer import DomainAnalyzer
from data_loader import load_language_dataset
from analysis import calculate_rank_correlation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

def main():
    # Configuration
    languages = ["hau", "nso", "tso", "zul"]
    model_types = ["nllb", "opus-mt", "m2m100"]
    
    # Initialize components
    metrics = Metrics()
    domain_analyzer = DomainAnalyzer()
    
    # Store results
    all_results = {}
    all_correlations = {}
    all_domain_analysis = {}
    
    # Process each language with progress bar
    for lang_code in tqdm(languages, desc="Processing languages"):
        logger.info(f"\nProcessing {lang_code}...")
        
        try:
            # Load datasets
            datasets = load_language_dataset(lang_code)
            logger.info(f"Loaded {len(datasets['original'])} original and {len(datasets['corrected'])} corrected texts")
            
            # Store results for this language
            lang_results = {}
            model_scores = {
                'original': {model: [] for model in model_types},
                'corrected': {model: [] for model in model_types}
            }
            
            # Evaluate each model with progress bar
            for model_type in tqdm(model_types, desc=f"Evaluating models for {lang_code}", leave=False):
                logger.info(f"Evaluating {model_type}...")
                
                # Get model and translate
                model = ModelFactory.get_model(model_type)
                translations = model.translate(datasets['original'], f"{lang_code}_Latn")
                
                # Calculate metrics
                scores = metrics.evaluate_batch(
                    sources=datasets['original'],
                    hypotheses=translations,
                    original_references=datasets['original'],
                    corrected_references=datasets['corrected']
                )
                
                # Store scores
                lang_results[model_type] = scores
                for ref_type in ['original', 'corrected']:
                    for metric in ['bleu', 'comet', 'bert_score']:
                        model_scores[ref_type][model_type].append(scores[ref_type][metric])
            
            # Calculate rank correlations
            correlations = {}
            for metric in ['bleu', 'comet', 'bert_score']:
                orig_scores = {m: s[metric] for m, s in lang_results.items()}
                corr_scores = {m: s[metric] for m, s in lang_results.items()}
                correlations[metric] = calculate_rank_correlation(orig_scores, corr_scores)
            
            # Analyze domain impact
            domain_analysis = domain_analyzer.analyze_domain_impact(
                texts=datasets['original'],
                original_scores=model_scores['original'],
                corrected_scores=model_scores['corrected']
            )
            
            # Store results
            all_results[lang_code] = lang_results
            all_correlations[lang_code] = correlations
            all_domain_analysis[lang_code] = domain_analysis
            
        except Exception as e:
            logger.error(f"Error processing {lang_code}: {str(e)}")
            continue
    
    # Save results to results directory
    results_df = pd.DataFrame(all_results).T
    results_df.to_csv('results/evaluation_results.csv')
    logger.info("Saved evaluation results to results/evaluation_results.csv")
    
    with open('results/correlation_analysis.json', 'w') as f:
        json.dump(all_correlations, f, indent=2)
    logger.info("Saved correlation analysis to results/correlation_analysis.json")
    
    with open('results/domain_analysis.json', 'w') as f:
        json.dump(all_domain_analysis, f, indent=2)
    logger.info("Saved domain analysis to results/domain_analysis.json")

if __name__ == "__main__":
    main() 