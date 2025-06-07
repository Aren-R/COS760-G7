from typing import Dict
from scipy.stats import spearmanr

def calculate_rank_correlation(original_scores: Dict[str, float], 
                             corrected_scores: Dict[str, float]) -> Dict:
    """Calculate Spearman's rank correlation between original and corrected scores"""
    models = list(original_scores.keys())
    orig_ranks = [original_scores[m] for m in models]
    corr_ranks = [corrected_scores[m] for m in models]
    
    correlation, p_value = spearmanr(orig_ranks, corr_ranks)
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'models': models,
        'original_ranks': orig_ranks,
        'corrected_ranks': corr_ranks
    } 