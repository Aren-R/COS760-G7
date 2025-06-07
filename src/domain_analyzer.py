import logging
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class DomainAnalyzer:
    def __init__(self, n_domains=5):
        """Initialize domain analyzer"""
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.kmeans = KMeans(n_clusters=n_domains)
        logger.info(f"Initialized domain analyzer with {n_domains} domains")

    def analyze_domain_impact(self, texts: List[str], 
                            original_scores: Dict[str, List[float]], 
                            corrected_scores: Dict[str, List[float]]) -> Dict:
        """Analyze impact of corrections across different domains"""
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        # Cluster texts into domains
        domains = self.kmeans.fit_predict(X)
        
        # Calculate average score changes per domain
        domain_analysis = {}
        for domain in range(self.kmeans.n_clusters):
            domain_indices = [i for i, d in enumerate(domains) if d == domain]
            domain_analysis[f"domain_{domain}"] = {
                'size': len(domain_indices),
                'score_changes': {}
            }
            
            for model in original_scores.keys():
                orig_scores = [original_scores[model][i] for i in domain_indices]
                corr_scores = [corrected_scores[model][i] for i in domain_indices]
                domain_analysis[f"domain_{domain}"]['score_changes'][model] = {
                    'mean_change': np.mean(np.array(corr_scores) - np.array(orig_scores)),
                    'std_change': np.std(np.array(corr_scores) - np.array(orig_scores))
                }
        
        return domain_analysis 