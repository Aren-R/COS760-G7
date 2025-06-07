from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class AnalysisMetrics:
    @staticmethod
    def calculate_spearman_correlation(
        original_scores: Dict[str, float],
        corrected_scores: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        computes Spearman's rank correlation between original and corrected scores
        """
        # convert dictionaries to lists to maintain order
        models = list(original_scores.keys())
        orig_ranks = [original_scores[m] for m in models]
        corr_ranks = [corrected_scores[m] for m in models]
        
        # calculate the Spearman correlation
        correlation, p_value = stats.spearmanr(orig_ranks, corr_ranks)
        return correlation, p_value

    @staticmethod
    def identify_domains(
        texts: List[str],
        n_domains: int = 3
    ) -> Tuple[List[int], Dict[int, List[str]]]:
        """
        clusters texts into domains using TF-IDF and K-means
        """
        # create TF-IDF vectors for the texts
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # cluster the texts into domains
        kmeans = KMeans(n_clusters=n_domains, random_state=42)
        domain_labels = kmeans.fit_predict(tfidf_matrix)
        
        # get the top keywords for each domain
        feature_names = vectorizer.get_feature_names_out()
        domain_keywords = {}
        
        for i in range(n_domains):
            center = kmeans.cluster_centers_[i]
            top_indices = center.argsort()[-10:][::-1]
            domain_keywords[i] = [feature_names[idx] for idx in top_indices]
        
        return domain_labels, domain_keywords

    @staticmethod
    def analyze_domain_impact(
        texts: List[str],
        original_scores: Dict[str, List[float]],
        corrected_scores: Dict[str, List[float]],
        n_domains: int = 3
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        analyzes the impact of corrections across different domains
        """
        # identify domains in the texts
        domain_labels, domain_keywords = AnalysisMetrics.identify_domains(texts, n_domains)
        
        # analyze each domain for each model
        domain_analysis = {}
        for model in original_scores.keys():
            domain_analysis[model] = {}
            
            for domain in range(n_domains):
                domain_indices = [i for i, label in enumerate(domain_labels) if label == domain]
                orig_domain_scores = [original_scores[model][i] for i in domain_indices]
                corr_domain_scores = [corrected_scores[model][i] for i in domain_indices]
                
                domain_analysis[model][domain] = {
                    'original_mean': np.mean(orig_domain_scores),
                    'corrected_mean': np.mean(corr_domain_scores),
                    'score_difference': np.mean(corr_domain_scores) - np.mean(orig_domain_scores),
                    'keywords': domain_keywords[domain]
                }
        
        return domain_analysis 