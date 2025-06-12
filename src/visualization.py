import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os

class TranslationVisualizer:
    def __init__(self, results_dir: str = "results/evaluations"):
        """Initialize the visualizer with the results directory."""
        self.results_dir = Path(results_dir)
        self.output_dir = Path("results/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_evaluation_results(self, model_name: str, target_lang: str) -> Dict:
        """Load evaluation results from the scores.json file."""
        scores_path = self.results_dir / model_name / target_lang / "scores.json"
        with open(scores_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def plot_metric_comparison(self, model_name: str, target_lang: str):
        """Create bar plots comparing original vs corrected scores for each metric."""
        results = self.load_evaluation_results(model_name, target_lang)
        metrics = results['metrics']
        
        metrics_list = ['bleu', 'comet', 'bertscore']
        fig, axes = plt.subplots(1, len(metrics_list), figsize=(15, 5))
        fig.suptitle(f'Original vs Corrected Scores - {model_name} ({target_lang})')
        
        for idx, metric in enumerate(metrics_list):
            if metric in metrics['original'] and metric in metrics['corrected']:
                original_score = metrics['original'][metric]['mean']
                corrected_score = metrics['corrected'][metric]['mean']
                
                ax = axes[idx]
                x = ['Original', 'Corrected']
                y = [original_score, corrected_score]
                
                bars = ax.bar(x, y)
                ax.set_title(f'{metric.upper()} Score')
                ax.set_ylim(0, max(y) * 1.1)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_{target_lang}_metric_comparison.png')
        plt.close()
    
    def plot_topic_analysis(self, model_name: str, target_lang: str):
        """Create heatmaps showing metric scores across different topics."""
        results = self.load_evaluation_results(model_name, target_lang)
        topic_scores = results.get('topic_analysis', {})
        
        if not topic_scores:
            return
        
        metrics = ['bleu', 'comet', 'bertscore']
        topics = list(topic_scores.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        fig.suptitle(f'Topic Analysis - {model_name} ({target_lang})')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            original_scores = [topic_scores[topic]['original'][metric]['mean'] 
                             for topic in topics]
            corrected_scores = [topic_scores[topic]['corrected'][metric]['mean'] 
                              for topic in topics]
            
            x = np.arange(len(topics))
            width = 0.35
            
            ax.bar(x - width/2, original_scores, width, label='Original')
            ax.bar(x + width/2, corrected_scores, width, label='Corrected')
            
            ax.set_title(f'{metric.upper()} by Topic')
            ax.set_xticks(x)
            ax.set_xticklabels(topics, rotation=45, ha='right')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_{target_lang}_topic_analysis.png')
        plt.close()
    
    def plot_correlation_heatmap(self, model_name: str, target_lang: str):
        """Create a heatmap showing correlations between original and corrected scores."""
        results = self.load_evaluation_results(model_name, target_lang)
        topic_scores = results.get('topic_analysis', {})
        
        if not topic_scores:
            return
        
        topics = list(topic_scores.keys())
        metrics = ['bleu', 'comet', 'bertscore']
        
        corr_matrix = np.zeros((len(topics), len(metrics)))
        for i, topic in enumerate(topics):
            for j, metric in enumerate(metrics):
                if metric in topic_scores[topic]['correlations']:
                    corr_matrix[i, j] = topic_scores[topic]['correlations'][metric]['correlation']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, 
                   xticklabels=metrics,
                   yticklabels=topics,
                   cmap='RdYlBu',
                   center=0,
                   annot=True,
                   fmt='.2f')
        
        plt.title(f'Correlation Heatmap - {model_name} ({target_lang})')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_{target_lang}_correlation_heatmap.png')
        plt.close()
    
    def plot_score_deltas(self, model_name: str, target_lang: str):
        """Create a plot showing the deltas between original and corrected scores."""
        results = self.load_evaluation_results(model_name, target_lang)
        topic_scores = results.get('topic_analysis', {})
        
        if not topic_scores:
            return
        
        topics = list(topic_scores.keys())
        metrics = ['bleu', 'comet', 'bertscore']
        
        delta_matrix = np.zeros((len(topics), len(metrics)))
        for i, topic in enumerate(topics):
            for j, metric in enumerate(metrics):
                if metric in topic_scores[topic]['original'] and metric in topic_scores[topic]['corrected']:
                    original = topic_scores[topic]['original'][metric]['mean']
                    corrected = topic_scores[topic]['corrected'][metric]['mean']
                    delta_matrix[i, j] = corrected - original
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(delta_matrix,
                   xticklabels=metrics,
                   yticklabels=topics,
                   cmap='RdYlBu',
                   center=0,
                   annot=True,
                   fmt='.2f')
        
        plt.title(f'Score Deltas (Corrected - Original) - {model_name} ({target_lang})')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_{target_lang}_score_deltas.png')
        plt.close()
    
    def plot_combined_analysis(self, model_name: str, target_lang: str = None):
        """Create a single comprehensive plot showing all analysis aspects."""
        if target_lang:
            results = self.load_evaluation_results(model_name, target_lang)
            self._create_analysis_plot(model_name, target_lang, results)
        else:
            african_languages = ['hau', 'nso', 'tso', 'zul']
            all_results = {}
            
            for lang in african_languages:
                try:
                    results = self.load_evaluation_results(model_name, lang)
                    all_results[lang] = results
                except FileNotFoundError:
                    print(f"No results found for {model_name} - {lang}")
                    continue
            
            if not all_results:
                return
                
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(2, 2)
            
            metrics_list = ['bleu', 'comet', 'bertscore']
            
            ax1 = fig.add_subplot(gs[0, 0])
            
            model_lang_pairs = []
            original_scores = []
            corrected_scores = []
            
            for model_name, lang_data in all_results.items():
                for lang, results in lang_data.items():
                    model_lang_pairs.append(f"{model_name}\n{lang}")
                    metrics = results['metrics']
                    orig_score = np.mean([metrics['original'][m]['mean'] for m in metrics_list])
                    corr_score = np.mean([metrics['corrected'][m]['mean'] for m in metrics_list])
                    original_scores.append(orig_score)
                    corrected_scores.append(corr_score)
            
            sorted_indices = np.argsort(original_scores)
            model_lang_pairs = [model_lang_pairs[i] for i in sorted_indices]
            original_scores = [original_scores[i] for i in sorted_indices]
            corrected_scores = [corrected_scores[i] for i in sorted_indices]
            
            x = np.arange(len(model_lang_pairs))
            width = 0.35
            
            orig_bars = ax1.bar(x - width/2, original_scores, width, label='Original')
            corr_bars = ax1.bar(x + width/2, corrected_scores, width, label='Corrected')
            
            for bar in orig_bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.5f}',
                        ha='center', va='bottom', rotation=90)
            
            for bar in corr_bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.5f}',
                        ha='center', va='bottom', rotation=90)
            
            ax1.set_title('Average Metric Scores Across Models and Languages')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_lang_pairs, rotation=45, ha='right')
            ax1.legend()
            
            ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
            ax1.yaxis.set_major_locator(plt.MaxNLocator(10))
            ax1.yaxis.set_tick_params(which='major', labelsize=8)
            
            ax1.set_ylim(0, max(max(original_scores), max(corrected_scores)) * 1.15)
            
            ax2 = fig.add_subplot(gs[0, 1])
            
            all_topics = set()
            topic_deltas = {}
            
            for lang, results in all_results.items():
                topic_scores = results.get('topic_analysis', {})
                for topic in topic_scores:
                    all_topics.add(topic)
                    if topic not in topic_deltas:
                        topic_deltas[topic] = []
                    
                    deltas = []
                    for metric in metrics_list:
                        if metric in topic_scores[topic]['original'] and metric in topic_scores[topic]['corrected']:
                            orig = topic_scores[topic]['original'][metric]['mean']
                            corr = topic_scores[topic]['corrected'][metric]['mean']
                            deltas.append(corr - orig)
                    topic_deltas[topic].append(np.mean(deltas))
            
            avg_deltas = {topic: np.mean(deltas) for topic, deltas in topic_deltas.items()}
            
            sorted_topics = sorted(avg_deltas.keys(), 
                                 key=lambda x: abs(avg_deltas[x]), 
                                 reverse=True)
            sorted_deltas = [avg_deltas[topic] for topic in sorted_topics]
            
            ax2.barh(sorted_topics, sorted_deltas)
            ax2.set_title('Average Score Change by Topic (Across Languages)')
            ax2.set_xlabel('Score Change (Corrected - Original)')
            
            ax2.xaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
            
            ax3 = fig.add_subplot(gs[1, 0])
            
            avg_corr_matrix = np.zeros((len(sorted_topics), len(metrics_list)))
            for i, topic in enumerate(sorted_topics):
                for j, metric in enumerate(metrics_list):
                    correlations = []
                    for lang, results in all_results.items():
                        topic_scores = results.get('topic_analysis', {})
                        if topic in topic_scores and metric in topic_scores[topic]['correlations']:
                            correlations.append(topic_scores[topic]['correlations'][metric]['correlation'])
                    if correlations:
                        avg_corr_matrix[i, j] = np.mean(correlations)
            
            sns.heatmap(avg_corr_matrix, 
                       xticklabels=[m.upper() for m in metrics_list],
                       yticklabels=sorted_topics,
                       cmap='RdYlBu',
                       center=0,
                       annot=True,
                       fmt='.5f',
                       ax=ax3)
            ax3.set_title('Average Correlation Analysis (Across Languages)')
            
            ax4 = fig.add_subplot(gs[1, 1])
            
            avg_delta_matrix = np.zeros((len(sorted_topics), len(metrics_list)))
            for i, topic in enumerate(sorted_topics):
                for j, metric in enumerate(metrics_list):
                    deltas = []
                    for lang, results in all_results.items():
                        topic_scores = results.get('topic_analysis', {})
                        if topic in topic_scores:
                            if metric in topic_scores[topic]['original'] and metric in topic_scores[topic]['corrected']:
                                orig = topic_scores[topic]['original'][metric]['mean']
                                corr = topic_scores[topic]['corrected'][metric]['mean']
                                deltas.append(corr - orig)
                    if deltas:
                        avg_delta_matrix[i, j] = np.mean(deltas)
            
            sns.heatmap(avg_delta_matrix,
                       xticklabels=[m.upper() for m in metrics_list],
                       yticklabels=sorted_topics,
                       cmap='RdYlBu',
                       center=0,
                       annot=True,
                       fmt='.5f',
                       ax=ax4)
            ax4.set_title('Average Score Changes by Topic and Metric (Across Languages)')
            
            fig.suptitle(f'Combined Translation Analysis - {model_name}', y=1.02, fontsize=16)
            plt.tight_layout()
            
            plt.savefig(self.output_dir / f'{model_name}_combined_analysis.png', 
                       bbox_inches='tight', dpi=300)
            plt.close()
    
    def _create_analysis_plot(self, model_name: str, target_lang: str, results: dict):
        """Helper method to create analysis plot for a single language."""
        metrics = results['metrics']
        topic_scores = results.get('topic_analysis', {})
        
        if not topic_scores:
            return
        
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        metrics_list = ['bleu', 'comet', 'bertscore']
        x = np.arange(len(metrics_list))
        width = 0.35
        
        original_scores = [metrics['original'][m]['mean'] for m in metrics_list]
        corrected_scores = [metrics['corrected'][m]['mean'] for m in metrics_list]
        
        ax1.bar(x - width/2, original_scores, width, label='Original')
        ax1.bar(x + width/2, corrected_scores, width, label='Corrected')
        ax1.set_title('Overall Metric Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.upper() for m in metrics_list])
        ax1.legend()
        
        ax2 = fig.add_subplot(gs[0, 1])
        topics = list(topic_scores.keys())
        
        topic_deltas = []
        for topic in topics:
            deltas = []
            for metric in metrics_list:
                if metric in topic_scores[topic]['original'] and metric in topic_scores[topic]['corrected']:
                    orig = topic_scores[topic]['original'][metric]['mean']
                    corr = topic_scores[topic]['corrected'][metric]['mean']
                    deltas.append(corr - orig)
            topic_deltas.append(np.mean(deltas))
        
        sorted_indices = np.argsort(np.abs(topic_deltas))[::-1]
        sorted_topics = [topics[i] for i in sorted_indices]
        sorted_deltas = [topic_deltas[i] for i in sorted_indices]
        
        ax2.barh(sorted_topics, sorted_deltas)
        ax2.set_title('Average Score Change by Topic')
        ax2.set_xlabel('Score Change (Corrected - Original)')
        
        ax3 = fig.add_subplot(gs[1, 0])
        corr_matrix = np.zeros((len(topics), len(metrics_list)))
        for i, topic in enumerate(topics):
            for j, metric in enumerate(metrics_list):
                if metric in topic_scores[topic]['correlations']:
                    corr_matrix[i, j] = topic_scores[topic]['correlations'][metric]['correlation']
        
        sns.heatmap(corr_matrix, 
                   xticklabels=[m.upper() for m in metrics_list],
                   yticklabels=topics,
                   cmap='RdYlBu',
                   center=0,
                   annot=True,
                   fmt='.4f',
                   ax=ax3)
        ax3.set_title('Correlation Analysis')
        
        ax4 = fig.add_subplot(gs[1, 1])
        delta_matrix = np.zeros((len(topics), len(metrics_list)))
        for i, topic in enumerate(topics):
            for j, metric in enumerate(metrics_list):
                if metric in topic_scores[topic]['original'] and metric in topic_scores[topic]['corrected']:
                    orig = topic_scores[topic]['original'][metric]['mean']
                    corr = topic_scores[topic]['corrected'][metric]['mean']
                    delta_matrix[i, j] = corr - orig
        
        sns.heatmap(delta_matrix,
                   xticklabels=[m.upper() for m in metrics_list],
                   yticklabels=topics,
                   cmap='RdYlBu',
                   center=0,
                   annot=True,
                   fmt='.4f',
                   ax=ax4)
        ax4.set_title('Score Changes by Topic and Metric')
        
        fig.suptitle(f'Translation Analysis - {model_name} ({target_lang})', y=1.02, fontsize=16)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f'{model_name}_{target_lang}_combined_analysis.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_all_models_analysis(self):
        """Create separate comprehensive plots showing analysis for all models and languages."""
        model_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            print("No model results found")
            return
            
        metrics_list = ['bleu', 'comet', 'bertscore']
        african_languages = ['hau', 'nso', 'tso', 'zul']
        
        all_data = {}
        all_topics = set()
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            all_data[model_name] = {}
            
            for lang in african_languages:
                try:
                    results = self.load_evaluation_results(model_name, lang)
                    all_data[model_name][lang] = results
                    
                    topic_scores = results.get('topic_analysis', {})
                    all_topics.update(topic_scores.keys())
                except FileNotFoundError:
                    continue
        
        if not all_data:
            return
            
        plt.figure(figsize=(15, 10))
        
        model_lang_pairs = []
        original_scores = []
        corrected_scores = []
        
        for model_name, lang_data in all_data.items():
            for lang, results in lang_data.items():
                model_lang_pairs.append(f"{model_name}\n{lang}")
                metrics = results['metrics']
                orig_score = np.mean([metrics['original'][m]['mean'] for m in metrics_list])
                corr_score = np.mean([metrics['corrected'][m]['mean'] for m in metrics_list])
                original_scores.append(orig_score)
                corrected_scores.append(corr_score)
        
        sorted_indices = np.argsort(original_scores)
        model_lang_pairs = [model_lang_pairs[i] for i in sorted_indices]
        original_scores = [original_scores[i] for i in sorted_indices]
        corrected_scores = [corrected_scores[i] for i in sorted_indices]
        
        x = np.arange(len(model_lang_pairs))
        width = 0.35
        
        orig_bars = plt.bar(x - width/2, original_scores, width, label='Original')
        corr_bars = plt.bar(x + width/2, corrected_scores, width, label='Corrected')
        
        for bar in orig_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.5f}',
                    ha='center', va='bottom', rotation=90)
        
        for bar in corr_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.5f}',
                    ha='center', va='bottom', rotation=90)
        
        plt.title('Average Metric Scores Across Models and Languages')
        plt.xticks(x, model_lang_pairs, rotation=45, ha='right')
        plt.legend()
        
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(10))
        plt.gca().yaxis.set_tick_params(which='major', labelsize=8)
        

        plt.ylim(0, max(max(original_scores), max(corrected_scores)) * 1.15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'all_models_metric_comparison.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        

        plt.figure(figsize=(15, 10))
        
        topic_deltas = {}
        for topic in all_topics:
            deltas = []
            for model_name, lang_data in all_data.items():
                for lang, results in lang_data.items():
                    topic_scores = results.get('topic_analysis', {})
                    if topic in topic_scores:
                        for metric in metrics_list:
                            if metric in topic_scores[topic]['original'] and metric in topic_scores[topic]['corrected']:
                                orig = topic_scores[topic]['original'][metric]['mean']
                                corr = topic_scores[topic]['corrected'][metric]['mean']
                                deltas.append(corr - orig)
            if deltas:
                topic_deltas[topic] = np.mean(deltas)
        
        sorted_topics = sorted(topic_deltas.keys(), 
                             key=lambda x: abs(topic_deltas[x]), 
                             reverse=True)
        sorted_deltas = [topic_deltas[topic] for topic in sorted_topics]
        
        plt.barh(sorted_topics, sorted_deltas)
        plt.title('Average Score Change by Topic (Across All Models and Languages)')
        plt.xlabel('Score Change (Corrected - Original)')
        
        plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'all_models_topic_analysis.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        plt.figure(figsize=(15, 10))
        
        avg_corr_matrix = np.zeros((len(sorted_topics), len(metrics_list)))
        for i, topic in enumerate(sorted_topics):
            for j, metric in enumerate(metrics_list):
                correlations = []
                for model_name, lang_data in all_data.items():
                    for lang, results in lang_data.items():
                        topic_scores = results.get('topic_analysis', {})
                        if topic in topic_scores and metric in topic_scores[topic]['correlations']:
                            correlations.append(topic_scores[topic]['correlations'][metric]['correlation'])
                if correlations:
                    avg_corr_matrix[i, j] = np.mean(correlations)
        
        sns.heatmap(avg_corr_matrix, 
                   xticklabels=[m.upper() for m in metrics_list],
                   yticklabels=sorted_topics,
                   cmap='RdYlBu',
                   center=0,
                   annot=True,
                   fmt='.5f')
        plt.title('Average Correlation Analysis (Across All Models and Languages)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'all_models_correlation_analysis.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        

        plt.figure(figsize=(15, 10))
        

        avg_delta_matrix = np.zeros((len(sorted_topics), len(metrics_list)))
        for i, topic in enumerate(sorted_topics):
            for j, metric in enumerate(metrics_list):
                deltas = []
                for model_name, lang_data in all_data.items():
                    for lang, results in lang_data.items():
                        topic_scores = results.get('topic_analysis', {})
                        if topic in topic_scores:
                            if metric in topic_scores[topic]['original'] and metric in topic_scores[topic]['corrected']:
                                orig = topic_scores[topic]['original'][metric]['mean']
                                corr = topic_scores[topic]['corrected'][metric]['mean']
                                deltas.append(corr - orig)
                if deltas:
                    avg_delta_matrix[i, j] = np.mean(deltas)
        
        sns.heatmap(avg_delta_matrix,
                   xticklabels=[m.upper() for m in metrics_list],
                   yticklabels=sorted_topics,
                   cmap='RdYlBu',
                   center=0,
                   annot=True,
                   fmt='.5f')
        plt.title('Average Score Changes by Topic and Metric (Across All Models and Languages)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'all_models_metric_deltas.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    def generate_all_visualizations(self, model_name: str = None, target_lang: str = None):
        """Generate visualization plots."""
        print("Generating visualizations...")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_all_models_analysis()
        self.plot_combined_analysis(model_name, target_lang)
        
        print(f"Visualizations saved to {self.output_dir}")