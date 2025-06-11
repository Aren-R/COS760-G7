# Impact of Reference Corrections on Machine Translation Model Rankings in Low-Resource African Languages

This project investigates how reference corrections in the FLORES evaluation dataset (devtest sets) for four African languages, Hausa, Northern Sotho (Sepedi), Xitsonga, and isiZulu, affect machine translation (MT) model rankings. Although the original FLORES dataset made important strides in covering low-resource languages, it contained numerous translation inaccuracies. These inconsistencies compromise the reliability of standard evaluation metrics and can distort model rankings, leading to misleading conclusions in MT research.

## Overview

We examine the impact of manually corrected references (from Abdulmumin et al., 2024) on MT model evaluation. Our analysis focuses on changes in model rankings, score sensitivity across metrics (BLEU, COMET, BERTScore), and variation across text domains.

## What We Did

- **Model Output Evaluation**: We evaluated outputs from several MT models (NLLB, OPUS-MT, MADLAD-400) using both original and corrected references.
- **Ranking Analysis**: We assessed how model rankings changed post-correction and analysed which models were most robust or sensitive to reference quality.
- **Metric Comparison**: We calculated score differences and Spearman’s correlation to measure the ranking volatility for each model-metric combination.
- **Domain-Level Exploration**: We examined if the thematic domain of the text influenced score deltas or ranking shifts.

## Key Insights from Corrections

- **Minimal Score Shifts**: Unlike Abdulmumin et al. (2024), who observed up to a 16% score change, we saw minimal differences between the original and corrected datasets, suggesting that models have since been trained on the improved data.

- **Stable Model Rankings**: Corrections did not significantly alter the relative performance of models. Rankings remained largely consistent across metrics.

- **Best & Worst Performers**:  
  - *Best*: **NLLB-200-distilled-600M** performed best overall across languages.  
  - *Worst*: **MADLAD400-3b-mt**, despite being the largest and most multilingual model, it underperformed which highlighted the fact that language coverage may matter more than parameter size.

- **Language Trends**: Models consistently performed best when translating **English to Hausa**, likely due to Hausa being the highest-resource language in the evaluation, further proving the resource to performance link.

- **Domain Performance**:  
  - *Weakest*: Entertainment domain, likely due to a lack of such content in African languages.  
  - *Strongest*: Health, politics, and travel—domains with more translatable vocabularies.

- **Statistical Consistency**: Spearman Rank correlation in model rankings was high and statistically significant (p < 10⁻⁷), confirming consistency in evaluations.

- **Challenges Identified**:  
  - Difficulty locating models supporting all target languages due to inconsistent documentation and code standards.  
  - High computational demands despite no model training, posing challenges for reproducibility.  
  - Time-sensitive nature of evaluation since dataset corrections get integrated rapidly, limiting window for comparative impact analysis.


## How to Use This Work

Use our findings to better understand how MT model rankings are affected by data quality in underrepresented languages. Our corrected reference datasets and evaluation scripts help support more equitable and accurate benchmarking practices.

## Responsible NLP Statement

Our work contributes to responsible NLP by highlighting how poor reference translations can unfairly penalise or benefit certain models, especially in low-resource settings. Ensuring reliable evaluation is crucial for the ethical development of MT tools that may influence education, information access, and digital equity in African communities.

## How to run
pip install -r requirements.txt
python src/main.py


## Acknowledgments

- Abdulmumin, I., Mkhwanazi, S., Mbooi, M., Muhammad, S.H., Ahmad, I.S., Putini, N., Mathebula, M., Shingange, M., Gwadabe, T., & Marivate, V. (2024). *Correcting FLORES Evaluation Dataset for Four African Languages*. Proceedings of the Ninth Conference on Machine Translation. https://doi.org/10.18653/v1/2024.wmt-1.44
- Tiedemann, J. (2020). *OPUS-MT: Building open translation services for the World*. arXiv preprint arXiv:2005.05943.
- Kudugunta, S. et al. (2023). *MADLAD-400: A Multilingual And Document-Level Large Audited Dataset*, arXiv [cs.CL] [Preprint]. Available at: http://arxiv.org/abs/2309.04662.
- Team, Facebook AI. (2022). *NLLB: No Language Left Behind*. https://ai.facebook.com/research/no-language-left-behind
- ‘No Language Left Behind: Scaling Human-Centered Machine Translation’ (2022).
