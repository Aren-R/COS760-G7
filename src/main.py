from data_loader import load_original_flores, load_corrected_flores
from models import initialize_models
from translation import TranslationPipeline
from evaluation import TranslationEvaluator
import json

# Configuration
DEBUG_SIZE = 100
TRANSLATION_BATCH_SIZE = 10
DEBUG = False

def main():
    print("\nLoading datasets...")
    english_data = load_original_flores(languages=['eng'])
    original_data = load_original_flores()
    corrected_data = load_corrected_flores()

    # Truncate datasets if in DEBUG mode
    if DEBUG:
        print(f"Running in DEBUG mode with {DEBUG_SIZE} samples per language")
        english_data = {lang: texts[:DEBUG_SIZE] for lang, texts in english_data.items()}
        original_data = {lang: texts[:DEBUG_SIZE] for lang, texts in original_data.items()}
        corrected_data = {lang: texts[:DEBUG_SIZE] for lang, texts in corrected_data.items()}
    else:
        print("Running with full dataset")

    print("\nInitializing translation models...")
    models = initialize_models()
    
    pipeline = TranslationPipeline(models)
    evaluator = TranslationEvaluator()
    
    african_languages = ['hau', 'nso', 'tso', 'zul']
    
    print("\nStarting translations and evaluations...")
    for model_name in models.keys():
        print(f"\nProcessing {model_name} model...")
        for target_lang in african_languages:
            print(f"\nTranslating to {target_lang}...")
            try:
                translations = pipeline.translate_batch(
                    texts=english_data['eng'],
                    source_lang='en',
                    target_lang=target_lang,
                    model_name=model_name,
                    batch_size=TRANSLATION_BATCH_SIZE
                )
                
                print("Evaluating translations...")
                original_refs = original_data[target_lang]
                corrected_refs = corrected_data[target_lang]
                
                # Calculate scores for all metrics
                original_scores = evaluator.evaluate_translations(
                    translations=translations,
                    references=original_refs,
                    metrics=['bleu', 'comet', 'bertscore']
                )
                
                corrected_scores = evaluator.evaluate_translations(
                    translations=translations,
                    references=corrected_refs,
                    metrics=['bleu', 'comet', 'bertscore']
                )
                
                # Save individual evaluation results
                evaluator.save_evaluation_results(
                    model_name=model_name,
                    target_lang=target_lang,
                    original_scores=original_scores,
                    corrected_scores=corrected_scores,
                    translations=translations,
                    original_refs=original_refs,
                    corrected_refs=corrected_refs
                )
                
                print(f"Results saved for {model_name} ({target_lang})")
                
            except Exception as e:
                print(f"Error processing {target_lang}: {str(e)}")
                print("Skipping to next language pair...")
    

if __name__ == "__main__":
    main()