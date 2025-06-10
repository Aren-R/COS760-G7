from data_loader import load_original_flores, load_corrected_flores
from models import initialize_models
from translation import TranslationPipeline
from evaluation import TranslationEvaluator, get_translations_by_topic
from visualization import TranslationVisualizer

# Configuration
TRUNCATED_DATASET_SIZE = 100
TRANSLATION_BATCH_SIZE = 10
TRUNCATE_DATASET = False

# Section Control Flags
RUN_TRANSLATIONS = False
RUN_EVALUATIONS = False
RUN_VISUALIZATIONS = True


def main():

    # LOAD DATASETS ------------------------------------------------------------------------------
    print("\nLoading datasets...")
    english_data = load_original_flores(languages=['eng'])
    original_data = load_original_flores()
    corrected_data = load_corrected_flores()
    #---------------------------------------------------------------------------------------------

    # TRUNCATE DATASET MODE TRUNCATION-----------------------------------------------------------------------
    if TRUNCATE_DATASET:
        print(f"Running in DEBUG mode with {TRUNCATED_DATASET_SIZE} samples per language")
        english_data = {lang: texts[:TRUNCATED_DATASET_SIZE] for lang, texts in english_data.items()}
        original_data = {lang: texts[:TRUNCATED_DATASET_SIZE] for lang, texts in original_data.items()}
        corrected_data = {lang: texts[:TRUNCATED_DATASET_SIZE] for lang, texts in corrected_data.items()}
    else:
        print("Running with full dataset")
    #---------------------------------------------------------------------------------------------

    # INITIALISATION------------------------------------------------------------------------------
    print(f"Initializing translation models...")
    models = initialize_models()
    
    pipeline = TranslationPipeline(models)
    evaluator = TranslationEvaluator()
    visualizer = TranslationVisualizer()
    #---------------------------------------------------------------------------------------------
    
    # LANGUAGES FOR EVALUATION
    languages_for_evaluation = ['hau', 'nso', 'tso', 'zul']
    
    # PERFORM TRANSLATIONS AND EVALUATIONS--------------------------------------------------------
    if RUN_TRANSLATIONS or RUN_EVALUATIONS:
        print("\nStarting translations and evaluations...")
        for model_name in models.keys():
            print(f"\nProcessing {model_name} model...")
            for target_lang in languages_for_evaluation:
                print(f"\nTranslating to {target_lang}...")
                try:
                    if RUN_TRANSLATIONS:
                        translations = pipeline.translate_batch(
                            texts=english_data['eng'],
                            source_lang='en',
                            target_lang=target_lang,
                            model_name=model_name,
                            batch_size=TRANSLATION_BATCH_SIZE
                        )
                    
                    if RUN_EVALUATIONS:
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
                    
                except Exception as e:
                    print(f"Error processing {target_lang}: {str(e)}")
                    print("Skipping to next language pair...")

    #---------------------------------------------------------------------------------------------

    # CREATE VISUALISATIONS-----------------------------------------------------------------------
    if RUN_VISUALIZATIONS:
        print("\nGenerating visualizations...")
        for model_name in models.keys():
            for target_lang in languages_for_evaluation:
                try:
                    print(f"\nGenerating visualizations for {model_name} - {target_lang}...")
                    visualizer.generate_all_visualizations(model_name, target_lang)
                except Exception as e:
                    print(f"Error generating visualizations for {model_name} - {target_lang}: {str(e)}")
                    print("Skipping to next model-language pair...")

    #---------------------------------------------------------------------------------------------
    
if __name__ == "__main__":
    main()