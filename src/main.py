from data_loader import load_original_flores, load_corrected_flores
from models import initialize_models
from translation import TranslationPipeline
from evaluation import TranslationEvaluator

DEBUG_SIZE = 10

def main():
     #Load English data
    english_data = load_original_flores(languages=['eng'])
    english_data = {lang: texts[:DEBUG_SIZE] for lang, texts in english_data.items()}

    # Load original FLORES dataset
    original_data = load_original_flores()
    original_data = {lang: texts[:DEBUG_SIZE] for lang, texts in original_data.items()}
    print("Successfully loaded original FLORES devtest dataset")
    print(f"Available languages: {list(original_data.keys())}")
    
    # Load corrected FLORES dataset
    corrected_data = load_corrected_flores()
    corrected_data = {lang: texts[:DEBUG_SIZE] for lang, texts in corrected_data.items()}
    print("\nSuccessfully loaded corrected FLORES devtest dataset")
    print(f"Available languages: {list(corrected_data.keys())}")
    
    # Initialize translation models
    print("\nInitializing translation models...")
    models = initialize_models()
    
    # Initialize translation pipeline and evaluator
    pipeline = TranslationPipeline(models)
    evaluator = TranslationEvaluator()
    
    # Translate English data to each African language using each model
    african_languages = ['hau', 'nso', 'tso', 'zul']
    
    print("\nStarting translations and evaluations...")
    for model_name in models.keys():
        print(f"\nUsing {model_name} model:")
        for target_lang in african_languages:
            print(f"\nTranslating English to {target_lang}...")
            try:
                translations = pipeline.translate_batch(
                    texts=english_data['eng'],
                    source_lang='en',
                    target_lang=target_lang,
                    model_name=model_name,
                    batch_size=4  # Small batch size for testing
                )
                
                # Print first translation as example
                print("\nExample translation:")
                print(f"English: {english_data['eng'][0]}")
                print(f"{target_lang}: {translations[0]}")
                
                # # Evaluate translations against both original and corrected references
                # print("\nEvaluating translations...")
                # print("target lang: ", target_lang)
                # # print(original_data[target_lang])
                # # Get references from both datasets
                # original_refs = original_data[target_lang]
                # corrected_refs = corrected_data[target_lang]
                # print("starting with scores")
                # # Calculate BLEU scores
                # original_scores = evaluator.evaluate_translations(
                #     translations=translations,
                #     references=original_refs,
                #     metrics=['bleu']
                # )
                
                # corrected_scores = evaluator.evaluate_translations(
                #     translations=translations,
                #     references=corrected_refs,
                #     metrics=['bleu']
                # )
                
                # print(f"\nBLEU Scores for {model_name} ({target_lang}):")
                # print(f"Original references: {original_scores['bleu']:.4f}")
                # print(f"Corrected references: {corrected_scores['bleu']:.4f}")
                
            except Exception as e:
                print(f"Error processing {target_lang}: {str(e)}")
                print("Skipping to next language pair...")

if __name__ == "__main__":
    main()