import os
from typing import Dict, List
from datasets import load_dataset

def load_language_dataset(lang_code: str) -> Dict[str, List[str]]:
    """Load both original and corrected datasets for a language"""
    # Load original dataset from Hugging Face
    original_dataset = load_dataset(
        "openlanguagedata/flores_plus",
        f"{lang_code}_Latn"
    )
    
    # Get the project root directory (go up from src/ to project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Load corrected dataset from local file using absolute path
    corrected_path = os.path.join(project_root, "data", "corrected", "devtest", f"{lang_code}_Latn.devtest")
    if not os.path.exists(corrected_path):
        raise FileNotFoundError(f"Corrected file not found at: {corrected_path}")
        
    with open(corrected_path, 'r', encoding='utf-8') as f:
        corrected_texts = f.read().splitlines()
    
    return {
        'original': original_dataset['devtest']['text'],
        'corrected': corrected_texts
    }