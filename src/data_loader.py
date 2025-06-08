from datasets import load_dataset
from typing import Dict, List, Tuple
import os

def load_original_flores(languages: List[str] = ['eng', 'hau', 'nso', 'tso', 'zul']) -> Dict[str, List[str]]:
    """
    Load the original FLORES devtest dataset for specified languages.
    
    Args:
        languages: List of language codes to load (default: ['en', 'hau', 'nso', 'tso', 'zul'])
    
    Returns:
        Dictionary containing devtest splits for each language
    """
    data = {}
    for lang in languages:
        dataset = load_dataset("openlanguagedata/flores_plus", f"{lang}_Latn")
        data[lang] = dataset['devtest']['text']
    
    return data

def load_corrected_flores(languages: List[str] = ['hau', 'nso', 'tso', 'zul']) -> Dict[str, List[str]]:
    """
    Load the corrected FLORES devtest dataset for specified languages from local directory.
    
    Args:
        languages: List of language codes to load (default: ['hau', 'nso', 'tso', 'zul'])
    
    Returns:
        Dictionary containing devtest splits for each language
    """
    data = {}
    base_path = "data/corrected"
    
    for lang in languages:
        # Load devtest set
        devtest_path = os.path.join(base_path, "devtest", f"{lang}_Latn.devtest")
        if os.path.exists(devtest_path):
            with open(devtest_path, 'r', encoding='utf-8') as f:
                data[lang] = [line.strip() for line in f if line.strip()]
                
    return data

def get_available_languages() -> List[str]:
    """
    Get list of available language codes in the dataset.
    
    Returns:
        List of language codes
    """
    return ['hau', 'nso', 'tso', 'zul'] 