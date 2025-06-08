from datasets import load_dataset
from typing import Dict, List, Tuple
import os
import json
from pathlib import Path
from tqdm import tqdm

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get_cache_path(dataset_type: str, language: str) -> Path:
    """Get the path for the dataset cache file"""
    return CACHE_DIR / f"{dataset_type}_{language}.json"

def _load_from_cache(cache_path: Path) -> List[str]:
    """Load dataset from cache"""
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def _save_to_cache(cache_path: Path, data: List[str]):
    """Save dataset to cache"""
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_original_flores(languages: List[str] = ['hau', 'nso', 'tso', 'zul'], use_cache: bool = True) -> Dict[str, List[str]]:
    """
    Load the original FLORES devtest dataset for specified languages.
    
    Args:
        languages: List of language codes to load (default: ['en', 'hau', 'nso', 'tso', 'zul'])
        use_cache: Whether to use cached dataset (default: True)
    
    Returns:
        Dictionary containing devtest splits for each language
    """
    data = {}
    print("\nLoading original FLORES dataset...")
    
    for lang in tqdm(languages, desc="Loading languages"):
        cache_path = _get_cache_path("original", lang)
        
        if use_cache:
            cached_data = _load_from_cache(cache_path)
            if cached_data is not None:
                data[lang] = cached_data
                continue
        
        dataset = load_dataset("openlanguagedata/flores_plus", f"{lang}_Latn")
        data[lang] = dataset['devtest']['text']
        
        if use_cache:
            _save_to_cache(cache_path, data[lang])
    
    return data

def load_corrected_flores(languages: List[str] = ['hau', 'nso', 'tso', 'zul'], use_cache: bool = True) -> Dict[str, List[str]]:
    """
    Load the corrected FLORES devtest dataset for specified languages from local directory.
    
    Args:
        languages: List of language codes to load (default: ['hau', 'nso', 'tso', 'zul'])
        use_cache: Whether to use cached dataset (default: True)
    
    Returns:
        Dictionary containing devtest splits for each language
    """
    data = {}
    base_path = "data/corrected"
    print("\nLoading corrected FLORES dataset...")
    
    for lang in tqdm(languages, desc="Loading languages"):
        cache_path = _get_cache_path("corrected", lang)
        
        if use_cache:
            cached_data = _load_from_cache(cache_path)
            if cached_data is not None:
                data[lang] = cached_data
                continue
        
        devtest_path = os.path.join(base_path, "devtest", f"{lang}_Latn.devtest")
        if os.path.exists(devtest_path):
            with open(devtest_path, 'r', encoding='utf-8') as f:
                data[lang] = [line.strip() for line in f if line.strip()]
            
            if use_cache:
                _save_to_cache(cache_path, data[lang])
                
    return data

def get_available_languages() -> List[str]:
    """
    Get list of available language codes in the dataset.
    
    Returns:
        List of language codes
    """
    return ['hau', 'nso', 'tso', 'zul'] 