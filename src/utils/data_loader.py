from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    @staticmethod
    def load_flores_data(lang_code: str):
        """
        loads both the original and corrected FLORES datasets for a given language code
        """
        try:
            # TODO: add the hf_token and replace the datasets with the local datasets
            original_dataset = load_dataset("openlanguagedata/flores_plus", f"eng-{lang_code}")
            corrected_dataset = load_dataset("dsfsi/flores-fix-4-africa", f"eng-{lang_code}")
            return original_dataset, corrected_dataset
        except Exception as e:
            # log an error if loading fails
            logger.error(f"Error loading datasets for {lang_code}: {str(e)}")
            raise 