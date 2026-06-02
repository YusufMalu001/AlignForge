"""
Dataset sampling strategy for CPU-feasible training.
Implements stratified sampling, varying prompt lengths, and difficult examples.
"""
from datasets import Dataset
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def sample_dataset(dataset: Dataset, num_samples: int, seed: int = 42) -> Dataset:
    """
    Samples the dataset carefully to ensure a good mix of prompt lengths
    and difficult/rejection-heavy examples (approximated by response length diffs).
    """
    random.seed(seed)
    logger.info(f"Sampling {num_samples} from dataset of size {len(dataset)}")
    
    # Calculate a proxy for "difficulty" or difference
    def add_diff(example):
        chosen_len = len(example['chosen'])
        rejected_len = len(example['rejected'])
        example['length_diff'] = abs(chosen_len - rejected_len)
        return example
        
    dataset = dataset.map(add_diff, desc="Calculating response diffs")
    
    # Sort or stratify based on length_diff
    dataset_list = list(dataset)
    # Sort by diff descending to get "difficult" or distinct examples
    dataset_list.sort(key=lambda x: x['length_diff'], reverse=True)
    
    # We take the top X% distinct examples, and random sample the rest
    top_n = min(num_samples // 2, len(dataset_list))
    selected = dataset_list[:top_n]
    
    remaining = dataset_list[top_n:]
    random.shuffle(remaining)
    selected.extend(remaining[:(num_samples - top_n)])
    
    random.shuffle(selected)
    logger.info(f"Final sampled dataset size: {len(selected)}")
    
    # Create dataset back from list
    return Dataset.from_list(selected)
