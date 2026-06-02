"""
Dataset loader for hh-rlhf.
"""
from datasets import load_dataset, Dataset
from pathlib import Path
import logging
from typing import Dict, Tuple

from data.sampler import sample_dataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def parse_hh_rlhf(text: str) -> Tuple[str, str]:
    """
    Parses the Anthropic hh-rlhf format into (prompt, response).
    Assumes text starts with \n\nHuman: and ends with \n\nAssistant: ...
    """
    delimiter = "\n\nAssistant:"
    parts = text.rpartition(delimiter)
    
    if not parts[1]:
        return "", text
        
    prompt = parts[0] + parts[1] # e.g. \n\nHuman: ... \n\nAssistant:
    response = parts[2].lstrip() # text after Assistant:
    
    return prompt.strip(), response.strip()

def prepare_dataset(config: Dict) -> Tuple[Dataset, Dataset]:
    """
    Loads, samples, and formats the dataset for DPO and SFT.
    """
    logger.info(f"Loading dataset {config['dataset_name']}")
    
    # Load helpful-base split (subset of hh-rlhf)
    # The dataset has multiple directories, 'helpful-base' is one of them.
    train_ds = load_dataset(config['dataset_name'], data_dir="helpful-base", split="train")
    eval_ds = load_dataset(config['dataset_name'], data_dir="helpful-base", split="test")
    
    # Sample
    train_ds = sample_dataset(train_ds, config['num_train_samples'], seed=config['seed'])
    eval_ds = sample_dataset(eval_ds, config['num_eval_samples'], seed=config['seed'])
    
    def format_row(example):
        # Anthropic format is full conversation string
        chosen_prompt, chosen_resp = parse_hh_rlhf(example['chosen'])
        _, rejected_resp = parse_hh_rlhf(example['rejected'])
        
        return {
            "prompt": chosen_prompt,
            "chosen": chosen_resp,
            "rejected": rejected_resp,
            "text": f"{chosen_prompt} {chosen_resp}" # For SFTTrainer format
        }
        
    train_ds = train_ds.map(format_row, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(format_row, remove_columns=eval_ds.column_names)
    
    return train_ds, eval_ds
