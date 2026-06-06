import os
import random
import torch
import numpy as np
import logging
import json
import datetime
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Sets seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Configure deterministic training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Global seed set to {seed}")

def get_git_commit():
    """Attempts to get the current git commit hash."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return commit
    except Exception:
        return "unknown"

def snapshot_experiment(config: dict, stage_name: str) -> str:
    """
    Takes a snapshot of the config and environment and saves it to the experiment registry.
    Returns the path to the saved metadata.
    """
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{stage_name}_{timestamp}"
    
    metadata = {
        "run_id": run_id,
        "stage": stage_name,
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "model_name": config.get("model_name"),
        "dataset": config.get("dataset_name"),
        "seed": config.get("seed", 42),
        "learning_rate": config.get("learning_rate"),
        "batch_size": config.get("batch_size"),
        "lora_rank": config.get("lora_r"),
        "config_snapshot": config
    }
    
    metadata_path = experiments_dir / f"{run_id}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Experiment metadata snapshotted to {metadata_path}")
    return str(metadata_path)
