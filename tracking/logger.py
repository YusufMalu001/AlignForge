"""
Experiment tracking integration using Weights & Biases (wandb).
"""
import wandb
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Ensures deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_tracking(project_name: str, job_type: str, config: Dict[str, Any]) -> None:
    """
    Initializes W&B for experiment tracking.
    """
    set_seed(config.get('seed', 42))
    
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not found. W&B logging is disabled or will prompt.")
    
    wandb.init(
        project=project_name,
        job_type=job_type,
        config=config,
        resume="allow"
    )

def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """
    Logs metrics to W&B.
    """
    wandb.log(metrics, step=step)

def finish_tracking() -> None:
    """
    Closes the W&B run.
    """
    wandb.finish()

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Helper to load the YAML configuration.
    """
    with open(Path(config_path), 'r') as f:
        return yaml.safe_load(f)
