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
os.environ["WANDB_DISABLED"] = "true" 
os.environ["WANDB_MODE"] = "disabled"

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
    
    # Skip W&B entirely
    if os.environ.get("WANDB_DISABLED") == "true": 
        print("W&B disabled. Using local logging only.") 
        return

    try: 
        import wandb

        wandb.init( 
        project=project_name, 
        job_type=job_type, 
        config=config, 
        resume="allow" 
        )

        print("W&B initialized successfully.") 
    
    except Exception as e: 
        print(f"W&B initialization failed: {e}") 
        print("Continuing with local logging only.")

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
