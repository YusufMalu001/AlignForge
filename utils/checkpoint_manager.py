import os
import shutil
from pathlib import Path
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class CheckpointManager:
    """Centralized checkpoint management for CPU training."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, trainer, output_dir: str, name: str = "checkpoint"):
        """Saves a model checkpoint."""
        checkpoint_path = Path(output_dir) / name
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        trainer.save_model(str(checkpoint_path))
        
    def find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        """Finds the most recent checkpoint in the output directory."""
        dir_path = Path(output_dir)
        if not dir_path.exists():
            return None
            
        checkpoints = [d for d in dir_path.iterdir() if d.is_dir() and "checkpoint" in d.name]
        if not checkpoints:
            return None
            
        # Assuming checkpoints are named with a step suffix or similar, we sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(checkpoints[0])

    def resume_training(self, trainer, output_dir: str, force_resume: bool = False):
        """Attempts to resume training from the latest checkpoint."""
        latest = self.find_latest_checkpoint(output_dir)
        if latest and force_resume:
            logger.info(f"Resuming training from latest checkpoint: {latest}")
            trainer.train(resume_from_checkpoint=latest)
        else:
            logger.info("Starting training from scratch.")
            trainer.train()

    def cleanup_old_checkpoints(self, output_dir: str, keep_last: int = 3):
        """Removes older checkpoints to save disk space."""
        dir_path = Path(output_dir)
        if not dir_path.exists():
            return
            
        checkpoints = [d for d in dir_path.iterdir() if d.is_dir() and "checkpoint" in d.name]
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if len(checkpoints) > keep_last:
            for old_ckpt in checkpoints[keep_last:]:
                logger.info(f"Cleaning up old checkpoint: {old_ckpt}")
                shutil.rmtree(old_ckpt, ignore_errors=True)
