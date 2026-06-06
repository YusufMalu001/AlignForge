"""
Stage 2: Reward Model Training with TRL.
"""
import argparse
from pathlib import Path
import logging
import numpy as np
import torch
from transformers import TrainingArguments
from trl import RewardTrainer, RewardConfig

from tracking.logger import init_tracking, load_config, finish_tracking
from data.loader import prepare_dataset
from data.formatting import UnifiedFormatter
from models.base_loader import get_reward_model_and_tokenizer
from utils.reproducibility import set_seed, snapshot_experiment
from utils.checkpoint_manager import CheckpointManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    """Computes custom metrics for pairwise ranking."""
    predictions, _ = eval_pred
    
    # predictions is a tuple/array depending on output
    # TRL RewardTrainer outputs logits for chosen and rejected
    if isinstance(predictions, tuple):
        # Trl usually returns a tuple of (chosen_logits, rejected_logits)
        # But depending on version, it might be an array of shape (batch, 2)
        pass
    
    # Standard format for RewardTrainer compute_metrics:
    # eval_pred.predictions usually contains a numpy array of shape (batch_size, 2) [chosen, rejected] or similar.
    # We will compute pairwise accuracy and margin manually
    # Let's assume standard HF formatting where predictions is (chosen_logits, rejected_logits)
    try:
        if isinstance(eval_pred.predictions, tuple):
            chosen_logits, rejected_logits = eval_pred.predictions
        else:
            # If shape is (batch, 2)
            chosen_logits = eval_pred.predictions[:, 0]
            rejected_logits = eval_pred.predictions[:, 1]
    except Exception as e:
        logger.error(f"Could not parse predictions for metrics: {e}")
        return {"accuracy": 0.0, "reward_margin": 0.0}

    accuracy = np.mean(chosen_logits > rejected_logits)
    margin = np.mean(chosen_logits - rejected_logits)
    variance = np.var(np.concatenate([chosen_logits, rejected_logits]))
    
    return {
        "pairwise_accuracy": accuracy,
        "reward_margin": margin,
        "reward_variance": float(variance)
    }

def preprocess_function(examples, tokenizer, max_length):
    formatter = UnifiedFormatter(tokenizer)
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    
    for p, c, r in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        # CORRECT IMPLEMENTATION: Tokenize prompt + response
        fmt = formatter.format_reward(p, c, r)
        
        tokenized_chosen = tokenizer(fmt["chosen_text"], truncation=True, max_length=max_length)
        tokenized_rejected = tokenizer(fmt["rejected_text"], truncation=True, max_length=max_length)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

def run_rm(resume_from_checkpoint: bool = False, max_samples: int = None):
    config = load_config()
    set_seed(config.get("seed", 42))
    snapshot_experiment(config, "reward_training")
    
    if max_samples is not None:
        config['num_train_samples'] = max_samples
        config['num_eval_samples'] = min(config.get('num_eval_samples', 50), max_samples)
        
    init_tracking(config['project_name'], "rm", config)
    
    train_ds, eval_ds = prepare_dataset(config)
    
    sft_checkpoint_path = Path(config['output_dir']) / "sft_checkpoint"
    ckpt_path = str(sft_checkpoint_path) if (sft_checkpoint_path / "adapter_config.json").exists() else None
        
    model, tokenizer = get_reward_model_and_tokenizer(
        config['model_name'], 
        lora_config=config,
        is_trainable=True,
        checkpoint_path=ckpt_path
    )
    
    train_ds = train_ds.map(lambda x: preprocess_function(x, tokenizer, config['max_length']), batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(lambda x: preprocess_function(x, tokenizer, config['max_length']), batched=True, remove_columns=eval_ds.column_names)
    
    output_dir = str(Path(config['output_dir']) / "rm_checkpoint")
    ckpt_manager = CheckpointManager(config['output_dir'])
    
    training_args = RewardConfig(
        output_dir=output_dir,
        learning_rate=float(config['learning_rate']),
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        save_strategy=config['save_strategy'],
        save_steps=config['save_steps'],
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=config['save_steps'],
        seed=config['seed'],
        report_to="none",
        use_cpu=True,
        max_length=config['max_length'],
    )
    
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )
    
    logger.info("Starting Reward Model training...")
    ckpt_manager.resume_training(trainer, output_dir, force_resume=resume_from_checkpoint)
    ckpt_manager.save_checkpoint(trainer, output_dir, "final_rm_checkpoint")
    finish_tracking()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max_samples", type=int, default=None, help="Override number of training samples")
    args = parser.parse_args()
    run_rm(args.resume, args.max_samples)
