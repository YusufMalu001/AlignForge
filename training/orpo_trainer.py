"""
Stage 3 (Alternative): Odds Ratio Preference Optimization (ORPO) with TRL.
ORPO does not require a reference model, merging SFT and Alignment into a single stage.
"""
import argparse
from pathlib import Path
import logging

from trl import ORPOTrainer, ORPOConfig
from tracking.logger import init_tracking, load_config, finish_tracking
from data.loader import prepare_dataset
from data.formatting import UnifiedFormatter
from models.base_loader import get_model_and_tokenizer
from utils.reproducibility import set_seed, snapshot_experiment
from utils.checkpoint_manager import CheckpointManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def preprocess_function(examples, tokenizer):
    formatter = UnifiedFormatter(tokenizer)
    new_examples = {"prompt": [], "chosen": [], "rejected": []}
    for p, c, r in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        # ORPO expects prompt, chosen, rejected in the same format as DPO
        fmt = formatter.format_dpo(p, c, r)
        new_examples["prompt"].append(fmt["prompt"])
        new_examples["chosen"].append(fmt["chosen"])
        new_examples["rejected"].append(fmt["rejected"])
    return new_examples

def run_orpo(resume_from_checkpoint: bool = False, max_samples: int = None):
    config = load_config()
    set_seed(config.get("seed", 42))
    snapshot_experiment(config, "orpo_training")
    
    tier = config.get("benchmark_tier", "publishable")
    if tier == "smoke_test":
        max_samples = 10
        
    if max_samples is not None:
        config['num_train_samples'] = max_samples
        config['num_eval_samples'] = min(config.get('num_eval_samples', 50), max_samples)
        
    init_tracking(config['project_name'], "orpo", config)
    
    train_ds, eval_ds = prepare_dataset(config)
    
    # ORPO does not strictly need SFT checkpoint, but we can load one or start from base
    # We will start from base model + LoRA
    model, tokenizer = get_model_and_tokenizer(
        config['model_name'], 
        lora_config=config,
        is_trainable=True,
        checkpoint_path=None
    )
    
    train_ds = train_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    eval_ds = eval_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    
    # Output path based on tier
    tier_path = "smoke_test" if tier == "smoke_test" else "final_benchmarks"
    output_dir = str(Path(config['output_dir']) / tier_path / "orpo_checkpoint")
    ckpt_manager = CheckpointManager(output_dir)
    
    training_args = ORPOConfig(
        output_dir=output_dir,
        beta=config['beta'], # Odds Ratio weight
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
        max_prompt_length=config['max_length'] // 2,
        max_length=config['max_length']
    )
    
    trainer = ORPOTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        args=training_args,
    )
    
    logger.info("Starting ORPO training...")
    ckpt_manager.resume_training(trainer, output_dir, force_resume=resume_from_checkpoint)
    ckpt_manager.save_checkpoint(trainer, output_dir, "final_orpo_checkpoint")
    finish_tracking()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max_samples", type=int, default=None, help="Override number of training samples")
    args = parser.parse_args()
    run_orpo(args.resume, args.max_samples)
