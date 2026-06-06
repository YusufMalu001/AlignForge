"""
Stage 3: Direct Preference Optimization (DPO) with TRL.
"""
import argparse
from pathlib import Path
import logging

from trl import DPOTrainer, DPOConfig

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
        fmt = formatter.format_dpo(p, c, r)
        new_examples["prompt"].append(fmt["prompt"])
        new_examples["chosen"].append(fmt["chosen"])
        new_examples["rejected"].append(fmt["rejected"])
    return new_examples

def run_dpo(resume_from_checkpoint: bool = False, max_samples: int = None):
    config = load_config()
    set_seed(config.get("seed", 42))
    snapshot_experiment(config, "dpo_training")
    
    if max_samples is not None:
        config['num_train_samples'] = max_samples
        config['num_eval_samples'] = min(config.get('num_eval_samples', 50), max_samples)
        
    init_tracking(config['project_name'], "dpo", config)
    
    train_ds, eval_ds = prepare_dataset(config)
    
    sft_checkpoint_path = Path(config['output_dir']) / "sft_checkpoint"
    ckpt_path = str(sft_checkpoint_path) if (sft_checkpoint_path / "adapter_config.json").exists() else None
        
    # Load model with PEFT adapters applied directly.
    # We DO NOT create a ref_model here, TRL DPOTrainer natively handles `model.disable_adapter()`
    # saving massive amounts of RAM (1x model instead of 2x models).
    model, tokenizer = get_model_and_tokenizer(
        config['model_name'], 
        lora_config=config,
        is_trainable=True,
        checkpoint_path=ckpt_path
    )
    
    train_ds = train_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    eval_ds = eval_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    
    output_dir = str(Path(config['output_dir']) / "dpo_checkpoint")
    ckpt_manager = CheckpointManager(config['output_dir'])
    
    training_args = DPOConfig(
        output_dir=output_dir,
        beta=config['beta'],
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
    
    # ref_model is explicitly None to trigger automatic PEFT reference handling
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        args=training_args,
    )
    
    logger.info("Starting DPO training (with built-in adapter switching)...")
    ckpt_manager.resume_training(trainer, output_dir, force_resume=resume_from_checkpoint)
    ckpt_manager.save_checkpoint(trainer, output_dir, "final_dpo_checkpoint")
    finish_tracking()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    run_dpo(args.resume)
