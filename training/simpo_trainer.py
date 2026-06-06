"""
Stage 4: Simple Preference Optimization (SimPO) with TRL.
SimPO normalizes reward margins by length and eliminates the reference model.
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

def run_simpo(resume_from_checkpoint: bool = False, max_samples: int = None):
    config = load_config()
    set_seed(config.get("seed", 42))
    snapshot_experiment(config, "simpo_training")
    
    tier = config.get("benchmark_tier", "publishable")
    if tier == "smoke_test":
        max_samples = 10
        
    if max_samples is not None:
        config['num_train_samples'] = max_samples
        config['num_eval_samples'] = min(config.get('num_eval_samples', 50), max_samples)
        
    init_tracking(config['project_name'], "simpo", config)
    
    train_ds, eval_ds = prepare_dataset(config)
    
    # SimPO builds on top of the SFT checkpoint
    sft_checkpoint_path = Path(config['output_dir']) / "sft_checkpoint"
    ckpt_path = str(sft_checkpoint_path) if (sft_checkpoint_path / "adapter_config.json").exists() else None
    
    model, tokenizer = get_model_and_tokenizer(
        config['model_name'], 
        lora_config=config,
        is_trainable=True,
        checkpoint_path=ckpt_path
    )
    
    train_ds = train_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    eval_ds = eval_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    
    tier_path = "smoke_test" if tier == "smoke_test" else "final_benchmarks"
    output_dir = str(Path(config['output_dir']) / tier_path / "simpo_checkpoint")
    ckpt_manager = CheckpointManager(output_dir)
    
    # Calculate derived Gamma from explicit config
    simpo_cfg = config.get("simpo", {})
    beta = float(simpo_cfg.get("beta", 2.0))
    gamma_ratio = float(simpo_cfg.get("gamma_beta_ratio", 0.5))
    gamma = beta * gamma_ratio
    lr = float(simpo_cfg.get("learning_rate", 5e-5))
    epochs = int(simpo_cfg.get("num_epochs", 1))
    
    logger.info(f"SimPO Config -> Beta: {beta}, Gamma: {gamma}")

    training_args = DPOConfig(
        output_dir=output_dir,
        beta=beta,
        loss_type="simpo",
        simpo_gamma=gamma, # Critical for SimPO length-normalized margin
        learning_rate=lr,
        num_train_epochs=epochs,
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
    
    # Note: ref_model=None ensures no reference model is loaded, confirming efficiency
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        args=training_args,
    )
    
    logger.info("Starting SimPO training...")
    ckpt_manager.resume_training(trainer, output_dir, force_resume=resume_from_checkpoint)
    ckpt_manager.save_checkpoint(trainer, output_dir, "final_simpo_checkpoint")
    finish_tracking()
