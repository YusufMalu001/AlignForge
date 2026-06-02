"""
Runs baseline and DPO model on eval prompts.
Uses dynamic quantization for CPU speedup.
"""
import json
from pathlib import Path
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

from tracking.logger import load_config
from data.loader import prepare_dataset
from models.base_loader import apply_dynamic_quantization

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_response(model, tokenizer, prompt: str, max_length: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def run_evaluation():
    config = load_config()
    _, eval_ds = prepare_dataset(config)
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading baseline model...")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    baseline_model = apply_dynamic_quantization(baseline_model)
    baseline_model.eval()
    
    logger.info("Loading DPO model...")
    dpo_checkpoint = str(Path(config['output_dir']) / "dpo_checkpoint")
    if not Path(dpo_checkpoint).exists():
        logger.error(f"DPO checkpoint not found at {dpo_checkpoint}. Run DPO first.")
        return
        
    dpo_model_base = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    dpo_model = PeftModel.from_pretrained(dpo_model_base, dpo_checkpoint).merge_and_unload()
    dpo_model = apply_dynamic_quantization(dpo_model)
    dpo_model.eval()
    
    results = []
    
    logger.info(f"Generating responses for {len(eval_ds)} evaluation prompts...")
    for item in tqdm(eval_ds):
        prompt = item['prompt']
        
        base_resp = generate_response(baseline_model, tokenizer, prompt, config['max_length'])
        dpo_resp = generate_response(dpo_model, tokenizer, prompt, config['max_length'])
        
        results.append({
            "prompt": prompt,
            "baseline_response": base_resp,
            "dpo_response": dpo_resp
        })
        
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "eval_outputs.json", "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Saved evaluation outputs to {output_dir / 'eval_outputs.json'}")

if __name__ == "__main__":
    run_evaluation()
