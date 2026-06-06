import argparse
import time
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ModelSanityChecker:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def run_checks(self, test_prompt: str = "\n\nHuman: Hello!\n\nAssistant:"):
        logger.info("Running pre-evaluation sanity checks...")
        
        inputs = self.tokenizer(test_prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)
        
        start_time = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        latency = time.time() - start_time
        
        # Check NaN logits
        has_nan = any(torch.isnan(scores).any().item() for scores in outputs.scores)
        if has_nan:
            logger.error("Sanity Check Failed: NaN logits detected!")
            return False
            
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        
        # Output length check
        if len(generated_ids) == 0:
            logger.error("Sanity Check Failed: Model generated 0 tokens.")
            return False
            
        # Repeated EOS check
        if self.tokenizer.eos_token_id in generated_ids:
            eos_count = (generated_ids == self.tokenizer.eos_token_id).sum().item()
            if eos_count > 1:
                logger.warning(f"Sanity Check Warning: Generated {eos_count} EOS tokens.")
        
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if not response.strip():
            logger.error("Sanity Check Failed: Model generated empty string response.")
            return False
            
        logger.info(f"Sanity check passed! Latency: {latency:.2f}s | Sample: {response[:30]}...")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model ID or 'base'")
    args = parser.parse_args()
    
    # Simple mapping for 'base'
    model_id = "Qwen/Qwen2-0.5B-Instruct" if args.model == "base" else args.model
    
    logger.info(f"Loading {model_id} for sanity check...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    model.eval()
    
    checker = ModelSanityChecker(model, tokenizer)
    success = checker.run_checks()
    if not success:
        exit(1)
