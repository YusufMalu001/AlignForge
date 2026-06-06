import argparse
import time
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def benchmark_latency(model, tokenizer, prompt: str, num_runs: int = 10, max_tokens: int = 50):
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = inputs.to(model.device)
        
    # Warmup
    model.generate(**inputs, max_new_tokens=5)
    
    latencies = []
    for _ in range(num_runs):
        start = time.time()
        model.generate(**inputs, max_new_tokens=max_tokens)
        latencies.append(time.time() - start)
        
    avg_latency = sum(latencies) / len(latencies)
    return avg_latency

def run_benchmarks(pt_model_id: str, onnx_model_dir: str):
    prompt = "\n\nHuman: What is the meaning of life?\n\nAssistant:"
    tokenizer = AutoTokenizer.from_pretrained(pt_model_id)
    
    logger.info("Loading PyTorch model...")
    pt_model = AutoModelForCausalLM.from_pretrained(pt_model_id, torch_dtype=torch.float32, device_map="cpu")
    
    logger.info("Loading ONNX model...")
    ort_model = ORTModelForCausalLM.from_pretrained(onnx_model_dir)
    
    logger.info("Benchmarking PyTorch Latency...")
    pt_latency = benchmark_latency(pt_model, tokenizer, prompt)
    
    logger.info("Benchmarking ONNX Latency...")
    ort_latency = benchmark_latency(ort_model, tokenizer, prompt)
    
    logger.info(f"PyTorch Avg Latency: {pt_latency:.4f}s")
    logger.info(f"ONNX Avg Latency: {ort_latency:.4f}s")
    logger.info(f"Speedup: {pt_latency / ort_latency:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_model", type=str, required=True, help="Original PyTorch Model ID")
    parser.add_argument("--onnx_model", type=str, required=True, help="ONNX Model Directory")
    args = parser.parse_args()
    
    run_benchmarks(args.pt_model, args.onnx_model)
