import argparse
from pathlib import Path
import logging
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def export_to_onnx(model_id: str, output_dir: str):
    logger.info(f"Exporting {model_id} to ONNX format...")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load model and export to ONNX
    ort_model = ORTModelForCausalLM.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Save ONNX model
    ort_model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    
    logger.info(f"Successfully exported ONNX model to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HuggingFace Model ID or local path")
    parser.add_argument("--output", type=str, default="./results/onnx_model", help="Output directory")
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.output)
