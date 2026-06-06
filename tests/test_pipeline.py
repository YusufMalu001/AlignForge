import pytest
import torch
import os
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.formatting import UnifiedFormatter
from models.base_loader import get_model_and_tokenizer, apply_dynamic_quantization
from utils.checkpoint_manager import CheckpointManager

# --- Mock Data ---
MOCK_PROMPT = "\n\nHuman: Hello!\n\nAssistant:"
MOCK_CHOSEN = "Hi there! How can I help?"
MOCK_REJECTED = "I don't know."
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"

@pytest.fixture(scope="module")
def tokenizer():
    t = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t

@pytest.fixture(scope="module")
def formatter(tokenizer):
    return UnifiedFormatter(tokenizer)

def test_dataset_formatting_sft(formatter, tokenizer):
    fmt = formatter.format_sft(MOCK_PROMPT, MOCK_CHOSEN)
    assert MOCK_PROMPT in fmt
    assert MOCK_CHOSEN in fmt
    assert fmt.endswith(tokenizer.eos_token)

def test_dataset_formatting_reward(formatter, tokenizer):
    fmt = formatter.format_reward(MOCK_PROMPT, MOCK_CHOSEN, MOCK_REJECTED)
    assert "chosen_text" in fmt
    assert "rejected_text" in fmt
    assert MOCK_PROMPT in fmt["chosen_text"]
    assert MOCK_PROMPT in fmt["rejected_text"]
    assert fmt["chosen_text"].endswith(tokenizer.eos_token)

def test_dataset_formatting_dpo(formatter):
    fmt = formatter.format_dpo(MOCK_PROMPT, MOCK_CHOSEN, MOCK_REJECTED)
    assert fmt["prompt"] == MOCK_PROMPT
    assert MOCK_CHOSEN in fmt["chosen"]
    assert MOCK_REJECTED in fmt["rejected"]
    # Ensure DPOTrainer dict structure
    assert "prompt" in fmt and "chosen" in fmt and "rejected" in fmt

def test_quantization_forward_pass(tokenizer):
    # Load small model for test
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cpu")
    q_model = apply_dynamic_quantization(model)
    
    inputs = tokenizer(MOCK_PROMPT, return_tensors="pt")
    
    # Check if we can do a forward pass without crashing
    with torch.inference_mode():
        outputs = q_model(**inputs)
        
    assert outputs.logits is not None
    assert not torch.isnan(outputs.logits).any()

def test_checkpoint_manager_save_load():
    test_dir = "./test_checkpoints"
    manager = CheckpointManager(test_dir)
    Path(test_dir).mkdir(exist_ok=True)
    
    # Mock a checkpoint dir
    ckpt_dir = Path(test_dir) / "checkpoint-10"
    ckpt_dir.mkdir(exist_ok=True)
    
    latest = manager.find_latest_checkpoint(test_dir)
    assert latest == str(ckpt_dir)
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)
