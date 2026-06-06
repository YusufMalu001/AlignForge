import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_judge import BaseJudge

class LocalJudge(BaseJudge):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32, 
            device_map="cpu", 
            trust_remote_code=True
        )
        self.model.eval()

    def evaluate_pairwise(self, prompt: str, response_a: str, response_b: str) -> str:
        judge_prompt = self._build_judge_prompt(prompt, response_a, response_b)
        inputs = self.tokenizer(judge_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=5, temperature=0.1)
            
        verdict_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        if "A" in verdict_text: return "A"
        if "B" in verdict_text: return "B"
        return "Tie"
