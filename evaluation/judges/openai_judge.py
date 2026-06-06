import os
import requests
import logging
from .base_judge import BaseJudge

logger = logging.getLogger(__name__)

class OpenAIJudge(BaseJudge):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment.")

    def evaluate_pairwise(self, prompt: str, response_a: str, response_b: str) -> str:
        judge_prompt = self._build_judge_prompt(prompt, response_a, response_b)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": judge_prompt}],
            "max_tokens": 10,
            "temperature": 0.0
        }
        
        try:
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
            resp.raise_for_status()
            verdict = resp.json()["choices"][0]["message"]["content"].strip()
            
            if "A" in verdict: return "A"
            if "B" in verdict: return "B"
            return "Tie"
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return "Tie"
