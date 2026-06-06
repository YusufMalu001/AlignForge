import requests
import logging
from .base_judge import BaseJudge

logger = logging.getLogger(__name__)

class OpenAICompatibleJudge(BaseJudge):
    def __init__(self, base_url: str, api_key: str, model_name: str = "default"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name

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
            resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
            resp.raise_for_status()
            verdict = resp.json()["choices"][0]["message"]["content"].strip()
            
            if "A" in verdict: return "A"
            if "B" in verdict: return "B"
            return "Tie"
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return "Tie"
