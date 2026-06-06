# AlignForge 🚀

AlignForge is a complete, production-grade 100% local RLHF (Reinforcement Learning from Human Feedback) fine-tuning pipeline for Large Language Models. Built for CPU and limited GPU environments, it demonstrates end-to-end alignment using Anthropic's `hh-rlhf` dataset.

## Features
- **4-Stage Pipeline**: SFT -> Reward Modeling -> DPO -> Evaluation.
- **Strict Dataset Validation**: Pre-flight checks prevent malformed data from poisoning training.
- **Memory Optimized**: Employs TRL's built-in reference model adapter-switching to cut DPO memory usage by 50%.
- **CPU Inference Acceleration**: PyTorch Dynamic INT8 Quantization on Linear layers for fast evaluation.
- **Scientific Reproducibility**: Built-in deterministic seeds, config snapshotting, and experiment registry.

## Setup Instructions

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Copy `.env.example` to `.env`.

## How to Run

### 1. Supervised Fine-Tuning (SFT)
Trains the base model on chosen responses to learn formatting.
```bash
python scripts/run_sft.py
```

### 2. Reward Model Training (RM)
Trains a sequence classification model to score human preference.
```bash
python scripts/run_rm.py
```

### 3. Direct Preference Optimization (DPO)
Optimizes the policy model via LoRA to maximize preference margin against the frozen reference.
```bash
python scripts/run_dpo.py
```

### 4. Automated Evaluation
Generates benchmark reports, calculating win rates, reward gains, and sequence diversity.
```bash
python scripts/run_eval.py
```

## Testing
Run the pytest suite to ensure data formatting and quantization logic is sound:
```bash
pytest tests/
```
