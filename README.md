# AlignForge 🚀

AlignForge is a complete, production-grade 100% local RLHF (Reinforcement Learning from Human Feedback) fine-tuning pipeline for Large Language Models. Built for a CPU-compatible environment, this project demonstrates end-to-end alignment using the Anthropic `hh-rlhf` dataset without any external API dependencies.

## Architecture

```text
Anthropic HH-RLHF (Chosen/Rejected)
          ↓
  Stage 1: Supervised Fine-Tuning (SFT)
  (Policy Model Base)
          ↓
  Stage 2: Reward Model Training (RM)
  (Local sequence classification judge)
          ↓
  Stage 3: Direct Preference Optimization (DPO)
  (Optimizes preference margin over frozen SFT)
          ↓
  Stage 4: Automated Evaluation
  (Dynamic INT8 Quantization + Local RM Scoring)
          ↓
  Streamlit Dashboard
```

## Setup Instructions

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Copy `.env.example` to `.env` and fill in your W&B key for tracking.
   ```bash
   cp .env.example .env
   ```

3. **Adjust Hyperparameters (Optional):**
   Edit `configs/config.yaml` to change model, batch size, learning rate, or sample size.

## How to Run

The pipeline is split into four modular stages. Each script supports a `--resume` flag to safely restart from the last checkpoint.

### 1. Supervised Fine-Tuning (SFT)
Trains the base model on chosen responses to learn the preferred format.
```bash
python scripts/run_sft.py
```

### 2. Reward Model Training (RM)
Trains a local sequence classification model on the chosen/rejected pairs to predict human preference scores.
```bash
python scripts/run_rm.py
```

### 3. Direct Preference Optimization (DPO)
Uses the SFT model as a frozen reference and trains a new policy via LoRA to maximize preference margins directly.
```bash
python scripts/run_dpo.py
```

### 4. Automated Evaluation
Generates responses from both the baseline and DPO models, scores them using the local Reward Model, and calculates key metrics.
```bash
python scripts/run_eval.py
```

### 5. Dashboard
Launch the interactive Streamlit dashboard to explore reward distributions, loss curves, and sample outputs side-by-side.
```bash
streamlit run dashboard/app.py
```

## Results Benchmark

| Metric | Value |
|--------|-------|
| Win Rate | TBD |
| Avg Reward Gain | TBD |
| Avg Baseline Reward | TBD |
| Avg DPO Reward | TBD |
| Avg Response Length (baseline) | TBD |
| Avg Response Length (DPO) | TBD |
| Vocab Diversity (baseline) | TBD |
| Vocab Diversity (DPO) | TBD |
| Repetition Score (baseline) | TBD |
| Repetition Score (DPO) | TBD |

---
*Note on CPU Training: Training even 500 samples on a CPU is computationally intensive. The pipeline uses small LoRA ranks, a batch size of 1 with gradient accumulation, and frequent checkpointing (default every 50 steps) to ensure progress isn't lost if interrupted.*
