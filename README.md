# AlignForge 🚀

AlignForge is a complete, production-grade Direct Preference Optimization (DPO) fine-tuning pipeline for Large Language Models. Built for a CPU-compatible environment, this project demonstrates end-to-end alignment using the Anthropic `hh-rlhf` dataset.

## Architecture

```text
Anthropic HH-RLHF (Chosen/Rejected)
          ↓
  Supervised Fine-Tuning (SFT)
  (Policy Model Base)
          ↓
  Frozen Reference Model
          ↓
  Direct Preference Optimization (DPO)
          ↓
  Evaluation (Dynamic INT8 Quantization)
          ↓
  GPT-4o-mini LLM-as-a-Judge
          ↓
  Streamlit Dashboard
```

## Setup Instructions

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Copy `.env.example` to `.env` and fill in your keys (OpenAI for evaluation, W&B for tracking).
   ```bash
   cp .env.example .env
   ```

3. **Adjust Hyperparameters (Optional):**
   Edit `configs/config.yaml` to change model, batch size, learning rate, or sample size.

## How to Run

The pipeline is split into three modular stages. Each script supports a `--resume` flag to safely restart from the last checkpoint.

### 1. Supervised Fine-Tuning (SFT)
Trains the base model on chosen responses to learn the preferred format.
```bash
python scripts/run_sft.py
```

### 2. Direct Preference Optimization (DPO)
Uses the SFT model as a frozen reference and trains a new policy via LoRA to maximize preference margins.
```bash
python scripts/run_dpo.py
```

### 3. Automated Evaluation
Generates responses from both the baseline and DPO models, judges them with GPT-4o-mini, and calculates key metrics.
```bash
python scripts/run_eval.py
```

### 4. Dashboard
Launch the interactive Streamlit dashboard to explore loss curves, win rates, and sample outputs side-by-side.
```bash
streamlit run dashboard/app.py
```

## Results Benchmark

| Metric | Value |
|--------|-------|
| Win Rate | TBD |
| Loss Rate | TBD |
| Tie Rate | TBD |
| Avg Response Length (baseline) | TBD |
| Avg Response Length (DPO) | TBD |
| Vocab Diversity (baseline) | TBD |
| Vocab Diversity (DPO) | TBD |
| Repetition Score (baseline) | TBD |
| Repetition Score (DPO) | TBD |

## Example Outputs

*Run the evaluation pipeline to populate example outputs here.*

---
*Note on CPU Training: Training even 500 samples on a CPU is computationally intensive. The pipeline uses small LoRA ranks, a batch size of 1 with gradient accumulation, and frequent checkpointing (default every 50 steps) to ensure progress isn't lost if interrupted.*
