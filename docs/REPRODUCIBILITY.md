# AlignForge Reproducibility Appendix

To ensure absolute scientific rigor and independent verification of the **Tier B2** and **Tier B3** benchmarking sequences, this document provides the exact configurations required to reproduce the AlignForge results.

## 1. Environment & Packages
The project relies on specific Hugging Face configurations to prevent `kwargs` discrepancies in the TRL margin scaling logic.
- **Python**: 3.10+
- **`trl`**: 0.15.1
- **`transformers`**: 4.41.2
- **`peft`**: 0.11.1
- **`datasets`**: 2.20.0
- **`accelerate`**: 0.31.0

## 2. Hardware Configuration
All benchmarks were explicitly designed to execute entirely locally on CPU infrastructure to validate the efficiency of ORPO and SimPO.
- **Compute Target**: CPU Only (`use_cpu=True`)
- **Quantization**: Dynamic INT8 (Linear Layers via PyTorch `qint8`)
- **Minimum RAM Requirement**: 8.0 GB (SimPO Peak: 4.3 GB, DPO Peak: 5.1 GB)

## 3. Dataset Integrity
- **Source**: `Anthropic/hh-rlhf`
- **Subset**: `helpful-base`
- **Hash/Commit**: `09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa`
- **Formatter**: `data/formatting.py` (Unified `[PROMPT]`, `[CHOSEN]`, `[REJECTED]` standard)

## 4. Randomization & Determinism
To replicate the exact sequence pairings and dropout layers during Tier B execution:
- **Global Seed**: `42`
- **Data Shuffling Seed**: `42`
- **Dataset Sampling Constraint**: Deterministic head slicing (e.g., `dataset.select(range(500))`)

## 5. Configuration Snapshot (`configs/config.yaml`)
```yaml
batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 2e-5
num_epochs: 1
max_length: 512
seed: 42
benchmark_tier: publishable
simpo:
  beta: 2.0
  gamma_beta_ratio: 0.5
  learning_rate: 5e-5
  num_epochs: 1
```

## 6. Execution Runtime Estimates (N=500 Samples)
If reproducing on standard x86 CPU architecture:
- **SFT**: ~3.5 Hours
- **RM (Debiased)**: ~4.0 Hours
- **DPO**: ~7.9 Hours
- **ORPO**: ~5.6 Hours
- **SimPO**: ~5.4 Hours
