# AlignForge: Research Summary & Alignment Methodology

## 1. Research Contribution
AlignForge provides a highly reproducible, locally executable testbed for comparing alignment techniques (specifically DPO) under hardware-constrained environments. By establishing rigorous data validation, unified prompt handling, and a strict reproducibility registry, it eliminates common engineering artifacts that pollute RLHF ablation studies.

## 2. Experimental Methodology
We employ a strictly controlled pipeline using the Anthropic `hh-rlhf` dataset. 
- **Base Model**: Qwen2-0.5B-Instruct.
- **Stage 1 (SFT)**: Conditions the model on the `chosen` responses to align the structural priors to the dataset's specific conversational tone.
- **Stage 2 (Reward Modeling)**: A sequence classification head is attached to the base model. The model is trained using a pairwise ranking loss to maximize the margin between $R(prompt+chosen)$ and $R(prompt+rejected)$.
- **Stage 3 (DPO)**: We bypass PPO and optimize the policy directly via the DPO objective, utilizing the SFT model as the implicit reference.

To ensure empirical validity, `utils/reproducibility.py` captures exact configuration snapshots and locks random seeds across CUDA, Numpy, and Torch.

## 3. Benchmark Interpretation
The evaluation pipeline (`evaluation/benchmark_report.py`) generates multi-dimensional metrics:
- **Win Rate**: The primary indicator of preference alignment, calculated by querying the local RM on generated responses.
- **Reward Gain**: $R_{DPO} - R_{Base}$. A positive scalar indicates successful gradient updates toward the preference distribution.
- **Distinct-N & Repetition**: Measures mode collapse. A common failure of DPO is decreasing vocab diversity; tracking Distinct-1/2 ensures the policy hasn't collapsed into a highly-rewarded but repetitive local minimum.

## 4. Limitations
- **Reward Hacking Susceptibility**: Because the Reward Model and the Policy Model share the same base architecture (0.5B), the RM is relatively weak. The DPO policy can easily exploit the RM's blind spots (e.g., length bias or repetitive phrasing).
- **Dataset Scale**: To remain CPU-compatible, the pipeline defaults to subsets of the `hh-rlhf` data.

## 5. Future Work
- **Length-Normalized Reward**: Implementing a penalty term in the reward scalar to prevent verbosity-induced reward hacking.
- **Iterative DPO (DPO-Iter)**: Using the newly aligned policy to generate synthetic chosen/rejected pairs, re-training the RM, and repeating the cycle to escape initial dataset limitations.
