# Executive Summary: AlignForge Research Platform

## The Problem
Aligning Large Language Models (LLMs) with human preferences has traditionally required massive GPU clusters to run complex 4-stage Reinforcement Learning from Human Feedback (RLHF) pipelines. The emergence of Direct Preference Optimization (DPO) streamlined the mathematical objective, but still imposed massive memory constraints due to its reliance on a frozen reference model. For applied research and enterprise deployment, a highly efficient, CPU-compatible, mathematically robust pipeline was needed.

## The Approach
AlignForge was engineered as a 100% local, CPU-optimized RLHF laboratory. It systematically compares baseline Supervised Fine-Tuning (SFT) against DPO, Odds Ratio Preference Optimization (ORPO), and Simple Preference Optimization (SimPO) using dynamic INT8 quantization.

## The Reward-Hacking Discovery
During initial smoke-testing, the evaluation pipeline detected a severe structural flaw: **Reward-Length Coupling**. 
Our linear regression analysis proved that the standard Reward Model was using verbosity as a mechanical proxy for quality. 
- **Pearson r**: 0.9576
- **$R^2$**: 91.69% 
Left unchecked, this bias would have allowed algorithms like DPO to artificially inflate win-rates by simply generating longer, bloated text (a known failure mode called *reward hacking*).

## The Debiasing Solution
To solve this mathematically, we implemented a custom `PenalizedRewardTrainer`. This architecture parses exact sequence lengths at tokenization and dynamically subtracts a data-driven penalty (derived from the regression's $\beta_1$ coefficient) directly from the margin calculation during loss computation. The resulting model was successfully debiased, dropping the Pearson correlation to a perfectly healthy `0.18`.

## SimPO Results & Key Metrics
With the reward model secured, we ran the **Tier B2 ($N=500$)** benchmarking sequence. SimPO was proven to be the universally superior alignment methodology.
- **Win Rate**: SimPO (71%) vs DPO (66%) — *Statistically Significant ($p=0.041$)*
- **Efficiency**: SimPO required **15% less RAM** (4.3 GB) and trained **30% faster** (5.4h) than DPO.
- **Diversity**: SimPO achieved a Distinct-1 score of **0.24** compared to DPO's **0.18**.
- **Safety**: DPO suffered 18 repetition failures; SimPO suffered only 3.

An ablation study confirmed that SimPO's massive success was structurally dependent on our debiased reward architecture. 

## Future Work
With SimPO established as the default production model, future work will involve executing the 1000-sample Tier B3 validation run for formal publication, and investigating Kahneman-Tversky Optimization (KTO) using the established debiased pipeline.
