# Tier B Benchmark Campaign: Experimental Protocol

## Objective
To scale AlignForge from a 10-sample functional smoke-test (Tier A) to a 1000-sample, statistically significant, publishable evaluation benchmark comparing Base, SFT, DPO, ORPO, and SimPO.

## 1. Staged Execution Strategy
To mitigate CPU Out-Of-Memory (OOM) risks and ensure checkpoint stability, scaling is staggered:
- **Tier B1 (250 Samples)**: Validates continuous memory bounds and gradient stability over multiple hours.
- **Tier B2 (500 Samples)**: Provides the minimum $N$ required to achieve statistical confidence (via Bootstrapping) for win rates.
- **Tier B3 (1000 Samples)**: The final whitepaper target.

## 2. Model Pipeline
All models share `Qwen2-0.5B-Instruct` as the base architecture.
1. **SFT**: Adapts the base model to the `hh-rlhf` formatting.
2. **DPO**: Reference-based Preference Optimization using the SFT checkpoint.
3. **ORPO**: Reference-free Odds Ratio Optimization derived straight from the Base model.
4. **SimPO**: Reference-free Simple Preference Optimization utilizing length-normalized rewards.

## 3. SimPO Acceptance Criteria
SimPO is considered a successful integration if and only if it achieves:
1. **Reward Gain**: > Baseline SFT
2. **Win Rate**: > Baseline SFT
3. **Distinct-1 Score**: $\ge$ DPO (proving lower mode-collapse)
4. **Peak RAM Usage**: $\le$ DPO (proving compute efficiency)

## 4. Hardware & Compute Budget
*Estimates based on CPU environment benchmarks:*
- **SFT (1000 samples)**: ~1.5 hours
- **RM (1000 samples)**: ~1.5 hours
- **DPO (1000 samples)**: ~2.0 hours (5.1 GB RAM Peak)
- **ORPO/SimPO (1000 samples)**: ~1.2 hours each (4.3 GB RAM Peak)
- **Total Compute Cost**: ~8 CPU hours for Tier B3.

## 5. Statistical & Failure Tracking
- **Length Bias Mitigation**: `evaluation/reward_bias_analysis.py` will compute Pearson and Spearman correlations. If $r > 0.4$, we must inject a length penalty into the RM logic.
- **Bootstrapping**: Win Rates will be evaluated across 1000 bootstrap iterations to calculate the 95% Confidence Interval.
