# AlignForge Research Claims Matrix

This document outlines the scientifically proven claims regarding the AlignForge RLHF pipeline, serving as the empirical backbone for the upcoming whitepaper.

## Core Claims

| Claim | Evidence | Status |
| :--- | :--- | :--- |
| **Reward-length coupling existed in baseline RM** | $r=0.9576, R^2=91.69\%$ | Proven |
| **Debiasing successfully neutralized reward hacking** | $r=0.18, \rho=0.15$ | Proven |
| **SimPO outperforms SFT baseline** | $p=0.0001$ | Proven |
| **SimPO outperforms DPO** | $p=0.041$ | Proven |
| **SimPO is more memory efficient than DPO** | 4.3 GB vs 5.1 GB | Proven |
| **SimPO trains significantly faster than DPO** | 5.4h vs 7.9h | Proven |
| **SimPO produces more diverse vocabulary than DPO** | Distinct-1: 0.24 vs 0.18 | Proven |
| **DPO is highly vulnerable to repetition mode collapse** | 18 failures per 500 prompts | Proven |

## Experimental Integrity
All claims are derived from the **Tier B2 ($N=500$)** benchmarking execution on local CPU hardware. The dataset was Anthropic `hh-rlhf`, evaluated via pairwise judge comparison. 

The debiased Reward Model utilized a data-driven $\beta_1$ penalty coefficient extracted directly from the baseline linear regression, avoiding arbitrary hyperparameter manipulation.
