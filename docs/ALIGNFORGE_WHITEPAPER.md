# AlignForge: Local Alignment Research Whitepaper

## Abstract
AlignForge presents a highly optimized, fully local pipeline for researching Large Language Model alignment. By integrating rigorous data validation, advanced statistical benchmarking, and a pluggable LLM-as-a-judge evaluation framework, this platform bridges the gap between proof-of-concept scripts and research-grade science. We provide detailed analyses of resource efficiency, memory management across DPO and ORPO paradigms, and model quantization via ONNX.

## 1. Motivation
The vast majority of open-source alignment pipelines neglect the constraints of commodity hardware and fail to provide statistically sound evaluation methods. Researchers frequently report arbitrary "Win Rates" without confidence intervals, leading to misleading conclusions. AlignForge was built to execute rigorous alignment studies entirely on local CPUs, tracking precise resource costs while offering robust statistical certainty.

## 2. Methodology

### 2.1 Supervised Fine-Tuning (SFT)
SFT conditions the base model (e.g., `Qwen2-0.5B-Instruct`) to the precise conversational format of the `hh-rlhf` dataset. AlignForge uses a `UnifiedFormatter` to guarantee that tokens generated during evaluation exactly match those seen during training, preventing KL-divergence explosions.

### 2.2 Reward Modeling
A sequence classification head is attached to the base model to predict human preference. Crucially, AlignForge corrects common tokenization errors by ensuring the context (prompt) is concatenated with the chosen/rejected responses before tokenization, allowing the model to learn conditional preference rather than arbitrary sequence properties.

### 2.3 Direct Preference Optimization (DPO)
DPO is executed with aggressive memory optimizations. By utilizing TRL's built-in PEFT management (`ref_model=None`), we bypass the need for a separate, frozen copy of the reference model in memory. This effectively halves the RAM required, making local CPU training feasible.

### 2.4 Odds Ratio Preference Optimization (ORPO) (Phase 2A)
As an alternative to DPO, we introduce ORPO. By omitting the reference model entirely and baking the alignment penalty directly into the Negative Log-Likelihood loss, ORPO further reduces computational overhead. 

## 3. Evaluation Framework

### 3.1 Pluggable Judge Architecture
Evaluating alignment requires high-quality judgment. AlignForge supports a configuration-driven judge system:
- `LocalJudge`: Offline, private, but computationally heavy.
- `OpenAIJudge` & `OpenAICompatibleJudge`: High-speed API inference for production benchmarking (e.g., via vLLM).

### 3.2 Statistical Significance
Reporting a single scalar for Win Rate is insufficient. AlignForge implements paired t-tests and bootstrapping to calculate 95% Confidence Intervals, ensuring observed improvements are statistically significant.

### 3.3 Failure & Cost Analytics
Beyond basic metrics, AlignForge outputs a `failure_report.json` identifying Hallucination, Repetition, and Refusals. Additionally, it logs training times, peak RAM utilization, and disk footprint to `resource_report.json`.

## 4. Deployment
To bridge research to production, models are exportable via HF Optimum to ONNX format. PyTorch dynamic INT8 quantization and ONNX Runtime are benchmarked head-to-head to determine the optimal inference strategy on CPU.

## 5. Limitations & Future Work
- **Hardware Bottlenecks**: While heavily optimized, CPU training for LLMs remains inherently slow. 
- **Future Integration**: Phase 2B (SimPO) and Phase 2C (KTO) will introduce further algorithmic comparisons to complete the alignment suite.
