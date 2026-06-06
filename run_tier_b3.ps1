# Tier B3 (1000 Samples) Execution Runner
# This script executes the final whitepaper validation run comparing SimPO against DPO and SFT.
# ORPO is excluded based on Tier B2 findings.

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "."
$env:PYTHONUTF8 = "1"

Write-Host "Starting AlignForge Tier B3 Validation Pipeline..."

Write-Host "Running SFT (1000 samples)..."
python scripts/run_sft.py --max_samples 1000
if ($LASTEXITCODE -ne 0) { throw "SFT failed" }

Write-Host "Running Penalized Reward Model (1000 samples)..."
python scripts/run_rm.py --max_samples 1000
if ($LASTEXITCODE -ne 0) { throw "RM failed" }

Write-Host "Running DPO (1000 samples)..."
python scripts/run_dpo.py --max_samples 1000
if ($LASTEXITCODE -ne 0) { throw "DPO failed" }

Write-Host "Running SimPO (1000 samples)..."
python scripts/run_simpo.py --max_samples 1000
if ($LASTEXITCODE -ne 0) { throw "SimPO failed" }

Write-Host "Running Validation Evaluation..."
python scripts/run_eval.py --models base,sft,dpo,simpo
if ($LASTEXITCODE -ne 0) { throw "Eval failed" }

Write-Host "Tier B3 completed successfully! Whitepaper metrics are ready."
