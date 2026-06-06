$ErrorActionPreference = "Stop"

Write-Host "Setting Environment Variables..."
$env:PYTHONPATH = "."
$env:PYTHONUTF8 = "1"

Write-Host "Running SFT..."
python scripts/run_sft.py
if ($LASTEXITCODE -ne 0) { throw "SFT failed" }

Write-Host "Running RM..."
python scripts/run_rm.py
if ($LASTEXITCODE -ne 0) { throw "RM failed" }

Write-Host "Running DPO..."
python scripts/run_dpo.py
if ($LASTEXITCODE -ne 0) { throw "DPO failed" }

Write-Host "Running ORPO..."
python scripts/run_orpo.py
if ($LASTEXITCODE -ne 0) { throw "ORPO failed" }

Write-Host "Running Evaluation..."
python scripts/run_eval.py --models base,sft,dpo,orpo
if ($LASTEXITCODE -ne 0) { throw "Eval failed" }

Write-Host "All smoke tests completed successfully!"
