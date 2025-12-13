$env:MISTRAL_API_KEY = "B6tGkllMMtadsnrqMZegMjDXW0vIgMYn"
$env:PYTHONPATH = "$PSScriptRoot/.."
Set-Location "$PSScriptRoot/.."
Write-Host "Starting HumanEval Workflow..."
python -m chameleon.workflow HumanEval --projects-dir "Projects"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Workflow Completed Successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Workflow Failed!" -ForegroundColor Red
}

