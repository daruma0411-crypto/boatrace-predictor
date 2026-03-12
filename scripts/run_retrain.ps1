# 全モデル一括再学習 (208次元統一)
# 事前に collect_status.ps1 で 10,000+ races を確認してから実行
# $env:DATABASE_URL を事前に設定してから実行
$env:PYTHONIOENCODING = "utf-8"
cd "$env:USERPROFILE\.openclaw\workspace\boatrace-predictor"

Write-Host "=== Retrain All Models (208-dim) ===" -ForegroundColor Cyan
python scripts/retrain_all_models.py --epochs 100 --patience 12 --batch-size 256

Write-Host ""
Write-Host "=== Done! Check models/ directory ===" -ForegroundColor Green
Get-ChildItem models/*.pth | ForEach-Object { Write-Host "$($_.Name) - $($_.Length / 1KB) KB - $($_.LastWriteTime)" }

pause
