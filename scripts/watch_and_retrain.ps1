# データ収集を監視 → 30,000件到達で自動再学習
# $env:DATABASE_URL を事前に設定してから実行
$env:PYTHONIOENCODING = "utf-8"
Set-Location "$env:USERPROFILE\.openclaw\workspace\boatrace-predictor"

$TARGET = 20000
Write-Host "=== Auto Retrain Monitor ===" -ForegroundColor Cyan
Write-Host "Target: $TARGET trainable races" -ForegroundColor Cyan
Write-Host "Checking every 60 seconds..." -ForegroundColor Cyan
Write-Host ""

while ($true) {
    $result = python -c @"
import os, psycopg2, psycopg2.extras
conn = psycopg2.connect(
    os.environ['DATABASE_URL'],
    cursor_factory=psycopg2.extras.RealDictCursor
)
cur = conn.cursor()
cur.execute('''
    SELECT COUNT(*) FROM (
        SELECT b.race_id FROM boats b
        JOIN races r ON b.race_id = r.id
        WHERE r.status='finished' AND r.result_1st IS NOT NULL
        GROUP BY b.race_id HAVING COUNT(*) = 6
    ) t
''')
print(cur.fetchone()['count'])
conn.close()
"@

    $count = [int]$result
    $pct = [math]::Round($count / $TARGET * 100, 1)
    $time = Get-Date -Format "HH:mm:ss"
    Write-Host "[$time] $count / $TARGET races ($pct%)" -ForegroundColor Yellow

    if ($count -ge $TARGET) {
        Write-Host ""
        Write-Host "=== TARGET REACHED! Starting retrain... ===" -ForegroundColor Green
        Write-Host ""

        # Git commit current models as backup
        git add models/
        git commit -m "backup: pre-retrain model backup" 2>$null

        # Retrain all 4 models
        python scripts/retrain_all_models.py --epochs 100 --patience 12 --batch-size 256

        Write-Host ""
        Write-Host "=== Retrain Complete! ===" -ForegroundColor Green
        Get-ChildItem models/*.pth | ForEach-Object {
            Write-Host "$($_.Name) - $([math]::Round($_.Length / 1KB)) KB - $($_.LastWriteTime)" -ForegroundColor Cyan
        }

        # Push to Railway
        Write-Host ""
        Write-Host "Pushing to Railway..." -ForegroundColor Cyan
        git add models/ config/
        git commit -m "feat: 208-dim unified retrain - all 4 models updated"
        git push origin master

        Write-Host ""
        Write-Host "=== ALL DONE! Deploy triggered ===" -ForegroundColor Green
        break
    }

    Start-Sleep -Seconds 60
}

pause
