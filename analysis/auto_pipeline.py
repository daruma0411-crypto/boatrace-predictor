"""V10 April fine-tune 検証パイプラインの自動進行

各ステップの完了を待って次を起動する。
進捗ログ: /tmp/scrape/auto_pipeline.log
"""
import os
import sys
import time
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def count_processes(pattern: str) -> int:
    """python.exe の中で CommandLine が pattern を含むプロセス数"""
    cmd = (
        f"Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        f"Where-Object {{ $_.CommandLine -like '*{pattern}*' }} | "
        f"Measure-Object | Select-Object -ExpandProperty Count"
    )
    try:
        r = subprocess.run(
            ['powershell.exe', '-Command', cmd],
            capture_output=True, text=True, timeout=30
        )
        return int((r.stdout or '0').strip() or 0)
    except Exception as e:
        log(f'count_processes error: {e}')
        return -1


def wait_until_done(pattern: str, label: str, interval: int = 30, max_wait: int = 36000):
    log(f"{label} 待機中... pattern={pattern}")
    waited = 0
    while waited < max_wait:
        n = count_processes(pattern)
        if n == 0:
            log(f"✅ {label} DONE")
            return True
        if waited % 300 == 0:
            log(f"  {label}: 稼働中 procs={n} ({waited}s 経過)")
        time.sleep(interval)
        waited += interval
    log(f"⚠️ {label} TIMEOUT after {max_wait}s")
    return False


def run(cmd: str, label: str) -> int:
    log(f"▶ {label} START")
    log(f"   $ {cmd}")
    rc = subprocess.call(cmd, shell=True)
    if rc == 0:
        log(f"✅ {label} OK")
    else:
        log(f"❌ {label} FAILED rc={rc}")
    return rc


def launch_2025_beforeinfo():
    log("2025 beforeinfo 8並列起動中...")
    for vs, ve in [(1,3),(4,6),(7,9),(10,12),(13,15),(16,18),(19,21),(22,24)]:
        cmd = (
            f"SCRAPE_SLEEP_SEC=0.3 nohup python analysis/scrape_beforeinfo_historical.py "
            f"--year 2025 --month 4 --venue-start {vs} --venue-end {ve} "
            f"> /tmp/scrape/2025_bi_v{vs}-{ve}.log 2>&1 &"
        )
        subprocess.Popen(['bash', '-c', cmd])
    time.sleep(10)
    log("2025 beforeinfo 起動完了")


def main():
    log("=== AUTO PIPELINE 開始 ===")

    # ---- 1. 2024 beforeinfo 完了待ち ----
    wait_until_done('scrape_beforeinfo_historical', '2024 beforeinfo (or 2025 if running)')

    # ---- 2. 2024 merge ----
    rc = run('python analysis/merge_historical_shards.py --year 2024 --month 4',
             '2024 merge')
    if rc != 0:
        log("Abort: 2024 merge failed"); return

    # ---- 3. 2025 scrape 完了待ち ----
    wait_until_done('scrape_historical.py --year 2025', '2025 scrape')

    # ---- 4. 2025 beforeinfo 起動 + 完了待ち ----
    launch_2025_beforeinfo()
    wait_until_done('scrape_beforeinfo_historical', '2025 beforeinfo')

    # ---- 5. 2025 merge ----
    rc = run('python analysis/merge_historical_shards.py --year 2025 --month 4',
             '2025 merge')
    if rc != 0:
        log("Abort: 2025 merge failed"); return

    # ---- 6. 2024+2025 extract ----
    rc = run('python analysis/extract_training_data_from_pkl.py --years 2024,2025 --month 4',
             'extract 2024+2025')
    if rc != 0:
        log("Abort: extract failed"); return

    # ---- 7. fine-tune ----
    train_pkl = 'analysis/models_v11/v10_april_finetune/train_data_2024_2025_04.pkl'
    rc = run(f'python analysis/finetune_v10_april.py --train-data {train_pkl}',
             'fine-tune')
    if rc != 0:
        log("Abort: fine-tune failed"); return

    # ---- 8. backtest ----
    rc = run('python analysis/backtest_v10_april.py --from 2026-04-01 --to 2026-04-30',
             'backtest')

    log("=== ALL DONE ===")


if __name__ == '__main__':
    main()
