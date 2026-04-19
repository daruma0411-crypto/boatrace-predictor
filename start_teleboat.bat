@echo off
REM テレボート自動購入ボット起動スクリプト
REM Windows Task Scheduler から毎日 07:00 JST に起動される
REM %~dp0 = このバッチファイルのあるディレクトリ（末尾バックスラッシュ付き）
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"%~dp0venv\Scripts\python.exe" scripts\teleboat_purchaser.py >> teleboat_purchaser.log 2>&1
