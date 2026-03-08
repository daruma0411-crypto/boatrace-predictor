"""過去データ収集スクリプト（非推奨 → collect_past_data.py / collect_parallel.py を使用）

互換性のためのラッパー。実際の収集ロジックは collect_past_data.py に移行済み。
"""
import sys
import os
import logging
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.collect_past_data import collect_date_range, main as collect_main
from utils.timezone import now_jst

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.warning(
    "collect_historical_data.py は非推奨です。"
    "collect_past_data.py または collect_parallel.py を使用してください。"
)


def main():
    collect_main()


if __name__ == '__main__':
    main()
