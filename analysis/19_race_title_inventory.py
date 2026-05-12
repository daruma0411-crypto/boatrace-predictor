"""race title 在庫確認 (A4 of Phase A roadmap, Issue #4)

本番 DB の races テーブルから title 候補列を動的検出し、月別の
充足率を集計。analysis/historical_data/ の scraped raw も並走走査。

READ-ONLY 厳守 (DB に対して SELECT のみ)。
出力: analysis/reports/race_title_inventory.md
"""
import os
import sys
import json
import pickle
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

REPORT_PATH = Path(__file__).parent / "reports" / "race_title_inventory.md"
HISTORICAL_DIR = Path(__file__).parent / "historical_data"
TITLE_CANDIDATE_COLS = ['race_title', 'race_name', 'title', 'subtitle', 'race_subtitle']


def detect_title_columns():
    """races テーブルの全カラムを取得し、title 候補列を検出"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'races'
            ORDER BY ordinal_position
        """)
        all_cols = [(r['column_name'], r['data_type']) for r in cur.fetchall()]
    detected = [c for c, _ in all_cols if c in TITLE_CANDIDATE_COLS]
    logger.info(f"races 全カラム: {len(all_cols)} 件")
    logger.info(f"title 候補列検出: {detected}")
    return all_cols, detected


def aggregate_db_inventory(detected_cols):
    """検出された title 列の月別充足率を集計"""
    monthly = {}
    for col in detected_cols:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT to_char(r.race_date, 'YYYY-MM') AS ym,
                       COUNT(*) AS total,
                       COUNT(NULLIF(r.{col}, '')) AS filled
                FROM races r
                WHERE r.race_date BETWEEN '2024-01-01' AND '2026-04-30'
                GROUP BY ym
                ORDER BY ym
            """)
            monthly[col] = [(r['ym'], r['total'], r['filled']) for r in cur.fetchall()]
    return monthly


def aggregate_historical_inventory():
    """analysis/historical_data/ 配下の JSON/PKL を走査"""
    if not HISTORICAL_DIR.exists():
        logger.warning(f"{HISTORICAL_DIR} なし、historical 走査スキップ")
        return {}
    result = defaultdict(lambda: {'total': 0, 'with_title': 0, 'keys_seen': set()})

    def scan_items(items, bucket):
        for it in items:
            if not isinstance(it, dict):
                continue
            result[bucket]['total'] += 1
            result[bucket]['keys_seen'].update(it.keys())
            if any(k in it and it[k] for k in ['title', 'race_title', 'race_name']):
                result[bucket]['with_title'] += 1

    for path in HISTORICAL_DIR.rglob('*.json'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"skip {path}: {e}")
            continue
        scan_items(data if isinstance(data, list) else [data], str(path.parent.name))

    for path in HISTORICAL_DIR.rglob('*.pkl'):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning(f"skip {path}: {e}")
            continue
        scan_items(data if isinstance(data, list) else [data], str(path.parent.name))

    return dict(result)


def make_report(all_cols, detected, monthly, historical):
    lines = []
    lines.append("# race title 在庫確認レポート\n\n")
    lines.append(f"生成日時: {datetime.now().isoformat()}\n")
    lines.append("対象: Phase A の A4 (Issue #4)\n\n")
    lines.append("## 1. races テーブル schema\n\n")
    lines.append(f"- 全カラム数: {len(all_cols)}\n")
    detected_str = ', '.join(detected) if detected else 'なし'
    lines.append(f"- title 候補列検出 (`{TITLE_CANDIDATE_COLS}`): **{detected_str}**\n\n")
    lines.append("### 全カラム一覧\n\n| カラム名 | 型 |\n|---|---|\n")
    for c, t in all_cols:
        lines.append(f"| {c} | {t} |\n")
    lines.append("\n## 2. 月別 title 充足率 (本番 DB)\n")
    if not detected:
        lines.append("\ntitle 候補列が races テーブルに存在しない。**A3 (スクレイピング拡張) 発火必要**。\n")
    else:
        for col, rows in monthly.items():
            lines.append(f"\n### 列 `{col}`\n\n| 年月 | total | filled | 充足率 |\n|---|---|---|---|\n")
            for ym, total, filled in rows:
                ratio = (filled / total * 100) if total else 0
                lines.append(f"| {ym} | {total} | {filled} | {ratio:.1f}% |\n")
    lines.append("\n## 3. scraped historical_data の在庫\n\n")
    if not historical:
        lines.append("historical_data なし\n")
    else:
        lines.append("| バケツ | 件数 | title 含む | 充足率 | 観測キー (先頭 8) |\n|---|---|---|---|---|\n")
        for bucket, info in sorted(historical.items()):
            t = info['total']
            wt = info['with_title']
            ratio = (wt / t * 100) if t else 0
            keys_str = ", ".join(sorted(info['keys_seen'])[:8])
            lines.append(f"| {bucket} | {t} | {wt} | {ratio:.1f}% | {keys_str} |\n")
    lines.append("\n## 4. 判定\n\n")
    if detected and monthly:
        recent_ratios = []
        for col, rows in monthly.items():
            for ym, total, filled in rows:
                if ym in ('2026-02', '2026-03', '2026-04') and total:
                    recent_ratios.append(filled / total * 100)
        avg_recent = sum(recent_ratios) / len(recent_ratios) if recent_ratios else 0
        verdict = "A3 スキップ可能" if avg_recent >= 95 else "A3 (スクレイピング拡張) 発火必要"
        lines.append(f"直近3ヶ月 (2026-02 〜 2026-04) の平均充足率: **{avg_recent:.1f}%**\n\n")
        lines.append(f"判定: **{verdict}** (基準: 95%)\n")
    else:
        lines.append("title 候補列が DB に無いため A3 発火必要\n")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    logger.info(f"レポート出力: {REPORT_PATH}")


def main():
    logger.info("A4 race title 在庫確認 開始")
    all_cols, detected = detect_title_columns()
    monthly = aggregate_db_inventory(detected)
    historical = aggregate_historical_inventory()
    make_report(all_cols, detected, monthly, historical)
    logger.info("完了")


if __name__ == '__main__':
    main()
