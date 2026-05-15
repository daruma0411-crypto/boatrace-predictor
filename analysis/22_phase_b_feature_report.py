"""Phase B 特徴量レポート生成 (Issue #4)

入力: analysis/features_phase_b.pkl
出力: analysis/reports/phase_b_features.md
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PKL_PATH = ROOT / "analysis" / "features_phase_b.pkl"
REPORT_PATH = ROOT / "analysis" / "reports" / "phase_b_features.md"


def summary_categorical(df, col, top_n=None):
    vc = df[col].value_counts(dropna=False)
    if top_n is not None:
        vc = vc.head(top_n)
    total = len(df)
    lines = [f"| {col} | n | % |", "|---|---|---|"]
    for val, n in vc.items():
        pct = 100 * n / total
        val_str = "NaN" if pd.isna(val) else str(val)
        lines.append(f"| {val_str} | {n} | {pct:.2f}% |")
    return "\n".join(lines)


def summary_numeric(df, col):
    desc = df[col].describe()
    na_pct = 100 * df[col].isna().mean()
    lines = ["| stat | value |", "|---|---|"]
    lines.append(f"| count | {int(desc['count'])} |")
    lines.append(f"| na rate | {na_pct:.2f}% |")
    lines.append(f"| mean | {desc['mean']:.4f} |")
    lines.append(f"| std | {desc['std']:.4f} |")
    lines.append(f"| min | {desc['min']:.4f} |")
    lines.append(f"| 25% | {desc['25%']:.4f} |")
    lines.append(f"| 50% | {desc['50%']:.4f} |")
    lines.append(f"| 75% | {desc['75%']:.4f} |")
    lines.append(f"| max | {desc['max']:.4f} |")
    return "\n".join(lines)


def main():
    df = pd.read_pickle(PKL_PATH)
    lines = []
    lines.append("# Phase B 特徴量レポート\n")
    lines.append(f"生成日時: {datetime.now().isoformat()}\n")
    lines.append(f"対象期間: 2026-02-01 〜 2026-04-30\n")
    lines.append(f"レース数: {len(df)}\n")
    lines.append("\n## 1. race_category 分布 (B1)\n\n")
    lines.append(summary_categorical(df, 'race_category'))
    lines.append("\n\n## 2. is_planned 分布 (B1)\n\n")
    lines.append(summary_categorical(df, 'is_planned'))
    lines.append("\n\n## 3. boat1_skill_gap (B2)\n\n")
    lines.append(summary_numeric(df, 'boat1_skill_gap'))
    lines.append("\n\n## 4. a_class_consumed (B3)\n\n")
    lines.append(summary_numeric(df, 'a_class_consumed'))
    lines.append("\n\n## 5. day_in_meeting 分布 (B4)\n\n")
    lines.append(summary_categorical(df, 'day_in_meeting'))
    lines.append("\n\n## 6. day_label_raw TOP 10 (B4 原値、保険)\n\n")
    lines.append(summary_categorical(df, 'day_label_raw', top_n=10))
    lines.append("\n")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding='utf-8')
    logger.info(f"レポート出力: {REPORT_PATH}")


if __name__ == '__main__':
    main()
