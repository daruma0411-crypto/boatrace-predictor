# Phase B 新特徴量 (B1-B5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Issue #4 Phase B の 5 特徴量 (B1-B5) を生成し、2026-02-01〜04-30 の 13,212 races を含む DataFrame pkl + 統計レポートを出力する。

**Architecture:** `src/phase_b_features.py` に純関数 (B1/B2/B4) を実装、`analysis/21_*.py` で DB JOIN + B3 SQL aggregation + 純関数適用、`analysis/22_*.py` で markdown レポート。本番非介入、analysis/ 完結。

**Tech Stack:** Python 3.x、pandas、PostgreSQL (Railway 本番、READ-ONLY)、unittest。

**Spec:** `docs/superpowers/specs/2026-05-15-phase-b-features-design.md`

**Issue:** https://github.com/daruma0411-crypto/boatrace-predictor/issues/4

---

## File Structure

| Path | Action | 責務 |
|---|---|---|
| `src/phase_b_features.py` | Create | B1 (classify_race_category, detect_planned_race), B2 (compute_skill_gap), B4 (parse_day_label) の純関数 |
| `tests/test_phase_b_features.py` | Create | 上記純関数の単体テスト |
| `analysis/21_build_phase_b_features.py` | Create | DB JOIN + B3 SQL aggregation + 純関数適用 → DataFrame pkl |
| `analysis/22_phase_b_feature_report.py` | Create | pkl 読み込み → 分布/欠損率 markdown |
| `analysis/features_phase_b.pkl` | Generated | DataFrame (race_id index × 6 列) |
| `analysis/reports/phase_b_features.md` | Generated | 統計レポート |

---

## Task 1: src/phase_b_features.py に B1/B2/B4 純関数 + テスト (TDD)

**Files:**
- Create: `src/phase_b_features.py`
- Create: `tests/test_phase_b_features.py`

### - [ ] Step 1: テストを先に書く

`tests/test_phase_b_features.py` 新規作成:

```python
"""Phase B 特徴量純関数の単体テスト

実行:
  python -m unittest tests.test_phase_b_features -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase_b_features import (
    classify_race_category,
    detect_planned_race,
    parse_day_label,
    compute_skill_gap,
)


class TestClassifyRaceCategory(unittest.TestCase):
    def test_qualifier(self):
        self.assertEqual(classify_race_category("予選1800m"), "qualifier")

    def test_semifinal(self):
        self.assertEqual(classify_race_category("準優1800m"), "semifinal")

    def test_final(self):
        self.assertEqual(classify_race_category("優勝戦1800m"), "final")

    def test_general(self):
        self.assertEqual(classify_race_category("一般 1800m"), "general")

    def test_other_when_no_match(self):
        self.assertEqual(classify_race_category("選抜1800m"), "other")

    def test_none_input(self):
        self.assertEqual(classify_race_category(None), "other")

    def test_empty_string(self):
        self.assertEqual(classify_race_category(""), "other")


class TestDetectPlannedRace(unittest.TestCase):
    def test_sunrise(self):
        self.assertTrue(detect_planned_race("サンライズ V"))

    def test_golden(self):
        self.assertTrue(detect_planned_race("ゴールデンカップ"))

    def test_gw_special(self):
        self.assertTrue(detect_planned_race("スポーツ報知杯争奪ゴールデンウィーク特選"))

    def test_v_premier(self):
        self.assertTrue(detect_planned_race("Ｖプレミアトーナメント"))

    def test_not_planned(self):
        self.assertFalse(detect_planned_race("第３３回多摩川さつき杯"))

    def test_none(self):
        self.assertFalse(detect_planned_race(None))


class TestParseDayLabel(unittest.TestCase):
    def test_first_day(self):
        self.assertEqual(parse_day_label("初日"), 1)

    def test_second_day_full_width(self):
        self.assertEqual(parse_day_label("２日目"), 2)

    def test_fifth_day_full_width(self):
        self.assertEqual(parse_day_label("５日目"), 5)

    def test_third_day_half_width(self):
        self.assertEqual(parse_day_label("3日目"), 3)

    def test_final_day_returns_none(self):
        # 節長依存のため None
        self.assertIsNone(parse_day_label("最終日"))

    def test_championship_returns_none(self):
        self.assertIsNone(parse_day_label("優勝戦"))

    def test_invalid_returns_none(self):
        self.assertIsNone(parse_day_label("予選"))

    def test_none_input(self):
        self.assertIsNone(parse_day_label(None))


class TestComputeSkillGap(unittest.TestCase):
    def test_normal(self):
        boats = [
            {'boat_number': 1, 'win_rate_2': 50.0},
            {'boat_number': 2, 'win_rate_2': 40.0},
            {'boat_number': 3, 'win_rate_2': 30.0},
            {'boat_number': 4, 'win_rate_2': 35.0},
            {'boat_number': 5, 'win_rate_2': 25.0},
            {'boat_number': 6, 'win_rate_2': 20.0},
        ]
        # 1号艇 50.0 - 平均(40,30,35,25,20)=30.0 = 20.0
        self.assertAlmostEqual(compute_skill_gap(boats), 20.0)

    def test_negative_gap(self):
        boats = [
            {'boat_number': 1, 'win_rate_2': 10.0},
            {'boat_number': 2, 'win_rate_2': 40.0},
            {'boat_number': 3, 'win_rate_2': 40.0},
            {'boat_number': 4, 'win_rate_2': 40.0},
            {'boat_number': 5, 'win_rate_2': 40.0},
            {'boat_number': 6, 'win_rate_2': 40.0},
        ]
        # 1号艇 10 - 40 = -30
        self.assertAlmostEqual(compute_skill_gap(boats), -30.0)

    def test_missing_boat1_returns_none(self):
        boats = [
            {'boat_number': 2, 'win_rate_2': 40.0},
            {'boat_number': 3, 'win_rate_2': 30.0},
        ]
        self.assertIsNone(compute_skill_gap(boats))

    def test_any_null_win_rate_returns_none(self):
        boats = [
            {'boat_number': 1, 'win_rate_2': 50.0},
            {'boat_number': 2, 'win_rate_2': None},
            {'boat_number': 3, 'win_rate_2': 30.0},
            {'boat_number': 4, 'win_rate_2': 35.0},
            {'boat_number': 5, 'win_rate_2': 25.0},
            {'boat_number': 6, 'win_rate_2': 20.0},
        ]
        self.assertIsNone(compute_skill_gap(boats))

    def test_wrong_boat_count_returns_none(self):
        boats = [{'boat_number': i, 'win_rate_2': 30.0} for i in range(1, 5)]
        self.assertIsNone(compute_skill_gap(boats))


if __name__ == '__main__':
    unittest.main()
```

### - [ ] Step 2: テストが失敗することを確認

```bash
cd C:/Users/iwashita.AKGNET/.openclaw/workspace/boatrace-predictor
python -m unittest tests.test_phase_b_features -v
```

期待: `ImportError: cannot import name 'classify_race_category' from 'src.phase_b_features'`

### - [ ] Step 3: 実装ファイルを作成

`src/phase_b_features.py` 新規作成:

```python
"""Phase B 新特徴量の純関数 (B1/B2/B4)

- B1: classify_race_category(subtitle) + detect_planned_race(title)
- B2: compute_skill_gap(boats)
- B4: parse_day_label(label)

B3 (a_class_consumed) は SQL window aggregation で計算するため
analysis/21_build_phase_b_features.py に直書き (本ファイル対象外)。
"""
import re


# B1: race_category 分類ルール
_CATEGORY_RULES = [
    ("優勝戦", "final"),
    ("準優", "semifinal"),
    ("予選", "qualifier"),
    ("一般", "general"),
]


def classify_race_category(subtitle):
    """subtitle から race_category を分類

    Args:
        subtitle: str | None (例: "予選1800m", "優勝戦1800m", None)

    Returns:
        str: "qualifier" / "semifinal" / "final" / "general" / "other"
    """
    if not subtitle:
        return "other"
    for keyword, label in _CATEGORY_RULES:
        if keyword in subtitle:
            return label
    return "other"


# B1: 企画レース判定辞書 (初期セット、backfill 結果から微調整)
_PLANNED_KEYWORDS = [
    "サンライズ",
    "ゴールデン",
    "Ｖプレミア",
    "Vプレミア",
    "V プレミア",
    "プレミアム",
    "GW特選",
    "GW 特選",
    "ゴールデンウィーク特選",
    "ＧＷ特選",
    "新春特選",
    "夏季特選",
    "特別選抜",
]


def detect_planned_race(title):
    """title が企画レース語を含むか

    Args:
        title: str | None

    Returns:
        bool
    """
    if not title:
        return False
    return any(kw in title for kw in _PLANNED_KEYWORDS)


# B4: day_label → integer parser
_DAY_PATTERN = re.compile(r'^([０-９0-9]+)日目$')
_ZEN_TO_HAN = str.maketrans('０１２３４５６７８９', '0123456789')


def parse_day_label(label):
    """day_label を 1..N の integer に変換

    Args:
        label: str | None (例: "初日", "２日目", "５日目", "最終日", "優勝戦")

    Returns:
        int | None: 整数。最終日/優勝戦/不正値/None は None (節長依存のため)
    """
    if not label:
        return None
    if label == "初日":
        return 1
    m = _DAY_PATTERN.match(label)
    if m:
        num_str = m.group(1).translate(_ZEN_TO_HAN)
        return int(num_str)
    # 最終日 / 優勝戦 / その他 → None (節長依存、Phase C で必要なら別途処理)
    return None


# B2: 実力乖離度
def compute_skill_gap(boats):
    """1号艇 win_rate_2 − 平均(2..6号艇 win_rate_2)

    Args:
        boats: list[dict]、各要素は {'boat_number': int, 'win_rate_2': float|None}

    Returns:
        float | None: 6艇揃わない / win_rate_2 に NULL 含む / 1号艇欠落 時は None
    """
    if not boats or len(boats) != 6:
        return None
    by_num = {b['boat_number']: b.get('win_rate_2') for b in boats}
    if any(by_num.get(i) is None for i in range(1, 7)):
        return None
    boat1 = by_num[1]
    others_mean = sum(by_num[i] for i in range(2, 7)) / 5
    return float(boat1 - others_mean)
```

### - [ ] Step 4: テストが PASS することを確認

```bash
python -m unittest tests.test_phase_b_features -v
```

期待: 全件 PASS (約 25 件)。

### - [ ] Step 5: コミット

```bash
git add src/phase_b_features.py tests/test_phase_b_features.py
git commit -m "feat(phase-b): B1/B2/B4 純関数 + 単体テスト追加"
```

---

## Task 2: analysis/21_build_phase_b_features.py パイプライン

**Files:**
- Create: `analysis/21_build_phase_b_features.py`
- Generated: `analysis/features_phase_b.pkl`

このタスクは長時間処理ではないが、本番 DB との JOIN を含む。SELECT のみ (READ-ONLY)。

### - [ ] Step 1: パイプラインスクリプトを作成

`analysis/21_build_phase_b_features.py` 新規作成:

```python
"""Phase B 特徴量 pkl 生成 (B1-B5 of Phase B roadmap, Issue #4)

入力: races + boats + race_titles テーブル (2026-02-01〜2026-04-30、READ-ONLY)
出力: analysis/features_phase_b.pkl (DataFrame、race_id index × 6 列)

特徴量:
- race_category (str): B1
- is_planned (bool): B1
- boat1_skill_gap (float): B2
- a_class_consumed (float [0,1]): B3 (SQL window で計算)
- day_in_meeting (int|NaN): B4
- day_label_raw (str|None): 保険、元値
"""
import os
import sys
import logging
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from src.database import get_db_connection
from src.phase_b_features import (
    classify_race_category,
    detect_planned_race,
    parse_day_label,
    compute_skill_gap,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / "analysis" / "features_phase_b.pkl"

DATE_FROM = '2026-02-01'
DATE_TO = '2026-04-30'


def fetch_rows():
    """races + race_titles JOIN で 1 レース 1 行を取得"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id AS race_id, r.race_date, r.venue_id, r.race_number,
                   rt.title, rt.subtitle, rt.day_label
            FROM races r
            LEFT JOIN race_titles rt ON r.id = rt.race_id
            WHERE r.race_date BETWEEN %s AND %s
            ORDER BY r.race_date, r.venue_id, r.race_number
        """, (DATE_FROM, DATE_TO))
        return cur.fetchall()


def fetch_boats(race_ids):
    """boats を race_id バッチで取得 (1 接続 / 1 クエリ)"""
    if not race_ids:
        return {}
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT race_id, boat_number, player_class, win_rate_2
            FROM boats
            WHERE race_id = ANY(%s)
            ORDER BY race_id, boat_number
        """, (race_ids,))
        rows = cur.fetchall()
    by_race = defaultdict(list)
    for r in rows:
        by_race[r['race_id']].append(dict(r))
    return dict(by_race)


def fetch_a_class_consumed():
    """B3: 同日同会場の累積 A 級出走比率を SQL window で計算

    Returns:
        dict[race_id] -> float | None
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        # 各 race について、同日同会場で race_number が小さいレースの A 級率を累積
        cur.execute("""
            WITH per_race AS (
                SELECT r.id AS race_id, r.race_date, r.venue_id, r.race_number,
                       COUNT(b.*) AS n_boats,
                       COUNT(b.*) FILTER (WHERE b.player_class IN ('A1', 'A2')) AS n_a
                FROM races r
                JOIN boats b ON r.id = b.race_id
                WHERE r.race_date BETWEEN %s AND %s
                GROUP BY r.id, r.race_date, r.venue_id, r.race_number
            ),
            cumulative AS (
                SELECT race_id, race_date, venue_id, race_number,
                       SUM(n_a) OVER (
                           PARTITION BY race_date, venue_id
                           ORDER BY race_number
                           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                       ) AS prior_a,
                       SUM(n_boats) OVER (
                           PARTITION BY race_date, venue_id
                           ORDER BY race_number
                           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                       ) AS prior_total
                FROM per_race
            )
            SELECT race_id,
                   CASE
                       WHEN prior_total IS NULL OR prior_total = 0 THEN NULL
                       ELSE prior_a::float / prior_total
                   END AS a_class_consumed
            FROM cumulative
        """, (DATE_FROM, DATE_TO))
        return {r['race_id']: r['a_class_consumed'] for r in cur.fetchall()}


def main():
    logger.info(f"Phase B 特徴量生成 開始: {DATE_FROM} 〜 {DATE_TO}")

    rows = fetch_rows()
    logger.info(f"races 取得: {len(rows)} 行")

    race_ids = [r['race_id'] for r in rows]
    boats_by_race = fetch_boats(race_ids)
    logger.info(f"boats 取得: {sum(len(v) for v in boats_by_race.values())} 行")

    a_class_map = fetch_a_class_consumed()
    logger.info(f"a_class_consumed 計算: {len(a_class_map)} 行")

    records = []
    for r in rows:
        race_id = r['race_id']
        records.append({
            'race_id': race_id,
            'race_date': r['race_date'],
            'venue_id': r['venue_id'],
            'race_number': r['race_number'],
            'race_category': classify_race_category(r.get('subtitle')),
            'is_planned': detect_planned_race(r.get('title')),
            'boat1_skill_gap': compute_skill_gap(boats_by_race.get(race_id, [])),
            'a_class_consumed': a_class_map.get(race_id),
            'day_in_meeting': parse_day_label(r.get('day_label')),
            'day_label_raw': r.get('day_label'),
        })

    df = pd.DataFrame(records).set_index('race_id')
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT_PATH)
    logger.info(f"保存: {OUT_PATH} ({len(df)} 行 × {len(df.columns)} 列)")


if __name__ == '__main__':
    main()
```

### - [ ] Step 2: 実行

```bash
cd C:/Users/iwashita.AKGNET/.openclaw/workspace/boatrace-predictor
python analysis/21_build_phase_b_features.py
```

期待: ~30 秒以内、ログに「保存: ...features_phase_b.pkl (13212 行 × 9 列)」程度。

### - [ ] Step 3: pkl の中身を確認

```bash
python -c "
import pandas as pd
df = pd.read_pickle('analysis/features_phase_b.pkl')
print('shape:', df.shape)
print('columns:', list(df.columns))
print('--- head ---')
print(df.head(3))
print('--- na rate ---')
print((df.isna().mean() * 100).round(2))
print('--- race_category counts ---')
print(df['race_category'].value_counts())
"
```

検証ポイント:
- shape: ~(13212, 9)
- columns: ['race_date', 'venue_id', 'race_number', 'race_category', 'is_planned', 'boat1_skill_gap', 'a_class_consumed', 'day_in_meeting', 'day_label_raw']
- race_category に "qualifier"/"semifinal"/"final"/"general"/"other" の分布が見える
- a_class_consumed の na 率は ~6% 前後 (race_number=1 は NaN)
- day_in_meeting の na 率は数 % (最終日/優勝戦は NaN)

### - [ ] Step 4: コミット

```bash
git add analysis/21_build_phase_b_features.py analysis/features_phase_b.pkl
git commit -m "feat(phase-b): B1-B5 特徴量パイプライン + pkl 生成 (13212 races)"
```

---

## Task 3: analysis/22_phase_b_feature_report.py 統計レポート

**Files:**
- Create: `analysis/22_phase_b_feature_report.py`
- Generated: `analysis/reports/phase_b_features.md`

### - [ ] Step 1: レポート生成スクリプトを作成

`analysis/22_phase_b_feature_report.py` 新規作成:

```python
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


def summary_categorical(df, col):
    """値カウントとパーセンテージを markdown 表で返す"""
    vc = df[col].value_counts(dropna=False)
    total = len(df)
    lines = [f"| {col} | n | % |", "|---|---|---|"]
    for val, n in vc.items():
        pct = 100 * n / total
        val_str = repr(val) if pd.isna(val) or val is None else str(val)
        lines.append(f"| {val_str} | {n} | {pct:.2f}% |")
    return "\n".join(lines)


def summary_numeric(df, col, bins=10):
    """describe() + ビン分布を markdown 表で返す"""
    desc = df[col].describe()
    na_pct = 100 * df[col].isna().mean()
    lines = [f"| stat | value |", "|---|---|"]
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
    lines.append(f"# Phase B 特徴量レポート\n")
    lines.append(f"生成日時: {datetime.now().isoformat()}\n")
    lines.append(f"対象期間: 2026-02-01 〜 2026-04-30\n")
    lines.append(f"レース数: {len(df)}\n\n")
    lines.append("## 1. race_category 分布 (B1)\n")
    lines.append(summary_categorical(df, 'race_category'))
    lines.append("\n\n## 2. is_planned 分布 (B1)\n")
    lines.append(summary_categorical(df, 'is_planned'))
    lines.append("\n\n## 3. boat1_skill_gap (B2)\n")
    lines.append(summary_numeric(df, 'boat1_skill_gap'))
    lines.append("\n\n## 4. a_class_consumed (B3)\n")
    lines.append(summary_numeric(df, 'a_class_consumed'))
    lines.append("\n\n## 5. day_in_meeting 分布 (B4)\n")
    lines.append(summary_categorical(df, 'day_in_meeting'))
    lines.append("\n\n## 6. day_label_raw TOP 10 (B4 原値、保険)\n")
    vc = df['day_label_raw'].value_counts(dropna=False).head(10)
    inline = ["| day_label_raw | n |", "|---|---|"]
    for val, n in vc.items():
        val_str = "NaN" if pd.isna(val) else str(val)
        inline.append(f"| {val_str} | {n} |")
    lines.append("\n".join(inline))
    lines.append("\n")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding='utf-8')
    logger.info(f"レポート出力: {REPORT_PATH}")


if __name__ == '__main__':
    main()
```

### - [ ] Step 2: 実行

```bash
python analysis/22_phase_b_feature_report.py
```

期待: ログに「レポート出力: ...phase_b_features.md」。

### - [ ] Step 3: レポートの中身を確認

```bash
head -80 analysis/reports/phase_b_features.md
```

検証:
- 6 セクション全部入っている
- race_category と is_planned の分布が出ている
- 数値特徴量 (boat1_skill_gap, a_class_consumed) の describe 表が出ている
- na rate が記載されている

### - [ ] Step 4: コミット

```bash
git add analysis/22_phase_b_feature_report.py analysis/reports/phase_b_features.md
git commit -m "feat(phase-b): 特徴量統計レポート生成"
```

---

## Self-Review

- **Spec 網羅性**:
  - B1 race_category + is_planned → Task 1 (`classify_race_category`, `detect_planned_race`)
  - B2 boat1_skill_gap → Task 1 (`compute_skill_gap`) + Task 2 で適用
  - B3 a_class_consumed → Task 2 内 `fetch_a_class_consumed()` SQL window
  - B4 day_in_meeting + day_label_raw → Task 1 (`parse_day_label`) + Task 2 で適用 + 原値保持
  - B5 統合 pkl → Task 2 の出力 `analysis/features_phase_b.pkl`
  - 統計レポート → Task 3
  - 全要件カバー済み
- **プレースホルダ**: なし。Task 1 全 4 関数の実装コードを完備、Task 2 の SQL 完備、Task 3 の整理コード明示。
- **型整合**: `parse_day_label`/`classify_race_category` 等の関数シグネチャは Task 1 で定義し Task 2 で同じシグネチャで呼ぶ。一貫。
- **粒度**: 各 Task 4-5 ステップ、コード貼付済み。

## 完了後の状態

- `analysis/features_phase_b.pkl` (13212 races × 9 列) 生成済み
- `analysis/reports/phase_b_features.md` で各特徴量の分布と欠損率が確認可能
- Phase C V11.5 LightGBM 訓練の入力データ準備完了
- Phase B 全タスク (B1-B5) クローズ
