---
date: 2026-05-15
topic: Phase B 新特徴量 (B1-B5)
issue: https://github.com/daruma0411-crypto/boatrace-predictor/issues/4
status: approved
risk: 🟢 通常 (READ-ONLY、本番非介入)
depends_on: A3 完了 (race_titles テーブル充足、commits cbb4a73〜3fdcd63)
unlocks: Phase C (V11.5 LightGBM 再学習)
---

# Phase B 新特徴量 (B1-B5) 設計

## 背景

Issue #4 改善ロードマップ Phase B の 5 特徴量 (B1-B5) を 1 つの spec で扱う。A3 完了で `race_titles` に title / subtitle / day_label が揃った前提。

V10 (`boatrace_model.pth`, 76dim) では認識していない「番組編成の構造的歪み」「節中位置」「実力乖離」「反動レース」を数値化し、Phase C V11.5 LightGBM の入力とする。

## スコープ

| 項目 | 内容 |
|---|---|
| 範囲 | 5 つの新特徴量を生成し DataFrame pkl + 基本統計レポート |
| 範囲外 | V10/V11.5 への統合 (Phase C)、本番投入、A 級プール残量推定 (将来検討) |
| 本番影響 | ゼロ。analysis/ 配下完結、DB は READ-ONLY |
| リスク判定 | 🟢 通常 |

## 成功条件

1. `analysis/features_phase_b.pkl` 生成: 2026-02-01〜2026-04-30 の 13,212 races を行として持つ DataFrame、5+ 特徴量列を含む
2. `analysis/reports/phase_b_features.md` 生成: 各特徴量の分布 (histogram / top values) と欠損率を記録
3. `tests/test_phase_b_features.py` の全テストが PASS
4. 各特徴量がリーケージなし (= 予測時点で取得可能なデータのみ使用)

## 各特徴量の定義

### B1: race_category (str) + is_planned (bool)

`race_titles.subtitle` および `race_titles.title` からカテゴリ分類。

**race_category 抽出ルール (subtitle ベース)**:
- subtitle に `'予選'` 含む → `"qualifier"`
- subtitle に `'準優'` 含む → `"semifinal"`
- subtitle に `'優勝戦'` 含む → `"final"`
- subtitle に `'一般'` 含む → `"general"`
- それ以外 → `"other"`

**is_planned フラグ (title ベース)**:
- title に「企画語辞書」のいずれかを含む → True
- 企画語辞書 (初期): `['Ｖプレミア', 'V プレミア', 'サンライズ', 'ゴールデン', 'GW 特選', 'GW特選', 'プレミアム', '記念']`
- 辞書は実装時に backfill データで頻出語を確認して微調整

### B2: boat1_skill_gap (float)

```
boat1_skill_gap = boats.win_rate_2[boat_number=1] - mean(boats.win_rate_2[boat_number IN 2..6])
```

正値: 1号艇が他より強い (典型的な「カタ」レース)
負値: 1号艇が他より弱い (荒れ要素)

NULL 扱い: win_rate_2 が NULL の場合は当該レース全体 NULL。

### B3: a_class_consumed (float [0, 1])

```sql
WITH prior AS (
    SELECT b.race_id
    FROM races r
    JOIN boats b ON r.id = b.race_id
    WHERE r.race_date = :target_date
      AND r.venue_id = :target_venue
      AND r.race_number < :target_race_number
)
SELECT
    COUNT(*) FILTER (WHERE b.player_class IN ('A1', 'A2'))::float / NULLIF(COUNT(*), 0) AS a_class_consumed
FROM boats b
WHERE b.race_id IN (SELECT race_id FROM prior)
```

target レース時点での「同日同会場の累積 A 級出走比率」。target.race_number=1 (先行レース 0 件) の場合は **NaN** (= まだ何も消費していないので「消費率」未定義)。Phase C モデル側で NaN を「特徴量未取得」として扱えるよう、0.0 とは区別する。

リーケージ防止: 対象レースより**前**の race_number のみを集計。

### B4: day_in_meeting (int) + day_label_raw (str)

```python
def parse_day_label(label: str) -> int | None:
    if label is None: return None
    if label == '初日': return 1
    m = re.match(r'^([０-９0-9]+)日目$', label)  # 全角数字対応
    if m: return int(zenkaku_to_hankaku(m.group(1)))
    if label in ('最終日', '優勝戦'): return None  # 節長依存、別途処理
    return None
```

day_label_raw は保険として元文字列をそのまま保持。Phase C で「節長から最終日を逆算」したくなった場合に使う。

## アーキテクチャ

```
src/phase_b_features.py (純関数)
  ├ classify_race_category(subtitle) → str
  ├ detect_planned_race(title, dict) → bool
  ├ parse_day_label(label) → int | None
  └ compute_skill_gap(boats: list[dict]) → float | None

analysis/21_build_phase_b_features.py (オーケストレータ)
  ├ DB から races + boats + race_titles を JOIN で SELECT
  ├ window 集約で B3 a_class_consumed を計算 (SQL)
  ├ Python 側で B1/B2/B4 を行ごとに適用
  └ DataFrame を pkl 保存

analysis/22_phase_b_feature_report.py (レポーター)
  └ pkl 読み込み → 各列の分布・欠損率を markdown 化
```

## ファイル構造

| Path | Action | 用途 | 推定行数 |
|---|---|---|---|
| `src/phase_b_features.py` | Create | B1/B2/B4 の純関数 | ~120 |
| `tests/test_phase_b_features.py` | Create | 純関数の単体テスト | ~150 |
| `analysis/21_build_phase_b_features.py` | Create | DB → pkl パイプライン | ~200 |
| `analysis/22_phase_b_feature_report.py` | Create | pkl → markdown レポート | ~120 |
| `analysis/features_phase_b.pkl` | Generated | DataFrame (race_id index × 6 列) | - |
| `analysis/reports/phase_b_features.md` | Generated | 分布・欠損率レポート | - |

## 出力 DataFrame 構造

```
                race_id  race_category  is_planned  boat1_skill_gap  a_class_consumed  day_in_meeting  day_label_raw
0                    1     "qualifier"       False              2.34               NaN               1            "初日"
1                    2     "qualifier"       False              1.85              0.083               1            "初日"
...
```

`race_id` を index、他 6 列。欠損は NaN (numeric) or None (str)。

## リーケージ防止

| 特徴量 | リーケージ懸念 | 対策 |
|---|---|---|
| B1 race_category | なし (レース固有データ) | - |
| B1 is_planned | なし | - |
| B2 boat1_skill_gap | なし (出走艇プロフィール) | - |
| B3 a_class_consumed | **あり** | SQL で race_number 厳密 `<` で過去のみ集計 |
| B4 day_in_meeting | なし (節情報) | - |

## テスト

| テスト | 種別 | 内容 |
|---|---|---|
| `test_classify_race_category` | unit | 5 ケース (qualifier/semifinal/final/general/other) で正解ラベル |
| `test_detect_planned_race` | unit | 企画語辞書ヒット時 True、それ以外 False |
| `test_parse_day_label_*` | unit | 初日/2日目/最終日/優勝戦/None/不正値 |
| `test_compute_skill_gap` | unit | 6 艇ダミーで計算結果一致、NULL 含む場合 None |
| `test_pipeline_smoke` | integration | 直近 1 日 (2026-04-30) で pkl 生成 → 168 行 + 6 列を確認 |

## 期間スコープ

2026-02-01 〜 2026-04-30 (race_titles backfill 済み 13,212 races) を pkl 対象とする。それ以外の期間 (2026-01 以前 / 2026-05 以降) は race_titles 未充足のため除外。

## 関連

- Issue: https://github.com/daruma0411-crypto/boatrace-predictor/issues/4
- A3 spec: `docs/superpowers/specs/2026-05-12-a3-race-title-scraping-design.md`
- vault: `30_decisions/2026-05-15-a3-completion-and-scope-expansion.md`
- memory: `project_boatrace_improvement_roadmap.md`
