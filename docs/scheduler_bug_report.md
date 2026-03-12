# スケジューラー停止バグ調査報告 (2026-03-12)

## 現象
- 3/12のベット: 0件（スケジューラー未起動）
- scheduler_healthテーブル: 3/11 13:05 UTC以降レコードなし
- Streamlit UI: 正常稼働中

## 根本原因

### bug #1: run_polling() にトップレベル例外ハンドラがない
`src/scheduler.py` の `run_polling()` メソッド内の `while True` ループが try-except で保護されていない。
1回の例外でループ全体が死ぬ。

```python
# 現状（壊れる）
def run_polling(self):
    while True:
        current = now_jst()  # ← ここで例外 → ループ脱出 → スレッド終了
        ...

# 修正（必須）
def run_polling(self):
    while True:
        try:
            current = now_jst()
            ...
        except Exception as e:
            logger.error(f"ポーリングサイクルエラー: {e}", exc_info=True)
            time.sleep(60)
```

### bug #2: リトライが発動しない
`streamlit_app/app.py` の `_run()` で、`run_polling()` が例外なしで正常returnした場合リトライ条件に該当しない。

### bug #3: Railway healthcheck がスケジューラー死亡を検知できない
`railway.json` の healthcheckPath が `/` (Streamlit UI) を見ているため、スケジューラースレッドが死んでも「正常」と判定される。

## DB確認結果
- 3/11: 71件ベット（全て05:00 UTC = 14:00 JST の1時間に集中）
- 3/12: 0件
- scheduler_health最終: 3/11 13:05 UTC「ポーリング開始」→ 以降記録なし

## 修正優先度
1. **即急**: run_polling() while内に try-except 追加
2. **短期**: _run() のリトライ条件修正（正常returnでもリトライ）
3. **中期**: スケジューラーを別プロセスに分離

---

# 構造的欠陥 監査レポート (2026-03-12)

別ターミナル（サードパーティ視点）による全コードベース監査結果。
scheduler.py の try-except 欠如は上記で既報のため除外。

---

## 即座に対応（資金安全に直結）

### 欠陥A: 日次最大損失制限がない
**ファイル**: `src/betting.py` L307-423

- レースごと: `max_total_bet_ratio = 0.02` → bankroll 20万の場合 4万円/レース/戦略
- **6戦略が同時にベットすると 24万円/レース**
- 1日最大288レース × 24万 = 理論上 **6,912万円/日** の損失可能
- **日次全戦略合計の損失キャップが存在しない**

```python
# 修正案: betting.py に追加
DAILY_LOSS_LIMIT = 50000  # 全戦略合計の1日最大損失額

def check_daily_loss_limit(self):
    today_loss = get_today_total_loss()  # DB集計
    if today_loss >= DAILY_LOSS_LIMIT:
        logger.warning(f"日次損失上限到達: {today_loss}円")
        return False
    return True
```

### 欠陥B: ケリー式の分母が0近い時に数値爆発
**ファイル**: `src/betting.py` L368-381

```python
b = discounted_odds - 1.0
kelly = (b * prob - q) / b  # b=0.001 → kelly=-499 になりうる
```

- `b < 0.01` の場合スキップすべき
- 四捨五入誤差で `kelly > 0` になりベット実行される可能性

```python
# 修正案
if b < 0.01:
    skip_counts['odds_too_low'] += 1
    continue
```

### 欠陥C: DB接続失敗時に重複ベット
**ファイル**: `src/scheduler.py` L249-270

```python
def _already_bet(self, race_id):
    try:
        # DB確認
    except Exception:
        return False  # ← DB障害時「ベットしてない」扱い → 重複ベット
```

```python
# 修正案: 安全側に倒す
    except Exception:
        logger.critical(f"重複チェックDB障害 race_id={race_id}")
        return True  # ベット済み扱いにしてスキップ
```

---

## 中程度（1-2週間で対応）

### 欠陥D: ThreadPoolExecutor で requests.Session 共用
**ファイル**: `src/scheduler.py` L278-295

- 2スレッドが同時に同じ `session` で HTTP GET → **requests.Session はスレッドセーフではない**
- 接続状態が混在して不正レスポンスを受ける可能性

修正案: スレッドごとに独立した Session を作るか、シーケンシャル処理に変更

### 欠陥E: モデル未発見時にダミーモデルで推論・ベット実行
**ファイル**: `src/predictor.py` L32-40

```python
except FileNotFoundError:
    self.model = BoatraceMultiTaskModel()  # 未学習モデル
    self.model.eval()  # ← ランダム確率でベット計算が走る
```

修正案: モデル未発見時は `raise` してベットをスキップ

### 欠陥F: 全艇データ欠損でも予測続行
**ファイル**: `src/features.py` L197-201

- NaN/None → 0.0 に変換して推論実行
- **全艇のデータが欠損してもゼロベクトルで予測を返す（信頼性ゼロ）**

修正案: 欠損率 > 30% なら予測スキップ

### 欠陥G: アンサンブルで次元不一致時にゼロパディング
**ファイル**: `src/predictor.py` L184-202

- モデルA: 208次元、モデルB: 194次元 → Bは14次元切り捨てて推論
- 警告ログは出るがベットは実行される

---

## 設計上の構造問題

### 欠陥H: スケジューラーがStreamlitのデーモンスレッド
- Streamlit はリロードのたびに全スクリプトを再実行
- デーモンスレッドは親プロセス終了で無条件終了
- **スケジューラーを別プロセス（worker）に分離すべき**

### 欠陥I: 結果精算が23:00 JST固定
**ファイル**: `src/scheduler.py` L184-191
- 遅延レースが23:00時点で未確定の場合、精算されない
- 翌日に持ち越されるが、翌日のロジックで拾えるか未確認

### 欠陥J: ケリー基準の前提が満たされていない
- ケリー式は「モデル確率 = 真の確率」を仮定
- 実際のモデル確率は miscalibrated
- フルケリーではなく **ハーフケリー or 1/4ケリー** が安全

---

## 対応優先度まとめ

| 優先度 | 欠陥 | 修正コスト |
|---|---|---|
| **即座** | A: 日次損失制限追加 | 小（10行程度） |
| **即座** | B: ケリー分母下限チェック | 小（2行） |
| **即座** | C: DB障害時の重複ベット防止 | 小（1行変更） |
| 1-2週間 | D: Session スレッドセーフ化 | 中 |
| 1-2週間 | E: ダミーモデル禁止 | 小 |
| 1-2週間 | F: 欠損率チェック追加 | 小 |
| 中期 | H: スケジューラー分離 | 大（アーキテクチャ変更） |
| 中期 | J: ケリー基準のキャリブレーション | 中 |
