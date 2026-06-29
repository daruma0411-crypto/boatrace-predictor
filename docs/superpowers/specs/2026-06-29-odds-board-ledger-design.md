# オッズ盤台帳（odds-board ledger）v1 — 設計書

- **日付**: 2026-06-29
- **状態**: 設計（実装プラン作成前）
- **対象**: boatrace-predictor / 予測スケジューラ（Railway 上の Streamlit 内バックグラウンドスレッド）
- **種別**: 本番スケジューラへの**追記オンリー**の機能追加（既存テーブル無改変）＋ read-only 研究ヘルパ

---

## 1. 背景と問題（なぜ作るか）

「ちょこちょこ戦略/キャリブレーションを増やし、良いものを選ぼうとする → 良さそうな時にはデータが無い → また増やして一から経過観測」というループで、**データが永遠に溜まらず真理が見えない**。

根本原因：DBに残るのは**フィルタを通って実際に買った上位3点だけ**。フィルタ（=戦略/キャリブレーション）を変えると買い目が変わり、過去の記録が使えず**サンプルがゼロに戻る**。

解決の核心：保存対象を「ブレる買い目」から「**ブレない事実**」に変える。各レースについて
**①モデルの見立て（確率）②その時の全オッズ盤 ③結果** を残せば、フィルタ・EV下限・オッズ枠・点数・キャリブレーション・ケリーは**台帳の上で後からいくらでもリプレイ**でき、ゼロに戻らない。

現状の在庫を確認した結果、**足りないのは「全オッズ盤」だけ**：
- ① 確率（probabilities_1st/2nd/3rd）… ✅ `predictions` に全レース分あり
- ③ 結果・配当 … ✅ `races`（result_1st/2nd/3rd, payout_sanrentan）にあり
- ② 全オッズ盤 … ❌ スケジューラが毎レース取得（`odds_dict`, 全120通り）するが betting 計算に使うだけで**捨てている**（odds 系テーブルは存在しない）

過去のオッズ盤は復元不能なため、**本番スケジューラに追記して「これから」捕るしかない**。

## 2. ゴールと成功条件

**v1 のゴール**：判断時の全3連単オッズ盤を毎レース確実に保存し始めること自体。

**成功条件（デプロイ後の現物確認で判定。"サイレント失敗しない"を担保＝V11.5教訓）**：
- 当日レースで `race_odds_board` に行が出続ける（毎レース1行）。
- `n_combos` が概ね 120 前後。
- **同レースで実際に採用された bet のオッズが、保存した盤の同じ組み合わせの値と一致**する（クロスチェック）。
- スケジューラの予測・購入・既存 `predictions/bets` 保存に**一切の悪影響が無い**。

## 3. スコープ

**含む**：
- 新テーブル `race_odds_board`（Railway DB に追加、既存テーブルは無改変）。
- スケジューラに**追記オンリー**のキャプチャ（`odds_dict` を1レース1行で upsert）。`try/except` で完全隔離。
- read-only 研究ヘルパ：`race_odds_board ⨝ predictions ⨝ races` で「買わなかった組み合わせも含む candidate 全体」を再構成する関数。
- 検証スクリプト（キャプチャが実際に効いているかを現物で確認）。

**含まない（将来フェーズ / YAGNI）**：
- 締切直前の確定オッズ（全盤CLV）— 判断時1回のみ。
- QMC 120通り出力 / QMC入力(ratings)の保存 — 必要時は既存 marginals から再計算（cache あり）。
- 過去バックフィル — オッズ盤は過去分が無いため不可。履歴は「採用betのオッズ」での部分分析のみ（既存）。
- リプレイ評価ハーネス本体（EV/オッズ枠/点数/ケリーの後付け評価）— **別 spec**。
- 2連単オッズ盤（`odds_2t`）の保存 — v1 は 3連単のみ。

## 4. アーキテクチャ / データフロー

```
スケジューラ（毎レース、締切3〜5分前 = 購入判断時）
  predict → odds_dict(_parse_odds で全120通り) → calculate_all_strategies → save_prediction
                          └──★ capture_odds_board(race_id, odds_dict)  ← 新規・try/except隔離
                              （race_id で upsert＝最初の1スナップショットのみ採用）

研究側（read-only セッション）
  race_odds_board(全オッズ) ⨝ predictions(確率 marginals) ⨝ races(結果) → candidate 台帳
```

書き込み点は scheduler 内、`odds_dict` 構築直後（既に全120通りがメモリにある場所）。`save_prediction` とは別の独立呼び出しにし、**1レース1回**（戦略数ぶん呼ばない）。

## 5. データモデル

新テーブル（Railway PostgreSQL、`CREATE TABLE IF NOT EXISTS`）：

```sql
CREATE TABLE IF NOT EXISTS race_odds_board (
    race_id     integer PRIMARY KEY,          -- 1レース1行（複数インスタンス並走に強い）
    captured_at timestamptz NOT NULL DEFAULT now(),
    odds_3t     jsonb NOT NULL,               -- {"1-2-3": 45.6, "1-2-4": 88.1, ...} 全120通り
    n_combos    integer NOT NULL              -- 健全性チェック（≈120 を期待）
);
```

- **PRIMARY KEY (race_id)** ＋ `INSERT ... ON CONFLICT (race_id) DO NOTHING` で **first-write-wins**。複数スケジューラインスタンスが同レースを処理しても**最初の1スナップショットだけ**残る（判断時オッズの一貫性を担保）。
- `odds_3t` の key は既存 `bets.combination` と同形式（例 `"3-1-5"`）。
- 容量：1行あたり 120 entries の小さな JSON。1日数百レース×数百日でも軽微。

## 6. 安全策（V11.5 サイレント失敗の教訓を反映）

- キャプチャ呼び出しは **`try/except` で完全隔離**。例外時は `logger.warning` のみ、**予測・購入・既存保存を絶対に止めない**。
- `CREATE TABLE IF NOT EXISTS` を起動時に1回（または初回キャプチャ時に冪等実行）。既存スキーマは無改変。
- **追記オンリー**：既存の `predictions`/`bets`/`races` への INSERT/UPDATE は一切変更しない。
- **発火監視**：キャプチャ成否を `scheduler_health` に記録（`odds_board_saved race_id=...` / 失敗時 `odds_board_failed ...`）。V11.5 のような「コミットしたが動いていない」を**現物で検知**できるようにする。

## 7. 研究側ヘルパ（read-only）

`analysis/` 配下に read-only 関数（既存 `analysis/soft_regime/db.py` の `conn.set_session(readonly=True)` 流儀に倣う）：
- `load_candidate_board(conn, date_from, date_to)`：`race_odds_board ⨝ predictions(marginals) ⨝ races(result)` を返す。
- v1 ではデータ取得・整形まで。EV/フィルタ/ケリーのリプレイ評価は別 spec。

## 8. テスト

- **単体（capture）**：`odds_dict` を渡すと期待 JSON 行が書かれる／同 race_id 二度目は無視（ON CONFLICT DO NOTHING）／例外時に呼び出し元へ伝播しない（隔離）。
- **単体（read helper）**：3テーブル結合が期待形を返す（テスト用に直近レースで件数・キー形を assert）。
- **統合/検証（デプロイ後・現物）**：当日レースで行が増える／`n_combos≈120`／**採用bet のオッズと盤の値が一致**／`scheduler_health` に成功ログ。

## 9. リスクと運用

- **リスク 🟠**（本番スケジューラに触る＝実購入を駆動するプロセス）。緩和＝try/except 隔離・追記のみ・upsert・発火監視。
- **ロールバック**：キャプチャ呼び出しを外すだけ（テーブルは残置しても無害）。
- **デプロイ**：Railway 再デプロイ（＝現在 V11.5 で保留中のものと同経路）。ただし**コミットは別**にし混ぜない（surgical changes）。
- **デプロイ後**：§2 成功条件を現物確認するまで「完了」と報告しない。

## 10. 期間の見立て（分析に入れるまで）

- **全盤レイヤー（確率較正・荒れ予測・EV地形・緩さ信号）**：1日数千件のため **1〜2週で初見、約1ヶ月で固い結論**（soft_regime が約1ヶ月/6000レースで実証済み）。
- **戦略の損益・ケリー割合**：的中が律速（3連単 的中4〜6%、信頼に的中≈30回＝500〜1000本）→ **2〜3ヶ月**。
- 助走：4月以降の部分履歴（採用betオッズ＋確率＋結果）で「採用betベース分析」は即可能。台帳はそれを全盤へ広げ「これから完全化」する。
- 推奨順：デプロイ翌日に現物確認 → 約2週で「2着3着の同時確率検証（ケリーの前提）＋全盤EV/オッズ枠」 → 約1ヶ月で較正/信号確定 → 2〜3ヶ月でROI/ケリー。
