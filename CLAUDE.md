# boatrace-predictor 専用ルール

本プロジェクトでの仮説検証・filter / 戦略採用判断は、必ず以下のプロトコルに従う。
ユーザー一般ルール (`C:\Users\iwashita.AKGNET\CLAUDE.md`) との重複部分は本ファイルが優先。

## 背景と問題意識 (2026-05-18 確立)

過去の経緯:
- A1 (V10 calibrator) / Phase C (V11.5 LightGBM) / Phase D (Kelly/EV) で ML 改良路線 3 連敗撤退
- Phase D' で「データ駆動 30 仮説 screening + 4 skip filter」を P7 として本番投入 (2026-05-15)
- **3 日経過で bets 3 件・全外し・2 日無発火 → 失敗確定**

失敗の根本原因 (サードパーティ Claude + Claude Code 突合判明、2026-05-18):
1. **30 仮説検証が 1着率だけ** — 中穴狙い戦略には 2着/3着率も等しく重要
2. **QMC 予測 (120 通り分布) と実データの突合をしていなかった** — QMC の系統誤差が未検出
3. **Filter が QMC の std 補正と二重対処** — `compute_ratings_early` の "+0.10 秒以上遅 std×1.12" と P7 H09 "1号艇展示+0.10 以上 skip" は同じ signal の重複反応

教訓: NN 3連敗 → Filter 路線 → 今度は QMC とぶつかった。**次は QMC を直視する番**。

## 仮説検証・フィルター採用プロトコル

新しい仮説、filter、戦略を評価する時は、結論を出す前に必ず以下を全て埋めること。
擁護論だけ書いて結論を急ぐのは禁止。判断は岩下さんが下す。Claude は論点を揃える役。

### Step 1: 擁護論 (Pro)
- なぜこの仮説/filter が機能すると考えられるか
- 物理的・力学的・市場構造的な説明
- データ上の signal (lift, ROI, effect size)

### Step 2: 批判論 (Con) — 必ず以下の全項目を書く

- **標本数チェック**: filter trigger 時の bets 数は train/test それぞれ何件か?
  - test 期間 < 30 件なら統計的検出力不足、信頼区間を必ず計算して提示
- **多重検定リスク**: 何個の仮説から選ばれたか? FDR 補正後でも事後選択バイアスが残る
- **filter 同士の相関**: 既存 filter と独立に効くか? 累積効果が単独効果より小さければ条件相関を疑う
- **市場織り込み仮説**: その signal は他の参加者も見えてないか? オッズが調整済みでないか?
- **PnL vs ROI**: ROI 改善と PnL 改善は別物。bets 削減量と PnL 増加量を必ず併記
- **物理ストーリーの事後合理化リスク**: 反対の結果でも同じく説明できないか?

### Step 2-追加 (2026-05-18 必須化、Phase D' 失敗教訓)

- **QMC vs 実データの突合**: 該当 filter trigger 時に QMC が予測した 120 通り分布と、
  実データの経験的着順分布を比較。系統的にズレている買い目があれば明示。
  QMC 再計算は 1 ヶ月 1.3 分で可能なので「コスト高」は言い訳にならない。
- **全 6 艇 × {1着率/2着率/3着率} の網羅検証**: 1着率だけでなく、各艇の 2着率・3着率も
  filter 条件下で集計。中穴狙い戦略では 2-3 着の構造も等しく重要。
- **代替買い目シナリオ**: 「skip」だけでなく「条件下で別の買い目軸 (例: 4-X-X) に切替
  えた場合の ROI」も試算する。skip は最後の選択肢。
- **QMC との重複チェック**: 新 filter の signal が `compute_ratings(_early)` の 11 項目
  std 補正 (クラス係数 / 展示タイム偏差 / モーター極端 / 部品交換 / ST / 進入コース /
  当地勝率 / 風速 / 波高 / クラス分散 / 展示タイム差) のいずれかと **重複していないか
  必ず明示**。重複する場合、filter ではなく **QMC 係数調整**を選ぶこと。
- **真犯人候補の反証 simulate 必須化** (2026-05-18 B' 失敗教訓): data 検証で
  「真犯人」と判定した仮説は、必ず修正シナリオを実装して simulate で反証可能性を
  確認する。観測された bias の最終発生地点 ≠ 因果の原因。「観察 → 解釈 → 修正」の
  最初の解釈で確定せず、複数解釈の simulate で淘汰すること。
  例: C で「H2 ロジット往復で B1 +12.58pt 増幅」を真犯人と判定したが、B' で
  6 修正シナリオ (S1 線形化 / S3 calibrator+線形 等) が全て baseline より悪化、
  「真犯人」判定が反証された (`analysis/reports/51_qmc_correction_scenarios.md`)。

### Step 3: 未検証論点 (Unknowns)
- 現データでは検証できない条件
- forward 期間でしか確認できない仮説
- 追加データ収集が必要な項目

### Step 4: 採用基準による自動振り分け

以下の数値ルールに照らして candidate を振り分け、結論は出さない:

**自動却下条件 (一つでも該当したら却下 candidate)**:
- test 期間 trigger 標本 < 30 件
- test 期間 effect size < ±5pt
- 累積効果が単独効果より小さい (条件相関の兆候)
- 物理的・市場構造的説明が不可能
- **QMC の `compute_ratings(_early)` で既に処理されている signal と重複** (Phase D' 失敗教訓)

**保留 candidate (追加検証必要)**:
- test 期間標本 30-100 件
- effect size 5-10pt
- 物理説明はあるが市場織り込みの可能性が払拭できない
- QMC との独立性が完全には確認できない

**採用 candidate (shadow 投入検討可)**:
- test 期間標本 100 件以上
- effect size 10pt 以上
- 物理説明明快、独立性確認済み
- QMC では構造的に表現できない signal であることが明示されている

### Step 5: 判断は岩下さんに投げる

Claude は「俺ならこうする」とは書かず、上記 1-4 を整理した状態で岩下さんの判断を待つ。
ただし「明らかな却下 candidate」については、その旨を明示する。

### Step 6: Shadow 投入ルール (採用判断後)

- 本番投入前に最低 shadow 並走 2 週間、撤退ライン明示
- shadow 省略の本番投入は岩下さんが明示的に「shadow 省略する」と書いた時のみ
- shadow ROI < baseline 80% で即撤退

## Filter と QMC の役割分担原則 (2026-05-18 確立)

P7 失敗の構造的教訓を踏まえた設計原則:

- **QMC が既に std 補正で表現している signal を、filter で重ねて skip するのは禁止**
- Filter は **QMC では構造的に表現できないもの** に限定:
  - 会場特殊性 (戸田の3コース捲り、桐生の夏季4コース等)
  - レース番号特性 (R1-R3 序盤、R12 後半等)
  - 企画レース構造 (シード番組、反動レース等)
  - 市場の集合的バイアス (本命人気のオッズ歪み等)
- 新 filter 追加前に、`compute_ratings(_early)` の 11 項目 (クラス係数 / 展示タイム偏差
  / モーター極端 / 部品交換 / ST / 進入コース / 当地勝率 / 風速 / 波高 / クラス分散 /
  展示タイム差) で既に処理されていないか **必ず**確認する
- 「QMC のヒューリスティック係数を調整する」と「filter を追加する」は別物。
  係数調整の方が筋がいい場合、係数調整を優先する
- 「filter で skip する」より前に「条件下で異なる買い目を取る (e.g. mc3_alt_pattern)」
  を検討する

## 過去の撤退記録 (vault `30_decisions/` 参照)

参照ファイル:
- `2026-05-12-phase-a-a4-a1-findings.md` — A1 V10 calibrator 旧版優位
- `2026-05-15-phase-c-retreat-v10-tie.md` — Phase C V11.5 LightGBM AUC タイ
- `2026-05-15-phase-d-retreat-already-applied.md` — Phase D Kelly/EV 既存設計優位
- `2026-05-15-p7-data-driven-launch.md` — P7 launch (3 日で撤退、本ファイル)
- (新規予定) `2026-05-18-p7-rollback-architectural-rethink.md` — P7 ロールバック + filter/QMC 二重対処判明

## 検証ワークフローの改訂 (2026-05-18 確立)

P7 失敗教訓を踏まえた、新規仮説・filter・戦略検証の標準ワークフロー:

### 必須実施項目
- **30 仮説スクリーニングの評価軸は 6 艇 × {1着率/2着率/3着率} = 18 metric に拡張済**
  (`analysis/48_full_position_hypothesis.py` 参照)。1着率のみの検証は禁止
- **QMC 過去予測との突合分析を必須化** (`analysis/49_qmc_vs_empirical.py` 参照)。
  QMC 再計算は 1 ヶ月 1.3 分で可能、コストは言い訳にならない
- **新戦略提案時は「代替買い目シナリオ」の data 検証を伴う**こと。
  「skip」だけで結論を出すのは禁止。「H09 trigger 時に 3-X-X / 2-X-X 軸の中穴で
  実際稼げるか」を data で検証してから採否判断
- **新 filter 提案時は QMC `compute_ratings_early` 11 項目との重複確認必須**。
  重複する signal を扱う filter は自動却下

### 検証ツール (本日確立)
| ツール | 用途 |
|---|---|
| `analysis/46_hypothesis_screening.py` | 30 仮説 1着率 screening (legacy、参考用) |
| `analysis/48_full_position_hypothesis.py` | 30 仮説 × 6 艇 × 3 着 拡張 (標準) |
| `analysis/49_qmc_vs_empirical.py` | QMC 120 通り vs 実頻度突合 (必須) |
| `analysis/qmc_predictions_cache.pkl` | QMC 過去予測キャッシュ |

### 過去の失敗パターンとの対比 (避けるべき道筋)
- **ML 改良路線 4 連敗** (A1 calibrator / Phase C V11.5 / Phase D Kelly / Phase D' P7):
  表面の ML レイヤーを弄っても、根本の QMC キャリブレーションを直視していなかった
- 教訓: **「強い ML がカバーしない領域をデータ検索で埋める」** より、**「QMC の calibration 系統誤差を直視する」** が次の本筋

## 関連ファイル / コードベース

| 領域 | ファイル |
|---|---|
| NN モデル | `src/models.py`, `src/features.py`, `models/boatrace_model.pth` |
| QMC エンジン | `src/monte_carlo.py` (`compute_ratings_early`, `qmc_sanrentan_v3`) |
| 戦略 / EV | `src/betting.py` (`_strategy_kelly`, `calculate_all_strategies`) |
| 戦略 config | `config/betting_config.json` |
| キャリブレーター | `models/calibrators.pkl`, `models/calibrators_v2.pkl` (検証で生成、本番非投入) |
| Phase B 特徴量 | `analysis/features_phase_b.pkl`, `src/phase_b_features.py` |
| 30 仮説 screening (legacy) | `analysis/46_hypothesis_screening.py` |
| 30 仮説 6艇×3着 拡張 | `analysis/48_full_position_hypothesis.py` |
| QMC 突合分析 | `analysis/49_qmc_vs_empirical.py` |
| QMC 予測キャッシュ | `analysis/qmc_predictions_cache.pkl` (5828 races、再計算 95 秒) |
