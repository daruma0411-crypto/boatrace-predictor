# P改ロールアウト（V11撤去・P改本実装・P全場・ダッシュ20万） — 設計書

- **日付**: 2026-07-30
- **状態**: 設計（実装プラン作成前）
- **対象**: boatrace-predictor / scheduler・betting・dashboard（Railway）＋オフライン分析
- **種別**: 本番コード変更を含む（**ただし全て紙運用＝実弾なし**。purchaser は DRY_RUN=true のまま触らない）

---

## 1. 背景

- 認定審査(2026-07-30): **P(mc_venue_focus)=4ヶ月+295,230/ROI133%/P(本物)=94%**・GW抜き全月プラス／**V11=P(勝ち)=0%** で実購入を停止済み／P2=GW依存で見送り。
- **P改 = P＋堅レース見送り＋低温日半額** を定義凍結済み（`analysis/soft_regime/p_kai_track.py`, commit 675ca03）。後付け検証: PnLほぼ維持で**最大DD -78,200→-36,100（54%減）**。
- Pの弱点=**機会が少ない**（得意場5場のみ: 平和島4・浜名湖6・住之江12・尼崎13・唐津23 → 全レースの~1.6%）。
- 台帳(6/30〜)は**全24場の全オッズ盤**を毎日記録中＝「Pの買い方を全場でやったら」は手持ちデータで計算可能（ただし1ヶ月分のみ）。

## 2. ゴール（4点）

1. **V11撤去**: 新規bet生成停止＋ダッシュボード非表示。履歴データは研究用に保持（削除しない）。
2. **全場リプレイ（オフライン・即回答）**: 台帳1ヶ月分で「P式買い方×全24場」を再現し、**次の得意場候補の一次リスト**を出す（仮説どまり、事後選択と明記）。
3. **「P全場」shadow 追加**: P と同じ設定から会場制限だけ外した `mc_venue_all` を紙運用登録。本番同一コードで forward データが毎日蓄積 → 候補場の確証エンジン。
4. **P改の本実装**: スケジューラ内で P の買い目に凍結ルールを適用し `p_kai` として記録。**P改・P全場をダッシュボードに ¥200,000 フレッシュスタートで表示**。

**成功条件（デプロイ後の現物確認）**: (a) v11_var13 の新規bet/predictionが止まる (b) mc_venue_all と p_kai のbetが記録され始める (c) p_kai のbetが offline tracker(`p_kai_track.py`)の判定と一致（忠実度） (d) ダッシュに2カードが¥200,000から表示 (e) 既存P/P2等の動作に影響なし。

## 3. スコープ外（YAGNI）

- 実購入の変更（purchaser は DRY_RUN=true のまま。TELEBOAT_STRATEGY も触らない）
- P2/V11.5/その他戦略の変更（V11.5 shadow は研究データとして継続）
- ケリー割合の変更・温度での増し張り（**守り(×0.5)のみ**。攻めは禁止＝H3教訓）
- 温度・堅さパラメータの再学習（凍結値を使う。変更時は別名の新定義として登録）

## 4. コンポーネント設計

### 4.1 V11撤去
- `src/betting.py` の `ACTIVE_STRATEGIES` から `'v11_var13'` を削除（`'v11_5_var13'` は残す）。
- 注意: **V11VAR13Predictor は全戦略共有の予測エンジンなので残す**（撤去するのは betting 戦略エントリのみ）。
- ダッシュボード: 非表示リスト（例 `HIDDEN_STRATEGIES = {'v11_var13'}`）で v11_var13 のカード/集計を隠す。DB削除はしない。

### 4.2 全場リプレイ（オフライン）
- 新規 `analysis/venue_expand/replay.py`（read-only）: 台帳(odds_3t)＋predictions(marginals)＋boats/races から、**本番の MC v1 経路（`qmc_sanrentan`）と P の config**（min_odds5/max_odds40/entropy/skip_56/EV/kelly を可能な限り本物の関数を再利用）で全24場の picks を再現し、確定配当で決済。
- **忠実度ゲート**: P の実際のbet（5場・7月分）を同じリプレイで再現し、記録された picks/PnL と概ね一致することを先に確認（combo validate と同じ思想）。
- 出力: 会場別 n/的中/PnL/ROI ＋ プール（非WL場全体）。**会場別は薄い（各場数bet）ため仮説どまり・事後選択リスクを明記**。
- ※ 完全一致は不可能な近似（乱数・オッズ取得タイミング等）。ゲートが大きくズレたら結果を出さず原因調査。

### 4.3 P全場 shadow（mc_venue_all）
- `config/betting_config.json` に `mc_venue_all` を追加: `mc_venue_focus` のクローンから `include_venues` を除去（全場対象）。他フィルタ（min_odds/max_odds/entropy/skip_56/kelly等）は同一。
- `ACTIVE_STRATEGIES` に追加。以後、通常フローで bets/predictions が記録される（コード追加は config＋setの2行のみ）。

### 4.4 P改（p_kai）本実装
- **`src/softness.py`（新規）**: 凍結パラメータを定数として焼き込み（`WEIGHTS`/`MU`/`SD`/`HARD_CUT=7.863`/`TEMP_THR=7.972`/`COLD_SCALE=0.5`/`TEMP_WINDOW=7`。値は `p_kai_track.py` の学習ロジックから一度だけ生成して埋め込む）。`score_race(boats_data) -> float|None`（wr_std/wr_top_gap/inside_wr/motor_std/st_std → 標準化 → 線形）。
- **軟らかさログ**: 新テーブル `race_softness(race_id int PK, race_date date, score real)`。スケジューラの予測処理で1レース1回 upsert（odds_board と同じ自己隔離パターン: try/except・ON CONFLICT DO NOTHING・scheduler_health 記録）。
- **温度**: `get_temperature(date) -> float|None` = race_softness から**前日までの直近7日**の日次平均の平均。日数不足(<3日)なら None→**scale 1.0 扱い**（保守的）。デプロイ時に**過去30日分を boats から一括バックフィル**する одноразовый スクリプトを用意（read-only計算→race_softness へ書込み。これは新テーブルへの追記のみ）。
- **bet 生成**: スケジューラで `calculate_all_strategies` の結果に `mc_venue_focus` の picks があれば、`score_race > HARD_CUT` のレースのみ・温度低なら amount×0.5（100円丸め・min_bet未満は切捨て）で `strategy_type='p_kai'` として保存。全体を try/except で隔離（失敗しても P 本体・他戦略に影響させない）。
- **P改はPの派生である事実を保存**: p_kai の bet は P と同レース・同買い目（金額のみ変わる）なので、`p_kai_track.py`（オフライン版）と突合できる＝**デプロイ後の忠実度検証**（成功条件(c)）。

### 4.5 ダッシュボード
- `STRATEGY_RESET_DATES` に `p_kai` と `mc_venue_all` をデプロイ日で追加（¥200,000 フレッシュスタート、v11_var13 の 5/27 リセットと同じ機構）。
- 表示名: 「P改（堅見送り+低温半額）」「P全場（得意場発掘）」。v11_var13 は HIDDEN で非表示。

## 5. テスト・検証

- **単体**: softness.score_race が `p_kai_track.py` と同一レースで同一スコア（許容誤差内）／温度計算のリーク無し（当日を含まない）／p_kai bet 導出（フィルタ・×0.5・丸め）の純ロジック／config 追加の smoke（P と mc_venue_all の並存）。
- **統合（デプロイ後・現物**＝V11.5サイレント失敗の教訓）: §2 成功条件 (a)〜(e) を検証スクリプトで機械チェック（bets/predictions の件数・p_kai と tracker の一致・health ログ）。
- 既存テスト（odds_board/combo/soft_regime）が壊れないこと。

## 6. リスクと対策

- 🟠 本番スケジューラ変更 → V11.5/odds_board で確立した**自己隔離パターン**（try/except・health記録・追記のみ）を全箇所に適用。実弾ゼロ（purchaser dry）のため金銭リスクなし。
- ビルド → mise.toml 修復済み。デプロイ後に Build 成功と発火を現物確認（サイレント失敗ゲート）。
- 過学習 → 全場リプレイの会場別結果は**仮説どまり**と明記。得意場の追加判断は mc_venue_all の forward 証拠で行う（事後選択しない）。パラメータは凍結値のみ。
- ロールバック → ACTIVE_STRATEGIES から外す＋dashboard エントリ削除（コミット revert）。新テーブルは残置無害。

## 7. デプロイ・運用

- 1回の Railway 再デプロイで 4.1/4.3/4.4/4.5 が有効化。バックフィル（4.4）はデプロイ後にローカルから1回実行。
- 7/31・8/31 の自動チェック（既存 routine）はそのまま。p_kai/mc_venue_all の forward は次回チェック時に読む。
- **実弾投入の判断**（P改 or 候補場拡張後の P）: forward を見て岩下さんが行う。
