# 広域調査: boatrace ML 文献 / GitHub / 実践事例まとめ

2026-05-19 実施。5 連敗の閉塞を脱するための外部知見調査。
我々が「井の中の蛙」になっていないかを文献・実装例・プロ実態から検証。

## 1. 我々の状況の客観評価 (まず現実認識)

### プロ実態
- 公営競技プロギャンブラー: **回収率 110% 程度**、**年間収益 100 万円程度** が現実
  ([HBOL 2018](https://hbol.jp/166579/))
- 商用 AI 予想サイト主張の「ROI 130%」は誇張、実運用での持続性証明なし
- ボートレース市場規模 2024 年 2.5 兆円、払戻率 75% (= ハウスエッジ 25%)

### 個人開発 ML 実績
| 事例 | 手法 | ROI | 出典 |
|---|---|---|---|
| LightGBM 多クラス分類 | 単勝 | 80-141% (test) | [KOJI BLOG](https://koji30learn.com/python-boatrace-lightgbm/) |
| LambdaRank (3連単) | XGB ランキング | hit率 8.15% | [Qiita](https://qiita.com/NGOhiroshi/items/d841795e395791cf847c) |
| 山崎氏 NN | 単一会場 (平和島) 特化 | **100% 超** (¥1.3M 利益) | [note](https://note.com/yu_yamasaki/n/n4e4a06e09ae4) |
| 全会場 ML | 3連単 12k races | 98% (Loss) | [複数事例] |
| 個人 NN | 3連単 | **109 万円 Loss** | (失敗例) |

### 我々の position
- production P6 +37% (18 日) → **健闘の部類** (平均的 ML は 100% 未満で苦戦)
- 5 連敗パターンは個人開発者の通則、edge thin 市場の典型現象
- 「あらゆる ML 修正で edge 微増 < variance」は誰もが直面

## 2. 試していない promising 方向 (insights)

### 🟢 A. 単一会場特化モデル ⭐⭐⭐

**最も再現性高い手法**。複数の実例で 100% 超達成。

- 山崎氏: 全会場モデルでは 100% 未満 → **平和島専用に絞った瞬間 100% 超**
- 別事例: 常滑のみ 15,108 races 学習で精度向上 ([eejanaica](https://eejanaica.com/kyotei-prediction/))
- 理由: 会場ごとの水面 / 風 / 1号艇率が大きく異なる
  - 芦屋 (24): 1号艇 1着率 **65.4%**
  - 平和島 (4): **44.5%**
  - 戸田 (2): **43.6%** ← 我々の data 34.39% と整合
  ([kcbn](https://kcbn.jp/distorted-odds/))

**我々の現状**: TOP10 venue 共有モデル → 場固有 signal が薄まる
**改修案**: 戸田 / 平和島など低 1号艇率会場で venue-specific NN

### 🟢 B. オッズ動向予測モデル ⭐⭐

CLV<0 問題の根本に直撃。

- blue_mihanada 氏: LightGBM で「現時点 odds → 締切時 odds」を予測 ([note](https://note.com/blue_mihanada/n/n3282239b812d))
- 発見: 「**強い picks ほどオッズ下がる**」「**高オッズは締切に向けて下がる**」
- 我々の CLV<0 (オッズ上がる) = 市場が picks を「弱い」と評価
- → オッズ動向 NN で「下がりそうな picks」のみ絞れば CLV>0 (本来の sharp money 整合) に転換可

**競艇 AI バズーカーの実践**: 締切 30 分前展示 + 15 分前オッズで「妙味」picks 発掘
([kyotei-ai](https://kyotei-ai.com/article/22))

### 🟡 C. 3連複 / 拡連複への展開 ⭐

mc3 系は 3連単のみ、他 bet 種は未試験。

- 7 種類の bet (単勝/複勝/2連単/2連複/3連単/3連複/拡連複) で hit 率/配当が全く異なる
- 3連単 平均 payout ¥7,360、3連複 ~¥1,500 (1/5)、hit 率 6x
- 「中穴狙い」で hit 率不安定な 3連単より、3連複の方が **variance 低・実利確保**しやすい

### 🟡 D. LambdaRank / 学習ランキング

XGBRanker / LightGBM lambdarank objective を使ったランキング学習。

- 現在の softmax は「1着確率」を独立予測
- LambdaRank は「相対順序」を最適化 → 3連単の構造と整合
- 実例: hit率 8.15% (我々と同等)、**過大期待しない**

### 🔴 E. RL / Transformer 系

academic では autonomous racing / 株式 trading に応用例あるが、**boatrace では未確立**。実装難度高、ROI 不明。

## 3. CLV の本来の意義と我々の特殊性

文献での通則:
> "Bettors who consistently achieve positive CLV see ROI 2-3 times higher than those who only track win rates"
> ([Sharp Football Analysis](https://www.sharpfootballanalysis.com/sportsbook/clv-betting/))

**我々の data は逆**: CLV<0 が ROI +27.6%、CLV>0 が -5.0%

**プロ解釈**:
- 我々の bet は EV+ で中穴狙い
- 市場は通常これを過小評価したまま (CLV<0、オッズ上昇)
- 当たれば高配当キャプチャ → ROI 正
- これは中穴狙いの典型的 pattern、「悪い signal」ではない

ただし「中穴狙い + 多会場 + 多戦略」の組み合わせは:
- それぞれが正しい選択肢
- だが edge thin、variance 大
- 数十 race の hit 偶然で全体が黒/赤に振れる

## 4. Kelly Criterion / Bankroll Management (一般則)

- **Full Kelly** はバラツキ大、professional はほぼ全員 **fractional Kelly** 使用 (1/4 〜 1/2)
- 我々: kelly_fraction 0.0625 (1/16) = ultra conservative
- これは妥当だが、edge 大きい 帯 (例: 単一会場特化で得た強 signal) では拡大検討余地あり
- ([OddsShopper](https://www.oddsshopper.com/articles/betting-101/kelly-criterion-in-sports-betting-beat-the-books-with-bet-sizing-y10))

## 5. 我々のアプローチの強み / 弱み

### 強み
- multi-strategy portfolio (mc/mc2/mc3 系) で risk 分散
- 11 項目 std 補正 (compute_ratings_early) は heuristic だが妥当 (展示 / クラス / 風波)
- QMC (Sobol) は MC より収束速度速い、計算効率は高水準
- DB infrastructure 整備、forward 検証可能
- CLV / closing_odds 既に取得済み (多くの個人開発者は未実装)
- shadow 並走機構あり

### 弱み (調査で判明)
- ❌ **単一会場特化なし** (TOP10 共有 → 場固有 signal 薄まる)
- ❌ **オッズ動向 NN なし** (CLV 予測 model 不在)
- ❌ **bet 種多様化なし** (3連単のみ)
- ❌ **LambdaRank / ペアワイズ未試** (ただし期待値小)
- ⚠️ **forward 期間短い** (cache 5 週間のみ、長期は production data 蓄積待ち)

## 6. 推奨される次の試行 (優先順)

### 案 X: 単一会場特化モデル (戸田 single venue) ⭐⭐⭐
- 期待: 山崎氏事例で 100%超達成
- 工数: 大 (1-2 週間)
- リスク: 既存 V10 とは別 NN 必要、データ ~250-400 races/venue で訓練十分か
- 第一歩: 戸田 (1号艇弱、オッズ歪み有名) のみで prototype

### 案 Y: オッズ動向予測 NN (CLV proxy) ⭐⭐
- 期待: CLV>0 picks を real-time 識別、本来の sharp money 整合
- 工数: 中 (1 週間)
- リスク: 過去 odds 時系列 data の整備、リアルタイム動向取得仕組み
- 第一歩: 既存 bets の odds vs closing_odds で「いつ買うべきか」を learn

### 案 Z: 3連複戦略 ⭐
- 期待: hit率 30%+、variance 低、実利確保
- 工数: 小 (3-4 日)
- リスク: 払戻 1/5 で ROI 改善見込みは限定的
- 第一歩: cache 5828 races で「QMC top-3 順序無視 hit 率」算出 → 3連複 simulation

### 案 STAY: 現状維持
- production P6 (mc3_venue_focus_r4) 継続
- 1-2 ヶ月 data 蓄積待ち
- 過去 5 連敗パターン回避
- 機会損失だが安全

## 7. プロ視点の結論 (推奨)

率直に言うと:

1. **「ML を弄り続けて edge 出す」アプローチは限界に達している** (5 連敗、個人開発者の通則と整合)
2. **「単一会場特化」は最も再現性高い未試行 path** (山崎氏含む複数事例で 100% 超達成)
3. **「オッズ動向予測」は我々の CLV 問題の根本対処** (中規模 R&D で投資価値あり)
4. **「3連複」は安定運用の補助手段** (variance 低、メイン戦略にはならない)

最有力: **案 X (戸田特化モデル) を 2 週間 prototype + 案 Y (オッズ動向 NN) を並行**
理由:
- 案 X は **完全に未試行 + 再現性高い**
- 案 Y は **CLV 問題の根本対処**
- 両者は独立 (戸田特化モデル + オッズ動向で picks 絞り)
- 工数合計 3-4 週間
- 撤退ライン: 戸田特化 ROI が backtest で 110% 超えなければ凍結

## 8. 参考リンク (主要)

### 個人開発事例
- [Pythonでボートレース予想AIを作ってみた](https://satolog.org/boat-gbm-tk/)
- [競艇で機械学習して回収率を出したら想像以上だった (KOJI BLOG)](https://koji30learn.com/python-boatrace-lightgbm/)
- [ニューラルネットワークでボートレース予想AI 100% 超 (山崎氏)](https://note.com/yu_yamasaki/n/n4e4a06e09ae4)
- [ボートレースの3連単をランキング学習で予測 (Qiita)](https://qiita.com/NGOhiroshi/items/d841795e395791cf847c)
- [機械学習でボートレース予想してみた (ひつじ)](https://hitsuzi-boatrace.com/how-to-boatrace-machine-learning-1/)

### オッズ / 市場分析
- [ボートレースのオッズについての研究 (機械学習) (blue_mihanada)](https://note.com/blue_mihanada/n/n3282239b812d)
- [競艇 AI バズーカー 直前情報活用](https://kyotei-ai.com/article/22)
- [オッズの歪みとは？高配当狙いの買い方 (kcbn)](https://kcbn.jp/distorted-odds/)

### 競馬 ML (近似領域、参考)
- [競馬予想で始める機械学習 (Zenn 完全版)](https://zenn.dev/dijzpeb/books/848d4d8e47001193f3fb)
- [機械学習で競馬の回収率 140% 超を達成 (Qiita)](https://qiita.com/umaro_ai/items/d1e0b61f90098ee7fbcb)

### 一般理論
- [Bradley-Terry Models for Supervised Learning (arXiv)](https://arxiv.org/pdf/1701.08055)
- [Closing Line Value Explained (Sharp Football Analysis)](https://www.sharpfootballanalysis.com/sportsbook/clv-betting/)
- [Kelly Criterion in Sports Betting (OddsShopper)](https://www.oddsshopper.com/articles/betting-101/kelly-criterion-in-sports-betting-beat-the-books-with-bet-sizing-y10)

### 公営競技 実態
- [公営ギャンブルで通年プラスにできる人は何をやっているのか (HBOL)](https://hbol.jp/166579/)
- [競艇 1号艇の有利度 (kyoutei-navi)](https://kyoutei-navi.com/beginner/boat-no1/)
