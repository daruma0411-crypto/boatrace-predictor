# 🚤 ボートレース予想AIシステム

PyTorch × ケリー基準による24時間稼働の自動予想システム

## 特徴

- **マルチタスク学習**: 1着/2着/3着を同時予測
- **条件付き確率**: 物理的に正しい3連単確率計算
- **ハーフ・ケリー基準**: 安全な資金管理
- **動的スケジューリング**: レース遅延に自動対応
- **A/Bテスト機能**: 2つの戦略を並行検証
- **24/7稼働**: Railway対応

## セットアップ

\\\ash
pip install -r requirements.txt
python -c "from src.database import init_database; init_database()"
\\\

## Railwayデプロイ

1. PostgreSQL追加
2. 環境変数設定: \TZ=Asia/Tokyo\
3. Git push

## ライセンス

MIT License
