"""環境チェックスクリプト: DATABASE_URL、TZ、依存関係確認"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def check_environment():
    """環境設定をチェック"""
    errors = []
    warnings = []

    # DATABASE_URL
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        errors.append("DATABASE_URL が設定されていません")
    else:
        if db_url.startswith('postgres://'):
            warnings.append(
                "DATABASE_URL が postgres:// で始まっています。"
                "自動的に postgresql:// に変換されます。"
            )
        print(f"[OK] DATABASE_URL: {db_url[:30]}...")

    # TZ
    tz = os.environ.get('TZ')
    if tz != 'Asia/Tokyo':
        warnings.append(f"TZ が 'Asia/Tokyo' ではありません: {tz}")
    else:
        print("[OK] TZ: Asia/Tokyo")

    # Python バージョン
    py_version = sys.version
    print(f"[OK] Python: {py_version}")

    # 依存パッケージ
    packages = [
        ('streamlit', 'streamlit'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('torch', 'torch'),
        ('sklearn', 'scikit-learn'),
        ('plotly', 'plotly'),
        ('psycopg2', 'psycopg2-binary'),
        ('pyjpboatrace', 'pyjpboatrace'),
        ('bs4', 'beautifulsoup4'),
        ('requests', 'requests'),
        ('schedule', 'schedule'),
        ('dateutil', 'python-dateutil'),
        ('pytz', 'pytz'),
    ]

    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"[OK] {package_name}")
        except ImportError:
            errors.append(f"{package_name} がインストールされていません")

    # DB接続テスト
    if db_url:
        try:
            from src.database import get_db_connection
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1")
            print("[OK] DB接続テスト成功")
        except Exception as e:
            errors.append(f"DB接続失敗: {e}")

    # 結果
    print("\n" + "=" * 50)
    if warnings:
        print("\n[WARN] 警告:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\n[ERROR] エラー:")
        for e in errors:
            print(f"  - {e}")
        print(f"\n環境チェック: {len(errors)}件のエラー")
        return False
    else:
        print("\n[PASS] 環境チェック: すべてOK")
        return True


if __name__ == '__main__':
    success = check_environment()
    sys.exit(0 if success else 1)
