"""A/Bテスト刷新用DBリセット: bets と predictions を全削除

使い方: DATABASE_URL=xxx python scripts/reset_ab_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import get_db_connection

def reset():
    with get_db_connection() as conn:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) as cnt FROM bets")
        bets_count = cur.fetchone()['cnt']

        cur.execute("SELECT COUNT(*) as cnt FROM predictions")
        preds_count = cur.fetchone()['cnt']

        print(f"削除対象: bets={bets_count}件, predictions={preds_count}件")

        if bets_count == 0 and preds_count == 0:
            print("既に空です。リセット不要。")
            return

        confirm = input("本当に削除しますか？ (yes/no): ")
        if confirm != 'yes':
            print("中止しました。")
            return

        # bets が predictions を参照しているので先に削除
        cur.execute("DELETE FROM bets")
        cur.execute("DELETE FROM predictions")

        print(f"リセット完了: bets {bets_count}件, predictions {preds_count}件 を削除")

        # 確認
        cur.execute("SELECT COUNT(*) as cnt FROM bets")
        print(f"bets 残: {cur.fetchone()['cnt']}件")
        cur.execute("SELECT COUNT(*) as cnt FROM predictions")
        print(f"predictions 残: {cur.fetchone()['cnt']}件")


if __name__ == '__main__':
    reset()
