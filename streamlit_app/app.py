"""ボートレース予想AIシステム メインUI"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from streamlit_app.components.db_utils import (
    get_db_connection,
    get_recent_predictions,
    get_performance_stats,
)
from streamlit_app.components.mobile_css import inject_mobile_css
from src.database import init_database

st.set_page_config(
    page_title="ボートレース予想AI",
    page_icon="🚤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_mobile_css()

try:
    init_database()
except Exception:
    pass

# --- 場名マッピング ---
VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}

STRATEGY_NAMES = {
    'kelly_strict': 'ケリー厳選',
    'top_prob_fixed': '確率上位固定',
}


def _venue_name(venue_id):
    return VENUE_NAMES.get(venue_id, f'場{venue_id}')


def _strategy_name(strategy_type):
    return STRATEGY_NAMES.get(strategy_type, strategy_type)


# --- サイドバー ---
with st.sidebar:
    st.title("🚤 ボートレース予想AI")
    st.divider()

    st.subheader("システム状態")
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as cnt FROM races WHERE race_date = CURRENT_DATE")
            today_races = cur.fetchone()['cnt']
            cur.execute(
                "SELECT COUNT(*) as cnt FROM predictions "
                "WHERE created_at::date = CURRENT_DATE"
            )
            today_preds = cur.fetchone()['cnt']
        st.success("DB接続: OK")
        st.metric("本日のレース", today_races)
        st.metric("本日の予測", today_preds)
    except Exception:
        st.error("DB接続: エラー")
        today_races = 0
        today_preds = 0

    st.divider()
    st.subheader("資金状況")
    try:
        stats = get_performance_stats(days=9999)
        total_amount = sum(s['total_amount'] or 0 for s in stats)
        total_payout = sum(s['total_payout'] or 0 for s in stats)
        profit = total_payout - total_amount
        roi = (total_payout / total_amount * 100) if total_amount > 0 else 0
        st.metric("総投資額", f"¥{total_amount:,}")
        st.metric("総回収額", f"¥{total_payout:,}")
        st.metric("収支", f"¥{profit:,}", delta=f"{roi:.1f}%")
    except Exception:
        st.info("データなし")


# --- メインコンテンツ ---
st.title("🚤 ボートレース予想AIダッシュボード")

tab1, tab2, tab3 = st.tabs(["📊 今日の予想", "💰 回収率", "🔄 最新予測"])

with tab1:
    st.subheader("今日の予想一覧")
    try:
        predictions = get_recent_predictions(limit=20)
        if predictions:
            df = pd.DataFrame(predictions)
            # 日本語カラムに変換
            col_rename = {
                'venue_id': '会場',
                'race_number': 'レース',
                'strategy_type': '戦略',
                'prediction_time': '予測時刻',
                'race_date': '日付',
                'probabilities_1st': '1着確率',
                'probabilities_2nd': '2着確率',
                'probabilities_3rd': '3着確率',
                'recommended_bets': '推奨買い目',
                'model_version': 'モデル版',
                'created_at': '作成日時',
                'race_id': 'レースID',
                'id': 'ID',
            }
            # 会場名を日本語に
            if 'venue_id' in df.columns:
                df['venue_id'] = df['venue_id'].map(
                    lambda v: _venue_name(v)
                )
            # 戦略名を日本語に
            if 'strategy_type' in df.columns:
                df['strategy_type'] = df['strategy_type'].map(
                    lambda s: _strategy_name(s)
                )
            display_cols = [
                c for c in ['venue_id', 'race_number', 'strategy_type',
                             'prediction_time']
                if c in df.columns
            ]
            if display_cols:
                display_df = df[display_cols].rename(columns=col_rename)
                st.dataframe(display_df, use_container_width=True)
            else:
                st.dataframe(
                    df.rename(columns=col_rename), use_container_width=True
                )
        else:
            st.info("まだ予想がありません。システムがレースデータを収集中です。")
    except Exception as e:
        st.error(f"データ取得エラー: {e}")

with tab2:
    st.subheader("回収率推移")
    try:
        stats = get_performance_stats(days=30)
        if stats:
            col1, col2 = st.columns(2)
            for s in stats:
                name = _strategy_name(s['strategy_type'])
                with col1 if s['strategy_type'] == 'kelly_strict' else col2:
                    st.markdown(f"**{name}**")
                    st.metric("ベット数", s['total_bets'])
                    st.metric("回収率", f"{s['roi']:.1f}%")
                    st.metric("的中率",
                              f"{s['wins'] / s['total_bets'] * 100:.1f}%"
                              if s['total_bets'] > 0 else "0%")
        else:
            st.info("パフォーマンスデータがまだありません。")
    except Exception as e:
        st.error(f"統計取得エラー: {e}")

with tab3:
    st.subheader("最新予測結果")
    try:
        recent = get_recent_predictions(limit=5)
        if recent:
            for pred in recent:
                venue = _venue_name(pred['venue_id'])
                strategy = _strategy_name(pred['strategy_type'])
                with st.expander(
                    f"{venue} {pred['race_number']}R "
                    f"({strategy})"
                ):
                    st.json({
                        '1着確率': pred.get('probabilities_1st'),
                        '2着確率': pred.get('probabilities_2nd'),
                        '3着確率': pred.get('probabilities_3rd'),
                        '推奨買い目': pred.get('recommended_bets'),
                    })
        else:
            st.info("予測結果がまだありません。")
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
