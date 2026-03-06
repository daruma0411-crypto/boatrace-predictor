"""ボートレース予想AIシステム メインUI"""
import streamlit as st
import pandas as pd
from streamlit_app.components.db_utils import (
    get_db_connection,
    get_recent_predictions,
    get_performance_stats,
)

st.set_page_config(
    page_title="ボートレース予想AI",
    page_icon="🚤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# モバイル対応CSS
st.markdown("""
<style>
    @media (max-width: 768px) {
        .stColumns > div { min-width: 100% !important; }
        .main .block-container { padding: 1rem; }
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


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
            display_cols = [
                c for c in ['venue_id', 'race_number', 'strategy_type',
                             'prediction_time']
                if c in df.columns
            ]
            if display_cols:
                st.dataframe(df[display_cols], use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        else:
            st.info("まだ予想がありません。システムがレースデータを収集中です。")
    except Exception as e:
        st.error(f"データ取得エラー: {e}")

with tab2:
    st.subheader("回収率推移")
    try:
        stats = get_performance_stats(days=30)
        if stats:
            df_stats = pd.DataFrame(stats)
            col1, col2 = st.columns(2)
            for s in stats:
                with col1 if s['strategy_type'] == 'kelly_strict' else col2:
                    st.markdown(f"**戦略: {s['strategy_type']}**")
                    st.metric("ベット数", s['total_bets'])
                    st.metric("ROI", f"{s['roi']:.1f}%")
                    st.metric("勝率",
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
                with st.expander(
                    f"場{pred['venue_id']} R{pred['race_number']} "
                    f"({pred['strategy_type']})"
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
