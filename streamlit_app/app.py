"""ボートレース予想AIシステム メインUI"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import date, timedelta
from streamlit_app.components.db_utils import (
    get_db_connection,
    get_recent_predictions,
    get_performance_stats,
    get_today_bets,
    get_today_venues,
    get_strategy_summary,
    get_daily_stats_by_period,
)
from streamlit_app.components.mobile_css import inject_mobile_css
from src.database import init_database, get_current_bankroll

st.set_page_config(
    page_title="ボートレース予想AI",
    page_icon="🚤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_mobile_css()

if 'db_initialized' not in st.session_state:
    try:
        init_database()
    except Exception:
        pass
    st.session_state.db_initialized = True

# --- キャッシュ付きDB取得 ---
@st.cache_data(ttl=60, show_spinner=False)
def _cached_predictions(limit):
    return get_recent_predictions(limit=limit)

@st.cache_data(ttl=60, show_spinner=False)
def _cached_today_bets():
    return get_today_bets()

@st.cache_data(ttl=60, show_spinner=False)
def _cached_today_venues():
    return get_today_venues()

@st.cache_data(ttl=60, show_spinner=False)
def _cached_today_counts():
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as cnt FROM races WHERE race_date = CURRENT_DATE")
            races = cur.fetchone()['cnt']
            cur.execute(
                "SELECT COUNT(*) as cnt FROM predictions "
                "WHERE created_at::date = CURRENT_DATE"
            )
            preds = cur.fetchone()['cnt']
        return races, preds, True
    except Exception:
        return 0, 0, False

@st.cache_data(ttl=60, show_spinner=False)
def _cached_strategy_summary(start_date, end_date):
    return get_strategy_summary(start_date, end_date)

@st.cache_data(ttl=60, show_spinner=False)
def _cached_bankroll(strategy_type):
    try:
        profit = get_current_bankroll(strategy_type=strategy_type)
        return 200000 + profit
    except Exception:
        return 200000

# --- 場名マッピング ---
VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}

STRATEGY_NAMES = {
    'conservative': 'A: 保守的 (1/8ケリー)',
    'standard': 'B: 普通 (1/4ケリー)',
    'divergence': 'C: 市場乖離',
    'high_confidence': 'D: 高確信',
    'ensemble': 'E: 合議制',
    'div_confidence': 'F: 乖離+確信',
}

STRATEGY_ORDER = [
    'conservative', 'standard', 'divergence',
    'high_confidence', 'ensemble', 'div_confidence',
]


def _venue_name(venue_id):
    return VENUE_NAMES.get(venue_id, f'場{venue_id}')


def _strategy_name(strategy_type):
    return STRATEGY_NAMES.get(strategy_type, strategy_type)


# --- サイドバー ---
with st.sidebar:
    st.title("🚤 ボートレース予想AI")
    st.divider()

    st.subheader("システム状態")
    today_races, today_preds, db_ok = _cached_today_counts()
    if db_ok:
        st.success("DB接続: OK")
    else:
        st.error("DB接続: エラー")
    st.metric("本日のレース", today_races)
    st.metric("本日の予測", today_preds)


# --- メインコンテンツ ---
st.title("🚤 ボートレース予想AIダッシュボード")

# --- 期間セレクター ---
today = date.today()
period = st.radio(
    "分析期間",
    ["デイリー", "1W", "1M", "1Y", "カスタム"],
    horizontal=True,
    key="period_selector",
)

if period == "デイリー":
    start_date = today
    end_date = today
elif period == "1W":
    start_date = today - timedelta(days=7)
    end_date = today
elif period == "1M":
    start_date = today - timedelta(days=30)
    end_date = today
elif period == "1Y":
    start_date = today - timedelta(days=365)
    end_date = today
else:
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("開始日", today - timedelta(days=30))
    with col_d2:
        end_date = st.date_input("終了日", today)

# --- 6戦略サマリーカード ---
st.subheader("戦略別サマリー")
try:
    summary_data = _cached_strategy_summary(str(start_date), str(end_date))
    summary_dict = {s['strategy_type']: s for s in summary_data} if summary_data else {}

    # 3列×2行
    row1 = st.columns(3)
    row2 = st.columns(3)
    all_cols = row1 + row2

    for idx, strategy_key in enumerate(STRATEGY_ORDER):
        with all_cols[idx]:
            label = _strategy_name(strategy_key)
            s = summary_dict.get(strategy_key)
            bankroll = _cached_bankroll(strategy_key)

            st.markdown(f"**{label}**")
            if s and s['total_bets'] > 0:
                total_amount = s['total_amount'] or 0
                total_payout = s['total_payout'] or 0
                roi = s['roi'] or 0
                wins = s['wins'] or 0
                total_bets = s['total_bets']
                total_races = s['total_races'] or 0
                win_rate = wins / total_bets * 100 if total_bets > 0 else 0

                st.metric("残金", f"¥{bankroll:,.0f}")
                st.metric("投資額", f"¥{total_amount:,}")
                st.metric("レース数", total_races)
                roi_delta = f"{'+'if roi >= 100 else ''}{roi - 100:.1f}%"
                st.metric("ROI", f"{roi:.1f}%", delta=roi_delta)
                st.metric("的中率", f"{win_rate:.1f}%")
            else:
                st.metric("残金", f"¥{bankroll:,.0f}")
                st.info("データなし")

except Exception as e:
    st.error(f"サマリー取得エラー: {e}")

st.divider()

# --- タブ ---
tab1, tab2, tab3 = st.tabs(["📊 本日の買い目", "💰 期間別推移", "🔄 予測詳細"])

# ========== タブ1: 本日の買い目（会場フィルタ付き） ==========
with tab1:
    try:
        all_bets = _cached_today_bets()
        venues = _cached_today_venues()

        if not all_bets:
            st.info("本日の買い目はまだありません。レース締切10分前に自動生成されます。")
        else:
            # 会場フィルタ
            venue_options = ['全会場'] + [
                _venue_name(v) for v in venues
            ]
            selected_venue = st.selectbox(
                '会場を選択', venue_options, key='venue_filter'
            )

            df = pd.DataFrame(all_bets)
            df['会場'] = df['venue_id'].map(_venue_name)
            df['戦略'] = df['strategy_type'].map(_strategy_name)

            # フィルタ適用
            if selected_venue != '全会場':
                df = df[df['会場'] == selected_venue]

            if df.empty:
                st.info(f"{selected_venue}の買い目はまだありません。")
            else:
                # 会場×レース番号でグループ表示
                for (venue_id, race_num), group in df.groupby(
                    ['venue_id', 'race_number']
                ):
                    venue = _venue_name(venue_id)
                    total_amount = group['amount'].sum()

                    with st.expander(
                        f"{venue} {race_num}R  "
                        f"（{len(group)}点 / 計 ¥{total_amount:,}）",
                        expanded=True,
                    ):
                        for strategy, sgroup in group.groupby('strategy_type'):
                            st.markdown(f"**{_strategy_name(strategy)}**")
                            display = sgroup[[
                                'combination', 'amount', 'odds',
                                'expected_value',
                            ]].copy()
                            display.columns = [
                                '組み合わせ', '金額', 'オッズ', '期待値',
                            ]
                            display['金額'] = display['金額'].map(
                                lambda x: f'¥{x:,}'
                            )
                            display['オッズ'] = display['オッズ'].map(
                                lambda x: f'{x:.1f}' if x else '-'
                            )
                            display['期待値'] = display['期待値'].map(
                                lambda x: f'{x:.2f}' if x else '-'
                            )
                            st.dataframe(
                                display.reset_index(drop=True),
                                use_container_width=True,
                                hide_index=True,
                            )

            # サマリー
            st.divider()
            summary = pd.DataFrame(all_bets)
            summary['会場'] = summary['venue_id'].map(_venue_name)
            venue_summary = summary.groupby('会場').agg(
                点数=('combination', 'count'),
                投資額=('amount', 'sum'),
            ).reset_index()
            venue_summary['投資額'] = venue_summary['投資額'].map(
                lambda x: f'¥{x:,}'
            )
            st.markdown("**会場別サマリー**")
            st.dataframe(
                venue_summary, use_container_width=True, hide_index=True
            )

    except Exception as e:
        st.error(f"データ取得エラー: {e}")


# ========== タブ2: 期間別推移 ==========
with tab2:
    st.subheader("日別収支推移")
    try:
        daily = get_daily_stats_by_period(str(start_date), str(end_date))
        if daily:
            import plotly.express as px

            df_daily = pd.DataFrame(daily)
            df_daily['profit'] = (
                df_daily['total_payout'].fillna(0) -
                df_daily['total_amount'].fillna(0)
            )
            df_daily['戦略'] = df_daily['strategy_type'].map(_strategy_name)

            fig = px.line(
                df_daily, x='race_date', y='profit',
                color='戦略',
                title=f'日別損益推移 ({start_date} ~ {end_date})',
                labels={'race_date': '日付', 'profit': '損益 (円)'},
            )
            fig.update_layout(
                margin=dict(l=30, r=20, t=50, b=30),
                font=dict(size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True,
                           config={'responsive': True, 'displayModeBar': False})

            # 累積損益
            for strategy in df_daily['strategy_type'].unique():
                mask = df_daily['strategy_type'] == strategy
                df_daily.loc[mask, 'cumulative_profit'] = (
                    df_daily.loc[mask, 'profit'].cumsum()
                )

            fig_cum = px.line(
                df_daily, x='race_date', y='cumulative_profit',
                color='戦略',
                title='累積損益推移',
                labels={'race_date': '日付', 'cumulative_profit': '累積損益 (円)'},
            )
            fig_cum.update_layout(
                margin=dict(l=30, r=20, t=50, b=30),
                font=dict(size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_cum, use_container_width=True,
                           config={'responsive': True, 'displayModeBar': False})
        else:
            st.info("この期間のデータがまだありません。")
    except Exception as e:
        st.error(f"日別データ取得エラー: {e}")


# ========== タブ3: 予測詳細 ==========
with tab3:
    st.subheader("最新予測結果（確率分布）")
    try:
        recent = _cached_predictions(limit=10)
        if recent:
            for pred in recent:
                venue = _venue_name(pred['venue_id'])
                strategy = _strategy_name(pred['strategy_type'])
                with st.expander(
                    f"{venue} {pred['race_number']}R ({strategy})"
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
