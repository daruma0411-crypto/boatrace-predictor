"""パフォーマンスページ: 場別・券種別・時系列・戦略A/B比較"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_app.components.db_utils import (
    get_performance_stats,
    get_venue_stats,
    get_daily_stats,
)

st.set_page_config(page_title="パフォーマンス", page_icon="📈", layout="wide")
st.title("📈 パフォーマンス分析")

VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}

days = st.slider("分析期間（日数）", min_value=7, max_value=90, value=30)

# --- 戦略別A/Bテスト比較 ---
st.subheader("戦略別 A/Bテスト比較")
try:
    stats = get_performance_stats(days=days)
    if stats:
        df = pd.DataFrame(stats)
        col1, col2 = st.columns(2)

        for i, row in df.iterrows():
            target = col1 if row['strategy_type'] == 'kelly_strict' else col2
            with target:
                strategy_label = (
                    "戦略A (ケリー基準)" if row['strategy_type'] == 'kelly_strict'
                    else "戦略B (確率優先)"
                )
                st.markdown(f"### {strategy_label}")
                c1, c2, c3 = st.columns(3)
                c1.metric("ベット数", row['total_bets'])
                c2.metric("ROI", f"{row['roi']:.1f}%")
                win_rate = (
                    row['wins'] / row['total_bets'] * 100
                    if row['total_bets'] > 0 else 0
                )
                c3.metric("的中率", f"{win_rate:.1f}%")

        fig = go.Figure(data=[
            go.Bar(
                name=row['strategy_type'],
                x=['ベット数', '投資額(千円)', '回収額(千円)', 'ROI(%)'],
                y=[
                    row['total_bets'],
                    (row['total_amount'] or 0) / 1000,
                    (row['total_payout'] or 0) / 1000,
                    row['roi'],
                ],
            )
            for _, row in df.iterrows()
        ])
        fig.update_layout(barmode='group', title='戦略比較')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("パフォーマンスデータがまだありません。")
except Exception as e:
    st.error(f"統計取得エラー: {e}")

# --- 時系列推移 ---
st.subheader("日別収支推移")
try:
    daily = get_daily_stats(days=days)
    if daily:
        df_daily = pd.DataFrame(daily)
        df_daily['profit'] = (
            df_daily['total_payout'].fillna(0) -
            df_daily['total_amount'].fillna(0)
        )

        fig = px.line(
            df_daily, x='race_date', y='profit',
            color='strategy_type',
            title='日別損益推移',
            labels={'race_date': '日付', 'profit': '損益 (円)',
                    'strategy_type': '戦略'},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        fig_roi = px.line(
            df_daily, x='race_date', y='roi',
            color='strategy_type',
            title='日別ROI推移',
            labels={'race_date': '日付', 'roi': 'ROI (%)',
                    'strategy_type': '戦略'},
        )
        fig_roi.add_hline(y=100, line_dash="dash", line_color="gray",
                          annotation_text="損益分岐点")
        st.plotly_chart(fig_roi, use_container_width=True)
    else:
        st.info("日別データがまだありません。")
except Exception as e:
    st.error(f"日別データ取得エラー: {e}")

# --- 場別パフォーマンス ---
st.subheader("場別パフォーマンス")
try:
    venue_data = get_venue_stats()
    if venue_data:
        df_venue = pd.DataFrame(venue_data)
        df_venue['venue_name'] = df_venue['venue_id'].map(
            lambda x: VENUE_NAMES.get(x, f'場{x}')
        )

        fig = px.bar(
            df_venue, x='venue_name', y='roi',
            color='strategy_type',
            barmode='group',
            title='場別ROI',
            labels={'venue_name': '競艇場', 'roi': 'ROI (%)',
                    'strategy_type': '戦略'},
        )
        fig.add_hline(y=100, line_dash="dash", line_color="red",
                      annotation_text="損益分岐点")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("場別データがまだありません。")
except Exception as e:
    st.error(f"場別データ取得エラー: {e}")
