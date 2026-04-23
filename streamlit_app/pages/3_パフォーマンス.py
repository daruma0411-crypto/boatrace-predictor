"""パフォーマンスページ: 6戦略比較・時系列・場別分析"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_app.components.db_utils import (
    get_performance_stats,
    get_venue_stats,
    get_daily_stats,
)
from streamlit_app.components.mobile_css import inject_mobile_css

st.set_page_config(page_title="パフォーマンス", page_icon="\U0001f4c8", layout="wide",
                   initial_sidebar_state="collapsed")
inject_mobile_css()
st.title("\U0001f4c8 パフォーマンス分析")

VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}

STRATEGY_NAMES = {
    'mc_quarter_kelly': 'L: MC v1 基準',
    'mc_early_race': 'O: MC v1 序盤',
    'mc_venue_focus': 'P: MC v1 得意場',
    'mc_high_ev': 'Q: MC v1 高EV',
    'mc_are_v2': 'R: MC v1 ModelB',
    'mc2_quarter_kelly': 'L2: MC v2 基準',
    'mc2_early_race': 'O2: MC v2 序盤',
    'mc2_venue_focus': 'P2: MC v2 得意場',
    'mc2_high_ev': 'Q2: MC v2 高EV',
    'mc2_are_v2': 'R2: MC v2 ModelB',
    'mc_early_race_filtered': 'Of: O+Miss分析フィルタ',
    'mc_early_race_v10_2_lr_hi': 'V10.2-A: fine-tune lr_hi',
    'mc_early_race_v10_2_gamma3': 'V10.2-B: fine-tune gamma3',
}

STRATEGY_COLORS = {
    'mc_quarter_kelly': '#1f77b4',
    'mc_early_race': '#ff7f0e',
    'mc_venue_focus': '#2ca02c',
    'mc_high_ev': '#d62728',
    'mc_are_v2': '#9467bd',
    'mc2_quarter_kelly': '#17becf',
    'mc2_early_race': '#bcbd22',
    'mc2_venue_focus': '#e377c2',
    'mc2_high_ev': '#8c564b',
    'mc2_are_v2': '#7f7f7f',
    'mc_early_race_filtered': '#ff6b35',
    'mc_early_race_v10_2_lr_hi': '#ffd700',
    'mc_early_race_v10_2_gamma3': '#ff4500',
}

STRATEGY_ORDER = [
    'mc_quarter_kelly', 'mc_early_race', 'mc_venue_focus',
    'mc_high_ev', 'mc_are_v2',
    'mc2_quarter_kelly', 'mc2_early_race', 'mc2_venue_focus',
    'mc2_high_ev', 'mc2_are_v2',
    'mc_early_race_filtered',
    'mc_early_race_v10_2_lr_hi',
    'mc_early_race_v10_2_gamma3',
]

# --- キャッシュ付きDB取得 (TTL=300秒) ---
@st.cache_data(ttl=300, show_spinner=False)
def _cached_performance_stats(days):
    return get_performance_stats(days=days)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_daily_stats(days):
    return get_daily_stats(days=days)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_venue_stats():
    return get_venue_stats()


# --- 全体を1つのfragmentで囲む（スライダー操作時に部分再実行）---
@st.fragment
def performance_fragment():
    days = st.select_slider(
        "分析期間（日数）",
        options=[7, 14, 30, 60, 90],
        value=30,
    )

    # --- 戦略別比較 ---
    st.subheader("戦略別比較 (A\u301cI)")
    try:
        stats = _cached_performance_stats(days)
        if stats:
            df = pd.DataFrame(stats)

            row1 = st.columns(3)
            row2 = st.columns(3)
            row3 = st.columns(3)
            all_cols = row1 + row2 + row3

            strategy_map = {s['strategy_type']: s for s in stats}

            for idx, strategy_key in enumerate(STRATEGY_ORDER):
                if strategy_key not in strategy_map:
                    continue
                if idx >= len(all_cols):
                    break
                row = strategy_map[strategy_key]
                with all_cols[idx]:
                    label = STRATEGY_NAMES.get(strategy_key, strategy_key)
                    st.markdown(f"### {label}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("ベット数", row['total_bets'])
                    c2.metric("ROI", f"{row['roi']:.1f}%")
                    win_rate = (
                        row['wins'] / row['total_bets'] * 100
                        if row['total_bets'] > 0 else 0
                    )
                    c3.metric("的中率", f"{win_rate:.1f}%")

            # 棒グラフ比較
            fig = go.Figure()
            for _, row in df.iterrows():
                name = STRATEGY_NAMES.get(row['strategy_type'], row['strategy_type'])
                color = STRATEGY_COLORS.get(row['strategy_type'], '#333')
                fig.add_trace(go.Bar(
                    name=name,
                    x=['ベット数', '投資額(千円)', '回収額(千円)', 'ROI(%)'],
                    y=[
                        row['total_bets'],
                        (row['total_amount'] or 0) / 1000,
                        (row['total_payout'] or 0) / 1000,
                        row['roi'],
                    ],
                    marker_color=color,
                ))
            fig.update_layout(
                barmode='group', title='戦略比較',
                height=400,
                margin=dict(l=30, r=20, t=50, b=30),
                font=dict(size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True,
                           config={'displayModeBar': False})
        else:
            st.info("パフォーマンスデータがまだありません。")
    except Exception as e:
        st.error(f"統計取得エラー: {e}")

    # --- 時系列推移 ---
    st.subheader("日別収支推移")
    try:
        daily = _cached_daily_stats(days)
        if daily:
            df_daily = pd.DataFrame(daily)
            df_daily['profit'] = (
                df_daily['total_payout'].fillna(0) -
                df_daily['total_amount'].fillna(0)
            )
            df_daily['戦略'] = df_daily['strategy_type'].map(
                lambda x: STRATEGY_NAMES.get(x, x)
            )

            color_map = {
                STRATEGY_NAMES.get(k, k): v
                for k, v in STRATEGY_COLORS.items()
            }

            fig = px.line(
                df_daily, x='race_date', y='profit',
                color='戦略',
                color_discrete_map=color_map,
                title='日別損益推移',
                labels={'race_date': '日付', 'profit': '損益 (円)'},
            )
            fig.update_layout(
                height=400,
                margin=dict(l=30, r=20, t=50, b=30),
                font=dict(size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True,
                           config={'displayModeBar': False})

            fig_roi = px.line(
                df_daily, x='race_date', y='roi',
                color='戦略',
                color_discrete_map=color_map,
                title='日別ROI推移',
                labels={'race_date': '日付', 'roi': 'ROI (%)'},
            )
            fig_roi.update_layout(
                height=400,
                margin=dict(l=30, r=20, t=50, b=30),
                font=dict(size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            fig_roi.add_hline(y=100, line_dash="dash", line_color="gray",
                              annotation_text="損益分岐点")
            st.plotly_chart(fig_roi, use_container_width=True,
                           config={'displayModeBar': False})
        else:
            st.info("日別データがまだありません。")
    except Exception as e:
        st.error(f"日別データ取得エラー: {e}")

    # --- 場別パフォーマンス ---
    st.subheader("場別パフォーマンス")
    try:
        venue_data = _cached_venue_stats()
        if venue_data:
            df_venue = pd.DataFrame(venue_data)
            df_venue['venue_name'] = df_venue['venue_id'].map(
                lambda x: VENUE_NAMES.get(x, f'場{x}')
            )
            df_venue['戦略'] = df_venue['strategy_type'].map(
                lambda x: STRATEGY_NAMES.get(x, x)
            )

            color_map = {
                STRATEGY_NAMES.get(k, k): v
                for k, v in STRATEGY_COLORS.items()
            }

            fig = px.bar(
                df_venue, x='venue_name', y='roi',
                color='戦略',
                color_discrete_map=color_map,
                barmode='group',
                title='場別ROI',
                labels={'venue_name': '競艇場', 'roi': 'ROI (%)'},
            )
            fig.update_layout(
                height=400,
                margin=dict(l=30, r=20, t=50, b=50),
                font=dict(size=11),
                xaxis=dict(tickangle=-45),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            fig.add_hline(y=100, line_dash="dash", line_color="red",
                          annotation_text="損益分岐点")
            st.plotly_chart(fig, use_container_width=True,
                           config={'displayModeBar': False})
        else:
            st.info("場別データがまだありません。")
    except Exception as e:
        st.error(f"場別データ取得エラー: {e}")

performance_fragment()
