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
    get_today_bets,
    get_today_venues,
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
def _cached_performance(days):
    return get_performance_stats(days=days)

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
    today_races, today_preds, db_ok = _cached_today_counts()
    if db_ok:
        st.success("DB接続: OK")
    else:
        st.error("DB接続: エラー")
    st.metric("本日のレース", today_races)
    st.metric("本日の予測", today_preds)

    st.divider()
    st.subheader("資金状況")
    try:
        stats = _cached_performance(days=9999)
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

tab1, tab2, tab3 = st.tabs(["📊 本日の買い目", "💰 回収率", "🔄 予測詳細"])

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


# ========== タブ2: 回収率 ==========
with tab2:
    st.subheader("戦略別パフォーマンス")
    try:
        stats = _cached_performance(days=30)
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
