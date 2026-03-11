"""ボートレース予想AIシステム メインUI"""
import sys
import os
import threading
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import date, timedelta

# --- スケジューラーをデーモンスレッドで起動（プロセス内で1回だけ） ---
# UIの初期ロードを妨げないよう30秒遅延してから起動
_scheduler_lock = threading.Lock()
_scheduler_started = False


def _start_scheduler_once():
    global _scheduler_started
    with _scheduler_lock:
        if _scheduler_started:
            return
        _scheduler_started = True

    import time as _time

    def _write_health(status, detail=''):
        """スケジューラーの状態をDBに記録"""
        try:
            import psycopg2
            db_url = os.environ.get('DATABASE_URL', '')
            if db_url.startswith('postgres://'):
                db_url = db_url.replace('postgres://', 'postgresql://', 1)
            if not db_url:
                return
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS scheduler_health (
                    id SERIAL PRIMARY KEY,
                    status VARCHAR(50),
                    detail TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            cur.execute(
                "INSERT INTO scheduler_health (status, detail) VALUES (%s, %s)",
                (status, detail[:500] if detail else ''),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _run():
        slog = logging.getLogger('scheduler_thread')
        slog.info("スケジューラー: 30秒後に起動予定")
        _write_health('waiting', '30秒待機中')
        _time.sleep(30)

        # 起動時統計JSON生成（旧Procfileから移動）
        try:
            slog.info("統計JSON生成中...")
            from scripts.gen_stats_json import main as gen_stats
            gen_stats()
            slog.info("統計JSON生成完了")
        except Exception as e:
            slog.warning(f"統計JSON生成スキップ: {e}")

        # クラッシュ時自動復帰ループ（最大10回リトライ）
        for attempt in range(10):
            try:
                _write_health('initializing', f'DB初期化中 (attempt={attempt+1})')
                from src.database import init_database
                init_database()
                _write_health('loading_model', 'モデル読込中')
                from src.scheduler import DynamicRaceScheduler
                scheduler = DynamicRaceScheduler()
                _write_health('running', 'ポーリング開始')
                slog.info("スケジューラースレッド起動完了")
                scheduler.run_polling()
            except Exception as e:
                slog.error(
                    f"スケジューラースレッド異常終了 (attempt={attempt+1}): {e}",
                    exc_info=True,
                )
                _write_health('crashed', f'attempt={attempt+1}: {str(e)[:400]}')
                _time.sleep(60)  # 1分後にリトライ
        slog.error("スケジューラー: 最大リトライ回数到達、停止")

    t = threading.Thread(target=_run, daemon=True, name="scheduler")
    t.start()


_start_scheduler_once()
from streamlit_app.components.db_utils import (
    get_db_connection,
    get_recent_predictions,
    get_today_bets,
    get_today_venues,
    get_daily_stats_by_period,
    get_dashboard_data,
)
from streamlit_app.components.mobile_css import inject_mobile_css

st.set_page_config(
    page_title="ボートレース予想AI",
    page_icon="\U0001f6a4",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_mobile_css()

# --- キャッシュ ---
@st.cache_data(ttl=300, show_spinner=False)
def _cached_dashboard(start_date, end_date):
    try:
        return get_dashboard_data(start_date, end_date)
    except Exception as e:
        return {
            'today_races': 0, 'today_preds': 0,
            'strategy_summary': [], 'bankrolls': {},
            'db_ok': False, 'error': str(e),
        }

@st.cache_data(ttl=300, show_spinner=False)
def _cached_today_bets():
    return get_today_bets()

@st.cache_data(ttl=300, show_spinner=False)
def _cached_today_venues():
    return get_today_venues()

@st.cache_data(ttl=300, show_spinner=False)
def _cached_daily_stats_by_period(start_date, end_date):
    return get_daily_stats_by_period(start_date, end_date)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_predictions(limit):
    return get_recent_predictions(limit=limit)

# --- 定数 ---
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
    'bt_none': 'G: BT基本 (odds≤30)',
    'bt_entropy': 'H: BT確信 (odds≤30+H<2.3)',
    'bt_ensemble': 'I: BT合議 (odds≤30+3/4)',
}

STRATEGY_ORDER = [
    'conservative', 'standard', 'divergence',
    'high_confidence', 'ensemble', 'div_confidence',
    'bt_none', 'bt_entropy', 'bt_ensemble',
]


def _venue_name(venue_id):
    return VENUE_NAMES.get(venue_id, f'場{venue_id}')


def _strategy_name(strategy_type):
    return STRATEGY_NAMES.get(strategy_type, strategy_type)


# --- session_state 初期化 ---
if 'start_date' not in st.session_state:
    st.session_state['start_date'] = str(date.today())
    st.session_state['end_date'] = str(date.today())


# --- サイドバー（fragment外で描画）---
with st.sidebar:
    st.title("\U0001f6a4 ボートレース予想AI")
    st.divider()
    st.subheader("システム状態")
    _sidebar_dashboard = _cached_dashboard(
        st.session_state['start_date'], st.session_state['end_date']
    )
    if _sidebar_dashboard['db_ok']:
        st.success("DB接続: OK")
    else:
        st.error("DB接続: エラー")
    st.metric("本日のレース", _sidebar_dashboard['today_races'])
    st.metric("本日の予測", _sidebar_dashboard['today_preds'])

# --- メインコンテンツ ---
st.title("\U0001f6a4 ボートレース予想AIダッシュボード")


# === 期間セレクター + 戦略カード（fragment）===
@st.fragment
def period_and_cards_fragment():
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

    # session_state に保存（他fragment参照用）
    st.session_state['start_date'] = str(start_date)
    st.session_state['end_date'] = str(end_date)

    # 一括DB取得
    dashboard = _cached_dashboard(str(start_date), str(end_date))

    # 戦略カード
    st.subheader("戦略別サマリー")
    summary_data = dashboard['strategy_summary']
    summary_dict = {s['strategy_type']: s for s in summary_data} if summary_data else {}
    bankrolls = dashboard['bankrolls']

    row1 = st.columns(3)
    row2 = st.columns(3)
    all_cols = row1 + row2

    for idx, strategy_key in enumerate(STRATEGY_ORDER):
        with all_cols[idx]:
            label = _strategy_name(strategy_key)
            s = summary_dict.get(strategy_key)
            bankroll = bankrolls.get(strategy_key, 200000)

            st.markdown(f"**{label}**")
            if s and s['total_bets'] > 0:
                total_amount = s['total_amount'] or 0
                roi = s['roi'] or 0
                wins = s['wins'] or 0
                total_bets = s['total_bets']
                total_races = s['total_races'] or 0
                win_rate = wins / total_bets * 100 if total_bets > 0 else 0

                st.metric("残金", f"\u00a5{bankroll:,.0f}")
                st.metric("投資額", f"\u00a5{total_amount:,}")
                st.metric("レース数", total_races)
                roi_delta = f"{'+'if roi >= 100 else ''}{roi - 100:.1f}%"
                st.metric("ROI", f"{roi:.1f}%", delta=roi_delta)
                st.metric("的中率", f"{win_rate:.1f}%")
            else:
                st.metric("残金", f"\u00a5{bankroll:,.0f}")
                st.info("データなし")

period_and_cards_fragment()

st.divider()

# --- タブ ---
tab1, tab2, tab3 = st.tabs(["\U0001f4ca 本日の買い目", "\U0001f4b0 期間別推移", "\U0001f504 予測詳細"])


# ========== タブ1: 本日の買い目 ==========
@st.fragment
def tab1_bets_fragment():
    try:
        all_bets = _cached_today_bets()
        venues = _cached_today_venues()

        if not all_bets:
            st.info("本日の買い目はまだありません。レース締切10分前に自動生成されます。")
            return

        venue_options = ['全会場'] + [_venue_name(v) for v in venues]
        selected_venue = st.selectbox(
            '会場を選択', venue_options, key='venue_filter'
        )

        df = pd.DataFrame(all_bets)
        df['会場'] = df['venue_id'].map(_venue_name)
        df['戦略'] = df['strategy_type'].map(_strategy_name)

        if selected_venue != '全会場':
            df = df[df['会場'] == selected_venue]

        if df.empty:
            st.info(f"{selected_venue}の買い目はまだありません。")
            return

        # サマリーを先に表示
        summary = pd.DataFrame(all_bets)
        summary['会場'] = summary['venue_id'].map(_venue_name)
        venue_summary = summary.groupby('会場').agg(
            点数=('combination', 'count'),
            投資額=('amount', 'sum'),
        ).reset_index()
        venue_summary['投資額'] = venue_summary['投資額'].map(lambda x: f'\u00a5{x:,}')
        st.markdown("**会場別サマリー**")
        st.dataframe(venue_summary, use_container_width=True, hide_index=True)

        st.divider()

        # レース別詳細（閉じた状態で表示 → クリックで展開）
        for (venue_id, race_num), group in df.groupby(['venue_id', 'race_number']):
            venue = _venue_name(venue_id)
            total_amount = group['amount'].sum()

            with st.expander(
                f"{venue} {race_num}R  "
                f"（{len(group)}点 / 計 \u00a5{total_amount:,}）",
                expanded=False,
            ):
                for strategy, sgroup in group.groupby('strategy_type'):
                    st.markdown(f"**{_strategy_name(strategy)}**")
                    display = sgroup[[
                        'combination', 'amount', 'odds', 'expected_value',
                    ]].copy()
                    display.columns = ['組み合わせ', '金額', 'オッズ', '期待値']
                    display['金額'] = display['金額'].map(lambda x: f'\u00a5{x:,}')
                    display['オッズ'] = display['オッズ'].map(
                        lambda x: f'{x:.1f}' if x else '-'
                    )
                    display['期待値'] = display['期待値'].map(
                        lambda x: f'{x:.2f}' if x else '-'
                    )
                    st.dataframe(
                        display.reset_index(drop=True),
                        use_container_width=True, hide_index=True,
                    )

    except Exception as e:
        st.error(f"データ取得エラー: {e}")

with tab1:
    tab1_bets_fragment()


# ========== タブ2: 期間別推移 ==========
@st.fragment
def tab2_trend_fragment():
    st.subheader("日別収支推移")
    try:
        s_date = st.session_state.get('start_date', str(date.today()))
        e_date = st.session_state.get('end_date', str(date.today()))
        daily = _cached_daily_stats_by_period(s_date, e_date)
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
                title=f'日別損益推移 ({s_date} ~ {e_date})',
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
                height=400,
                margin=dict(l=30, r=20, t=50, b=30),
                font=dict(size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_cum, use_container_width=True,
                           config={'displayModeBar': False})
        else:
            st.info("この期間のデータがまだありません。")
    except Exception as e:
        st.error(f"日別データ取得エラー: {e}")

with tab2:
    tab2_trend_fragment()


# ========== タブ3: 予測詳細 ==========
@st.fragment
def tab3_predictions_fragment():
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

with tab3:
    tab3_predictions_fragment()
