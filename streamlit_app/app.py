"""ボートレース予想AIシステム メインUI"""
import sys
import os
import threading
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DEPLOY_VERSION = "v10.6-mc-v1-qmc-v3early-11strategies"

# モジュールロード時に即座にDB書き込み（クラッシュ箇所特定用）
try:
    import psycopg2 as _pg2
    _db_url = os.environ.get('DATABASE_URL', '')
    if _db_url.startswith('postgres://'):
        _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
    if _db_url:
        _c = _pg2.connect(_db_url)
        _cur = _c.cursor()
        _cur.execute(
            "INSERT INTO scheduler_health (status, detail) VALUES (%s, %s)",
            ('module_loaded', f'version={_DEPLOY_VERSION}'),
        )
        _c.commit()
        _c.close()
        print(f"[APP] module_loaded marker written, version={_DEPLOY_VERSION}", flush=True)
    else:
        print("[APP] WARNING: DATABASE_URL is empty!", flush=True)
except Exception as _e:
    print(f"[APP] module_loaded write failed: {_e}", flush=True)

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
    import traceback as _tb

    def _write_health(status, detail=''):
        """スケジューラーの状態をDBに記録"""
        print(f"[HEALTH] {status}: {detail}", flush=True)
        try:
            import psycopg2
            db_url = os.environ.get('DATABASE_URL', '')
            if db_url.startswith('postgres://'):
                db_url = db_url.replace('postgres://', 'postgresql://', 1)
            if not db_url:
                print("[HEALTH] WARNING: DATABASE_URL not set!", flush=True)
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
        except Exception as e:
            print(f"[HEALTH] DB write failed: {e}", flush=True)

    def _run():
        try:
            slog = logging.getLogger('scheduler_thread')
            slog.info(f"スケジューラー起動: version={_DEPLOY_VERSION}")
            print(f"[SCHEDULER] Starting version={_DEPLOY_VERSION}", flush=True)
        except Exception as e:
            print(f"[SCHEDULER] Logger init failed: {e}", flush=True)

        try:
            _write_health('waiting', f'version={_DEPLOY_VERSION}, 30秒待機中')
        except Exception as e:
            print(f"[SCHEDULER] Health write failed: {e}", flush=True)

        _time.sleep(30)

        # 起動時統計JSON生成（旧Procfileから移動）
        try:
            print("[SCHEDULER] Generating stats JSON...", flush=True)
            from scripts.gen_stats_json import main as gen_stats
            gen_stats()
            print("[SCHEDULER] Stats JSON done", flush=True)
        except Exception as e:
            print(f"[SCHEDULER] Stats JSON skipped: {e}", flush=True)

        # クラッシュ時自動復帰ループ（無限リトライ）
        attempt = 0
        while True:
            attempt += 1
            try:
                print(f"[SCHEDULER] Attempt {attempt}: initializing...", flush=True)
                _write_health('initializing', f'DB初期化中 (attempt={attempt})')
                from src.database import init_database
                init_database()
                print(f"[SCHEDULER] Attempt {attempt}: loading model...", flush=True)
                _write_health('loading_model', 'モデル読込中')
                from src.scheduler import DynamicRaceScheduler
                scheduler = DynamicRaceScheduler()
                print(f"[SCHEDULER] Attempt {attempt}: starting polling...", flush=True)
                _write_health('running', 'ポーリング開始')
                scheduler.run_polling()
                # run_polling()が正常returnした場合もリトライ
                print(f"[SCHEDULER] run_polling() returned normally, restarting (attempt={attempt})", flush=True)
                _write_health('restarting', f'正常終了後の再起動 attempt={attempt}')
            except Exception as e:
                tb_str = _tb.format_exc()
                print(f"[SCHEDULER] CRASHED (attempt={attempt}): {e}\n{tb_str}", flush=True)
                _write_health('crashed', f'attempt={attempt}: {str(e)[:400]}')
            # 指数バックオフ: 60→120→240→480→600秒上限
            wait = min(60 * (2 ** min(attempt - 1, 3)), 600)
            _write_health('backoff', f'attempt={attempt}, wait={wait}秒')
            _time.sleep(wait)

    try:
        t = threading.Thread(target=_run, daemon=True, name="scheduler")
        t.start()
        print(f"[APP] Scheduler thread started", flush=True)
    except Exception as e:
        print(f"[APP] Failed to start scheduler thread: {e}", flush=True)


_start_scheduler_once()
print(f"[APP] Module loaded, version={_DEPLOY_VERSION}", flush=True)
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
    page_title=f"BoatAI [{_DEPLOY_VERSION}]",
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
def _cached_real_stats(start_date, end_date):
    """purchase_log から本番購入の実績を取得"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    COUNT(*) as total_purchases,
                    COALESCE(SUM(pl.amount), 0) as total_invested,
                    COUNT(*) FILTER (WHERE b.is_hit = true) as wins,
                    COALESCE(SUM(CASE WHEN b.is_hit = true THEN b.payout ELSE 0 END), 0) as total_payout
                FROM purchase_log pl
                JOIN bets b ON pl.bet_id = b.id
                WHERE pl.status = 'success'
                  AND pl.purchased_at >= %s
                  AND pl.purchased_at < %s::date + 1
            """, (start_date, end_date))
            row = cur.fetchone()
            return {
                'total_purchases': row[0],
                'total_invested': int(row[1]),
                'wins': row[2],
                'total_payout': int(row[3]),
            }
    except Exception:
        return {'total_purchases': 0, 'total_invested': 0, 'wins': 0, 'total_payout': 0}

@st.cache_data(ttl=300, show_spinner=False)
def _cached_real_daily(start_date, end_date):
    """purchase_log から本番購入の日次推移を取得"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    pl.purchased_at::date as day,
                    SUM(pl.amount) as invested,
                    COALESCE(SUM(CASE WHEN b.is_hit = true THEN b.payout ELSE 0 END), 0) as payout
                FROM purchase_log pl
                JOIN bets b ON pl.bet_id = b.id
                WHERE pl.status = 'success'
                  AND pl.purchased_at >= %s
                  AND pl.purchased_at < %s::date + 1
                GROUP BY pl.purchased_at::date
                ORDER BY day
            """, (start_date, end_date))
            return [{'day': r[0], 'invested': int(r[1]), 'payout': int(r[2])} for r in cur.fetchall()]
    except Exception:
        return []

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
    'mc_quarter_kelly': 'L: MC v1 基準',
    'mc_early_race': 'O: MC v1 序盤',
    'mc_venue_focus': 'P: MC v1 得意場',
    'mc_high_ev': 'Q: MC v1 高EV',
    'mc_are_v2': 'R: MC v1 ModelB',
    'mc2_quarter_kelly': 'L2: QMC 基準',
    'mc2_early_race': 'O2: QMC 序盤',
    'mc2_venue_focus': 'P2: QMC 得意場',
    'mc2_high_ev': 'Q2: QMC 高EV',
    'mc2_are_v2': 'R2: QMC ModelB',
    'mc3_early_race': 'O3: QMC v3 序盤',
}

STRATEGY_ORDER = [
    'mc_quarter_kelly', 'mc_early_race', 'mc_venue_focus',
    'mc_high_ev', 'mc_are_v2',
    'mc2_quarter_kelly', 'mc2_early_race', 'mc2_venue_focus',
    'mc2_high_ev', 'mc2_are_v2',
    'mc3_early_race',
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
    st.metric("本日のベット", _sidebar_dashboard.get('today_bets', 0))
    st.metric("本日の的中", _sidebar_dashboard.get('today_hits', 0))

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

    # === 本番投入カード（先頭表示） ===
    REAL_BANKROLL = 200000  # 本番初期資金
    try:
        real_stats = _cached_real_stats(str(start_date), str(end_date))
    except Exception:
        real_stats = {'total_purchases': 0, 'total_invested': 0, 'wins': 0, 'total_payout': 0}

    st.subheader("本番投入")
    real_col1, real_col2 = st.columns([1, 2])
    with real_col1:
        st.markdown(
            "<p style='font-size:1.75rem;font-weight:700;margin:0 0 0.5rem 0;"
            "color:#ff6b35'>\U0001f4b0 本番投入 (O: MC v1 序盤)</p>",
            unsafe_allow_html=True,
        )
        if real_stats['total_purchases'] > 0:
            invested = real_stats['total_invested']
            payout = real_stats['total_payout']
            net = payout - invested
            roi = payout / invested * 100 if invested > 0 else 0
            balance = REAL_BANKROLL + net
            wins = real_stats['wins']
            win_rate = wins / real_stats['total_purchases'] * 100

            st.metric("残金", f"\u00a5{balance:,.0f}")
            st.metric("投資額", f"\u00a5{invested:,}")
            st.metric("購入数", real_stats['total_purchases'])
            roi_delta = f"{'+'if roi >= 100 else ''}{roi - 100:.1f}%"
            st.metric("ROI", f"{roi:.1f}%", delta=roi_delta)
            st.metric("的中率", f"{win_rate:.1f}%")
        else:
            st.metric("残金", f"\u00a5{REAL_BANKROLL:,.0f}")
            st.info("明日から本番稼働開始")
    with real_col2:
        st.markdown(
            "<div style='background:#1a1a2e;border:2px solid #ff6b35;border-radius:10px;"
            "padding:1rem;margin-top:2.5rem'>"
            "<b>戦略:</b> O (MC v1 序盤R1-R4限定)<br>"
            "<b>ベット額:</b> Kelly計算額（実額投入）<br>"
            "<b>初期資金:</b> \u00a5200,000<br>"
            "<b>自動購入:</b> テレボートSP版 (WebKit)<br>"
            "</div>",
            unsafe_allow_html=True,
        )

    # === 本番投入グラフ ===
    try:
        real_daily = _cached_real_daily(str(start_date), str(end_date))
    except Exception:
        real_daily = []

    if real_daily:
        import plotly.graph_objects as go

        # 日次の累積残金を計算
        cumul_net = 0
        dates = []
        balances = []
        for d in real_daily:
            cumul_net += d['payout'] - d['invested']
            dates.append(d['day'])
            balances.append(REAL_BANKROLL + cumul_net)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=balances,
            mode='lines+markers',
            name='本番投入',
            line=dict(color='#ff6b35', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(255,107,53,0.1)',
        ))
        # 初期資金ライン
        fig.add_hline(
            y=REAL_BANKROLL, line_dash="dash",
            line_color="rgba(255,255,255,0.3)",
            annotation_text=f"初期資金 \u00a5{REAL_BANKROLL:,}",
            annotation_position="bottom right",
            annotation_font_color="rgba(255,255,255,0.5)",
        )
        fig.update_layout(
            title=None,
            xaxis_title="日付",
            yaxis_title="残高 (円)",
            template="plotly_dark",
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            yaxis=dict(tickformat=","),
        )
        st.plotly_chart(fig, use_container_width=True,
                       config={'displayModeBar': False, 'staticPlot': True})

    st.divider()

    # === シミュレーション戦略カード ===
    st.subheader("戦略別サマリー")
    summary_data = dashboard['strategy_summary']
    summary_dict = {s['strategy_type']: s for s in summary_data} if summary_data else {}
    bankrolls = dashboard['bankrolls']

    n = len(STRATEGY_ORDER)
    cols_per_row = 3
    rows_needed = (n + cols_per_row - 1) // cols_per_row
    all_cols = []
    for r in range(rows_needed):
        remaining = n - r * cols_per_row
        num_cols = min(remaining, cols_per_row)
        all_cols.extend(st.columns(num_cols))

    for idx, strategy_key in enumerate(STRATEGY_ORDER):
        with all_cols[idx]:
            label = _strategy_name(strategy_key)
            s = summary_dict.get(strategy_key)
            bankroll = bankrolls.get(strategy_key, 200000)

            st.markdown(f"<p style='font-size:1.75rem;font-weight:700;margin:0 0 0.5rem 0'>{label}</p>", unsafe_allow_html=True)
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

        df = pd.DataFrame(all_bets)

        # ベット一覧テーブル
        display = df.copy()

        # 時間: deadline_time → HH:MM 形式
        display['時間'] = display['deadline_time'].apply(
            lambda x: x.strftime('%H:%M') if pd.notna(x) and hasattr(x, 'strftime') else '-'
        )

        # レース: venue_id + race_number → "桐生 1R" 形式
        display['レース'] = display.apply(
            lambda r: f"{_venue_name(r['venue_id'])} {r['race_number']}R", axis=1
        )

        # 結果: is_finished, is_hit, return_amount, actual_result_trifecta を統合
        def _format_result(row):
            if not row.get('is_finished'):
                return '\u23f3 判定中'
            if row.get('is_hit') is True:
                amt = row.get('return_amount', 0)
                return f'\U0001f3af 的中 (+\u00a5{int(amt):,})'
            trifecta = row.get('actual_result_trifecta', '')
            if trifecta:
                return f'\u274c ハズレ (正解: {trifecta})'
            return '\u274c ハズレ'

        display['結果'] = display.apply(_format_result, axis=1)

        # 数値フォーマット
        display['金額'] = display['amount'].map(lambda x: f'\u00a5{int(x):,}')
        display['オッズ'] = display['odds'].map(
            lambda x: f'{x:.1f}倍' if x else '-'
        )
        display['期待値'] = display['expected_value'].map(
            lambda x: f'{x:.2f}' if x else '-'
        )

        # カラム絞り込み
        display = display[['時間', 'レース', 'combination',
                           '金額', 'オッズ', '期待値', '結果']].copy()
        display.columns = ['時間', 'レース', '買い目',
                           '金額', 'オッズ', '期待値', '結果']

        st.dataframe(
            display.reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )

    except Exception as e:
        st.error(f"データ取得エラー: {e}")

with tab1:
    tab1_bets_fragment()


# ========== タブ2: 残高推移 ==========
@st.fragment
def tab2_trend_fragment():
    st.subheader("戦略別残高推移")

    # 期間切替
    period = st.radio(
        "期間", ["日", "月", "年"], horizontal=True, key="trend_period"
    )
    today = date.today()
    if period == "日":
        t_start = today - timedelta(days=7)
    elif period == "月":
        t_start = today - timedelta(days=30)
    else:
        t_start = today - timedelta(days=365)

    try:
        daily = _cached_daily_stats_by_period(str(t_start), str(today))
        if not daily:
            st.info("この期間のデータがまだありません。")
            return

        import plotly.graph_objects as go

        df_daily = pd.DataFrame(daily)
        df_daily['profit'] = (
            df_daily['total_payout'].fillna(0) -
            df_daily['total_amount'].fillna(0)
        )

        # 戦略別チェックボックス（停止済み戦略を除外）
        strategies = sorted(s for s in df_daily['strategy_type'].unique()
                            if s in STRATEGY_NAMES)
        selected = []
        cols_cb = st.columns(len(strategies))
        for i, s in enumerate(strategies):
            with cols_cb[i]:
                if st.checkbox(_strategy_name(s), value=True, key=f'cb_{s}'):
                    selected.append(s)

        if not selected:
            st.info("戦略を1つ以上選択してください。")
            return

        # 残高計算: 20万スタート → 日ごとの累積損益を加算
        INITIAL = 200000
        fig = go.Figure()
        for s in selected:
            mask = df_daily['strategy_type'] == s
            df_s = df_daily[mask].sort_values('race_date').copy()
            df_s['balance'] = INITIAL + df_s['profit'].cumsum()
            fig.add_trace(go.Scatter(
                x=df_s['race_date'], y=df_s['balance'],
                mode='lines+markers', name=_strategy_name(s),
                hovertemplate='%{x}<br>残高: ¥%{y:,.0f}<extra></extra>',
            ))

        fig.add_hline(y=INITIAL, line_dash="dash", line_color="gray",
                      annotation_text="初期資金 ¥200,000")
        fig.update_layout(
            height=500,
            margin=dict(l=30, r=20, t=30, b=30),
            font=dict(size=14),
            yaxis_title="残高 (円)",
            xaxis_title="日付",
            yaxis_tickformat=",",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            dragmode=False,
        )
        st.plotly_chart(fig, use_container_width=True,
                       config={'displayModeBar': False, 'staticPlot': True})

    except Exception as e:
        st.error(f"残高推移取得エラー: {e}")

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
