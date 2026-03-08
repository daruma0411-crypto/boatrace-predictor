"""システム状態ページ: DB統計、モデル情報、手動予測トリガー"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from streamlit_app.components.db_utils import get_db_connection
    from streamlit_app.components.mobile_css import inject_mobile_css
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

st.set_page_config(page_title="システム状態", page_icon="🔧", layout="wide",
                   initial_sidebar_state="collapsed")
if DB_AVAILABLE:
    inject_mobile_css()
st.title("🔧 システム状態")

VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}


# === DB統計 ===
st.subheader("データベース統計")
if not DB_AVAILABLE:
    st.warning("DB接続モジュールが利用できません")
try:
    with get_db_connection() as conn:
        cur = conn.cursor()

        # 総レース数
        cur.execute("SELECT COUNT(*) as cnt FROM races")
        total_races = cur.fetchone()['cnt']

        # 結果あり
        cur.execute("SELECT COUNT(*) as cnt FROM races WHERE result_1st IS NOT NULL")
        finished_races = cur.fetchone()['cnt']

        # 総boats数
        cur.execute("SELECT COUNT(*) as cnt FROM boats")
        total_boats = cur.fetchone()['cnt']

        # 日付範囲
        cur.execute("SELECT MIN(race_date) as min_d, MAX(race_date) as max_d FROM races")
        date_range = cur.fetchone()

        # 月別レース数
        cur.execute("""
            SELECT DATE_TRUNC('month', race_date)::date as month,
                   COUNT(*) as cnt
            FROM races
            WHERE result_1st IS NOT NULL
            GROUP BY month
            ORDER BY month
        """)
        monthly = cur.fetchall()

    col1, col2, col3 = st.columns(3)
    col1.metric("総レース数", f"{total_races:,}")
    col2.metric("結果あり", f"{finished_races:,}")
    col3.metric("選手データ", f"{total_boats:,}")

    if date_range['min_d'] and date_range['max_d']:
        st.info(
            f"データ期間: {date_range['min_d']} ~ {date_range['max_d']} "
            f"({(date_range['max_d'] - date_range['min_d']).days}日間)"
        )

    if monthly:
        df_monthly = pd.DataFrame(monthly)
        df_monthly['month'] = pd.to_datetime(df_monthly['month']).dt.strftime('%Y-%m')
        st.bar_chart(df_monthly.set_index('month')['cnt'])

except Exception as e:
    st.error(f"DB接続エラー: {e}")

st.divider()

# === モデル情報 ===
st.subheader("モデル情報")
model_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'models', 'boatrace_model.pth'
)

if os.path.exists(model_path) and TORCH_AVAILABLE:
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        col1, col2 = st.columns(2)

        if isinstance(checkpoint, dict):
            if 'val_loss' in checkpoint:
                col1.metric("検証ロス", f"{checkpoint['val_loss']:.4f}")
            if 'epoch' in checkpoint:
                col2.metric("エポック数", checkpoint['epoch'])
            if 'train_size' in checkpoint:
                st.metric("学習データ数", f"{checkpoint['train_size']:,}")
        else:
            st.info("モデルファイルあり (メタデータなし)")

        file_size = os.path.getsize(model_path)
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        st.caption(
            f"ファイルサイズ: {file_size/1024:.0f} KB | "
            f"最終更新: {mod_time.strftime('%Y-%m-%d %H:%M')}"
        )
    except Exception as e:
        st.warning(f"モデル読み込みエラー: {e}")
else:
    st.warning("モデルファイルが見つかりません")

st.divider()

# === 手動予測 ===
st.subheader("手動予測テスト")
st.caption("特定のレースに対してモデルの予測を実行します")

col1, col2, col3 = st.columns(3)
with col1:
    venue_id = st.selectbox(
        "会場",
        options=list(VENUE_NAMES.keys()),
        format_func=lambda x: f"{x} {VENUE_NAMES[x]}",
    )
with col2:
    race_number = st.selectbox("レース番号", options=list(range(1, 13)))
with col3:
    race_date = st.date_input("レース日", value=datetime.now().date())

if st.button("予測実行", type="primary"):
    try:
        from src.scraper import _get_session, scrape_racelist
        from src.predictor import RealtimePredictor
        import numpy as np

        with st.spinner("出走表取得中..."):
            session = _get_session()
            boats = scrape_racelist(session, race_date, venue_id, race_number)

        if not boats or len(boats) != 6:
            st.error("出走表を取得できませんでした（非開催 or データなし）")
        else:
            st.success(f"出走表取得成功: {len(boats)}艇")

            # 選手情報テーブル表示
            boat_df = pd.DataFrame(boats)
            display_cols = ['boat_number', 'player_name', 'player_class',
                           'win_rate', 'win_rate_2', 'avg_st']
            available_cols = [c for c in display_cols if c in boat_df.columns]
            st.dataframe(
                boat_df[available_cols].rename(columns={
                    'boat_number': '艇番',
                    'player_name': '選手名',
                    'player_class': '級別',
                    'win_rate': '勝率',
                    'win_rate_2': '2連率',
                    'avg_st': '平均ST',
                }),
                use_container_width=True,
                hide_index=True,
            )

            # 予測実行
            with st.spinner("予測計算中..."):
                predictor = RealtimePredictor()
                race_data = {
                    'venue_id': venue_id,
                    'month': race_date.month,
                    'distance': 1800,
                    'wind_speed': 0,
                    'wind_direction': 'calm',
                    'temperature': 20,
                }
                boats_data = []
                for b in boats:
                    boats_data.append({
                        'boat_number': b['boat_number'],
                        'player_class': b.get('player_class'),
                        'win_rate': b.get('win_rate'),
                        'win_rate_2': b.get('win_rate_2'),
                        'win_rate_3': b.get('win_rate_3'),
                        'local_win_rate': b.get('local_win_rate'),
                        'local_win_rate_2': b.get('local_win_rate_2'),
                        'avg_st': b.get('avg_st'),
                        'motor_win_rate_2': b.get('motor_win_rate_2'),
                        'motor_win_rate_3': b.get('motor_win_rate_3'),
                        'boat_win_rate_2': b.get('boat_win_rate_2'),
                        'weight': b.get('weight'),
                        'exhibition_time': None,
                        'approach_course': b['boat_number'],
                        'is_new_motor': False,
                        'fallback_flag': False,
                    })

                prediction = predictor.predict(race_data, boats_data)

            # 結果表示
            probs_1st = prediction['probs_1st']
            st.markdown("### 1着予測確率")

            prob_df = pd.DataFrame({
                '艇番': [f'{i+1}号艇' for i in range(6)],
                '確率': [f'{p*100:.1f}%' for p in probs_1st],
                'バー': probs_1st,
            })

            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(
                    prob_df[['艇番', '確率']],
                    use_container_width=True,
                    hide_index=True,
                )
            with col2:
                st.bar_chart(
                    pd.DataFrame({
                        '確率': probs_1st,
                    }, index=[f'{i+1}号艇' for i in range(6)]),
                )

            # 3連単上位
            pred_1st = int(np.argmax(probs_1st)) + 1
            probs_2nd = prediction['probs_2nd']
            probs_3rd = prediction['probs_3rd']
            pred_2nd = int(np.argmax(probs_2nd)) + 1
            pred_3rd = int(np.argmax(probs_3rd)) + 1

            st.markdown(
                f"### 予測: **{pred_1st}-{pred_2nd}-{pred_3rd}** "
                f"(1着{probs_1st[pred_1st-1]*100:.1f}%)"
            )

    except Exception as e:
        st.error(f"予測エラー: {e}")
        import traceback
        st.code(traceback.format_exc())
