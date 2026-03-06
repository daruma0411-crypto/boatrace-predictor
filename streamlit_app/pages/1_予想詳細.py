"""予想詳細ページ: レース選択、確率ヒートマップ、推奨買い目"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_app.components.db_utils import get_db_connection
from streamlit_app.components.mobile_css import inject_mobile_css

st.set_page_config(page_title="予想詳細", page_icon="🎯", layout="wide",
                   initial_sidebar_state="collapsed")
inject_mobile_css()
st.title("🎯 予想詳細")

VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}

# --- レース選択 ---
try:
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT r.venue_id, r.race_number, r.race_date,
                   p.strategy_type, p.id as prediction_id
            FROM predictions p
            JOIN races r ON p.race_id = r.id
            WHERE r.race_date = CURRENT_DATE
            ORDER BY r.venue_id, r.race_number
        """)
        available = cur.fetchall()
except Exception:
    available = []

if not available:
    st.info("本日の予想データがありません。")
    st.stop()

options = [
    f"{VENUE_NAMES.get(r['venue_id'], r['venue_id'])} {r['race_number']}R "
    f"({r['strategy_type']})"
    for r in available
]
selected_idx = st.selectbox("レースを選択", range(len(options)),
                             format_func=lambda i: options[i])
selected = available[selected_idx]

# --- 予測データ取得 ---
try:
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM predictions WHERE id = %s",
            (selected['prediction_id'],)
        )
        pred = cur.fetchone()
except Exception as e:
    st.error(f"データ取得エラー: {e}")
    st.stop()

if not pred:
    st.warning("予測データが見つかりません。")
    st.stop()

# --- 確率ヒートマップ ---
st.subheader("確率ヒートマップ")

probs_1st = pred.get('probabilities_1st', [0]*6)
probs_2nd = pred.get('probabilities_2nd', [0]*6)
probs_3rd = pred.get('probabilities_3rd', [0]*6)

if isinstance(probs_1st, str):
    import json
    probs_1st = json.loads(probs_1st)
    probs_2nd = json.loads(probs_2nd)
    probs_3rd = json.loads(probs_3rd)

heatmap_data = np.array([probs_1st, probs_2nd, probs_3rd])

fig = go.Figure(data=go.Heatmap(
    z=heatmap_data,
    x=[f'{i+1}号艇' for i in range(6)],
    y=['1着', '2着', '3着'],
    colorscale='RdYlGn',
    text=np.round(heatmap_data * 100, 1),
    texttemplate='%{text}%',
    textfont={'size': 14},
))
fig.update_layout(
    title='着順別確率 (%)',
    height=300,
    margin=dict(l=30, r=20, t=50, b=30),
    font=dict(size=12),
)
st.plotly_chart(fig, use_container_width=True,
               config={'responsive': True, 'displayModeBar': False})

# --- 推奨買い目 ---
st.subheader("推奨買い目")
recommended = pred.get('recommended_bets', [])
if isinstance(recommended, str):
    import json
    recommended = json.loads(recommended)

if recommended:
    df = pd.DataFrame(recommended)
    display_cols = [c for c in ['combination', 'amount', 'odds',
                                 'expected_value', 'probability']
                    if c in df.columns]
    st.dataframe(
        df[display_cols].style.format({
            'odds': '{:.1f}',
            'expected_value': '{:.3f}',
            'probability': '{:.4f}',
            'amount': '¥{:,.0f}',
        }),
        use_container_width=True,
    )
    total = sum(b.get('amount', 0) for b in recommended)
    st.metric("合計投資額", f"¥{total:,}")
else:
    st.info("推奨買い目なし（条件を満たす買い目がありません）")
