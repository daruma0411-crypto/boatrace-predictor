"""設定ページ: ケリー係数、期待値、賭け金制限、モデル選択"""
import json
import os
import streamlit as st

st.set_page_config(page_title="設定", page_icon="⚙️")
st.title("⚙️ 設定")

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'config', 'betting_config.json'
)


def load_config():
    defaults = {
        'kelly_fraction': 0.5,
        'max_bet_ratio': 0.05,
        'min_expected_value': 1.05,
        'min_bet_amount': 100,
        'max_bet_amount': 10000,
    }
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        defaults.update(config)
    except FileNotFoundError:
        pass
    return defaults


def save_config(config):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


config = load_config()

st.subheader("ケリー基準設定")
kelly_fraction = st.slider(
    "ケリー係数（ハーフ・ケリー = 0.5）",
    min_value=0.1, max_value=1.0, value=config['kelly_fraction'],
    step=0.05,
    help="フルケリー=1.0、ハーフケリー=0.5。小さいほど保守的。"
)

st.subheader("期待値フィルタ")
min_ev = st.slider(
    "最低期待値",
    min_value=1.0, max_value=2.0, value=config['min_expected_value'],
    step=0.01,
    help="この値以上の期待値がある買い目のみ購入（戦略A）"
)

st.subheader("賭け金制限")
col1, col2 = st.columns(2)
with col1:
    min_bet = st.number_input(
        "最低賭け金 (円)",
        min_value=100, max_value=10000,
        value=config['min_bet_amount'], step=100
    )
with col2:
    max_bet = st.number_input(
        "最高賭け金 (円)",
        min_value=100, max_value=100000,
        value=config['max_bet_amount'], step=1000
    )

max_bet_ratio = st.slider(
    "最大資金比率",
    min_value=0.01, max_value=0.20, value=config['max_bet_ratio'],
    step=0.01,
    help="1レースあたり資金の何%まで賭けるか"
)

st.subheader("モデル選択")
model_options = ['models/boatrace_model.pth']
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models'
)
if os.path.exists(models_dir):
    for f in os.listdir(models_dir):
        if f.endswith('.pth') and f != 'boatrace_model.pth':
            model_options.append(f'models/{f}')

selected_model = st.selectbox("使用モデル", model_options)

st.divider()

if st.button("設定を保存", type="primary"):
    new_config = {
        'kelly_fraction': kelly_fraction,
        'max_bet_ratio': max_bet_ratio,
        'min_expected_value': min_ev,
        'min_bet_amount': min_bet,
        'max_bet_amount': max_bet,
        'model_path': selected_model,
    }
    save_config(new_config)
    st.success("設定を保存しました")
    st.json(new_config)
