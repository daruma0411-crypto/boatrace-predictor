"""共通モバイル対応CSS - 全ページから呼び出す"""
import streamlit as st


def inject_mobile_css():
    """モバイル対応CSSを注入する"""
    st.markdown("""
<style>
    /* ===== 共通スタイル ===== */
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        border: 1px solid #333;
    }

    /* タッチ操作向けボタン拡大 */
    .stButton > button {
        min-height: 48px;
        font-size: 1rem;
    }

    /* スライダーのタッチ領域拡大 */
    .stSlider [role="slider"] {
        width: 24px !important;
        height: 24px !important;
    }

    /* ===== タブレット (1024px以下) ===== */
    @media (max-width: 1024px) {
        .main .block-container {
            padding: 1.5rem 1rem;
        }
        /* Plotlyチャート高さ調整 */
        .js-plotly-plot {
            max-height: 400px;
        }
    }

    /* ===== タブレット小 / 大型スマホ (768px以下) ===== */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem;
        }
        /* st.columns を縦積みに */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            min-width: 100% !important;
            width: 100% !important;
        }
        /* タイトルフォント縮小 */
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.25rem !important; }
        h3 { font-size: 1.1rem !important; }
        /* メトリクス縮小 */
        [data-testid="stMetricValue"] {
            font-size: 1.3rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
        }
        /* タブのフォント */
        .stTabs [data-baseweb="tab"] {
            font-size: 0.85rem !important;
            padding: 8px 12px !important;
        }
        /* Plotlyチャート高さ */
        .js-plotly-plot {
            max-height: 350px;
        }
        /* テーブルのスクロール */
        [data-testid="stDataFrame"] {
            overflow-x: auto !important;
        }
    }

    /* ===== スマホ (480px以下) ===== */
    @media (max-width: 480px) {
        .main .block-container {
            padding: 0.5rem 0.25rem;
        }
        h1 { font-size: 1.25rem !important; }
        h2 { font-size: 1.1rem !important; }
        h3 { font-size: 1rem !important; }
        [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
        }
        /* タブを横スクロール可能に */
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem !important;
            padding: 6px 8px !important;
            white-space: nowrap;
        }
        /* ボタンフルWidth */
        .stButton > button {
            width: 100% !important;
            min-height: 52px;
        }
        /* スライダーのタッチ領域さらに拡大 */
        .stSlider [role="slider"] {
            width: 28px !important;
            height: 28px !important;
        }
        /* Plotlyチャート高さ */
        .js-plotly-plot {
            max-height: 300px;
        }
        /* expander */
        [data-testid="stExpander"] summary {
            font-size: 0.9rem !important;
        }
        /* number input */
        .stNumberInput input {
            font-size: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)
