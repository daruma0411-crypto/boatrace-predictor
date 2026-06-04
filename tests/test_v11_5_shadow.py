"""V11.5 shadow 実装の smoke test

- V115VAR13Predictor が正常にロードできる
- ACTIVE_STRATEGIES に登録されている
- betting_config.json に entry がある + predict_router 正しい
- dashboard STRATEGY_RESET_DATES に登録されている
"""
import pytest


def test_v115_predictor_loads():
    """V115VAR13Predictor が specialist 72 個 + V10 baseline をロードできる"""
    from src.v11_5_var13_predictor import V115VAR13Predictor
    p = V115VAR13Predictor()
    # 2着/3着 specialist が venue 別に load 済みであること
    assert len(p.spec_76_2nd) >= 20, f"2着 specialist 不足: {len(p.spec_76_2nd)}"
    assert len(p.spec_76_3rd) >= 20, f"3着 specialist 不足: {len(p.spec_76_3rd)}"


def test_active_strategies_contains_v11_5():
    """ACTIVE_STRATEGIES に v11_5_var13 が登録されている"""
    from src.betting import ACTIVE_STRATEGIES
    assert 'v11_5_var13' in ACTIVE_STRATEGIES
    assert 'v11_var13' in ACTIVE_STRATEGIES  # 本番 V11 も維持


def test_betting_config_has_v11_5():
    """betting_config.json に v11_5_var13 entry がある + predict_router 正しい"""
    import json
    from pathlib import Path
    cfg_path = Path(__file__).parent.parent / 'config' / 'betting_config.json'
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    assert 'v11_5_var13' in cfg['strategies']
    s = cfg['strategies']['v11_5_var13']
    assert s['predict_router'] == 'v11_5_var13'
    assert s['mc_version'] == 4
    assert s['include_venues'] == [1, 2, 3, 4, 7, 10, 12, 13, 14, 16, 22, 23, 24]


def test_strategy_reset_dates_has_v11_5():
    """STRATEGY_RESET_DATES に v11_5_var13 (2026-06-04) 登録"""
    from streamlit_app.components.db_utils import STRATEGY_RESET_DATES
    assert STRATEGY_RESET_DATES.get('v11_5_var13') == '2026-06-04'
