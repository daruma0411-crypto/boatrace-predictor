"""V11.5 (VAR-13) Predictor — V11 + venue 別 2着/3着 specialist

V11 (1着 specialist のみ訓練、2着/3着 は V10 baseline) の 1号艇軸偏重 (88.9%) を
解消するため、2着・3着も venue 別 specialist 化。全 boat の prob が venue 別に
上振れし、正規化後の相対比は実 1着率 (~59%) に近づく想定。

差分:
  - 76dim specialist を venue × {1st, 2nd, 3rd} = 24 × 3 個 load
  - predict 時に probs_1st / probs_2nd / probs_3rd 全てを venue specialist で override
  - 82dim, Pool は V11 のまま (1着のみ)
  - 2着/3着 specialist がない場合は V10 baseline へ fallback

QMC pipeline (compute_ratings_early / qmc_sanrentan_v3) は **完全に従来通り**。
"""
import logging
from pathlib import Path

import numpy as np
import lightgbm as lgb

from src.v11_var13_predictor import V11VAR13Predictor

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


class V115VAR13Predictor(V11VAR13Predictor):
    """V11 + venue 別 2着/3着 specialist."""

    def __init__(self, model_path='models/boatrace_model.pth', config_path=None):
        super().__init__(model_path, config_path)

        spec_76_dir = ROOT / self.shadow_config['specialists_76_dir']
        self.spec_76_2nd = {}
        self.spec_76_3rd = {}
        for vid in range(1, 25):
            p2 = spec_76_dir / f'lightgbm_v{vid:02d}_2nd.txt'
            p3 = spec_76_dir / f'lightgbm_v{vid:02d}_3rd.txt'
            if p2.exists():
                self.spec_76_2nd[vid] = lgb.Booster(model_file=str(p2))
            if p3.exists():
                self.spec_76_3rd[vid] = lgb.Booster(model_file=str(p3))

        logger.info(f"V115VAR13Predictor 初期化: "
                    f"2着 specialists={len(self.spec_76_2nd)}, "
                    f"3着 specialists={len(self.spec_76_3rd)}")

    def predict(self, race_data, boats_data):
        baseline = super().predict(race_data, boats_data)
        venue_id = race_data.get('venue_id')
        if venue_id is None:
            return baseline

        m2 = self.spec_76_2nd.get(venue_id)
        m3 = self.spec_76_3rd.get(venue_id)
        if m2 is None and m3 is None:
            return baseline

        try:
            features_82 = self._build_features_82(race_data, boats_data)
            x_76 = self._features_76(features_82).reshape(1, -1)
            result = dict(baseline)
            if m2 is not None:
                p2 = m2.predict(x_76, num_iteration=m2.best_iteration)[0]
                result['probs_2nd'] = list(p2.tolist() if hasattr(p2, 'tolist') else p2)
            if m3 is not None:
                p3 = m3.predict(x_76, num_iteration=m3.best_iteration)[0]
                result['probs_3rd'] = list(p3.tolist() if hasattr(p3, 'tolist') else p3)
            result['v11_5_specialists'] = (
                f'2nd={"v"+str(venue_id) if m2 is not None else "v10"},'
                f'3rd={"v"+str(venue_id) if m3 is not None else "v10"}'
            )
            return result
        except Exception as e:
            logger.warning(f"V11.5 predict failed for venue {venue_id}: {e}, fallback to V11")
            return baseline
