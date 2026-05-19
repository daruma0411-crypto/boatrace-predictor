"""V11 (VAR-13) Predictor — Venue-Adapted Recipe 13 venues

v11_var13 strategy 用の predictor。venue 別 best approach (recipe blend / pool /
specialist) で probs_1st を生成。V10 NN baseline からの拡張 (V11 = V10 + VAR)。

Functional 13 venues: venue 別 best approach
それ以外 11 venues: V10 baseline そのまま

QMC pipeline (compute_ratings_early / qmc_sanrentan_v3) は **完全に従来通り**。
変更箇所は probs_1st の計算ロジックのみ。
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import lightgbm as lgb

from src.predictor import RealtimePredictor
from src.features import FeatureEngineer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
V11_CONFIG_PATH = ROOT / 'models' / 'v11_var13_config.json'
# 旧名 fallback (移行期)
SHADOW_CONFIG_PATH = ROOT / 'models' / 'shadow_config.json'


class V11VAR13Predictor(RealtimePredictor):
    """venue 別 best approach で probs_1st を生成する predictor.

    - V10 NN + Isotonic calibrator (継承、2着3着で使用)
    - venue 別 best approach (1着のみ override)
    """

    def __init__(self, model_path='models/boatrace_model.pth',
                 config_path=None):
        super().__init__(model_path)
        if config_path is None:
            # v11_var13_config.json 優先、なければ shadow_config.json
            config_path = str(V11_CONFIG_PATH if V11_CONFIG_PATH.exists() else SHADOW_CONFIG_PATH)

        with open(config_path, encoding='utf-8') as f:
            self.shadow_config = json.load(f)

        self.functional_venues = set(self.shadow_config['functional_venues'])
        self.venue_strategies = {int(k): v for k, v in self.shadow_config['venue_strategies'].items()}
        self.venue_distances = {int(k): v for k, v in self.shadow_config['venue_distances'].items()}

        # 76dim specialists 全 24 load
        spec_76_dir = ROOT / self.shadow_config['specialists_76_dir']
        self.spec_76 = {}
        for vid in range(1, 25):
            path = spec_76_dir / f'lightgbm_v{vid:02d}_1st.txt'
            if path.exists():
                self.spec_76[vid] = lgb.Booster(model_file=str(path))

        # 82dim specialists 全 24 load
        spec_82_dir = ROOT / self.shadow_config['specialists_82_dir']
        self.spec_82 = {}
        for vid in range(1, 25):
            path = spec_82_dir / f'lightgbm_v{vid:02d}_1st.txt'
            if path.exists():
                self.spec_82[vid] = lgb.Booster(model_file=str(path))

        # Pool models load
        self.pool_models = {}
        for pool_id, info in self.shadow_config['pool_models'].items():
            path = ROOT / info['file']
            if path.exists():
                self.pool_models[pool_id] = lgb.Booster(model_file=str(path))

        # Feature engineer (76dim) と scaler は親から継承
        self._shadow_fe = FeatureEngineer()

        logger.info(f"V11VAR13Predictor (Venue-Adapted Recipe) 初期化: "
                    f"specialists 76dim={len(self.spec_76)}, 82dim={len(self.spec_82)}, "
                    f"pool={len(self.pool_models)}, functional={len(self.functional_venues)}")

    def _build_features_82(self, race_data, boats_data):
        """76dim + 6dim local advantage = 82dim"""
        features = self._shadow_fe.transform(race_data, boats_data)
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features.reshape(1, -1)).flatten()
        local_adv = []
        for b in boats_data:
            lr = b.get('local_win_rate_2', 0) or 0
            gr = b.get('win_rate_2', 0) or 0
            local_adv.append((lr - gr) / 100.0)
        return np.concatenate([features, np.array(local_adv, dtype=np.float32)])

    def _features_76(self, features_82):
        return features_82[:76]

    def _predict_venue_v11(self, venue_id, features_82, v10_probs_1st):
        """venue 別 best approach で probs_1st を返す."""
        strategy = self.venue_strategies.get(venue_id)
        if strategy is None:
            return v10_probs_1st
        stype = strategy['type']
        x_82 = features_82.reshape(1, -1)
        x_76 = self._features_76(features_82).reshape(1, -1)

        if stype == 'specialist_76':
            vid = strategy['venue']
            model = self.spec_76.get(vid)
            if model:
                return model.predict(x_76, num_iteration=model.best_iteration)[0]
            return v10_probs_1st

        if stype == 'specialist_82':
            vid = strategy['venue']
            model = self.spec_82.get(vid)
            if model:
                return model.predict(x_82, num_iteration=model.best_iteration)[0]
            return v10_probs_1st

        if stype == 'pool':
            model = self.pool_models.get(strategy['pool_id'])
            if model:
                return model.predict(x_76, num_iteration=model.best_iteration)[0]
            return v10_probs_1st

        if stype == 'recipe_v10_own':
            # V10 × v10_weight + own specialist × own_weight
            v10_w = strategy['v10_weight']
            own_w = strategy['own_weight']
            target = venue_id
            own_model = self.spec_76.get(target)
            if not own_model:
                return v10_probs_1st
            own_probs = own_model.predict(x_76, num_iteration=own_model.best_iteration)[0]
            return v10_w * v10_probs_1st + own_w * own_probs

        if stype == 'recipe_top_K_sim':
            # target + top-K 類似 venue specialists の平均
            target = strategy['target']
            K = strategy['K']
            sim_venues = [d['venue_id'] for d in self.venue_distances.get(target, [])[:K]]
            members = [target] + sim_venues
            probs = np.zeros(6)
            n = 0
            for v in members:
                m = self.spec_76.get(v)
                if m:
                    probs += m.predict(x_76, num_iteration=m.best_iteration)[0]
                    n += 1
            if n == 0:
                return v10_probs_1st
            return probs / n

        if stype == 'recipe_75_sub':
            # 引き算 recipe: target + top-K sim - sub_alpha × opp3
            target = strategy['target']
            K = strategy['K']
            own_w = strategy['own_w']
            sub_alpha = strategy['sub_alpha']
            sim_venues = [d['venue_id'] for d in self.venue_distances.get(target, [])[:K]]
            opp3 = sorted(self.venue_distances.get(target, []), key=lambda x: -x['distance'])[:3]
            opp_ids = [d['venue_id'] for d in opp3]
            probs = np.zeros(6)
            # own
            own_model = self.spec_76.get(target)
            if own_model:
                probs += own_w * own_model.predict(x_76, num_iteration=own_model.best_iteration)[0]
            # sim
            for v in sim_venues:
                m = self.spec_76.get(v)
                if m:
                    probs += 1.0 * m.predict(x_76, num_iteration=m.best_iteration)[0]
            # opp 引き算
            for v in opp_ids:
                m = self.spec_76.get(v)
                if m:
                    probs += (-sub_alpha / len(opp_ids)) * m.predict(x_76, num_iteration=m.best_iteration)[0]
            probs = np.clip(probs, 0.001, None)
            s = probs.sum()
            if s > 0:
                probs = probs / s
            return probs

        if stype == 'recipe_own_functional':
            # own × own_w + functional_others × 1
            target = strategy['target']
            own_w = strategy['own_w']
            others = strategy['functional_others']
            members_w = [(target, own_w)] + [(v, 1.0) for v in others]
            total_w = sum(w for _, w in members_w)
            probs = np.zeros(6)
            for v, w in members_w:
                m = self.spec_76.get(v)
                if m:
                    probs += (w / total_w) * m.predict(x_76, num_iteration=m.best_iteration)[0]
            s = probs.sum()
            if s > 0:
                probs = probs / s
            return probs

        return v10_probs_1st

    def predict(self, race_data, boats_data):
        """venue 別 best approach で 1着 probs を override、2/3着は V10 baseline 維持."""
        # まず V10 baseline 予測を取得
        baseline = super().predict(race_data, boats_data)
        v10_probs_1st = np.array(baseline['probs_1st'])

        venue_id = race_data.get('venue_id')
        if venue_id is None or venue_id not in self.functional_venues:
            # 非 functional venue: V10 baseline そのまま
            return baseline

        try:
            features_82 = self._build_features_82(race_data, boats_data)
            v11_probs_1st = self._predict_venue_v11(venue_id, features_82, v10_probs_1st)
            # 1着のみ override、2/3着は V10 baseline 維持
            return {
                'probs_1st': list(v11_probs_1st.tolist() if hasattr(v11_probs_1st, 'tolist') else v11_probs_1st),
                'probs_2nd': baseline['probs_2nd'],
                'probs_3rd': baseline['probs_3rd'],
                'prediction_time': baseline['prediction_time'],
                'v11_var13_strategy': f'venue_{venue_id}_{self.venue_strategies[venue_id]["type"]}',
            }
        except Exception as e:
            logger.warning(f"V11 predict failed for venue {venue_id}: {e}, fallback to V10")
            return baseline
