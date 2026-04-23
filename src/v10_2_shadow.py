"""V10.2 shadow predictor (X3 variants)

V10本番には一切影響しない。失敗時は例外をraise、呼び出し側でtry/except想定。

使い方:
    predictor = V10_2_Predictor('v10_2_lr_hi')
    probs_1st, probs_2nd, probs_3rd = predictor.predict(race_data, boats_data)
    # -> cal_prob 適用済み。MC simulation 前の NN出力。
"""
import os
import pickle
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 変種のモデル格納ディレクトリ
VARIANTS_DIR = Path(__file__).parent.parent / "analysis" / "models_v11" / "v10_2_variants"


class V10_2_Predictor:
    """V10.2 variant predictor (NN + calibrator)。V10と独立でロード。"""

    def __init__(self, variant_name):
        """
        Args:
            variant_name: 'v10_2_lr_hi' / 'v10_2_gamma3' 等
        """
        import torch
        from src.models import BoatraceMultiTaskModel
        from src.features import FeatureEngineer

        vdir = VARIANTS_DIR / variant_name
        if not vdir.exists():
            raise FileNotFoundError(f"V10.2 variant not found: {vdir}")

        self.variant_name = variant_name
        self.fe = FeatureEngineer()

        # Model
        state = torch.load(vdir / 'model.pth', map_location='cpu',
                           weights_only=False)
        self.model = BoatraceMultiTaskModel(
            input_dim=state['input_dim'],
            hidden_dims=state['hidden_dims'],
            num_boats=state['num_boats'],
            dropout=state['dropout'])
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()

        # Scaler
        with open(vdir / 'feature_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        # Calibrators (Isotonic Regression, 3ポジション × 6クラス)
        with open(vdir / 'calibrators.pkl', 'rb') as f:
            self.calibrators = pickle.load(f)

        self._torch = torch
        logger.info(f"V10.2 shadow predictor loaded: {variant_name}")

    def _apply_iso(self, probs, iso_list):
        out = np.array([float(iso_list[i].transform([probs[i]])[0])
                        for i in range(len(probs))])
        s = out.sum()
        return (out / s).tolist() if s > 0 else out.tolist()

    def predict(self, race_data, boats_data):
        """NN predict + calibrator apply

        Args:
            race_data: dict (venue_id, month, wind_speed, ...)
            boats_data: list[dict] 6艇

        Returns:
            (probs_1st, probs_2nd, probs_3rd) each list[6]
            または None (特徴量生成失敗時)
        """
        import torch.nn.functional as F
        try:
            features = self.fe.transform(race_data, boats_data)
        except Exception as e:
            logger.warning(f"[{self.variant_name}] feature transform失敗: {e}")
            return None

        features = self.scaler.transform(features.reshape(1, -1))
        X = self._torch.FloatTensor(features)
        with self._torch.no_grad():
            out = self.model(X)
        p1_raw = F.softmax(out[0], dim=1).numpy()[0]
        p2_raw = F.softmax(out[1], dim=1).numpy()[0]
        p3_raw = F.softmax(out[2], dim=1).numpy()[0]
        p1 = self._apply_iso(p1_raw, self.calibrators['1st'])
        p2 = self._apply_iso(p2_raw, self.calibrators['2nd'])
        p3 = self._apply_iso(p3_raw, self.calibrators['3rd'])
        return p1, p2, p3
