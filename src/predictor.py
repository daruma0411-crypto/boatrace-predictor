"""リアルタイム予測エンジン

v6: StandardScaler対応
    - models/feature_scaler.pkl が存在すれば自動で正規化を適用
    - 学習時と推論時で同一のスケーラーを使用（スケール不統一問題を解消）

v5: 特徴量選別モデル対応
    - feature_mask_208.npy + input_dim=23 → FeatureEngineerLegacy(208) + mask → 23次元
    - input_dim > 43 → FeatureEngineerLegacy (v2互換)
    - input_dim <= 43 → FeatureEngineer (v3)
"""
import json
import logging
import os
import pickle
import numpy as np
import torch
# メモリ節約: PyTorchの内部スレッド数を制限（デフォルトはCPUコア数）
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from src.models import load_model
from src.features import FeatureEngineer, FeatureEngineerLegacy
from src.database import get_db_connection
from utils.timezone import now_jst

logger = logging.getLogger(__name__)

# アンサンブル用モデルパス一覧
ENSEMBLE_MODEL_PATHS = [
    'models/boatrace_model.pth',
    'models/boatrace_model_s05.pth',
    'models/boatrace_model_s07.pth',
    'models/boatrace_model_s085.pth',
]

# 特徴量マスク (208→N次元選別)
FEATURE_MASK_PATH = 'models/feature_mask_208.npy'

# StandardScaler (v5学習で保存)
FEATURE_SCALER_PATH = 'models/feature_scaler.pkl'


def _load_feature_scaler():
    """StandardScalerをロード。ファイルなければ None"""
    if os.path.exists(FEATURE_SCALER_PATH):
        try:
            with open(FEATURE_SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            logger.info(f"StandardScalerロード: {FEATURE_SCALER_PATH}")
            return scaler
        except Exception as e:
            logger.warning(f"StandardScalerロード失敗: {e}")
    return None


def _load_feature_mask():
    """特徴量マスクをロード。ファイルなければ None"""
    if os.path.exists(FEATURE_MASK_PATH):
        mask = np.load(FEATURE_MASK_PATH)
        return mask
    return None


def _get_model_input_dim(model):
    """モデルの入力次元を取得"""
    try:
        return model.shared[0].in_features
    except Exception:
        return getattr(model, 'input_dim', 208)


def _apply_mask(features, mask):
    """208次元特徴量にマスクを適用してN次元に絞る"""
    if mask is not None and len(features) == len(mask):
        return features[mask]
    return features


class RealtimePredictor:
    """リアルタイム予測: 特徴量生成→正規化→モデル推論→結果保存"""

    def __init__(self, model_path='models/boatrace_model.pth'):
        self.model_path = model_path
        self.model = None
        self.feature_engineer = None
        self.feature_mask = None
        self.feature_scaler = None
        self.device = torch.device('cpu')

    def _ensure_model(self):
        """モデルをロード（未ロード時）。特徴量マスク・スケーラーも確定"""
        if self.model is not None:
            return

        self.model = load_model(self.model_path, self.device)
        input_dim = _get_model_input_dim(self.model)
        mask = _load_feature_mask()

        # V5判定: マスクが存在し、モデル入力次元 == マスク選別後の次元
        if mask is not None and input_dim == int(mask.sum()):
            self.feature_engineer = FeatureEngineerLegacy()
            self.feature_mask = mask
            logger.info(
                f"V5 (23dim) model loaded and features masked successfully "
                f"[FeatureEngineerLegacy(208) → mask → {input_dim}dim]"
            )
        elif input_dim <= FeatureEngineer.TOTAL_DIM:
            self.feature_engineer = FeatureEngineer()
            self.feature_mask = None
            logger.info(f"V4 model loaded [FeatureEngineer({input_dim}dim)]")
        else:
            self.feature_engineer = FeatureEngineerLegacy()
            self.feature_mask = None
            logger.info(f"V2互換 model loaded [FeatureEngineerLegacy({input_dim}dim)]")

        # StandardScaler（v5学習で保存されていれば適用）
        self.feature_scaler = _load_feature_scaler()

    def predict(self, race_data, boats_data):
        """特徴量生成→(マスク適用)→(StandardScaler正規化)→PyTorchモデル推論→確率を返却"""
        self._ensure_model()

        features = self.feature_engineer.transform(race_data, boats_data)
        features = _apply_mask(features, self.feature_mask)

        # StandardScaler正規化（スケーラーが存在すれば適用）
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features.reshape(1, -1)).flatten()

        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out_1st, out_2nd, out_3rd = self.model(x)

        probs_1st = torch.softmax(out_1st, dim=1).squeeze().numpy()
        probs_2nd = torch.softmax(out_2nd, dim=1).squeeze().numpy()
        probs_3rd = torch.softmax(out_3rd, dim=1).squeeze().numpy()

        return {
            'probs_1st': probs_1st.tolist(),
            'probs_2nd': probs_2nd.tolist(),
            'probs_3rd': probs_3rd.tolist(),
            'prediction_time': now_jst().isoformat(),
        }

    def save_prediction(self, race_id, prediction_result,
                         recommended_bets=None, model_version='v1.0',
                         strategy_type='conservative'):
        """予測結果をPostgreSQLに保存"""
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO predictions
                (race_id, probabilities_1st, probabilities_2nd,
                 probabilities_3rd, recommended_bets,
                 model_version, strategy_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                race_id,
                json.dumps(prediction_result['probs_1st']),
                json.dumps(prediction_result['probs_2nd']),
                json.dumps(prediction_result['probs_3rd']),
                json.dumps(recommended_bets) if recommended_bets else None,
                model_version,
                strategy_type,
            ))
            prediction_id = cur.fetchone()['id']
            logger.info(
                f"予測保存: id={prediction_id}, race_id={race_id}, "
                f"strategy={strategy_type}"
            )
            return prediction_id

    def _get_pre_race_data(self, race_id):
        """DBから選手・モーターデータを取得"""
        with get_db_connection() as conn:
            cur = conn.cursor()

            cur.execute(
                "SELECT * FROM races WHERE id = %s", (race_id,)
            )
            race = cur.fetchone()
            if not race:
                logger.error(f"レースが見つかりません: race_id={race_id}")
                return None, None

            cur.execute(
                "SELECT * FROM boats WHERE race_id = %s ORDER BY boat_number",
                (race_id,)
            )
            boats = cur.fetchall()

            race_data = {
                'venue_id': race['venue_id'],
                'month': race['race_date'].month,
                'distance': 1800,
                'wind_speed': race.get('wind_speed') or 0,
                'wind_direction': race.get('wind_direction') or 'calm',
                'temperature': race.get('temperature') or 20,
                'wave_height': race.get('wave_height') or 0,
                'water_temperature': race.get('water_temperature') or 20,
            }

            boats_data = []
            for b in boats:
                boats_data.append({
                    'boat_number': b['boat_number'],
                    'player_class': b['player_class'],
                    'win_rate': b['win_rate'],
                    'win_rate_2': b['win_rate_2'],
                    'win_rate_3': b['win_rate_3'],
                    'local_win_rate': b['local_win_rate'],
                    'local_win_rate_2': b['local_win_rate_2'],
                    'avg_st': b['avg_st'],
                    'motor_win_rate_2': b['motor_win_rate_2'],
                    'motor_win_rate_3': b['motor_win_rate_3'],
                    'boat_win_rate_2': b['boat_win_rate_2'],
                    'weight': b['weight'],
                    'exhibition_time': b['exhibition_time'],
                    'approach_course': b['approach_course'],
                    'is_new_motor': b['is_new_motor'],
                    'tilt': b.get('tilt'),
                    'parts_changed': b.get('parts_changed', False),
                    'fallback_flag': False,
                })

            return race_data, boats_data


class EnsemblePredictor:
    """4モデルアンサンブル予測: 特徴量計算1回、推論だけ各モデルで実行"""

    def __init__(self, model_paths=None, shared_predictor=None):
        self.model_paths = model_paths or ENSEMBLE_MODEL_PATHS
        self.models = {}          # path → model
        self.feature_mask = None  # 全モデル共通マスク
        self.feature_scaler = None  # StandardScaler
        self._fe = None           # 全モデル共通FeatureEngineer
        self._initialized = False
        self.device = torch.device('cpu')
        self._shared_predictor = shared_predictor  # RealtimePredictorのモデルを共有

    def _ensure_models(self):
        """全モデルを遅延ロード + マスク確定"""
        if self._initialized:
            return

        mask = _load_feature_mask()

        for path in self.model_paths:
            if not os.path.exists(path):
                logger.warning(f"アンサンブルモデル未発見: {path}")
                continue
            # RealtimePredictorと同じモデルパスなら共有（2重ロード防止）
            if (self._shared_predictor is not None
                    and self._shared_predictor.model is not None
                    and path == self._shared_predictor.model_path):
                self.models[path] = self._shared_predictor.model
                logger.info(f"アンサンブルモデル共有: {path} (RealtimePredictorと共有)")
                continue
            try:
                model = load_model(path, self.device)
                self.models[path] = model
                logger.info(f"アンサンブルモデルロード: {path}")
            except Exception as e:
                logger.warning(f"アンサンブルモデルロード失敗: {path}: {e}")

        if not self.models:
            logger.warning("アンサンブル: ロード済みモデルなし")
            self._initialized = True
            return

        # 最初のモデルで FE + mask を確定（全モデル同じ次元前提）
        first_model = next(iter(self.models.values()))
        input_dim = _get_model_input_dim(first_model)

        if mask is not None and input_dim == int(mask.sum()):
            self._fe = FeatureEngineerLegacy()
            self.feature_mask = mask
            logger.info(
                f"V5 (23dim) ensemble loaded and features masked successfully "
                f"[FeatureEngineerLegacy(208) → mask → {input_dim}dim, "
                f"{len(self.models)} models]"
            )
        elif input_dim <= FeatureEngineer.TOTAL_DIM:
            self._fe = FeatureEngineer()
            self.feature_mask = None
            logger.info(
                f"V4 ensemble loaded [FeatureEngineer({input_dim}dim), "
                f"{len(self.models)} models]"
            )
        else:
            self._fe = FeatureEngineerLegacy()
            self.feature_mask = None
            logger.info(
                f"V2互換 ensemble loaded [FeatureEngineerLegacy({input_dim}dim), "
                f"{len(self.models)} models]"
            )

        # StandardScaler（v5学習で保存されていれば適用）
        self.feature_scaler = _load_feature_scaler()

        self._initialized = True

    def predict_all(self, race_data, boats_data):
        """全モデルで推論し、結果リストを返す

        Returns:
            list of dict: [{probs_1st, probs_2nd, probs_3rd, model_path}, ...]
        """
        self._ensure_models()

        if not self.models:
            return []

        # 特徴量計算は1回だけ（全モデル共通）
        features = self._fe.transform(race_data, boats_data)
        features = _apply_mask(features, self.feature_mask)

        # StandardScaler正規化（スケーラーが存在すれば適用）
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features.reshape(1, -1)).flatten()

        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        results = []
        for path, model in self.models.items():
            with torch.no_grad():
                out_1st, out_2nd, out_3rd = model(x)

            probs_1st = torch.softmax(out_1st, dim=1).squeeze().numpy()
            probs_2nd = torch.softmax(out_2nd, dim=1).squeeze().numpy()
            probs_3rd = torch.softmax(out_3rd, dim=1).squeeze().numpy()

            results.append({
                'probs_1st': probs_1st.tolist(),
                'probs_2nd': probs_2nd.tolist(),
                'probs_3rd': probs_3rd.tolist(),
                'model_path': path,
            })

        return results
