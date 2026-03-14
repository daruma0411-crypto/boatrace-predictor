"""リアルタイム予測エンジン

v3: モデルの入力次元に応じてFeatureEngineerを自動選択
    - 43次元モデル → FeatureEngineer (v3)
    - 208次元モデル → FeatureEngineerLegacy (v2互換)
"""
import json
import logging
import os
import numpy as np
import torch
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


def _get_feature_engineer_for_model(model):
    """モデルの入力次元に応じたFeatureEngineerを返す"""
    try:
        input_dim = model.shared[0].in_features
    except Exception:
        input_dim = getattr(model, 'input_dim', 208)

    if input_dim <= FeatureEngineer.TOTAL_DIM:
        return FeatureEngineer()
    else:
        return FeatureEngineerLegacy()


class RealtimePredictor:
    """リアルタイム予測: 特徴量生成→モデル推論→結果保存"""

    def __init__(self, model_path='models/boatrace_model.pth'):
        self.model_path = model_path
        self.model = None
        self.feature_engineer = None
        self.device = torch.device('cpu')

    def _ensure_model(self):
        """モデルをロード（未ロード時）。ファイル未発見時は例外を伝搬"""
        if self.model is None:
            self.model = load_model(self.model_path, self.device)
            self.feature_engineer = _get_feature_engineer_for_model(self.model)
            fe_class = self.feature_engineer.__class__.__name__
            logger.info(f"FeatureEngineer: {fe_class} ({self.feature_engineer.TOTAL_DIM}次元)")

    def predict(self, race_data, boats_data):
        """特徴量生成→PyTorchモデル推論→確率を返却"""
        self._ensure_model()

        features = self.feature_engineer.transform(race_data, boats_data)
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

    def __init__(self, model_paths=None):
        self.model_paths = model_paths or ENSEMBLE_MODEL_PATHS
        self.models = {}  # 遅延ロード
        self.feature_engineers = {}  # モデルごとのFE
        self.device = torch.device('cpu')

    def _ensure_models(self):
        """全モデルを遅延ロード"""
        for path in self.model_paths:
            if path in self.models:
                continue
            if not os.path.exists(path):
                logger.warning(f"アンサンブルモデル未発見: {path}")
                continue
            try:
                model = load_model(path, self.device)
                self.models[path] = model
                self.feature_engineers[path] = _get_feature_engineer_for_model(model)
                logger.info(f"アンサンブルモデルロード: {path}")
            except Exception as e:
                logger.warning(f"アンサンブルモデルロード失敗: {path}: {e}")

    def predict_all(self, race_data, boats_data):
        """全モデルで推論し、結果リストを返す

        Returns:
            list of dict: [{probs_1st, probs_2nd, probs_3rd, model_path}, ...]
        """
        self._ensure_models()

        if not self.models:
            logger.warning("アンサンブル: ロード済みモデルなし")
            return []

        # 特徴量キャッシュ（同じFEクラスなら再計算不要）
        features_cache = {}

        results = []
        for path, model in self.models.items():
            fe = self.feature_engineers[path]
            fe_key = fe.__class__.__name__

            if fe_key not in features_cache:
                features_cache[fe_key] = fe.transform(race_data, boats_data)

            features = features_cache[fe_key]
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)

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
