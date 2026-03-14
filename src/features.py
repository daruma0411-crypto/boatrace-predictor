"""特徴量エンジニアリング

v3: 重要度分析に基づくFeature Selection — 208次元→42次元
    削除対象: Variance=0（データ未取得）+ |importance| < 0.001 の175次元
    残留: 選手成績/モーター/ボート成績 + venue_id + weight

    将来データ取得バグ修正後に気象・展示タイムを復活させる想定で、
    FeatureEngineerLegacy（208次元）も残す。

v2 (legacy): 直前情報追加 — 波高・水温(global) + チルト・部品交換(per-boat) = 208次元
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

VENUE_COUNT = 24
WIND_DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'calm']
PLAYER_CLASSES = ['A1', 'A2', 'B1', 'B2']


class FeatureEngineer:
    """v3: 重要度分析済み42次元 (グローバル1 + 艇別7×6 = 43 → 丸めて42)

    重要度分析結果（Permutation Importance, log_loss）:
      有効カテゴリ:
        1. win_rate_2   (全国2連率)         avg_imp=0.0107  ← 最重要
        2. win_rate_3   (全国3連率)         avg_imp=0.0126  ← 最重要
        3. local_win_rate_2 (当地2連率)     avg_imp=0.0068  ← 重要
        4. motor_win_rate_2 (モーター2連率) avg_imp=0.0020
        5. motor_win_rate_3 (モーター3連率) avg_imp=0.0017
        6. boat_win_rate_2  (ボート2連率)   avg_imp=0.0016
        7. venue_id                         avg_imp=0.0001  ← 微小だが場の特性反映

      削除（Variance=0 or imp≈0）:
        - 風速/風向/気温/波高/水温 (DB未取得)
        - 展示タイム/チルト/部品交換/新モーター (DB未取得)
        - 進入コースOneHot/フォールバックフラグ (定数)
        - 枠番OneHot (定数)
        - 級別OneHot (imp極小、勝率系で代替済み)
        - win_rate_rel (imp負)
        - local_win_rate (imp微小)
        - avg_st / inner_st_diff (imp微小)
        - weight_diff (imp微小: 0.0002)
        - month / distance (imp≈0)
    """

    GLOBAL_DIM = 1       # venue_id のみ
    PER_BOAT_DIM = 7     # 6つの成績指標 + weight_diff
    NUM_BOATS = 6
    TOTAL_DIM = GLOBAL_DIM + PER_BOAT_DIM * NUM_BOATS  # 43

    # 特徴量名（分析・デバッグ用）
    FEATURE_NAMES = None

    @classmethod
    def get_feature_names(cls):
        if cls.FEATURE_NAMES is None:
            names = ['G_venue_id']
            for b in range(1, 7):
                prefix = f'B{b}'
                names.extend([
                    f'{prefix}_win_rate_2',
                    f'{prefix}_win_rate_3',
                    f'{prefix}_local_win_rate_2',
                    f'{prefix}_motor_win_rate_2',
                    f'{prefix}_motor_win_rate_3',
                    f'{prefix}_boat_win_rate_2',
                    f'{prefix}_weight_diff',
                ])
            cls.FEATURE_NAMES = names
        return cls.FEATURE_NAMES

    def transform(self, race_data, boats_data):
        """レースデータと艇データから特徴量ベクトルを生成 (43次元)

        Raises:
            ValueError: 3艇以上の主要データが欠損している場合
        """
        global_features = self._extract_global(race_data)
        boat_features = []
        missing_count = 0

        for i in range(self.NUM_BOATS):
            if i < len(boats_data):
                boat = boats_data[i]
                key_fields = ['win_rate', 'win_rate_2', 'motor_win_rate_2']
                if all(not boat.get(f) for f in key_fields):
                    missing_count += 1
                bf = self._extract_boat(boat, boats_data)
            else:
                missing_count += 1
                bf = np.zeros(self.PER_BOAT_DIM)
            boat_features.append(bf)

        if missing_count >= 3:
            raise ValueError(
                f"データ欠損率過大: {missing_count}/6艇の主要データなし"
            )

        features = np.concatenate([global_features] + boat_features)
        features = self._clean_features(features)

        assert len(features) == self.TOTAL_DIM, \
            f"特徴量次元不一致: {len(features)} != {self.TOTAL_DIM}"
        return features

    def _extract_global(self, race_data):
        """グローバル特徴量 (1次元): venue_id のみ"""
        venue_id = race_data.get('venue_id', 0)
        return np.array([venue_id / VENUE_COUNT], dtype=np.float32)

    def _extract_boat(self, boat_data, all_boats):
        """艇別特徴量 (7次元)

        全国2連率(1) + 全国3連率(1) + 当地2連率(1) +
        モーター2連率(1) + モーター3連率(1) + ボート2連率(1) +
        体重差分(1) = 7
        """
        features = []

        # 全国2連率
        features.append(boat_data.get('win_rate_2', 0.0) or 0.0)

        # 全国3連率
        features.append(boat_data.get('win_rate_3', 0.0) or 0.0)

        # 当地2連率
        features.append(boat_data.get('local_win_rate_2', 0.0) or 0.0)

        # モーター2連率
        features.append(boat_data.get('motor_win_rate_2', 0.0) or 0.0)

        # モーター3連率
        features.append(boat_data.get('motor_win_rate_3', 0.0) or 0.0)

        # ボート2連率
        features.append(boat_data.get('boat_win_rate_2', 0.0) or 0.0)

        # 体重差分（平均との差）
        weight = boat_data.get('weight', 52.0) or 52.0
        avg_weight = np.mean(
            [b.get('weight', 52.0) or 52.0 for b in all_boats]
        ) if all_boats else 52.0
        features.append(weight - avg_weight)

        return np.array(features, dtype=np.float32)

    def _clean_features(self, features):
        """NaN→0、Inf→clip"""
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        features = np.clip(features, -100.0, 100.0)
        return features


class FeatureEngineerLegacy:
    """v2互換: 208次元（旧モデル読み込み用）

    グローバル16次元 + 艇別32次元×6艇 = 208次元
    """

    GLOBAL_DIM = 16
    PER_BOAT_DIM = 32
    NUM_BOATS = 6
    TOTAL_DIM = GLOBAL_DIM + PER_BOAT_DIM * NUM_BOATS  # 208

    def transform(self, race_data, boats_data):
        global_features = self._extract_global(race_data)
        boat_features = []
        missing_count = 0

        for i in range(self.NUM_BOATS):
            if i < len(boats_data):
                boat = boats_data[i]
                key_fields = ['win_rate', 'win_rate_2', 'motor_win_rate_2']
                if all(not boat.get(f) for f in key_fields):
                    missing_count += 1
                bf = self._extract_boat(boat, boats_data)
            else:
                missing_count += 1
                bf = np.zeros(self.PER_BOAT_DIM)
            boat_features.append(bf)

        if missing_count >= 3:
            raise ValueError(
                f"データ欠損率過大: {missing_count}/6艇の主要データなし"
            )

        features = np.concatenate([global_features] + boat_features)
        features = self._clean_features(features)

        assert len(features) == self.TOTAL_DIM, \
            f"特徴量次元不一致: {len(features)} != {self.TOTAL_DIM}"
        return features

    def _extract_global(self, race_data):
        features = []
        features.append(race_data.get('venue_id', 0) / VENUE_COUNT)
        features.append(race_data.get('month', 6) / 12.0)
        features.append(race_data.get('distance', 1800) / 1800.0)
        features.append(race_data.get('wind_speed', 0) / 10.0)
        wind_dir = race_data.get('wind_direction', 'calm')
        wind_onehot = [0.0] * len(WIND_DIRECTIONS)
        if wind_dir in WIND_DIRECTIONS:
            wind_onehot[WIND_DIRECTIONS.index(wind_dir)] = 1.0
        features.extend(wind_onehot)
        features.append(race_data.get('temperature', 20) / 40.0)
        features.append((race_data.get('wave_height', 0) or 0) / 20.0)
        features.append((race_data.get('water_temperature', 20) or 20) / 40.0)
        return np.array(features, dtype=np.float32)

    def _extract_boat(self, boat_data, all_boats):
        features = []
        player_class = boat_data.get('player_class', 'B1')
        class_onehot = [0.0] * len(PLAYER_CLASSES)
        if player_class in PLAYER_CLASSES:
            class_onehot[PLAYER_CLASSES.index(player_class)] = 1.0
        features.extend(class_onehot)
        win_rate = boat_data.get('win_rate', 0.0) or 0.0
        avg_win_rate = np.mean(
            [b.get('win_rate', 0.0) or 0.0 for b in all_boats]
        ) if all_boats else 0.0
        features.append(win_rate - avg_win_rate)
        features.append(boat_data.get('win_rate_2', 0.0) or 0.0)
        features.append(boat_data.get('win_rate_3', 0.0) or 0.0)
        features.append(boat_data.get('local_win_rate', 0.0) or 0.0)
        features.append(boat_data.get('local_win_rate_2', 0.0) or 0.0)
        boat_avg_st = boat_data.get('avg_st', 0.0) or 0.0
        features.append(boat_avg_st)
        boat_number = boat_data.get('boat_number', 1)
        inner_sts = [
            b.get('avg_st', 0.0) or 0.0
            for b in all_boats
            if (b.get('boat_number', 0) or 0) < boat_number
        ]
        features.append(boat_avg_st - np.mean(inner_sts) if inner_sts else 0.0)
        features.append(boat_data.get('motor_win_rate_2', 0.0) or 0.0)
        features.append(boat_data.get('motor_win_rate_3', 0.0) or 0.0)
        features.append(boat_data.get('boat_win_rate_2', 0.0) or 0.0)
        features.append(1.0 if boat_data.get('is_new_motor', False) else 0.0)
        weight = boat_data.get('weight', 52.0) or 52.0
        avg_weight = np.mean(
            [b.get('weight', 52.0) or 52.0 for b in all_boats]
        ) if all_boats else 52.0
        features.append(weight - avg_weight)
        ex_time = boat_data.get('exhibition_time', 0.0) or 0.0
        avg_ex_time = np.mean(
            [b.get('exhibition_time', 0.0) or 0.0 for b in all_boats]
        ) if all_boats else 0.0
        features.append(ex_time - avg_ex_time)
        approach = boat_data.get('approach_course', boat_number)
        approach_onehot = [0.0] * 6
        if approach and 1 <= approach <= 6:
            approach_onehot[approach - 1] = 1.0
        features.extend(approach_onehot)
        features.append(1.0 if boat_data.get('fallback_flag', False) else 0.0)
        boat_num_onehot = [0.0] * 6
        if 1 <= boat_number <= 6:
            boat_num_onehot[boat_number - 1] = 1.0
        features.extend(boat_num_onehot)
        features.append((boat_data.get('tilt', 0.0) or 0.0) / 3.0)
        features.append(1.0 if boat_data.get('parts_changed', False) else 0.0)
        return np.array(features, dtype=np.float32)

    def _clean_features(self, features):
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        features = np.clip(features, -100.0, 100.0)
        return features
