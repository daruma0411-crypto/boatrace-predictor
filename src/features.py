"""特徴量エンジニアリング

v4: 直前情報復活 — 展示タイム/チルト/ST/気象データ追加
    v3で「DB未取得」として削除した特徴量をDBデータ充填完了に伴い復活。
    グローバル4次元 + 艇別12次元×6艇 = 76次元

v3: 重要度分析に基づくFeature Selection — 208次元→43次元（旧・使用停止）

v2 (legacy): 直前情報追加 — 波高・水温(global) + チルト・部品交換(per-boat) = 208次元
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

VENUE_COUNT = 24
WIND_DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'calm']
PLAYER_CLASSES = ['A1', 'A2', 'B1', 'B2']


class FeatureEngineerV3:
    """v3（旧）: 43次元。DB未取得時代の暫定版。後方互換用に残す。"""

    GLOBAL_DIM = 1
    PER_BOAT_DIM = 7
    NUM_BOATS = 6
    TOTAL_DIM = GLOBAL_DIM + PER_BOAT_DIM * NUM_BOATS  # 43

    FEATURE_NAMES = None

    @classmethod
    def get_feature_names(cls):
        if cls.FEATURE_NAMES is None:
            names = ['G_venue_id']
            for b in range(1, 7):
                prefix = f'B{b}'
                names.extend([
                    f'{prefix}_win_rate_2', f'{prefix}_win_rate_3',
                    f'{prefix}_local_win_rate_2',
                    f'{prefix}_motor_win_rate_2', f'{prefix}_motor_win_rate_3',
                    f'{prefix}_boat_win_rate_2', f'{prefix}_weight_diff',
                ])
            cls.FEATURE_NAMES = names
        return cls.FEATURE_NAMES

    def transform(self, race_data, boats_data):
        global_features = self._extract_global(race_data)
        boat_features = []
        missing_count = 0
        for i in range(self.NUM_BOATS):
            if i < len(boats_data):
                boat = boats_data[i]
                if all(not boat.get(f) for f in ['win_rate', 'win_rate_2', 'motor_win_rate_2']):
                    missing_count += 1
                bf = self._extract_boat(boat, boats_data)
            else:
                missing_count += 1
                bf = np.zeros(self.PER_BOAT_DIM)
            boat_features.append(bf)
        if missing_count >= 3:
            raise ValueError(f"データ欠損率過大: {missing_count}/6艇")
        features = np.concatenate([global_features] + boat_features)
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(features, -100.0, 100.0)

    def _extract_global(self, race_data):
        return np.array([race_data.get('venue_id', 0) / VENUE_COUNT], dtype=np.float32)

    def _extract_boat(self, boat_data, all_boats):
        weight = boat_data.get('weight', 52.0) or 52.0
        avg_weight = np.mean([b.get('weight', 52.0) or 52.0 for b in all_boats]) if all_boats else 52.0
        return np.array([
            boat_data.get('win_rate_2', 0.0) or 0.0,
            boat_data.get('win_rate_3', 0.0) or 0.0,
            boat_data.get('local_win_rate_2', 0.0) or 0.0,
            boat_data.get('motor_win_rate_2', 0.0) or 0.0,
            boat_data.get('motor_win_rate_3', 0.0) or 0.0,
            boat_data.get('boat_win_rate_2', 0.0) or 0.0,
            weight - avg_weight,
        ], dtype=np.float32)


class FeatureEngineer:
    """v4: 直前情報復活版 76次元

    グローバル 4次元:
      venue_id(1) + wind_speed(1) + wind_direction_sin/cos(2)

    艇別 12次元 × 6艇 = 72次元:
      成績系: win_rate_2, win_rate_3, local_win_rate_2 (3)
      機材系: motor_win_rate_2, motor_win_rate_3, boat_win_rate_2 (3)
      直前情報: exhibition_time_diff, avg_st, tilt (3)
      コース: approach_course_diff (1) — 枠番との差（前づけ検出）
      その他: weight_diff, parts_changed (2)

    合計: 4 + 72 = 76次元
    """

    GLOBAL_DIM = 4       # venue_id + wind_speed + wind_sin + wind_cos
    PER_BOAT_DIM = 12    # 成績3 + 機材3 + 直前3 + コース1 + 他2
    NUM_BOATS = 6
    TOTAL_DIM = GLOBAL_DIM + PER_BOAT_DIM * NUM_BOATS  # 76

    FEATURE_NAMES = None

    # 風向を角度に変換するマップ
    _WIND_ANGLE = {
        'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
        'S': 180, 'SW': 225, 'W': 270, 'NW': 315, 'calm': -1,
    }

    @classmethod
    def get_feature_names(cls):
        if cls.FEATURE_NAMES is None:
            names = ['G_venue_id', 'G_wind_speed', 'G_wind_sin', 'G_wind_cos']
            for b in range(1, 7):
                p = f'B{b}'
                names.extend([
                    f'{p}_win_rate_2', f'{p}_win_rate_3', f'{p}_local_win_rate_2',
                    f'{p}_motor_win_rate_2', f'{p}_motor_win_rate_3', f'{p}_boat_win_rate_2',
                    f'{p}_exhibit_time_diff', f'{p}_avg_st', f'{p}_tilt',
                    f'{p}_course_diff',
                    f'{p}_weight_diff', f'{p}_parts_changed',
                ])
            cls.FEATURE_NAMES = names
        return cls.FEATURE_NAMES

    def transform(self, race_data, boats_data):
        """レースデータと艇データから特徴量ベクトルを生成 (76次元)"""
        global_features = self._extract_global(race_data)
        boat_features = []
        missing_count = 0

        for i in range(self.NUM_BOATS):
            if i < len(boats_data):
                boat = boats_data[i]
                if all(not boat.get(f) for f in ['win_rate', 'win_rate_2', 'motor_win_rate_2']):
                    missing_count += 1
                bf = self._extract_boat(boat, boats_data)
            else:
                missing_count += 1
                bf = np.zeros(self.PER_BOAT_DIM)
            boat_features.append(bf)

        if missing_count >= 3:
            raise ValueError(f"データ欠損率過大: {missing_count}/6艇の主要データなし")

        features = np.concatenate([global_features] + boat_features)
        features = self._clean_features(features)

        assert len(features) == self.TOTAL_DIM, \
            f"特徴量次元不一致: {len(features)} != {self.TOTAL_DIM}"
        return features

    def _extract_global(self, race_data):
        """グローバル特徴量 (4次元)"""
        venue_id = race_data.get('venue_id', 0) / VENUE_COUNT
        wind_speed = (race_data.get('wind_speed', 0) or 0) / 10.0

        # 風向をsin/cosエンコード（循環特徴量）
        wind_dir = race_data.get('wind_direction', 'calm')
        angle = self._WIND_ANGLE.get(wind_dir, -1)
        if angle < 0:  # calm
            wind_sin, wind_cos = 0.0, 0.0
        else:
            rad = np.radians(angle)
            wind_sin, wind_cos = np.sin(rad), np.cos(rad)

        return np.array([venue_id, wind_speed, wind_sin, wind_cos], dtype=np.float32)

    def _extract_boat(self, boat_data, all_boats):
        """艇別特徴量 (12次元)"""
        # --- 成績系 (3) ---
        win_rate_2 = boat_data.get('win_rate_2', 0.0) or 0.0
        win_rate_3 = boat_data.get('win_rate_3', 0.0) or 0.0
        local_wr2 = boat_data.get('local_win_rate_2', 0.0) or 0.0

        # --- 機材系 (3) ---
        motor_wr2 = boat_data.get('motor_win_rate_2', 0.0) or 0.0
        motor_wr3 = boat_data.get('motor_win_rate_3', 0.0) or 0.0
        boat_wr2 = boat_data.get('boat_win_rate_2', 0.0) or 0.0

        # --- 直前情報 (3) ---
        # 展示タイム差分（平均との差。速い=負の値=有利）
        ex_time = boat_data.get('exhibition_time', 0.0) or 0.0
        avg_ex = np.mean([b.get('exhibition_time', 0.0) or 0.0 for b in all_boats]) if all_boats else 0.0
        exhibit_diff = (ex_time - avg_ex) if (ex_time > 0 and avg_ex > 0) else 0.0

        # 平均ST（小さいほど良い）
        avg_st = boat_data.get('avg_st', 0.0) or 0.0

        # チルト角（-0.5〜3.0、出力重視↔回転重視）
        tilt = (boat_data.get('tilt', 0.0) or 0.0) / 3.0

        # --- コース差分 (1) ---
        boat_number = boat_data.get('boat_number', 1) or 1
        approach = boat_data.get('approach_course', boat_number) or boat_number
        course_diff = (approach - boat_number) / 5.0  # 前づけ=負、後ろ=正

        # --- その他 (2) ---
        weight = boat_data.get('weight', 52.0) or 52.0
        avg_weight = np.mean([b.get('weight', 52.0) or 52.0 for b in all_boats]) if all_boats else 52.0
        weight_diff = weight - avg_weight

        parts = 1.0 if boat_data.get('parts_changed', False) else 0.0

        return np.array([
            win_rate_2, win_rate_3, local_wr2,
            motor_wr2, motor_wr3, boat_wr2,
            exhibit_diff, avg_st, tilt,
            course_diff,
            weight_diff, parts,
        ], dtype=np.float32)

    def _clean_features(self, features):
        """NaN→0、Inf→clip"""
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(features, -100.0, 100.0)


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
