"""特徴量エンジニアリング (208次元)

v2: 直前情報追加 — 波高・水温(global) + チルト・部品交換(per-boat)
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

VENUE_COUNT = 24
WIND_DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'calm']
PLAYER_CLASSES = ['A1', 'A2', 'B1', 'B2']


class FeatureEngineer:
    """レースデータを208次元特徴量に変換

    グローバル16次元 + 艇別32次元×6艇 = 208次元

    v1 (194次元) からの追加:
      global: +波高(1) +水温(1) = +2
      per-boat: +チルト(1) +部品交換フラグ(1) = +2
    """

    GLOBAL_DIM = 16
    PER_BOAT_DIM = 32
    NUM_BOATS = 6
    TOTAL_DIM = GLOBAL_DIM + PER_BOAT_DIM * NUM_BOATS  # 208

    def transform(self, race_data, boats_data):
        """レースデータと艇データから特徴量ベクトルを生成"""
        global_features = self._extract_global(race_data)
        boat_features = []

        for i in range(self.NUM_BOATS):
            if i < len(boats_data):
                bf = self._extract_boat(boats_data[i], boats_data)
            else:
                bf = np.zeros(self.PER_BOAT_DIM)
            boat_features.append(bf)

        features = np.concatenate([global_features] + boat_features)
        features = self._clean_features(features)

        assert len(features) == self.TOTAL_DIM, \
            f"特徴量次元不一致: {len(features)} != {self.TOTAL_DIM}"
        return features

    def _extract_global(self, race_data):
        """グローバル特徴量 (16次元)

        場ID(1) + 月(1) + 距離(1) + 風速(1) + 風向OneHot(9) + 気温(1)
        + 波高(1) + 水温(1) = 16
        """
        features = []

        venue_id = race_data.get('venue_id', 0)
        features.append(venue_id / VENUE_COUNT)

        month = race_data.get('month', 6)
        features.append(month / 12.0)

        distance = race_data.get('distance', 1800)
        features.append(distance / 1800.0)

        wind_speed = race_data.get('wind_speed', 0)
        features.append(wind_speed / 10.0)

        wind_dir = race_data.get('wind_direction', 'calm')
        wind_onehot = [0.0] * len(WIND_DIRECTIONS)
        if wind_dir in WIND_DIRECTIONS:
            wind_onehot[WIND_DIRECTIONS.index(wind_dir)] = 1.0
        features.extend(wind_onehot)

        temperature = race_data.get('temperature', 20)
        features.append(temperature / 40.0)

        # v2: 波高 (cm → 正規化: /20)
        wave_height = race_data.get('wave_height', 0) or 0
        features.append(wave_height / 20.0)

        # v2: 水温 (℃ → 正規化: /40)
        water_temperature = race_data.get('water_temperature', 20) or 20
        features.append(water_temperature / 40.0)

        return np.array(features, dtype=np.float32)

    def _extract_boat(self, boat_data, all_boats):
        """艇別特徴量 (32次元)

        級別OneHot(4) + 全国勝率相対(1) + 全国2連率(1) + 全国3連率(1) +
        当地勝率(1) + 当地2連率(1) + 平均ST(1) + 内側平均ST差(1) +
        モーター2連率(1) + モーター3連率(1) + ボート2連率(1) +
        新モーターフラグ(1) + 体重差分(1) + 展示タイム差分(1) +
        進入コースOneHot(6) + フォールバックフラグ(1) + 枠番OneHot(6)
        + チルト(1) + 部品交換フラグ(1) = 32
        """
        features = []

        # 級別 One-Hot (4)
        player_class = boat_data.get('player_class', 'B1')
        class_onehot = [0.0] * len(PLAYER_CLASSES)
        if player_class in PLAYER_CLASSES:
            class_onehot[PLAYER_CLASSES.index(player_class)] = 1.0
        features.extend(class_onehot)

        # 全国勝率（相対: レース平均との差）
        win_rate = boat_data.get('win_rate', 0.0) or 0.0
        avg_win_rate = np.mean(
            [b.get('win_rate', 0.0) or 0.0 for b in all_boats]
        ) if all_boats else 0.0
        features.append(win_rate - avg_win_rate)

        # 全国2連率
        features.append(boat_data.get('win_rate_2', 0.0) or 0.0)

        # 全国3連率
        features.append(boat_data.get('win_rate_3', 0.0) or 0.0)

        # 当地勝率
        features.append(boat_data.get('local_win_rate', 0.0) or 0.0)

        # 当地2連率
        features.append(boat_data.get('local_win_rate_2', 0.0) or 0.0)

        # 平均ST
        boat_avg_st = boat_data.get('avg_st', 0.0) or 0.0
        features.append(boat_avg_st)

        # 内側全艇平均ST差
        boat_number = boat_data.get('boat_number', 1)
        inner_sts = [
            b.get('avg_st', 0.0) or 0.0
            for b in all_boats
            if (b.get('boat_number', 0) or 0) < boat_number
        ]
        if inner_sts:
            features.append(boat_avg_st - np.mean(inner_sts))
        else:
            features.append(0.0)

        # モーター2連率
        features.append(boat_data.get('motor_win_rate_2', 0.0) or 0.0)

        # モーター3連率
        features.append(boat_data.get('motor_win_rate_3', 0.0) or 0.0)

        # ボート2連率
        features.append(boat_data.get('boat_win_rate_2', 0.0) or 0.0)

        # 新モーターフラグ
        features.append(1.0 if boat_data.get('is_new_motor', False) else 0.0)

        # 体重差分（平均との差、標準偏差で割らない）
        weight = boat_data.get('weight', 52.0) or 52.0
        avg_weight = np.mean(
            [b.get('weight', 52.0) or 52.0 for b in all_boats]
        ) if all_boats else 52.0
        features.append(weight - avg_weight)

        # 展示タイム差分（平均との差、標準偏差で割らない）
        ex_time = boat_data.get('exhibition_time', 0.0) or 0.0
        avg_ex_time = np.mean(
            [b.get('exhibition_time', 0.0) or 0.0 for b in all_boats]
        ) if all_boats else 0.0
        features.append(ex_time - avg_ex_time)

        # 進入コース One-Hot (6)
        approach = boat_data.get('approach_course', boat_number)
        approach_onehot = [0.0] * 6
        if approach and 1 <= approach <= 6:
            approach_onehot[approach - 1] = 1.0
        features.extend(approach_onehot)

        # フォールバックフラグ（進入コース欠損で枠なり仮定）
        features.append(
            1.0 if boat_data.get('fallback_flag', False) else 0.0
        )

        # 枠番 One-Hot (6)
        boat_num_onehot = [0.0] * 6
        if 1 <= boat_number <= 6:
            boat_num_onehot[boat_number - 1] = 1.0
        features.extend(boat_num_onehot)

        # v2: チルト (度 → 正規化: /3.0, 通常 -0.5〜3.0)
        tilt = boat_data.get('tilt', 0.0) or 0.0
        features.append(tilt / 3.0)

        # v2: 部品交換フラグ
        features.append(
            1.0 if boat_data.get('parts_changed', False) else 0.0
        )

        return np.array(features, dtype=np.float32)

    def _clean_features(self, features):
        """NaN→0、Inf→clip"""
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        features = np.clip(features, -100.0, 100.0)
        return features
