"""Phase B 特徴量純関数の単体テスト

実行:
  python -m unittest tests.test_phase_b_features -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase_b_features import (
    classify_race_category,
    detect_planned_race,
    parse_day_label,
    compute_skill_gap,
)


class TestClassifyRaceCategory(unittest.TestCase):
    def test_qualifier(self):
        self.assertEqual(classify_race_category("予選1800m"), "qualifier")

    def test_semifinal(self):
        self.assertEqual(classify_race_category("準優1800m"), "semifinal")

    def test_final(self):
        self.assertEqual(classify_race_category("優勝戦1800m"), "final")

    def test_general(self):
        self.assertEqual(classify_race_category("一般 1800m"), "general")

    def test_other_when_no_match(self):
        self.assertEqual(classify_race_category("選抜1800m"), "other")

    def test_none_input(self):
        self.assertEqual(classify_race_category(None), "other")

    def test_empty_string(self):
        self.assertEqual(classify_race_category(""), "other")


class TestDetectPlannedRace(unittest.TestCase):
    def test_sunrise(self):
        self.assertTrue(detect_planned_race("サンライズ V"))

    def test_golden(self):
        self.assertTrue(detect_planned_race("ゴールデンカップ"))

    def test_gw_special(self):
        self.assertTrue(detect_planned_race("スポーツ報知杯争奪ゴールデンウィーク特選"))

    def test_v_premier(self):
        self.assertTrue(detect_planned_race("Ｖプレミアトーナメント"))

    def test_not_planned(self):
        self.assertFalse(detect_planned_race("第３３回多摩川さつき杯"))

    def test_none(self):
        self.assertFalse(detect_planned_race(None))


class TestParseDayLabel(unittest.TestCase):
    def test_first_day(self):
        self.assertEqual(parse_day_label("初日"), 1)

    def test_second_day_full_width(self):
        self.assertEqual(parse_day_label("２日目"), 2)

    def test_fifth_day_full_width(self):
        self.assertEqual(parse_day_label("５日目"), 5)

    def test_third_day_half_width(self):
        self.assertEqual(parse_day_label("3日目"), 3)

    def test_final_day_returns_none(self):
        self.assertIsNone(parse_day_label("最終日"))

    def test_championship_returns_none(self):
        self.assertIsNone(parse_day_label("優勝戦"))

    def test_invalid_returns_none(self):
        self.assertIsNone(parse_day_label("予選"))

    def test_none_input(self):
        self.assertIsNone(parse_day_label(None))


class TestComputeSkillGap(unittest.TestCase):
    def test_normal(self):
        boats = [
            {'boat_number': 1, 'win_rate_2': 50.0},
            {'boat_number': 2, 'win_rate_2': 40.0},
            {'boat_number': 3, 'win_rate_2': 30.0},
            {'boat_number': 4, 'win_rate_2': 35.0},
            {'boat_number': 5, 'win_rate_2': 25.0},
            {'boat_number': 6, 'win_rate_2': 20.0},
        ]
        self.assertAlmostEqual(compute_skill_gap(boats), 20.0)

    def test_negative_gap(self):
        boats = [
            {'boat_number': 1, 'win_rate_2': 10.0},
            {'boat_number': 2, 'win_rate_2': 40.0},
            {'boat_number': 3, 'win_rate_2': 40.0},
            {'boat_number': 4, 'win_rate_2': 40.0},
            {'boat_number': 5, 'win_rate_2': 40.0},
            {'boat_number': 6, 'win_rate_2': 40.0},
        ]
        self.assertAlmostEqual(compute_skill_gap(boats), -30.0)

    def test_missing_boat1_returns_none(self):
        boats = [
            {'boat_number': 2, 'win_rate_2': 40.0},
            {'boat_number': 3, 'win_rate_2': 30.0},
        ]
        self.assertIsNone(compute_skill_gap(boats))

    def test_any_null_win_rate_returns_none(self):
        boats = [
            {'boat_number': 1, 'win_rate_2': 50.0},
            {'boat_number': 2, 'win_rate_2': None},
            {'boat_number': 3, 'win_rate_2': 30.0},
            {'boat_number': 4, 'win_rate_2': 35.0},
            {'boat_number': 5, 'win_rate_2': 25.0},
            {'boat_number': 6, 'win_rate_2': 20.0},
        ]
        self.assertIsNone(compute_skill_gap(boats))

    def test_wrong_boat_count_returns_none(self):
        boats = [{'boat_number': i, 'win_rate_2': 30.0} for i in range(1, 5)]
        self.assertIsNone(compute_skill_gap(boats))


if __name__ == '__main__':
    unittest.main()
