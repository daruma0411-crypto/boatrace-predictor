"""scrape_race_title() の単体テスト (offline、fixture HTML 使用)

実行:
  pytest tests/test_scrape_race_title.py -v
  または
  python -m unittest tests.test_scrape_race_title
"""
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper import (
    _parse_title_from_html,
    _parse_subtitle_from_html,
    _parse_day_label_from_html,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"

EXPECTED_TITLE = "江戸川６３４杯\u3000モーターボート大賞"
EXPECTED_SUBTITLE = "予選1200m"  # 正規化後 (改行・全角空白除去)
# fixture は hd=20260511 / jcd=03 / rno=1 のページ。タブで 5月11日初日 → 5月16日最終日
EXPECTED_DAY_LABEL_FOR_0511 = "初日"
EXPECTED_DAY_LABEL_FOR_0513 = "３日目"
EXPECTED_DAY_LABEL_FOR_0516 = "最終日"


class TestScrapeRaceTitle(unittest.TestCase):
    def test_extract_title_from_real_html(self):
        html = (FIXTURE_DIR / "racelist_sample.html").read_text(encoding='utf-8')
        title = _parse_title_from_html(html)
        self.assertIsNotNone(title)
        self.assertEqual(title, EXPECTED_TITLE)

    def test_extract_title_from_no_title_html(self):
        html = (FIXTURE_DIR / "racelist_no_title.html").read_text(encoding='utf-8')
        title = _parse_title_from_html(html)
        self.assertIsNone(title)

    def test_extract_title_from_malformed_html(self):
        title = _parse_title_from_html("<html><not closed")
        self.assertIsNone(title)


class TestScrapeRaceMeta(unittest.TestCase):
    def setUp(self):
        self.html = (FIXTURE_DIR / "racelist_sample.html").read_text(encoding='utf-8')

    def test_extract_subtitle(self):
        self.assertEqual(_parse_subtitle_from_html(self.html), EXPECTED_SUBTITLE)

    def test_extract_subtitle_malformed(self):
        self.assertIsNone(_parse_subtitle_from_html("<html><not closed"))

    def test_extract_day_label_first_day(self):
        self.assertEqual(_parse_day_label_from_html(self.html, date(2026, 5, 11)), EXPECTED_DAY_LABEL_FOR_0511)

    def test_extract_day_label_third_day(self):
        self.assertEqual(_parse_day_label_from_html(self.html, date(2026, 5, 13)), EXPECTED_DAY_LABEL_FOR_0513)

    def test_extract_day_label_final_day(self):
        self.assertEqual(_parse_day_label_from_html(self.html, date(2026, 5, 16)), EXPECTED_DAY_LABEL_FOR_0516)

    def test_extract_day_label_date_not_in_meeting(self):
        # 節期間外 (例 2026-06-01) は None
        self.assertIsNone(_parse_day_label_from_html(self.html, date(2026, 6, 1)))

    def test_extract_day_label_none_input(self):
        self.assertIsNone(_parse_day_label_from_html(self.html, None))


if __name__ == '__main__':
    unittest.main()
