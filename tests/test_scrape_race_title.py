"""scrape_race_title() の単体テスト (offline、fixture HTML 使用)

実行:
  pytest tests/test_scrape_race_title.py -v
  または
  python -m unittest tests.test_scrape_race_title
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper import _parse_title_from_html

FIXTURE_DIR = Path(__file__).parent / "fixtures"

EXPECTED_TITLE = "江戸川６３４杯\u3000モーターボート大賞"


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


if __name__ == '__main__':
    unittest.main()
