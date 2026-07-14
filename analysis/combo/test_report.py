import os
from report import build_report

def test_report_builds():
    path = build_report("2026-06-30", "2026-12-31", limit=150)
    assert os.path.exists(path)
    txt = open(path, encoding="utf-8").read()
    for kw in ("歪みマップ", "候補手法", "フラット", "本番同等", "既存", "前向き"):
        assert kw in txt, kw
    print("OK report at", path)

if __name__ == "__main__":
    test_report_builds(); print("ALL PASS")
