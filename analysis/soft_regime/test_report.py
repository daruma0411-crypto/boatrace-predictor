import os
from db import connect
from report import build_report

def test_report_builds_and_has_verdict():
    conn = connect()
    path = build_report(conn)
    assert os.path.exists(path)
    txt = open(path, encoding="utf-8").read()
    assert "結論" in txt and "事前検知" in txt
    assert "Stage B" in txt and "Stage A" in txt
    print("OK report at", path)

if __name__ == "__main__":
    test_report_builds_and_has_verdict(); print("ALL PASS")
