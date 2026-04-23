"""分析A: Miss Pattern Analysis

現行V10モデル (mc_early_race系) がどのセグメントで的中率/ROIが低いかを炙り出す。
再学習時にweight 2xすべきセグメントを特定する。

READ-ONLY: DBからSELECTのみ。V10本番プロセスには一切影響しない。

出力: analysis/reports/01_miss_pattern_YYYYMMDD_HHMMSS.md
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# 分析対象戦略（V10本番系 + 比較用）
TARGET_STRATEGIES = [
    'mc_early_race',      # O1: V10 本番
    'mc2_early_race',     # O2: QMC v2
    'mc3_early_race',     # O3: QMC v3 (追加特徴量)
    'mc_quarter_kelly',   # L: V10 基準
    'mc2_quarter_kelly',  # L2: QMC 基準
]

VENUE_NAMES = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島',
    5: '多摩川', 6: '浜名湖', 7: '蒲郡', 8: '常滑',
    9: '津', 10: '三国', 11: 'びわこ', 12: '住之江',
    13: '尼崎', 14: '鳴門', 15: '丸亀', 16: '児島',
    17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村',
}


def pct(x):
    return f"{x*100:.1f}%"


def fetch_all_bets(cur):
    """全戦略の全bets + レース情報を取得 (result確定分のみ)"""
    cur.execute("""
        SELECT
            b.id as bet_id, b.strategy_type, b.combination, b.amount,
            b.result, b.payout, b.odds, b.expected_value, b.is_hit,
            b.created_at,
            r.id as race_id, r.venue_id, r.race_number, r.race_date,
            r.actual_result_trifecta, r.payout_sanrentan,
            r.wind_speed, r.wind_direction, r.wave_height,
            r.water_temperature, r.temperature,
            (r.deadline_time AT TIME ZONE 'Asia/Tokyo')::date as deadline_jst_date,
            EXTRACT(HOUR FROM r.deadline_time AT TIME ZONE 'Asia/Tokyo') as deadline_hour
        FROM bets b
        JOIN races r ON b.race_id = r.id
        WHERE b.strategy_type = ANY(%s)
          AND b.result IS NOT NULL
          AND r.deadline_time >= NOW() - INTERVAL '60 days'
        ORDER BY r.deadline_time ASC
    """, (TARGET_STRATEGIES,))
    return cur.fetchall()


def segment_stats(bets, key_fn, label):
    """セグメント別の集計。key_fn(bet) -> segment key"""
    buckets = defaultdict(lambda: {
        'bets': 0, 'hits': 0, 'invest': 0, 'payout': 0,
    })
    for b in bets:
        key = key_fn(b)
        if key is None:
            continue
        bucket = buckets[key]
        bucket['bets'] += 1
        bucket['invest'] += int(b['amount'])
        if b['is_hit']:
            bucket['hits'] += 1
            bucket['payout'] += int(b['payout'] or 0)
    rows = []
    for k, v in sorted(buckets.items(), key=lambda x: -x[1]['bets']):
        hit_rate = v['hits'] / v['bets'] if v['bets'] else 0
        roi = v['payout'] / v['invest'] if v['invest'] else 0
        rows.append({
            'segment': k,
            'bets': v['bets'],
            'hits': v['hits'],
            'hit_rate': hit_rate,
            'invest': v['invest'],
            'payout': v['payout'],
            'profit': v['payout'] - v['invest'],
            'roi': roi,
        })
    return rows


def format_table(rows, headers, column_types=None):
    """Markdownテーブルで出力"""
    if column_types is None:
        column_types = ['str'] * len(headers)
    lines = []
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for r in rows:
        cells = []
        for v, t in zip(r, column_types):
            if t == 'int':
                cells.append(f"{v:,}")
            elif t == 'pct':
                cells.append(pct(v))
            elif t == 'money':
                sign = '+' if v > 0 else ''
                cells.append(f"{sign}{int(v):,}")
            elif t == 'roi':
                cells.append(f"{v*100:.0f}%")
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def wind_bin(b):
    w = b.get('wind_speed')
    if w is None:
        return None
    if w <= 2: return '弱(0-2m)'
    if w <= 5: return '中(3-5m)'
    return '強(6m+)'


def wave_bin(b):
    w = b.get('wave_height')
    if w is None:
        return None
    if w <= 2: return '低(0-2cm)'
    if w <= 5: return '中(3-5cm)'
    return '高(6cm+)'


def odds_bin(b):
    o = b.get('odds')
    if o is None:
        return None
    o = float(o)
    if o < 10: return '01_<10'
    if o < 20: return '02_10-20'
    if o < 30: return '03_20-30'
    if o < 50: return '04_30-50'
    if o < 80: return '05_50-80'
    return '06_80+'


def ev_bin(b):
    ev = b.get('expected_value')
    if ev is None:
        return None
    ev = float(ev)
    if ev < 0.5: return '01_EV<0.5'
    if ev < 0.8: return '02_EV_0.5-0.8'
    if ev < 1.0: return '03_EV_0.8-1.0'
    if ev < 1.5: return '04_EV_1.0-1.5'
    return '05_EV_1.5+'


def main():
    logger.info("=== Miss Pattern Analysis 開始 ===")
    conn = psycopg2.connect(os.environ['DATABASE_URL'], cursor_factory=RealDictCursor)
    cur = conn.cursor()

    bets = fetch_all_bets(cur)
    logger.info(f"取得: {len(bets)}件 (確定済み・60日以内)")
    if not bets:
        logger.error("データがありません")
        return

    report = []
    report.append(f"# Miss Pattern Analysis\n")
    report.append(f"生成日時: {datetime.now():%Y-%m-%d %H:%M:%S} JST")
    report.append(f"対象戦略: {', '.join(TARGET_STRATEGIES)}")
    report.append(f"対象bets: {len(bets)}件 (確定済み・60日以内)\n")

    # === 1. 戦略別サマリ ===
    report.append("## 1. 戦略別サマリ\n")
    by_strat = segment_stats(bets, lambda b: b['strategy_type'], 'strategy')
    table_rows = []
    for r in by_strat:
        table_rows.append([
            r['segment'], r['bets'], r['hits'],
            r['hit_rate'], r['invest'], r['payout'],
            r['profit'], r['roi'],
        ])
    report.append(format_table(
        table_rows,
        ['戦略', 'bets', 'hits', '的中率', '投資', '回収', '損益', 'ROI'],
        ['str', 'int', 'int', 'pct', 'int', 'int', 'money', 'roi']
    ))
    report.append("")

    # === 2. 会場別（mc_early_raceのみ絞る）===
    mc_bets = [b for b in bets if b['strategy_type'] == 'mc_early_race']
    report.append(f"\n## 2. mc_early_race 会場別\n")
    report.append(f"対象: {len(mc_bets)}件\n")
    by_venue = segment_stats(mc_bets,
        lambda b: f"V{b['venue_id']:02d} {VENUE_NAMES.get(b['venue_id'], '?')}", '')
    table_rows = []
    for r in by_venue:
        table_rows.append([
            r['segment'], r['bets'], r['hits'],
            r['hit_rate'], r['invest'], r['payout'],
            r['profit'], r['roi'],
        ])
    report.append(format_table(
        table_rows,
        ['会場', 'bets', 'hits', '的中率', '投資', '回収', '損益', 'ROI'],
        ['str', 'int', 'int', 'pct', 'int', 'int', 'money', 'roi']
    ))
    report.append("")

    # === 3. レース番号別 ===
    report.append(f"\n## 3. mc_early_race レース番号別\n")
    by_rn = segment_stats(mc_bets, lambda b: f"R{b['race_number']}", '')
    table_rows = []
    for r in sorted(by_rn, key=lambda x: x['segment']):
        table_rows.append([
            r['segment'], r['bets'], r['hits'],
            r['hit_rate'], r['invest'], r['payout'],
            r['profit'], r['roi'],
        ])
    report.append(format_table(
        table_rows,
        ['R番号', 'bets', 'hits', '的中率', '投資', '回収', '損益', 'ROI'],
        ['str', 'int', 'int', 'pct', 'int', 'int', 'money', 'roi']
    ))
    report.append("")

    # === 4. 風速別 ===
    report.append(f"\n## 4. mc_early_race 風速別\n")
    by_wind = segment_stats(mc_bets, wind_bin, '')
    table_rows = []
    for r in sorted(by_wind, key=lambda x: x['segment']):
        table_rows.append([
            r['segment'], r['bets'], r['hits'],
            r['hit_rate'], r['invest'], r['payout'],
            r['profit'], r['roi'],
        ])
    report.append(format_table(
        table_rows,
        ['風速', 'bets', 'hits', '的中率', '投資', '回収', '損益', 'ROI'],
        ['str', 'int', 'int', 'pct', 'int', 'int', 'money', 'roi']
    ))
    report.append("")

    # === 5. 波高別 ===
    report.append(f"\n## 5. mc_early_race 波高別\n")
    by_wave = segment_stats(mc_bets, wave_bin, '')
    table_rows = []
    for r in sorted(by_wave, key=lambda x: x['segment']):
        table_rows.append([
            r['segment'], r['bets'], r['hits'],
            r['hit_rate'], r['invest'], r['payout'],
            r['profit'], r['roi'],
        ])
    report.append(format_table(
        table_rows,
        ['波高', 'bets', 'hits', '的中率', '投資', '回収', '損益', 'ROI'],
        ['str', 'int', 'int', 'pct', 'int', 'int', 'money', 'roi']
    ))
    report.append("")

    # === 6. オッズ帯別 ===
    report.append(f"\n## 6. mc_early_race オッズ帯別\n")
    by_odds = segment_stats(mc_bets, odds_bin, '')
    table_rows = []
    for r in sorted(by_odds, key=lambda x: x['segment']):
        table_rows.append([
            r['segment'], r['bets'], r['hits'],
            r['hit_rate'], r['invest'], r['payout'],
            r['profit'], r['roi'],
        ])
    report.append(format_table(
        table_rows,
        ['オッズ帯', 'bets', 'hits', '的中率', '投資', '回収', '損益', 'ROI'],
        ['str', 'int', 'int', 'pct', 'int', 'int', 'money', 'roi']
    ))
    report.append("")

    # === 7. Expected Value帯別 ===
    report.append(f"\n## 7. mc_early_race EV帯別\n")
    by_ev = segment_stats(mc_bets, ev_bin, '')
    table_rows = []
    for r in sorted(by_ev, key=lambda x: x['segment']):
        table_rows.append([
            r['segment'], r['bets'], r['hits'],
            r['hit_rate'], r['invest'], r['payout'],
            r['profit'], r['roi'],
        ])
    report.append(format_table(
        table_rows,
        ['EV帯', 'bets', 'hits', '的中率', '投資', '回収', '損益', 'ROI'],
        ['str', 'int', 'int', 'pct', 'int', 'int', 'money', 'roi']
    ))
    report.append("")

    # === 8. 時間帯（デイ/ナイター）===
    report.append(f"\n## 8. mc_early_race 締切時間帯別\n")
    def time_bin(b):
        h = b.get('deadline_hour')
        if h is None: return None
        h = int(h)
        if h < 15: return 'デイ(〜15時)'
        if h < 18: return '夕方(15-18時)'
        return 'ナイター(18時〜)'
    by_time = segment_stats(mc_bets, time_bin, '')
    table_rows = []
    for r in sorted(by_time, key=lambda x: x['segment']):
        table_rows.append([
            r['segment'], r['bets'], r['hits'],
            r['hit_rate'], r['invest'], r['payout'],
            r['profit'], r['roi'],
        ])
    report.append(format_table(
        table_rows,
        ['時間帯', 'bets', 'hits', '的中率', '投資', '回収', '損益', 'ROI'],
        ['str', 'int', 'int', 'pct', 'int', 'int', 'money', 'roi']
    ))
    report.append("")

    # === サマリ：弱点セグメント ===
    report.append(f"\n## 9. 🚨 弱点セグメント（ROI<80% かつ bets>=5）\n")
    report.append("再学習時に重点対象にすべきセグメント：\n")
    weak = []
    # 会場
    for r in by_venue:
        if r['bets'] >= 5 and r['roi'] < 0.8:
            weak.append(('会場', r['segment'], r['bets'], r['roi'], r['profit']))
    # 風速
    for r in by_wind:
        if r['bets'] >= 5 and r['roi'] < 0.8:
            weak.append(('風速', r['segment'], r['bets'], r['roi'], r['profit']))
    # 波高
    for r in by_wave:
        if r['bets'] >= 5 and r['roi'] < 0.8:
            weak.append(('波高', r['segment'], r['bets'], r['roi'], r['profit']))
    # オッズ
    for r in by_odds:
        if r['bets'] >= 5 and r['roi'] < 0.8:
            weak.append(('オッズ', r['segment'], r['bets'], r['roi'], r['profit']))

    if weak:
        report.append(format_table(
            weak,
            ['次元', 'セグメント', 'bets', 'ROI', '損益'],
            ['str', 'str', 'int', 'roi', 'money']
        ))
    else:
        report.append("_該当なし_")
    report.append("")

    # === サマリ：強みセグメント ===
    report.append(f"\n## 10. ⭐ 強みセグメント（ROI>=150% かつ bets>=5）\n")
    report.append("現状モデルが得意な領域。新モデルでも維持すべき：\n")
    strong = []
    for r in by_venue:
        if r['bets'] >= 5 and r['roi'] >= 1.5:
            strong.append(('会場', r['segment'], r['bets'], r['roi'], r['profit']))
    for r in by_wind:
        if r['bets'] >= 5 and r['roi'] >= 1.5:
            strong.append(('風速', r['segment'], r['bets'], r['roi'], r['profit']))
    for r in by_wave:
        if r['bets'] >= 5 and r['roi'] >= 1.5:
            strong.append(('波高', r['segment'], r['bets'], r['roi'], r['profit']))
    for r in by_odds:
        if r['bets'] >= 5 and r['roi'] >= 1.5:
            strong.append(('オッズ', r['segment'], r['bets'], r['roi'], r['profit']))

    if strong:
        report.append(format_table(
            strong,
            ['次元', 'セグメント', 'bets', 'ROI', '損益'],
            ['str', 'str', 'int', 'roi', 'money']
        ))
    else:
        report.append("_該当なし_")
    report.append("")

    # 保存
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = REPORT_DIR / f"01_miss_pattern_{ts}.md"
    path.write_text('\n'.join(report), encoding='utf-8')
    logger.info(f"\nレポート保存: {path}")
    logger.info(f"\n{'='*60}")
    logger.info(f"要約:")
    logger.info(f"  総bets: {len(bets)}")
    logger.info(f"  戦略数: {len(by_strat)}")
    logger.info(f"  弱点セグメント: {len(weak)}")
    logger.info(f"  強みセグメント: {len(strong)}")

    conn.close()


if __name__ == '__main__':
    main()
