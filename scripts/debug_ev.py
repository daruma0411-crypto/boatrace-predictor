"""直近予測のEV分布を調査"""
import psycopg2, psycopg2.extras

DB = 'postgresql://boatrace:brpred2026secure@shinkansen.proxy.rlwy.net:24787/boatrace_db?sslmode=disable'
conn = psycopg2.connect(DB)
cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

cur.execute("""
    SELECT p.race_id, p.probabilities_1st, p.probabilities_2nd, p.probabilities_3rd,
           r.venue_id, r.race_number
    FROM predictions p
    JOIN races r ON p.race_id = r.id
    WHERE p.created_at > '2026-03-14 09:40:00+00'
    LIMIT 1
""")
pred = cur.fetchone()
if not pred:
    print("No prediction")
    exit()

p1 = pred['probabilities_1st']
p2 = pred['probabilities_2nd']
p3 = pred['probabilities_3rd']
print(f"Race: venue={pred['venue_id']} R={pred['race_number']} (id={pred['race_id']})")
print(f"1st: {[round(x,4) for x in p1]}")

combos = []
for i in range(6):
    for j in range(6):
        if j == i:
            continue
        for k in range(6):
            if k == i or k == j:
                continue
            p_1st_i = p1[i]
            others_2nd = sum(p2[x] for x in range(6) if x != i)
            p_2nd_j = p2[j] / others_2nd if others_2nd > 0 else 0
            others_3rd = sum(p3[x] for x in range(6) if x != i and x != j)
            p_3rd_k = p3[k] / others_3rd if others_3rd > 0 else 0
            prob = p_1st_i * p_2nd_j * p_3rd_k
            combo = f"{i+1}-{j+1}-{k+1}"
            combos.append((combo, prob))

combos.sort(key=lambda x: -x[1])
print(f"\nTop 15 combos (prob >= 0.02, odds 5-50):")
count_ev_ge1 = 0
for combo, prob in combos:
    if prob >= 0.02:
        # EV = prob * odds. To pass min_ev=1.0, need odds >= 1/prob
        min_odds_needed = 1.0 / prob
        print(f"  {combo}: prob={prob:.4f} | need odds>={min_odds_needed:.1f} to pass EV>=1.0")
        if min_odds_needed <= 50:
            count_ev_ge1 += 1

print(f"\nCombos with prob>=0.02: {sum(1 for c,p in combos if p >= 0.02)}")
print(f"Of those, EV>=1.0 possible (need odds<=50): {count_ev_ge1}")
print(f"\nAll combos with prob>=0.01:")
for combo, prob in combos[:30]:
    if prob >= 0.01:
        print(f"  {combo}: prob={prob:.4f}")
conn.close()
