#!/usr/bin/env python3
"""437 权威批分析(10-15 收官):权威名单过滤 + 三层战场分解 + oracle + 胜率。
口径:compare/batch437/CALIBRATION.md(我方 compute-only vs ocean total_time,同日同机)。
"""
import csv, math, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = {}
for r in csv.DictReader(open(os.path.join(REPO, "compare/batch437/batch_437.csv"))):
    rows[(r["mat"], r["arm"])] = (float(r["ms"]) if r["ms"] else None, r["note"])

# 权威名单 = 337 ∪ July100(文件在位者)
names, seen = [], set()
o337 = [r["mat"] for r in csv.DictReader(open(os.path.join(REPO, "compare/l0_337/l0_337_results.csv")))]
for m in o337: names.append(m); seen.add(m)
july = [r["matrix"] for r in csv.DictReader(open(os.path.join(REPO, "compare/methods_cmp_first100.csv")))]
f100 = [m for m in july if m not in seen and os.path.exists(os.path.join(REPO, "data/first100", m + ".mtx"))]
ARMS = ["C", "CD", "A", "ADPT", "OCN"]

def val(m, a): return rows.get((m, a), (None, ""))[0]
def gm(vals): return math.exp(sum(math.log(v) for v in vals) / len(vals))
def paired(arm, mats):
    ps = [(val(m, arm), val(m, "OCN")) for m in mats if val(m, arm) and val(m, "OCN")]
    if not ps: return None, 0
    return gm([a / b for a, b in ps]), len(ps)
def report(tag, mats):
    print(f"\n== {tag}(n={len(mats)}) ==")
    for a in ARMS[:-1]:
        g, n = paired(a, mats)
        if g: print(f"  {a}/OCN  = {g:.4f} (配对 n={n})")
    # oracle(我方四臂逐阵最优;仅四臂全值阵)
    ps = []
    for m in mats:
        vs = [val(m, a) for a in ARMS[:-1]]
        o = val(m, "OCN")
        if all(vs) and o: ps.append(min(vs) / o)
    if ps: print(f"  oracle/OCN = {gm(ps):.4f} (n={len(ps)})")
    for a in ARMS[:-1]:
        w = sum(1 for m in mats if val(m, a) and val(m, "OCN") and val(m, a) < val(m, "OCN"))
        t = sum(1 for m in mats if val(m, a) and val(m, "OCN"))
        if t: print(f"  {a} 胜率 = {w}/{t}")

allm = o337 + f100
report("并集战场(权威 428)", allm)
report("ocean337 子集(其主场)", o337)
report("first100 子集(挑战赛集)", f100)

# 臂间:ADPT vs A(dispatcher 是否增值)、CD vs C
for x, y in [("ADPT", "A"), ("CD", "C"), ("A", "C")]:
    ps = [(val(m, x), val(m, y)) for m in allm if val(m, x) and val(m, y)]
    print(f"{x}/{y} 全集配对 = {gm([a/b for a,b in ps]):.4f} (n={len(ps)})")

# 红旗终账
flags = [(m, a, n) for (m, a), (v, n) in rows.items() if not v and n]
print(f"\n红旗终账({len(flags)}):")
for m, a, n in sorted(flags): print(f"  {m:<12} {a:<5} {n}")
print("\nANALYZE437_DONE")
