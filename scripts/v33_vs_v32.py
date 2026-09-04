#!/usr/bin/env python3
"""v33 vs v32 per-matrix diff 报告(全刷完成后跑)。
用法:.venv/bin/python scripts/v33_vs_v32.py
输出:geomean 对比 / 大赢大输榜 / nnz 红旗 / DNF 清单。"""
import csv, math, sys

def load(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        def num(k):
            try: return float(r.get(k) or "")
            except: return None
        out[r["matrix"]] = dict(auto=num("Auto"), ocean=num("Ocean"), cnnz=num("cnnz"),
                                choice=r.get("Auto_choice", ""))
    return out

v33, v32 = load("compare/ocean337/methods_cmp_v33.csv"), load("compare/ocean337/methods_cmp_v32.csv")
common = [m for m in v33 if m in v32]

def geo(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")

g33 = geo([v33[m]["auto"] / v33[m]["ocean"] for m in common if v33[m]["auto"] and v33[m]["ocean"]])
g32 = geo([v32[m]["auto"] / v32[m]["ocean"] for m in common if v32[m]["auto"] and v32[m]["ocean"]])
print(f"geomean Auto/Ocean: v32 = {g32:.4f} → v33 = {g33:.4f}   ({(g33/g32-1)*100:+.2f}%)")

diffs = []
for m in common:
    a33, a32 = v33[m]["auto"], v32[m]["auto"]
    if a33 and a32 and a32 > 0:
        diffs.append((m, (a33 / a32 - 1) * 100))
    elif (a33 is None) != (a32 is None):
        diffs.append((m, None))
ok = [(m, d) for m, d in diffs if d is not None]
print(f"\n有效对 {len(ok)}/{len(common)};DNF 变化: {[m for m, d in diffs if d is None]}")
big_win = sorted([x for x in ok if x[1] < -2], key=lambda x: x[1])
big_loss = sorted([x for x in ok if x[1] > 2], key=lambda x: -x[1])
print(f"\n大赢(≤−2%,{len(big_win)} 阵):")
for m, d in big_win[:15]: print(f"  {m:<28}{d:+7.1f}%   {v32[m]['auto']:.1f}→{v33[m]['auto']:.1f}")
print(f"\n大输(≥+2%,{len(big_loss)} 阵):")
for m, d in big_loss[:15]: print(f"  {m:<28}{d:+7.1f}%   {v32[m]['auto']:.1f}→{v33[m]['auto']:.1f}")

red = [m for m in common if v33[m]["cnnz"] and v32[m]["cnnz"]
       and abs(v33[m]["cnnz"] / v32[m]["cnnz"] - 1) > 0.01]
if red: print(f"\n⚠ nnz 红旗(>1% 漂移,口径铁律 = 无效): {red}")
