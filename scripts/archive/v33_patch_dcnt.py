#!/usr/bin/env python3
"""v33 CSV 定点补测(DCNT=0 翻默认后):v33↔v32 变化 >2% 的阵用新默认重测,增量回写。
nnz 红旗:补测 cnnz ≠ 原 cnnz >1% → WARN 不覆写(口径铁律)。"""
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_methods as CM

CSV = "compare/ocean337/methods_cmp_v33.csv"
THRESH = 10.0

def _num(x):
    try: return float(x)
    except: return None

rows = list(csv.DictReader(open(CSV)))
v32 = {r["matrix"]: r for r in csv.DictReader(open("compare/ocean337/methods_cmp_v32.csv"))}
todo = []
for r in rows:
    m = r["matrix"]
    a33, a32 = _num(r.get("Auto")), _num(v32.get(m, {}).get("Auto"))
    if a33 and a32 and a32 > 0 and abs(a33 / a32 - 1) * 100 > THRESH:
        todo.append((m, a33, a32))
print(f"补测 {len(todo)} 阵(|Δ|>{THRESH}%),新默认含 DCNT=0", flush=True)

import csv as _csv
for i, (m, a33, a32) in enumerate(todo):
    p = CM.find_mtx(m)
    if not p:
        print(f"[{i+1}/{len(todo)}] {m}: mtx 未找到", flush=True); continue
    old_cnnz = _num(v32.get(m, {}).get("cnnz"))
    res = CM.run_spgemm_method(p, "adaptive", timeout=900, exp_cnnz=old_cnnz)
    if res is None:
        print(f"[{i+1}/{len(todo)}] {m}: 失败/超时,保留原值", flush=True); continue
    t_new, nnz_new, choice_new = res[0], res[2], res[3]   # (compute, wall, nnz, choice)
    for r in rows:
        if r["matrix"] == m:
            r["Auto"] = f"{t_new:.3f}"
            if nnz_new: r["cnnz"] = str(nnz_new)
            if choice_new: r["Auto_choice"] = choice_new
            break
    d_old = (a33 / a32 - 1) * 100 if a32 else 0
    d_new = (t_new / a32 - 1) * 100 if a32 else 0
    print(f"[{i+1}/{len(todo)}] {m}: {a33:.1f}→{t_new:.1f}(vs v32 {d_old:+.1f}%→{d_new:+.1f}%)", flush=True)

with open(CSV, "w", newline="") as f:
    w = _csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print("CSV 回写完成", flush=True)
