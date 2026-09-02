#!/usr/bin/env python3
"""v29 DNF 补跑:Auto 列为空的 29 阵用 timeout=1800 重跑(v28 同配方),增量回写 v29 CSV。
nnz 红旗:补跑 cnnz ≠ v28 cnnz → 打 WARN 不覆写(口径铁律:结构变化 = 无效测量)。"""
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_methods as CM

V29 = "compare/ocean337/methods_cmp_v29.csv"
V28 = "compare/ocean337/methods_cmp_v28_ocgate.csv"
TIMEOUT = 1800
FIELDS = ["matrix", "n", "sym", "density_pct", "cu", "Auto", "Auto_choice", "Ocean",
          "opSparse", "HSMU", "bhSparse", "dense", "cublas", "cnnz"]

def _num(x):
    try: return float(x)
    except: return None

rows = list(csv.DictReader(open(V29)))
v28 = {r["matrix"]: r for r in csv.DictReader(open(V28))}
todo = [r for r in rows if _num(r.get("Auto")) is None]
print(f"补跑 {len(todo)}/{len(rows)} 阵(timeout={TIMEOUT}s,MP_HOST_MB=8192)", flush=True)

for i, r in enumerate(todo):
    name = r["matrix"]
    p = CM.find_mtx(name)
    if not p:
        print(f"[{i+1}/{len(todo)}] {name}: mtx 未找到", flush=True); continue
    old_cnnz = _num(v28.get(name, {}).get("cnnz"))
    t0 = time.time()
    res = CM.run_spgemm_method(p, "adaptive", timeout=TIMEOUT, exp_cnnz=old_cnnz)
    if res is None:
        print(f"[{i+1}/{len(todo)}] {name:14} 仍 DNF({time.time()-t0:.0f}s)", flush=True); continue
    comp, wall, cnnz, choice = res
    flag = ""
    if old_cnnz and cnnz > 0 and abs(cnnz - old_cnnz) / old_cnnz > 0.01:
        flag = f"  ⚠NNZ-RED {cnnz} vs v28 {old_cnnz}(不覆写)"
        print(flag, flush=True)
    else:
        r["Auto"] = round(comp, 3); r["Auto_choice"] = choice; r["cnnz"] = cnnz
        with open(V29, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=FIELDS); w.writeheader()
            for rr in rows: w.writerow({k: rr.get(k, "") for k in FIELDS})
    print(f"[{i+1}/{len(todo)}] {name:14} Auto={r['Auto']!s:<9} ({time.time()-t0:.0f}s) {choice}", flush=True)

# 终版汇总
import math
g, w = [], 0
for r in rows:
    a, o = _num(r.get("Auto")), _num(r.get("Ocean"))
    if a and o:
        g.append(a / o); w += a <= o
if g:
    print(f"\n=== v29 终版:{len(g)} 有效阵,赢 {w},geomean {math.exp(sum(map(math.log, g))/len(g)):.4f}× ===", flush=True)
