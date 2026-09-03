#!/usr/bin/env python3
"""论文主图(Ocean 风格):Fig6 风格 GFLOPS 趋势 + Fig7 风格逐阵加速比 + Table2 风格汇总。
数据:compare/ocean337/methods_cmp_v29.csv(v30 出数后改 CSV 路径重刷)。
GFLOPS = 2×C_nnz / t(输出语义,与 spECK/Ocean 同款可比分母)。
用法:.venv/bin/python scripts/fig_paper_main.py [csv 路径] [输出前缀]"""
import sys, csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "compare/ocean337/methods_cmp_v29.csv"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "fig/paper_v29"
METHODS = [("Auto", "Ours (Auto)", "#d62728", 2.2),
           ("Ocean", "Ocean", "#1f77b4", 1.8),
           ("opSparse", "opSparse", "#2ca02c", 1.5),
           ("HSMU", "HSMU", "#9467bd", 1.5),
           ("bhSparse", "bhSparse", "#8c564b", 1.5),
           ("dense", "dense", "#7f7f7f", 1.2),
           ("cublas", "cuBLAS", "#bcbd22", 1.2)]

rows = list(csv.DictReader(open(CSV)))
def num(x):
    try: return float(x)
    except: return None

# ── 公共数据:有效行(x = C_nnz/行,y = GFLOPS)──
data = {}
for key, label, *_ in [(m[0], m[1]) for m in METHODS]:
    pts = []
    for r in rows:
        t, c = num(r.get(key)), num(r.get("cnnz"))
        if t and c and t > 0 and c > 0:
            pts.append((c / max(int(r["n"]), 1), 2 * c / (t * 1e6)))
    pts.sort()
    data[key] = np.array(pts)

# ── Fig6 风格:GFLOPS 趋势(秩平滑曲线 + 偏差带)──
fig, ax = plt.subplots(figsize=(9, 5.2))
def smooth(y, w=9):
    if len(y) < 3: return y
    k = np.ones(min(w, len(y))) / min(w, len(y))
    return np.convolve(y, k, mode="same")
for key, label, color, lw in METHODS:
    pts = data[key]
    if len(pts) < 3: continue
    x, y = pts[:, 0], pts[:, 1]
    ys = smooth(y)
    ax.plot(x, ys, color=color, lw=lw, label=label, zorder=3 if key == "Auto" else 2)
    band = np.percentile(y, [25, 75]) if len(y) > 8 else None
    if band is not None:
        ax.fill_between(x, np.minimum(ys, band[0]), np.maximum(ys, band[1]),
                        color=color, alpha=0.10, lw=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Output nonzeros per row (sorted)", fontsize=12)
ax.set_ylabel("GFLOPS  (2·nnz(C)/t)", fontsize=12)
ax.set_title("SpGEMM throughput on ocean337 square suite (H100 PCIe)", fontsize=13)
ax.grid(True, which="both", alpha=0.25, ls=":")
ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{PREFIX}_gflops_trend.png", dpi=200)
fig.savefig(f"{PREFIX}_gflops_trend.pdf")
print(f"saved {PREFIX}_gflops_trend.png/.pdf")

# ── Fig7 风格:逐阵加速比 vs Ocean(排序,截断 1/8~8×)──
fig, ax = plt.subplots(figsize=(9, 4.6))
for key, label, color, lw in METHODS:
    if key in ("Ocean",): continue
    sp = []
    for r in rows:
        t, o = num(r.get(key)), num(r.get("Ocean"))
        if t and o and t > 0 and o > 0:
            sp.append(min(max(o / t, 1/8), 8))
    sp.sort()
    ax.plot(np.arange(len(sp)), sp, color=color, lw=lw, label=f"{label} vs Ocean")
ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
ax.set_ylim(1/8, 8)
ax.set_yscale("log"); ax.set_yticks([1/8, 1/4, 1/2, 1, 2, 4, 8])
ax.set_yticklabels(["1/8", "1/4", "1/2", "1×", "2×", "4×", "8×"])
ax.set_xlabel("Matrices (sorted)", fontsize=12)
ax.set_ylabel("Speedup vs Ocean", fontsize=12)
ax.set_title("Per-matrix speedup over Ocean (ocean337, H100 PCIe)", fontsize=13)
ax.grid(True, which="both", alpha=0.25, ls=":")
ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{PREFIX}_speedup_vs_ocean.png", dpi=200)
fig.savefig(f"{PREFIX}_speedup_vs_ocean.pdf")
print(f"saved {PREFIX}_speedup_vs_ocean.png/.pdf")

# ── Table2 风格汇总(#best/geomean)──
print(f"\n{'method':<10}{'valid':>6}{'#best':>7}{'geomean ms':>12}{'GFLOPS geo':>11}")
base = {}
for key, *_ in METHODS:
    v = [num(r.get(key)) for r in rows]
    v = [x for x in v if x and x > 0]
    if v:
        base[key] = math.exp(sum(math.log(x) for x in v) / len(v))
for key, label, *_ in METHODS:
    best = 0; v = []
    for r in rows:
        ts = {k: num(r.get(k)) for k, *_ in METHODS}
        ts = {k: x for k, x in ts.items() if x and x > 0}
        if key in ts and ts and abs(ts[key] - min(ts.values())) < 1e-9:
            best += 1
        x = ts.get(key)
        if x: v.append(x)
    gf = [2 * num(r["cnnz"]) / (num(r.get(key)) * 1e6) for r in rows
          if num(r.get(key)) and num(r.get("cnnz"))]
    gfg = math.exp(sum(math.log(g) for g in gf) / len(gf)) if gf else 0
    print(f"{label:<10}{len(v):>6}{best:>7}{(math.exp(sum(math.log(x) for x in v)/len(v)) if v else 0):>12.3f}{gfg:>11.2f}")
