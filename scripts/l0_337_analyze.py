#!/usr/bin/env python3
"""337 结果分析:SiO2 修复前后 GM 对比 + 逐阵变化。
用法:.venv/bin/python scripts/l0_337_analyze.py [old_csv] [new_log]
"""
import sys, os, math, re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_old(path):
    """旧版 L0 结果 CSV."""
    data = {}
    for line in open(path).readlines()[1:]:
        f = line.strip().split(",")
        if len(f) >= 5 and f[3]:
            data[f[0]] = (float(f[1]), float(f[2]), f[3], float(f[4]))  # A, C, better, ratio
    return data

def load_new(path):
    """新版 337 屏 log."""
    data = {}
    pat = re.compile(r"\[(\d+)/\d+\]\s+(\S+)\s+A=([\d.]+|DNF)\s+C=([\d.]+|DNF)")
    for line in open(path):
        m = pat.search(line)
        if m:
            mat = m.group(2)
            a = float(m.group(3)) if m.group(3) != "DNF" else None
            c = float(m.group(4)) if m.group(4) != "DNF" else None
            if a and c:
                data[mat] = (a, c)
    return data

def load_ocean(path):
    data = {}
    import csv
    for r in csv.DictReader(open(path)):
        if r["ocean_ms"] != "DNF":
            data[r["mat"]] = float(r["ocean_ms"])
    return data

if __name__ == "__main__":
    old_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "compare/l0_337/l0_337_results.csv")
    new_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "compare/l0_337/screen_v2_heavyfix.log")
    oc_path = os.path.join(REPO, "compare/ocean_337/ocean_337.csv")

    old = load_old(old_path)
    new = load_new(new_path)
    ocean = load_ocean(oc_path)

    # 新旧对比(有数据的阵)
    changes = []
    for m in sorted(set(old) & set(new) & set(ocean)):
        old_best = min(old[m][0], old[m][1])
        new_best = min(new[m][0], new[m][1])
        old_ratio = old_best / ocean[m]
        new_ratio = new_best / ocean[m]
        if abs(new_ratio / old_ratio - 1) > 0.05:  # >5% change
            changes.append((m, old_ratio, new_ratio, new_ratio / old_ratio))

    # GM 计算
    old_ratios = [min(old[m][0], old[m][1]) / ocean[m] for m in old if m in ocean]
    new_ratios = [min(new[m][0], new[m][1]) / ocean[m] for m in new if m in ocean]
    if old_ratios:
        gm_old = math.exp(sum(math.log(r) for r in old_ratios) / len(old_ratios))
        print(f"旧 337 GM(修复前): {gm_old:.4f} (n={len(old_ratios)})")
    if new_ratios:
        gm_new = math.exp(sum(math.log(r) for r in new_ratios) / len(new_ratios))
        print(f"新 337 GM(修复后): {gm_new:.4f} (n={len(new_ratios)})")
    if old_ratios and new_ratios:
        print(f"GM 变化: {(gm_new/gm_old-1)*100:+.1f}%")

    if changes:
        print(f"\n=== 显著变化(>5%)的阵 ===")
        for m, o, n, ch in sorted(changes, key=lambda x: -abs(x[3]-1)):
            print(f"  {m:<22} {o:.3f} → {n:.3f} ({ch:+.1%})")

    # SiO2 专项
    if "SiO2" in new and "SiO2" in ocean:
        s = min(new["SiO2"]) / ocean["SiO2"]
        print(f"\nSiO2: {s:.3f}× vs ocean")
