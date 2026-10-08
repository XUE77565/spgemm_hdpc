#!/usr/bin/env python3
"""D2 专轮:search-dense 巨行平台现象的两假设判决(study/35 §五b 续)。

假设①(强制臂 co-residency 副作用):强制开门 → 百万块满装置争用 → 每块墙钟抬到平台;
  自然路由下 dense 块少且与 hash 块交错 → 同行成本应显著更低且与特征恢复相关。
假设②(带宽饱和平台=真物理):同行两语境成本接近,平台是 dense search 的真单价。

判决口径(同行配对,D 臂 vs DN 臂 ×3 rep 中位):
  A. 中位比值 DN/D(平台带行:|cyc_D − 平台|<20% 的行)
  B. DN 侧特征相关恢复度(corr(flop), corr(span))
  C. DN 侧子律拟合 R²(search 行,窗数 W+flop 形态)
用法:.venv/bin/python scripts/compare_d_dn_context.py <mat> [mat ...]
"""
import os, sys, statistics, math
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "compare", "d1_oracle")
PB2_W, TIER_SPAN_MAX = 5400, 7277


def med3(paths, ncols):
    acc = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for ln in open(p):
            f = ln.split()
            if len(f) != ncols:
                continue
            acc.setdefault(int(f[0]), []).append([float(x) for x in f[1:]])
    return {r: [statistics.median(v[i] for v in vs) for i in range(len(vs[0]))]
            for r, vs in acc.items() if len(vs) == 3}


def feats_W(span, flop):
    return [math.ceil(span / PB2_W), flop, 1.0]


def main():
    mats = sys.argv[1:] or ["crankseg_2", "F2", "c-64", "in-2004"]
    for m in mats:
        D = med3([os.path.join(SRC, f"{m}.D{i}.dd.tsv") for i in (1, 2, 3)], 3)
        DN = med3([os.path.join(SRC, f"{m}.DN{i}.dd.tsv") for i in (1, 2, 3)], 3)
        if not D or not DN:
            print(f"{m}: D n={len(D)} DN n={len(DN)} —— 不足,跳过")
            continue
        common = [r for r in DN if r in D]
        # DN 侧特征:从 H 臂真值表拿(行特征两语境一致)
        ft = {}
        tp = os.path.join(SRC, "truth", f"{m}.truth.tsv")
        if os.path.exists(tp):
            for ln in open(tp).readlines()[1:]:
                f = ln.split()
                ft[int(f[0])] = (float(f[2]), float(f[4]))   # flop, span
        se_D = {r: v for r, v in D.items() if v[1] == 0}
        se_DN = {r: v for r, v in DN.items() if v[1] == 0}
        print(f"\n== {m}: D(dd) n={len(D)} search={len(se_D)} | DN(dd) n={len(DN)} search={len(se_DN)} | 交集={len(common)}")
        if se_DN:
            cs = [r for r in se_DN if r in se_D]
            if cs:
                rat = [se_DN[r][0] / max(1, se_D[r][0]) for r in cs]
                print(f"  search 同行配对 n={len(cs)}  DN/D 中位={statistics.median(rat):.3f} "
                      f"p25={np.percentile(rat,25):.3f} p75={np.percentile(rat,75):.3f}")
            if ft:
                fl = np.array([ft[r][0] for r in se_DN if r in ft])
                sp = np.array([ft[r][1] for r in se_DN if r in ft])
                cy = np.array([se_DN[r][0] for r in se_DN if r in ft])
                if len(fl) > 100:
                    print(f"  DN-search 特征相关:corr(flop)={np.corrcoef(fl,cy)[0,1]:.2f} "
                          f"corr(span)={np.corrcoef(sp,cy)[0,1]:.2f}  cyc中位={np.median(cy):.0f}")
                    X = np.array([feats_W(ft[r][1], ft[r][0]) for r in se_DN if r in ft])
                    y = cy
                    if len(X) == len(y) and len(X) > 100:
                        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                        r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                        print(f"  DN-search 子律拟合(W+flop):n={len(X)} R²={r2:.3f} "
                              f"γW={coef[0]:.0f} δflop={coef[1]:.1f} c={coef[2]:.0f}")
        cu_DN = {r: v for r, v in DN.items() if v[1] == 1}
        if cu_DN and ft:
            fl = np.array([ft[r][0] for r in cu_DN if r in ft])
            cy = np.array([cu_DN[r][0] for r in cu_DN if r in ft])
            if len(fl) > 100:
                print(f"  DN-cursor n={len(fl)} corr(flop)={np.corrcoef(fl,cy)[0,1]:.2f} cyc中位={np.median(cy):.0f}")


if __name__ == "__main__":
    main()
