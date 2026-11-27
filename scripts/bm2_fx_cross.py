#!/usr/bin/env python3
"""两遍式行门 HH_BM2_FX 的交点分析:读 bm2_gate_cal.py 的 <前缀>_cells.csv,给出 R(flop/HT) 曲线与 R=1 的交点。

R = Σ两遍式周期 / Σ哈希周期(同一批行在两次运行里的逐行块周期,GPU 侧计数);flop/HT = 每行乘积数 ÷ 该 bin 的哈希表长。
桶 k ↔ flop/HT ∈ [2^(k/2), 2^((k+1)/2))(半倍频程)。只看位图形态(leg=0)。
交点的取法(保守):从小到大扫,取**第一个** R ≥ 1 且行数够的桶;交点 = 它与前一个够数的桶之间按 log(flop/HT) 线性插值到 R=1 的位置
  (前一个桶不存在或 R 已 ≥1 时,取该桶的下界)。此后若又出现 R<1 的桶也不回头(保守:门只会更紧)。
建议门值(dialog/qa-013 定的规则):FX = 0.65 × 各值区模式、各 bin 交点里最小的那个;行门写法是 flop×100 ≤ FX×HT,所以给出百分数。
多份 cells(比如合成阵一份、真实训练池一份)可以一起给:各自出曲线,并出合并后的曲线;建议门值用合并后的。
只能喂训练数据(data/train_disp、tools/train_synth 生成的合成阵);428 / flex142 / external_72 不得作为输入。
用法:bm2_fx_cross.py NAME=cells.csv [NAME2=cells2.csv ...] [--min-rows 200] [--margin 0.65]
输出(机读行):FX_CROSS <来源> <模式> <bin|all> <交点或 none>;FX_SUGGEST_PCT <整数或 none>。
"""
import argparse, csv, math, sys


def load(path):
    cells = {}
    try:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["leg"] != "0":
                continue
            key = (int(r["mode"]), int(r["bin"]), int(r["log2x2_flop_over_ht"]))
            c = cells.setdefault(key, [0, 0.0, 0.0])
            c[0] += int(r["rows"]); c[1] += float(r["sum_th"]); c[2] += float(r["sum_tk"])
    except (OSError, KeyError, ValueError) as e:
        print("# 读 %s 出错: %s" % (path, e))
    return cells


def curve(cells, mode, b, min_rows):
    """→ [(k, rows, R)](按 k 升序;只留行数够的桶)。b=None 表示各 bin 合并。"""
    agg = {}
    for (m, bb, k), (n, th, tk) in cells.items():
        if m != mode or (b is not None and bb != b):
            continue
        a = agg.setdefault(k, [0, 0.0, 0.0])
        a[0] += n; a[1] += th; a[2] += tk
    return [(k, n, tk / th) for k, (n, th, tk) in sorted(agg.items()) if n >= min_rows and th > 0]


def cross(cv):
    prev = None
    for k, n, r in cv:
        if r >= 1.0:
            lo = 2.0 ** (k / 2.0)
            if prev is None or prev[2] >= 1.0:
                return lo
            # 桶中心取几何中点:2^((k+0.5)/2)
            x0, x1 = (prev[0] + 0.5) / 2.0, (k + 0.5) / 2.0
            x = x0 + (1.0 - prev[2]) * (x1 - x0) / (r - prev[2])
            return 2.0 ** x
        prev = (k, n, r)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+", help="NAME=cells.csv")
    ap.add_argument("--min-rows", type=int, default=200)
    ap.add_argument("--margin", type=float, default=0.65)
    a = ap.parse_args()
    sets = []
    for s in a.src:
        name, _, path = s.partition("=")
        if not path:
            name, path = "cells", s
        sets.append((name, load(path)))
    if len(sets) > 1:
        merged = {}
        for _, cells in sets:
            for k, v in cells.items():
                c = merged.setdefault(k, [0, 0.0, 0.0])
                c[0] += v[0]; c[1] += v[1]; c[2] += v[2]
        sets.append(("合并", merged))
    best = None
    for name, cells in sets:
        modes = sorted({m for (m, _, _) in cells})
        bins = sorted({b for (_, b, _) in cells})
        print("")
        print("===== 来源 %s:位图形态,%d 个格 =====" % (name, len(cells)))
        if not cells:
            continue
        ks = sorted({k for (_, _, k) in cells})
        for mode in modes:
            print("-- 模式 %d(%s):R 按 flop/HT 桶(行数不足 %d 的格记 -)" % (mode, "值在 SMEM" if mode == 1 else "值直写全局", a.min_rows))
            print("%-16s" % "flop/HT 下界" + "".join(" %7.2f" % (2.0 ** (k / 2.0)) for k in ks))
            for b in [None] + bins:
                cv = {k: (n, r) for k, n, r in curve(cells, mode, b, a.min_rows)}
                print("%-16s" % ("各 bin 合并" if b is None else "bin%d" % b) + "".join((" %7.3f" % cv[k][1]) if k in cv else "       -" for k in ks))
            for b in [None] + bins:
                cv = curve(cells, mode, b, a.min_rows)
                x = cross(cv)
                top = (2.0 ** ((cv[-1][0] + 1) / 2.0)) if cv else 0.0
                print("FX_CROSS %s %d %s %s   (该曲线量程到 %.1f,%d 个够数的桶)" % (
                    name, mode, "all" if b is None else "bin%d" % b, ("%.2f" % x) if x else "none", top, len(cv)))
                if name == sets[-1][0] and b is not None and x is not None:
                    best = x if best is None else min(best, x)
    print("")
    if best is None:
        print("FX_SUGGEST_PCT none   (没有哪条逐 bin 曲线越过 R=1:量程内两遍式都不输,或数据不够)")
    else:
        print("FX_SUGGEST_PCT %d   (= %.2f × 各模式各 bin 交点的最小值 %.2f,取整到百分数;来源 %s)" % (
            int(math.floor(a.margin * best * 100)), a.margin, best, sets[-1][0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
