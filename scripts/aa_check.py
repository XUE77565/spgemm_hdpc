#!/usr/bin/env python3
"""A/A 对照臂判读:ab_arms.py 的逐发 CSV 里,对照臂(env 与基线臂完全相同)对基线臂的逐阵中位比。

计时本身稳不稳的直接量(dialog/qa-006 双方对齐):
  干净 = |GM − 1| ≤ gm_tol(默认 0.01)且 非微级阵(基线中位 co ≥ 1ms)的比全部落在 1 ± mat_tol(默认 0.05)内。
  阈值出处:bm2_r2 的无侵入电池(基线件对开关关)GM 0.9918、最差 1.034 且为微级。
输出(机读行在前):AA_N / AA_GM / AA_NONMICRO_MIN / AA_NONMICRO_MAX / AA_VERDICT clean|polluted|nodata,随后是逐阵表。
用法:aa_check.py CSV [--base off] [--aa aa] [--gm-tol 0.01] [--mat-tol 0.05]
"""
import argparse, csv, math, statistics, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--base", default="off")
    ap.add_argument("--aa", default="aa")
    ap.add_argument("--gm-tol", type=float, default=0.01)
    ap.add_argument("--mat-tol", type=float, default=0.05)
    a = ap.parse_args()
    d = {}
    order = []
    try:
        for r in csv.DictReader(open(a.csv, encoding="utf-8")):
            try:
                co = float(r["co"])
            except (KeyError, ValueError, TypeError):
                continue
            if co <= 0:
                continue
            d.setdefault((r["mat"], r["arm"]), []).append(co)
            if r["mat"] not in order:
                order.append(r["mat"])
    except OSError as e:
        print("AA_VERDICT nodata")
        print("# 读不到 %s: %s" % (a.csv, e))
        return 0
    rows = []
    for m in order:
        b, x = d.get((m, a.base)), d.get((m, a.aa))
        if not b or not x:
            continue
        bm = statistics.median(b)
        rows.append((m, bm, statistics.median(x) / bm, bm < 1.0))
    if not rows:
        print("AA_VERDICT nodata")
        return 0
    gm = math.exp(sum(math.log(r[2]) for r in rows) / len(rows))
    nm = [r[2] for r in rows if not r[3]]
    lo, hi = (min(nm), max(nm)) if nm else (float("nan"), float("nan"))
    clean = abs(gm - 1.0) <= a.gm_tol and all(abs(v - 1.0) <= a.mat_tol for v in nm)
    print("AA_N %d" % len(rows))
    print("AA_GM %.4f" % gm)
    print("AA_NONMICRO_MIN %.4f" % lo)
    print("AA_NONMICRO_MAX %.4f" % hi)
    print("AA_VERDICT %s" % ("clean" if clean else "polluted"))
    print("%-18s %10s %8s  %s" % ("mat", a.base + "_ms", a.aa + "/" + a.base, "级"))
    for m, bm, r, micro in rows:
        flag = "" if micro or abs(r - 1.0) <= a.mat_tol else "  ← 超出 ±%.0f%%" % (100 * a.mat_tol)
        print("%-18s %10.3f %8.4f  %s%s" % (m, bm, r, "微" if micro else ("中" if bm < 10 else "稳"), flag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
