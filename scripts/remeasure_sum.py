#!/usr/bin/env python3
"""影子里的"待复测阵"加轮复测的判读(读 shadow428.py 的 PREFIX.csv;每阵每臂 ≥15 发)。

背景:全 428 影子(prof_p1)里,少数阵在 A/A 对照臂上自己就偏离超 ±5%,或在实验臂上 >1.10;其中 rajat 族是基线自身双峰
(同一配置的三发里就有 49ms 与 62ms 两种读数),三发中位说明不了问题。这里对每个阵单独给判定。
预先写定的判法(dialog/qa-011,待本机侧认可;跑之前写定,不看数据改):
  稳定阵(该阵自己的 aa/off 中位比在 ±5% 内):实验臂中位比 ≤ 1.10 → 放行;否则 → 仍红。
  不稳阵(该阵 aa/off 中位比超 ±5%,即基线自身不稳/双峰):分位数逐档比——
      实验臂的 最小值、下四分位、上四分位 分别对 off 的同一分位,比值依次 ≤ 1.05 / 1.05 / 1.10 → 放行;否则 → 仍红。
      (双峰时下四分位落在快模、上四分位落在慢模;逐档不变慢 = 两个模都没被拖慢,也没有更常掉进慢模。)
输出:逐阵逐臂的 n / 最小 / 下四分位 / 中位 / 上四分位 / 最大,各比值,该阵判定;末尾汇总 REMEASURE_RED n。
用法:remeasure_sum.py PREFIX [--base off] [--aa aa]
"""
import argparse, csv, statistics, sys


def q(v, p):
    s = sorted(v)
    if len(s) == 1:
        return s[0]
    x = p * (len(s) - 1)
    i = int(x)
    f = x - i
    return s[i] if i + 1 >= len(s) else s[i] * (1 - f) + s[i + 1] * f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--base", default="off")
    ap.add_argument("--aa", default="aa")
    a = ap.parse_args()
    co, order, arms = {}, [], []
    try:
        for r in csv.DictReader(open(a.prefix + ".csv", encoding="utf-8")):
            try:
                c = float(r["co"])
            except ValueError:
                continue
            if c <= 0:
                continue
            co.setdefault((r["mat"], r["arm"]), []).append(c)
            if r["mat"] not in order:
                order.append(r["mat"])
            if r["arm"] not in arms:
                arms.append(r["arm"])
    except OSError as e:
        print("# 读不到 %s.csv: %s" % (a.prefix, e))
        return 0
    exp = [x for x in arms if x not in (a.base, a.aa)]
    red = {x: [] for x in exp}
    print("# remeasure_sum: %d 阵;臂 %s;实验臂 %s" % (len(order), " ".join(arms), " ".join(exp)))
    for m in order:
        b = co.get((m, a.base))
        if not b:
            print("%-22s 基线臂无数据" % m)
            continue
        print("== %s" % m)
        print("   %-5s %3s %9s %9s %9s %9s %9s" % ("臂", "n", "最小", "下四分", "中位", "上四分", "最大"))
        for x in arms:
            v = co.get((m, x))
            if v:
                print("   %-5s %3d %9.3f %9.3f %9.3f %9.3f %9.3f" % (x, len(v), min(v), q(v, .25), statistics.median(v), q(v, .75), max(v)))
        bm = statistics.median(b)
        va = co.get((m, a.aa))
        ra = statistics.median(va) / bm if va else float("nan")
        stable = va is not None and abs(ra - 1.0) <= 0.05
        spread = (q(b, .75) - q(b, .25)) / bm
        print("   该阵 A/A 中位比 %.4f → %s(基线臂四分位距/中位 = %.1f%%)" % (ra, "稳定阵" if stable else "不稳阵", 100 * spread))
        for x in exp:
            v = co.get((m, x))
            if not v:
                continue
            rmed = statistics.median(v) / bm
            rmin, r25, r75 = min(v) / min(b), q(v, .25) / q(b, .25), q(v, .75) / q(b, .75)
            if stable:
                ok = rmed <= 1.10
            else:
                ok = rmin <= 1.05 and r25 <= 1.05 and r75 <= 1.10
            if not ok:
                red[x].append(m)
            print("   %-5s 中位比 %.4f | 最小比 %.4f 下四分比 %.4f 上四分比 %.4f → %s" % (x, rmed, rmin, r25, r75, "放行" if ok else "仍红"))
    print("")
    for x in exp:
        print("REMEASURE_RED %s %d%s" % (x, len(red[x]), (" : " + " ".join(red[x])) if red[x] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
