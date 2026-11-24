#!/usr/bin/env python3
"""bm2_gate_cal.py — 想法二(HH_BM2)行门的标定:用训练池的逐行配对周期,选 (值区模式, FX, CX)

输入(--dir 目录下,每个训练阵三组文件;由作业用同一二进制、HASH_ROWTIME 落盘得到):
  <阵>.off.rt   HH_BM2 关:每行 "r a_len flop est span ovf 周期 c_nnz bkid"(src/spgemm_kernel_hash.cu 管线公共尾)
  <阵>.m1.rt    HH_BM2=1 且不设行门(凡 SMEM 装得下的 HH 行都走两遍式);走了两遍式的行,周期的最高位(2^63)被置 1
  <阵>.m2.rt    HH_BM2=2 同上
  <阵>.m1.log / <阵>.m2.log   对应运行的 stderr(取 [bm2-stat] 行里每个 bin 的 leg:0=位图形态 1=legacy 形态)

做法:同一行在 off 跑里的周期 th(哈希路径)与在 m 跑里的周期 tk(两遍式)配对——任何行门的效果都能离线算:
  R(门) = (Σ_门内行 tk + Σ_其余 HH 行 th) / Σ_全部 HH 行 th        (逐阵算,再对阵取算术平均)
门的形态与核内一致(hh_bm2_gate_kernel):flop×100 ≤ FX×HT ∧ cw×100 ≤ CX×HT,0=不设;HT=该 bin 的表长,
cw=⌈span/32⌉。位图形态(leg=0)与 legacy 形态(leg=1)分开选,因为两种形态下哈希路径的固定成本不同。
逐行周期是块从进入到写完的时长(同 bin 两种路径块宽、每 SM 块数相同,时长比 = 占用 SM 时间比)。

输出(--out 前缀):
  <前缀>.txt        每个 (leg, 模式) 的网格前几名、"全收"与"最优门"的 R、逐阵表、非合格行的实例税读数
  <前缀>.env        机读:AUTO_OK=0/1 与 AUTO_ENV=K=V,K=V…(给 ab_arms 的臂用)
  <前缀>_cells.csv  (leg, bin, 模式, flop/HT 半倍频程桶, cw/HT 半倍频程桶) 的行数、Σth、Σtk——离线复核门形态用
只用训练池;428 主账 / flex142 / external_72 不得作为本脚本的输入(纪律 7)。
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HT = {4: 512, 5: 2048, 6: 2719, 7: 4096, 8: 8192, 9: 16253}
HH_BINS = (4, 5, 7, 8, 9)
TOP = np.uint64(1) << np.uint64(63)
LOW = TOP - np.uint64(1)
FXS = [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 0]
CXS = [5, 10, 20, 40, 80, 160, 0]
RE_STAT = re.compile(r"\[bm2-stat\] bin=(\d+) mode=(\d+) leg=(\d+)")


def load_rt(path):
    a = np.loadtxt(path, dtype=np.uint64, ndmin=2)
    if a.shape[1] < 9:
        raise ValueError("%s: 列数 %d < 9" % (path, a.shape[1]))
    return a


def legs_from_log(path):
    legs = {}
    try:
        with open(path, errors="replace") as f:
            for ln in f:
                m = RE_STAT.search(ln)
                if m:
                    legs[int(m.group(1))] = int(m.group(3))     # 同一 bin 多次出现取最后一次(计时轮)
    except OSError:
        pass
    return legs


def winsor(x, q=99.9):
    if x.size == 0:
        return x
    cap = np.percentile(x, q)
    return np.minimum(x, cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-rows", type=int, default=200, help="一个阵在某形态下的 HH 行少于此数则不计入该形态的目标")
    ap.add_argument("--min-gain", type=float, default=0.005, help="最优门的平均 R 高于 1−本值 则该形态不开")
    a = ap.parse_args()

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    names = sorted(os.path.basename(p)[:-len(".off.rt")] for p in glob.glob(os.path.join(a.dir, "*.off.rt")))
    # 逐行记录:阵号、bin、leg、flop/HT×100、cw/HT×100、th、t1、t2(未走=nan)
    R = {k: [] for k in ("mat", "bin", "leg", "fr", "cr", "th", "t1", "t2", "nt1", "nt2")}
    used = []
    for name in names:
        try:
            off = load_rt(os.path.join(a.dir, name + ".off.rt"))
            m = {1: load_rt(os.path.join(a.dir, name + ".m1.rt")), 2: load_rt(os.path.join(a.dir, name + ".m2.rt"))}
        except Exception as ex:
            say("# 跳过 %s: %r" % (name, ex))
            continue
        n = off.shape[0]
        if m[1].shape[0] != n or m[2].shape[0] != n:
            say("# 跳过 %s: 三份行数不等" % name)
            continue
        legs = legs_from_log(os.path.join(a.dir, name + ".m2.log"))
        legs.update({b: l for b, l in legs_from_log(os.path.join(a.dir, name + ".m1.log")).items() if b not in legs})
        bk = off[:, 8].astype(np.int64)
        hh = np.isin(bk, HH_BINS) & (off[:, 5] == 0) & (off[:, 6] > 0)
        if not hh.any():
            continue
        leg = np.full(n, -1, dtype=np.int64)
        for b in HH_BINS:
            if b in legs:
                leg[bk == b] = legs[b]
        hh &= leg >= 0
        if not hh.any():
            continue
        ht = np.zeros(n)
        for b in HH_BINS:
            ht[bk == b] = HT[b]
        flop = off[:, 2].astype(np.float64)
        cw = ((off[:, 4].astype(np.int64) + 31) >> 5).astype(np.float64)
        th = winsor(off[hh, 6].astype(np.float64))
        tk, ntk = {}, {}
        for k in (1, 2):
            rt = m[k][hh, 6]
            took = (rt & TOP) != 0
            cyc = (rt & LOW).astype(np.float64)
            okk = took & (cyc > 0) & (m[k][hh, 5] == 0)
            t = np.where(okk, cyc, np.nan)
            if okk.any():
                t[okk] = winsor(t[okk])
            tk[k] = t
            nt = np.where(~took & (cyc > 0) & (m[k][hh, 5] == 0), cyc, np.nan)    # 没走两遍式的行在 BM2 实例里的哈希路径周期
            ntk[k] = nt
        mid = len(used)
        used.append(name)
        cnt = int(hh.sum())
        R["mat"].append(np.full(cnt, mid)); R["bin"].append(bk[hh]); R["leg"].append(leg[hh])
        R["fr"].append(flop[hh] * 100.0 / ht[hh]); R["cr"].append(cw[hh] * 100.0 / ht[hh])
        R["th"].append(th); R["t1"].append(tk[1]); R["t2"].append(tk[2]); R["nt1"].append(ntk[1]); R["nt2"].append(ntk[2])
    if not used:
        say("# 没有可用的训练阵(HH bin 无行或文件缺失)")
        with open(a.out + ".env", "w") as f:
            f.write("AUTO_OK=0\n")
        with open(a.out + ".txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        return 0
    D = {k: np.concatenate(v) for k, v in R.items()}
    M = len(used)
    say("# bm2_gate_cal: 训练阵 %d 个,HH 行合计 %d(位图形态 %d / legacy 形态 %d)"
        % (M, D["th"].size, int((D["leg"] == 0).sum()), int((D["leg"] == 1).sum())))

    best = {}
    for lg in (0, 1):
        sel_leg = D["leg"] == lg
        base = np.bincount(D["mat"][sel_leg], weights=D["th"][sel_leg], minlength=M)
        nrow = np.bincount(D["mat"][sel_leg], minlength=M)
        okm = (nrow >= a.min_rows) & (base > 0)
        say("")
        say("===== 形态 leg=%d(%s):计入目标的阵 %d 个(HH 行 ≥ %d)=====" % (lg, "位图" if lg == 0 else "legacy", int(okm.sum()), a.min_rows))
        if not okm.any():
            best[lg] = (0, 0, 0, 1.0)
            continue
        cand = []
        for k in (1, 2):
            t = D["t%d" % k]
            v = sel_leg & ~np.isnan(t)
            delta = np.where(v, t - D["th"], 0.0)
            for fx in FXS:
                for cx in CXS:
                    g = v & ((fx == 0) | (D["fr"] <= fx)) & ((cx == 0) | (D["cr"] <= cx))
                    sd = np.bincount(D["mat"][g], weights=delta[g], minlength=M)
                    r = 1.0 + sd[okm] / base[okm]
                    cov = np.bincount(D["mat"][g], weights=D["th"][g], minlength=M)[okm] / base[okm]
                    cand.append((float(r.mean()), k, fx, cx, float(r.max()), float(cov.mean())))
        cand.sort()
        say("%6s %4s %6s %6s %9s %9s %9s" % ("名次", "模式", "FX", "CX", "平均R", "最差阵R", "覆盖(时间)"))
        for i, (r, k, fx, cx, rmax, cov) in enumerate(cand[:8]):
            say("%6d %4d %6s %6s %9.4f %9.4f %8.1f%%" % (i + 1, k, fx or "∞", cx or "∞", r, rmax, 100 * cov))
        for k in (1, 2):
            allr = [c for c in cand if c[1] == k and c[2] == 0 and c[3] == 0][0]
            say("  全收(模式 %d,不设门): 平均 R = %.4f,最差阵 R = %.4f" % (k, allr[0], allr[4]))
        r, k, fx, cx, rmax, cov = cand[0]
        if r > 1.0 - a.min_gain:
            say("  → 最优门的平均收益 < %.1f%%:该形态不开" % (100 * a.min_gain))
            best[lg] = (0, 0, 0, 1.0)
        else:
            say("  → 选:模式 %d,FX=%s,CX=%s(平均 R=%.4f)" % (k, fx or "∞", cx or "∞", r))
            best[lg] = (k, fx, cx, r)
        # 逐阵:全收 vs 所选门
        k_show = best[lg][0] or cand[0][1]
        t = D["t%d" % k_show]
        v = sel_leg & ~np.isnan(t)
        delta = np.where(v, t - D["th"], 0.0)
        fx, cx = (best[lg][1], best[lg][2]) if best[lg][0] else (cand[0][2], cand[0][3])
        g = v & ((fx == 0) | (D["fr"] <= fx)) & ((cx == 0) | (D["cr"] <= cx))
        r_all = 1.0 + np.bincount(D["mat"][v], weights=delta[v], minlength=M) / np.maximum(base, 1.0)
        r_gate = 1.0 + np.bincount(D["mat"][g], weights=delta[g], minlength=M) / np.maximum(base, 1.0)
        say("  逐阵(模式 %d):%-22s %9s %9s %9s %9s" % (k_show, "阵", "HH行", "合格行", "R全收", "R门"))
        for mi in np.argsort(-base):
            if nrow[mi] == 0:
                continue
            say("                 %-22s %9d %9d %9.4f %9.4f%s" % (used[mi], nrow[mi], int((v & (D["mat"] == mi)).sum()),
                                                                 r_all[mi], r_gate[mi], "" if okm[mi] else "  (行数不足,不计入目标)"))
        # 实例税:没走两遍式的 HH 行,在 BM2 实例里的周期 ÷ 在原实例里的周期
        for k in (1, 2):
            nt = D["nt%d" % k]
            w = sel_leg & ~np.isnan(nt)
            if w.sum() >= 100:
                say("  实例税读数(模式 %d,未走两遍式的 %d 行):Σ周期(BM2 实例) / Σ周期(原实例) = %.4f"
                    % (k, int(w.sum()), float(nt[w].sum() / D["th"][w].sum())))

    # 机读 env
    k0, fx0, cx0, _ = best[0]
    k1, fx1, cx1, _ = best[1]
    with open(a.out + ".env", "w") as f:
        if k0 == 0 and k1 == 0:
            f.write("AUTO_OK=0\n")
            say("\n# 两种形态都没有 ≥%.1f%% 的平均收益:不产生 auto 臂" % (100 * a.min_gain))
        else:
            env = "HH_BM2=%d,HH_BM2_L1=%d,HH_BM2_FX=%d,HH_BM2_CX=%d,HH_BM2_FX1=%d,HH_BM2_CX1=%d" % (k0, k1, fx0, cx0, fx1, cx1)
            f.write("AUTO_OK=1\nAUTO_ENV=%s\n" % env)
            say("\n# auto 臂: " + env)

    # 桶表(半倍频程):离线复核门形态
    with open(a.out + "_cells.csv", "w") as f:
        f.write("leg,bin,mode,log2x2_flop_over_ht,log2x2_cw_over_ht,rows,sum_th,sum_tk\n")
        fb = np.floor(2 * np.log2(np.maximum(D["fr"], 1e-9) / 100.0)).astype(np.int64)
        cb = np.floor(2 * np.log2(np.maximum(D["cr"], 1e-9) / 100.0)).astype(np.int64)
        for k in (1, 2):
            t = D["t%d" % k]
            v = ~np.isnan(t)
            key = np.stack([D["leg"][v], D["bin"][v], fb[v], cb[v]], axis=1)
            if key.shape[0] == 0:
                continue
            uq, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            cnt = np.bincount(inv, minlength=uq.shape[0])
            sth = np.bincount(inv, weights=D["th"][v], minlength=uq.shape[0])
            stk = np.bincount(inv, weights=t[v], minlength=uq.shape[0])
            for i in range(uq.shape[0]):
                f.write("%d,%d,%d,%d,%d,%d,%.0f,%.0f\n" % (uq[i, 0], uq[i, 1], k, uq[i, 2], uq[i, 3], cnt[i], sth[i], stk[i]))
    with open(a.out + ".txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
