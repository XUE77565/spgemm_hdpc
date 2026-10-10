#!/usr/bin/env python3
"""D4 广域屏:51 阵 D6EXP off/on ×3 交替,暖净窗(TOTAL−h2d−d2h)中位。
输出:逐阵 on/off 比 + 新 GM = 1.3895 × (on/off GM)。
用法:CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/d4_screen.py [--reps 3]
"""
import argparse, math, os, re, subprocess, sys, statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_test")
DIR = os.path.join(REPO, "data", "ocean", "square")
TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
H2D = re.compile(r"\[hash-prof\] h2d\s+([\d.]+)")
D2H = re.compile(r"\[hash-prof\] d2h\s+([\d.]+)")

MATS51 = """bloweybq pf2177 af_2_k101 c-71 HTC_336_4438 para-4 af_shell4 bloweybl
hugetrace-00010 pre2 FEM_3D_thermal2 nd24k lowThrust_8 std1_Jac3 c-66b c-big
mult_dcop_02 crankseg_2 Si10H16 email-Enron lowThrust_11 nemeth25 lowThrust_5
Fault_639 nasasrb c-43 pkustk03 opt1 c-57 Serena a2nnsnsl vanbody shipsec1
msc23052 conf6_0-8x8-20 TSOPF_RS_b2052_c1 rajat28 net100 poisson3Db trdheim
bone010 F2 dielFilterV3clx c-64 invextr1_new hangGlider_4 pli TSOPF_FS_b39_c7
bbmat count""".split()


def run(m, d6, timeout=1500):
    env = dict(os.environ, METHOD="hash", WARMUP_N="5",
               D6EXP=("1" if d6 else "0"), USE_MEMPOOL="1", MP_HOST_MB="8192")
    try:
        r = subprocess.run([BIN, os.path.join(DIR, m + ".mtx")],
                           capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    t = TOT.findall(r.stderr); h = H2D.findall(r.stderr); d = D2H.findall(r.stderr)
    if not (t and h and d):
        return None
    return float(t[-1]) - float(h[-1]) - float(d[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mats", nargs="*", default=MATS51)
    a = ap.parse_args()
    ratios = []
    dnf = []
    print(f"D4 广域屏:{len(a.mats)} 阵 ×{a.reps} 交替  D6EXP off/on")
    print(f"{'mat':<22}{'off':>8}{'on':>8}{'on/off':>8}")
    for m in a.mats:
        off, on = [], []
        for rep in range(a.reps):
            v = run(m, False)
            if v: off.append(v)
            v = run(m, True)
            if v: on.append(v)
        if not off or not on:
            dnf.append(m)
            print(f"{m:<22} DNF", flush=True)
            continue
        o, n = statistics.median(off), statistics.median(on)
        r = n / o
        ratios.append((m, r))
        print(f"{m:<22}{o:>8.2f}{n:>8.2f}{r:>8.3f}", flush=True)
    if ratios:
        gm = math.exp(sum(math.log(r) for _, r in ratios) / len(ratios))
        new_gm = 1.3895 * gm
        print(f"\n{'='*50}")
        print(f"完成 {len(ratios)} 阵 / DNF {len(dnf)} 阵")
        print(f"on/off GM = {gm:.4f}")
        print(f"新 51 阵 GM ≈ 1.3895 × {gm:.4f} = {new_gm:.4f}")
        print(f"(基线 1.3895 → 目标 1.33±)")


if __name__ == "__main__":
    main()
