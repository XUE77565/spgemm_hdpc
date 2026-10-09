#!/usr/bin/env python3
"""D3② A/B:D6EXP 学生派发 off/on 交替序 ×3,暖净窗口径(TOTAL−h2d−d2h 末轮)。
判据(study/34 D4 前哨):bloweybq 显著赢(L0→k7);其余阵零未解释回归;pre2 看 k7 份额。
用法:.venv/bin/python scripts/ab_d6exp.py [mat ...]
"""
import os, re, subprocess, sys, statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_test")
DEFAULT = ["bloweybq", "pre2", "c-43", "Si10H16", "c-64", "in-2004"]
TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
H2D = re.compile(r"\[hash-prof\] h2d\s+([\d.]+)")
D2H = re.compile(r"\[hash-prof\] d2h\s+([\d.]+)")


def run(m, d6):
    env = dict(os.environ, METHOD="hash", WARMUP_N="5",
               D6EXP=("1" if d6 else "0"), USE_MEMPOOL="1", MP_HOST_MB="8192")
    r = subprocess.run([BIN, os.path.join(REPO, "data/ocean/square", m + ".mtx")],
                       capture_output=True, text=True, env=env, timeout=1500)
    t = TOT.findall(r.stderr); h = H2D.findall(r.stderr); d = D2H.findall(r.stderr)
    if not (t and h and d):
        return None
    return float(t[-1]) - float(h[-1]) - float(d[-1])


def main():
    mats = sys.argv[1:] or DEFAULT
    print(f"{'mat':<10}{'off 中位':>10}{'on 中位':>10}{'on/off':>9}")
    for m in mats:
        off, on = [], []
        for rep in range(3):          # 交替序防时钟爬坡偏置
            v = run(m, False); v and off.append(v)
            v = run(m, True);  v and on.append(v)
        if not off or not on:
            print(f"{m:<10} FAILED")
            continue
        o, n = statistics.median(off), statistics.median(on)
        print(f"{m:<10}{o:>10.2f}{n:>10.2f}{n/o:>9.3f}", flush=True)


if __name__ == "__main__":
    main()
