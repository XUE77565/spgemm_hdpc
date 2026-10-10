#!/usr/bin/env python3
"""L0 337 全集屏:A(全hash) vs C(默认) ×3 交替,暖净窗中位。
每阵记录两配置净窗,输出逐阵 + L0 择优 GM(相对恒 C)。
双卡并行:大阵(>50MB mtx)分 GPU1,小阵分 GPU0。
用法:.venv/bin/python scripts/l0_screen337.py [--reps 3]
"""
import argparse, math, os, re, subprocess, statistics, sys, threading, queue
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_test")
DIR = os.path.join(REPO, "data", "ocean", "square")
OUT = os.path.join(REPO, "compare", "l0_337")
TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
H2D = re.compile(r"\[hash-prof\] h2d\s+([\d.]+)")
D2H = re.compile(r"\[hash-prof\] d2h\s+([\d.]+)")


def run(m, allhash, gpu, timeout=1800):
    env = dict(os.environ, METHOD="hash", WARMUP_N="3", USE_MEMPOOL="1", MP_HOST_MB="8192",
               D6EXP="0", K7="0", CUDA_VISIBLE_DEVICES=str(gpu))
    if allhash:
        env["DITER_MIN_FLOP"] = "999999999"
    try:
        r = subprocess.run([BIN, os.path.join(DIR, m + ".mtx")],
                           capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    t = TOT.findall(r.stderr); h = H2D.findall(r.stderr); d = D2H.findall(r.stderr)
    if not (t and h and d):
        return None
    return float(t[-1]) - float(h[-1]) - float(d[-1])


def worker(mats, gpu, reps, results, lock):
    for m in mats:
        a_vals, c_vals = [], []
        for rep in range(reps):
            v = run(m, True, gpu)
            if v: a_vals.append(v)
            v = run(m, False, gpu)
            if v: c_vals.append(v)
        with lock:
            results[m] = (statistics.median(a_vals) if a_vals else None,
                          statistics.median(c_vals) if c_vals else None)
            a_s = f"{results[m][0]:.1f}" if results[m][0] else "DNF"
            c_s = f"{results[m][1]:.1f}" if results[m][1] else "DNF"
            done = sum(1 for r in results.values() if r[0] is not None or r[1] is not None)
            print(f"[{done}/{len(ALL_MATS)}] {m:<24} A={a_s:<8} C={c_s:<8}", flush=True)


if __name__ == "__main__":
    ALL_MATS = sorted(os.path.splitext(f)[0] for f in os.listdir(DIR) if f.endswith(".mtx"))
    # 按文件大小分:大阵→GPU1,小阵→GPU0(负载均衡)
    big, small = [], []
    for m in ALL_MATS:
        sz = os.path.getsize(os.path.join(DIR, m + ".mtx"))
        (big if sz > 50e6 else small).append(m)
    results = {}
    lock = threading.Lock()
    t0 = threading.Thread(target=worker, args=(small, 0, 3, results, lock))
    t1 = threading.Thread(target=worker, args=(big, 1, 3, results, lock))
    t0.start(); t1.start(); t0.join(); t1.join()
    # 汇总
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "l0_337_results.csv"), "w") as f:
        f.write("mat,allhash_ms,default_ms,better,ratio\n")
        for m in ALL_MATS:
            a, c = results.get(m, (None, None))
            if a and c:
                better = "A" if a < c else "C"
                ratio = min(a, c) / c
                f.write(f"{m},{a:.2f},{c:.2f},{better},{ratio:.4f}\n")
            else:
                f.write(f"{m},{a or 'DNF'},{c or 'DNF'},,1.0\n")
    ratios = [min(a, c) / c for m, (a, c) in results.items() if a and c]
    a_wins = sum(1 for m, (a, c) in results.items() if a and c and a < c * 0.98)
    c_wins = sum(1 for m, (a, c) in results.items() if a and c and c < a * 0.98)
    n = len(ratios)
    if n:
        gm = math.exp(sum(math.log(r) for r in ratios) / n)
        print(f"\n{'='*50}")
        print(f"完成 {n} 阵 / DNF {len(ALL_MATS)-n} 阵")
        print(f"A(全hash)赢 {a_wins} 阵 / C(默认)赢 {c_wins} 阵 / 持平 {n-a_wins-c_wins} 阵")
        print(f"L0 择优 vs 恒C: GM={gm:.4f}")
