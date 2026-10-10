#!/usr/bin/env python3
"""ocean 337 全集跑:convert + spgemm → 每阵 compute 时间 → 与我方 L0 择优配对 GM。
用法:python scripts/ocean_337.py [--gpu 0]
"""
import os, re, subprocess, sys, json, math, glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCEAN_CONV = os.path.join(REPO, "ocean", "convert")
OCEAN_RUN = os.path.join(REPO, "ocean", "spgemm")
OCEAN_DIR = os.path.join(REPO, "ocean")
DIR = os.path.join(REPO, "data", "ocean", "square")
OUT = os.path.join(REPO, "compare", "ocean_337")

MATS = sorted(os.path.splitext(f)[0] for f in os.listdir(DIR) if f.endswith(".mtx"))


def run_ocean(m, timeout=300):
    csr = "/tmp/ocean_337_tmp.csr"
    stats = os.path.join(OCEAN_DIR, "stats.json")
    if os.path.exists(stats):
        os.remove(stats)
    try:
        r1 = subprocess.run([OCEAN_CONV, os.path.join(DIR, m + ".mtx"), csr],
                            capture_output=True, timeout=timeout)
        if r1.returncode != 0:
            return None
        r2 = subprocess.run([OCEAN_RUN, csr, "config/bench_detail.json"],
                            cwd=OCEAN_DIR, capture_output=True, timeout=timeout)
        if r2.returncode != 0:
            return None
    except (subprocess.TimeoutExpired, Exception):
        return None
    if not os.path.exists(stats):
        return None
    try:
        d = json.load(open(stats))
        return d.get("total_time", None)
    except:
        return None


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, "ocean_337.csv")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path).readlines()[1:]:
            done.add(line.split(",")[0])
    with open(out_path, "a") as f:
        if not done:
            f.write("mat,ocean_ms\n")
        for i, m in enumerate(MATS):
            if m in done:
                continue
            t = run_ocean(m)
            v = f"{t:.4f}" if t else "DNF"
            f.write(f"{m},{v}\n"); f.flush()
            if i % 20 == 0:
                print(f"[{i+1}/{len(MATS)}] {m}: {v}", flush=True)
    print(f"ocean 337 完成 → {out_path}")
