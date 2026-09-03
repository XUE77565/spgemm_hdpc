#!/usr/bin/env python3
"""通用 env A/B 电池(净窗用):基线(全关)vs 实验(envs 开),同窗交替 RUNS 轮取中位 compute-only。
nnz 红旗:>1% 不一致 = 无效测量。
用法:.venv/bin/python scripts/ab_env.py "BATCH_S=1 COMPACT_T=1 COUNT_TINY=1" mat1 mat2 ...
      .venv/bin/python scripts/ab_env.py "H2DOVL=1" mat1 mat2 ..."""
import os, sys, re, subprocess, statistics

BIN = os.environ.get("SPGEMM_BIN", "./spgemm_test")
DIR = "data/ocean/square"
RUNS = 3

def one(mtx, env_str):
    env = dict(os.environ, MP_HOST_MB="8192", USE_MEMPOOL="1", METHOD="adaptive", DBG="1")
    for kv in env_str.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
    try:
        r = subprocess.run([BIN, f"{DIR}/{mtx}.mtx"], capture_output=True, text=True, env=env, timeout=300)
    except subprocess.TimeoutExpired:
        return None, None
    prof = re.findall(r"\[hash-prof\]\s+(\S+)\s+([0-9.]+)\s*ms", r.stderr)
    nnz = re.findall(r"Result C:.*?nnz\s*=\s*(\d+)", r.stdout)
    if not prof or not nnz:
        return None, None
    keys = [k for k, _ in prof]
    idx = len(keys) - 1 - keys[::-1].index("h2d") if "h2d" in keys else 0
    ph = {k: float(v) for (k, v) in list(prof)[idx:]}
    tot = ph.get("TOTAL(GPU)")
    if tot is None:
        return None, None
    return tot - ph.get("h2d", 0) - ph.get("d2h", 0), int(nnz[-1])

def med(v):
    return statistics.median(v) if v else None

env_on = sys.argv[1]
mats = sys.argv[2:]
print(f"A/B: off vs [{env_on}] ×{RUNS} 轮交替\n{'matrix':<20}{'off(ms)':>10}{'on(ms)':>10}{'Δ%':>8}  nnz")
for mtx in mats:
    off, on, nz = [], [], set()
    for _ in range(RUNS):
        for e, acc in (("", off), (env_on, on)):
            c, n = one(mtx, e)
            if c is not None:
                acc.append(c)
                if n: nz.add(n)
    mo, mn = med(off), med(on)
    if mo and mn and len(nz) == 1:
        print(f"{mtx:<20}{mo:>10.3f}{mn:>10.3f}{(mn/mo-1)*100:>7.1f}%  {list(nz)[0]}")
    else:
        print(f"{mtx:<20}{mo!s:>10}{mn!s:>10}  ⚠nnz={nz if len(nz)!=1 else list(nz)[0]}")
