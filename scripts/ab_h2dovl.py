#!/usr/bin/env python3
"""H2DOVL A/B(docs/71):交替双 binary 配置(compute-only = TOTAL−h2d−d2h),nnz 红旗。
用法:.venv/bin/python scripts/ab_h2dovl.py mat1 mat2 ..."""
import os, sys, re, subprocess, statistics
sys.path.insert(os.path.dirname(os.path.abspath(__file__))) if False else None
BIN = "./spgemm_test"
DIR = "data/ocean/square"
RUNS = 3   # 每配置 3 次(交替),取中位

def one(mtx, mode):
    env = dict(os.environ, MP_HOST_MB="8192", USE_MEMPOOL="1", METHOD="adaptive", H2DOVL=str(mode), DBG="1")
    try:
        r = subprocess.run([BIN, f"{DIR}/{mtx}.mtx"], capture_output=True, text=True, env=env, timeout=300)
    except subprocess.TimeoutExpired:
        return None, None
    # 取 timed run(最后一次 [hash] 段)的相位
    prof = re.findall(r"\[hash-prof\]\s+(\S+)\s+([0-9.]+)\s+ms", r.stderr)
    nnz = re.findall(r"Result C:.*?nnz\s*=\s*(\d+)", r.stdout)
    if not prof or not nnz:
        return None, None
    ph = {}
    for k, v in prof[-10:]:   # 末段
        ph[k] = float(v)
    # 末段完整 = 从最后一次 'h2d' 起
    keys = [k for k, _ in prof]
    if "h2d" in keys:
        idx = len(keys) - 1 - keys[::-1].index("h2d")
        ph = {k: float(v) for (k, v) in list(prof)[idx:]}
    total = ph.get("TOTAL(GPU)")
    if total is None:
        return None, None
    comp = total - ph.get("h2d", 0) - ph.get("d2h", 0)
    return comp, int(nnz[-1])

def med(v):
    return statistics.median(v) if v else None

print(f"{'matrix':<18}{'off(ms)':>10}{'on(ms)':>10}{'Δ%':>8}  nnz")
for mtx in sys.argv[1:]:
    off, on, nz = [], [], set()
    for _ in range(RUNS):
        for mode, acc in ((0, off), (1, on)):
            c, n = one(mtx, mode)
            if c is not None:
                acc.append(c)
                if n: nz.add(n)
    mo, mn = med(off), med(on)
    if mo and mn and len(nz) == 1:
        print(f"{mtx:<18}{mo:>10.3f}{mn:>10.3f}{(mn/mo-1)*100:>7.1f}%  {list(nz)[0]}")
    else:
        print(f"{mtx:<18}{mo!s:>10}{mn!s:>10}   ⚠nnz={nz if len(nz)!=1 else list(nz)[0]}")
