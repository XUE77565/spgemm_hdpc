#!/usr/bin/env python3
"""v29 回归复验(docs/65 口径):当前 build vs v28 build,同窗交替各 3 轮取中位(compute-only)。
判定:|Δ|<3% = 噪声带;>5% 才算真回归/改善。nnz 红旗。
用法:.venv/bin/python scripts/reverify_v29_regressions.py [mat ...](默认 = v29 报告的回归清单)"""
import os, sys, re, subprocess, statistics

CUR = os.path.abspath("./spgemm_test")
V28 = "/tmp/v28build/spgemm_test"
DIR = "data/ocean/square"
RUNS = 3
DEFAULT = ["c-58", "c-66", "std1_Jac3_db", "a0nsdsil", "torso1", "std1_Jac2",
           "std1_Jac3", "Ge87H76", "Stanford_Berkeley", "pkustk14",
           "c-64", "c-73"]   # 后两个 = 改善守卫(方向应保持 −)

def one(binpath, mtx):
    env = dict(os.environ, USE_MEMPOOL="1", METHOD="adaptive")
    # MP_HOST_MB 不设 → run 侧自动;这里直接跑 binary,大阵需 32G
    if "MP_HOST_MB" not in os.environ:
        env["MP_HOST_MB"] = "8192"
    try:
        r = subprocess.run([binpath, f"{DIR}/{mtx}.mtx"], capture_output=True, text=True, env=env, timeout=600)
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

print(f"{'matrix':<18}{'v28(ms)':>9}{'v29(ms)':>9}{'Δ%':>8}  verdict")
for mtx in (sys.argv[1:] or DEFAULT):
    a28, a29, nz = [], [], set()
    for _ in range(RUNS):
        for bp, acc in ((V28, a28), (CUR, a29)):
            c, n = one(bp, mtx)
            if c is not None:
                acc.append(c)
                if n: nz.add(n)
    m28, m29 = med(a28), med(a29)
    if m28 and m29 and len(nz) == 1:
        d = (m29 / m28 - 1) * 100
        v = "噪声带" if abs(d) < 3 else ("真回归" if d > 5 else ("真改善" if d < -5 else "边缘"))
        print(f"{mtx:<18}{m28:>9.2f}{m29:>9.2f}{d:>7.1f}%  {v}  nnz={list(nz)[0]}")
    else:
        print(f"{mtx:<18}{m28!s:>9}{m29!s:>9}  ⚠ 数据不全 nnz={nz}")
