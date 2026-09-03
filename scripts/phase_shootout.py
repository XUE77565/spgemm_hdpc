#!/usr/bin/env python3
"""分相对拍靶纸(docs/76 第二枪):我们 HashProf vs Ocean stats.json,逐相并排。
用法:.venv/bin/python scripts/phase_shootout.py pre2 TSOPF_RS_b2383_c1 ...
⚠ 须净窗(GPU 独占);Ocean 侧:ocean/spgemm <csr> config/bench_detail.json → stats.json"""
import os, sys, re, json, subprocess
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

BIN = "./spgemm_test"
DIR = "data/ocean/square"
OC = os.path.join(REPO, "ocean")

def ours(mtx):
    env = dict(os.environ, MP_HOST_MB="8192", USE_MEMPOOL="1", METHOD="adaptive", DBG="1")
    r = subprocess.run([BIN, f"{DIR}/{mtx}.mtx"], capture_output=True, text=True, env=env, timeout=900)
    prof = re.findall(r"\[hash-prof\]\s+(\S+)\s+([0-9.]+)\s*ms", r.stderr)
    if not prof: return None
    keys = [k for k, _ in prof]
    idx = len(keys) - 1 - keys[::-1].index("h2d")
    ph = {}
    for k, v in list(prof)[idx:]:
        ph[k] = ph.get(k, 0) + float(v)   # 同名相位(dense 双 pass)累计
    tot = ph.pop("TOTAL(GPU)", None)
    ph.pop("h2d", None); ph.pop("d2h", None)
    return ph, tot - 0 if tot else None

def theirs(mtx):
    csr = f"/tmp/shootout_{mtx}.csr"
    stats = f"{OC}/stats.json"
    if os.path.exists(stats): os.remove(stats)
    r1 = subprocess.run([f"{OC}/convert", f"{DIR}/{mtx}.mtx", csr], capture_output=True, timeout=300)
    if r1.returncode != 0: return None
    r2 = subprocess.run([f"{OC}/spgemm", csr, f"{OC}/config/bench_detail.json"],
                        capture_output=True, timeout=900, cwd=OC)
    if not os.path.exists(stats): return None
    d = json.load(open(stats))
    tim = d.get("timing", {})
    flat = {}
    def walk(o, pre=""):
        if isinstance(o, dict):
            for k, v in o.items(): walk(v, f"{pre}.{k}" if pre else k)
        elif isinstance(o, (int, float)) and o > 0.001:
            flat[pre] = o
    walk(tim)
    return flat, d.get("total_time")

for mtx in sys.argv[1:]:
    o = ours(mtx); t = theirs(mtx)
    print(f"\n===== {mtx} =====")
    if not o or not t:
        print("  数据不全", o is None, t is None); continue
    our_ph, our_tot = o; th_ph, th_tot = t
    print(f"{'我们相位':<24}{'ms':>8}   | {'Ocean 相位':<28}{'ms':>8}")
    print(f"{'TOTAL(compute)':<24}{sum(our_ph.values()):>8.2f}   | {'total_time':<28}{th_tot:>8.2f}")
    a = sorted(our_ph.items(), key=lambda x: -x[1])
    b = sorted(th_ph.items(), key=lambda x: -x[1])
    for i in range(max(len(a), len(b))):
        l = f"{a[i][0]:<24}{a[i][1]:>8.2f}" if i < len(a) else " " * 32
        r = f"| {b[i][0]:<28}{b[i][1]:>8.2f}" if i < len(b) else ""
        print(f"{l}   {r}")
