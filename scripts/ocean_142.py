#!/usr/bin/env python3
"""ocean 在 Flex 142 bench 上的基线(11-16):convert → spgemm Iteration 10 计时。
输出 compare/flexbench142/ocean.csv;与 pair.csv(我方+Flex)拼成三方对决。
"""
import csv, os, re, subprocess, sys

REPO = "/home/xueyizhou/spgemm-challenge"
OCEAN = os.path.join(REPO, "ocean")
DIR = os.path.join(REPO, "external_sota/FlexSpGEMM/data/test142")
CSR_DIR = "/tmp/ocean_csr_142"
OUT = os.path.join(REPO, "compare/flexbench142/ocean.csv")
os.makedirs(CSR_DIR, exist_ok=True)
CONFIG = os.path.join(OCEAN, "config/bench.json")
ITER10 = re.compile(r"Iteration 10.*?([\d.]+)\s*ms")

mats = sorted(f[:-4] for f in os.listdir(DIR) if f.endswith(".mtx"))
print(f"就绪 {len(mats)} 阵", flush=True)

f = open(OUT, "w", newline="")
w = csv.writer(f); w.writerow(["mat", "ocean_ms", "note"])
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
for i, m in enumerate(mats):
    mtx = os.path.join(DIR, m + ".mtx")
    csr = os.path.join(CSR_DIR, m + ".csr")
    if not os.path.exists(csr):
        r = subprocess.run([os.path.join(OCEAN, "convert"), mtx, csr],
                          cwd=OCEAN, capture_output=True, timeout=600)
    try:
        r = subprocess.run([os.path.join(OCEAN, "spgemm"), csr, CONFIG],
                          cwd=OCEAN, env=env, capture_output=True, text=True, timeout=1800)
        t = ITER10.findall(r.stdout + r.stderr)
        if t: w.writerow([m, f"{float(t[-1]):.3f}", ""])
        else: w.writerow([m, "", "NOPROF"])
    except subprocess.TimeoutExpired:
        w.writerow([m, "", "TIMEOUT"])
    f.flush()
    if (i+1) % 20 == 0: print(f"[{i+1}/{len(mats)}]", flush=True)
f.close()
print("done", flush=True)
