#!/usr/bin/env python3
"""merge3 臂标签小批(10-17 A2' 第 2 步):为内层 logit(hash vs merge3)重训提供逐阵 merge3 时间。

范围 = f100 91 阵 + 337 的 n<5000 小阵(预门之内 dispatcher 仍在做选择的人群)。
口径同批437(compute-only,WARMUP_N=3);断点续跑。
"""
import csv, os, re, subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_test")
TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
H2D = re.compile(r"\[mrg3-prof\] h2d\s+([\d.]+)")
D2H = re.compile(r"\[mrg3-prof\] d2h\s+([\d.]+)")

o337 = [r["mat"] for r in csv.DictReader(open(os.path.join(REPO, "compare/l0_337/l0_337_results.csv")))]
fx = {r["mat"]: r for r in csv.DictReader(open(os.path.join(REPO, "compare/batch437/features.csv")))}
seen = set(o337)
july = [r["matrix"] for r in csv.DictReader(open(os.path.join(REPO, "compare/methods_cmp_first100.csv")))]
f100 = [m for m in july if m not in seen and os.path.exists(os.path.join(REPO, "data/first100", m + ".mtx"))]

mats = [(m, os.path.join(REPO, "data/first100", m + ".mtx")) for m in f100]
mats += [(m, os.path.join(REPO, "data/ocean/square", m + ".mtx")) for m in o337
         if int(fx[m]["n"]) < 5000]
mats.sort(key=lambda x: os.path.getsize(x[1]))

out_path = os.path.join(REPO, "compare/batch437/mrg3_labels.csv")
done = set()
if os.path.exists(out_path):
    done = {r["mat"] for r in csv.DictReader(open(out_path)) if r["ms"]}
f = open(out_path, "a", newline="")
w = csv.writer(f)
if not done:
    w.writerow(["mat", "ms", "note"])

env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", WARMUP_N="3", USE_MEMPOOL="1",
           MP_HOST_MB="8192", D6EXP="0", K7="0", METHOD="merge3")
for i, (m, p) in enumerate(mats):
    if m in done: continue
    try:
        r = subprocess.run([BIN, p], capture_output=True, text=True, env=env, timeout=900)
        t = TOT.findall(r.stderr)
        if not t:
            w.writerow([m, "", "NOPROF"]); f.flush(); continue
        h = H2D.findall(r.stderr); d = D2H.findall(r.stderr)
        v = float(t[-1]) - (float(h[-1]) if h else 0) - (float(d[-1]) if d else 0)
        w.writerow([m, f"{v:.3f}", ""]); f.flush()
    except subprocess.TimeoutExpired:
        w.writerow([m, "", "TIMEOUT"]); f.flush()
    if i % 20 == 0: print(f"[{i+1}/{len(mats)}] {m}", flush=True)
print("MRG3_LABELS_DONE", len(mats))
