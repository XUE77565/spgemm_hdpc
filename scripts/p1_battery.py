#!/usr/bin/env python3
"""P1 per-file 验收电池(study/58 夜1;15 阵 × {per-file, 整sm90, 整c61} × 交错 2 轮)。
口径:METHOD=hash(merge3 微阵 1138_bus/arc130 走 adaptive 练 merge3 路由)/WARMUP_N=3/
USE_MEMPOOL=1/MP_HOST_MB=8192/D6EXP=0/CUDA_VISIBLE_DEVICES=1(confirm13 同款)。
判据(study/57 §五十一对齐):TSOPF accumulate 回 4.26ms 级;F2 TOTAL 保持 −3.1% 级;
nnz 三构建逐位同(不同即停);输出 results/p1_battery_1027.csv(断点续跑:有值即跳)。
"""
import csv, os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "p1_battery_1027.csv")
MATS = [
    ("arc130", "data/first100/arc130.mtx", "adaptive"),
    ("1138_bus", "data/first100/1138_bus.mtx", "adaptive"),
    ("bcsstk17", "data/first100/bcsstk17.mtx", "hash"),
    ("case9", "data/ocean/square/case9.mtx", "hash"),
    ("web-Google", "data/ocean/square/web-Google.mtx", "hash"),
    ("TSOPF", "data/ocean/square/TSOPF_RS_b678_c2.mtx", "hash"),
    ("bloweybq", "data/ocean/square/bloweybq.mtx", "hash"),
    ("F2", "data/train_disp/af23560.mtx", "hash"),
    ("StocF", "data/ocean/square/StocF-1465.mtx", "hash"),
    ("ldoor", "data/ocean/square/ldoor.mtx", "hash"),
    ("c-64", "data/ocean/square/c-64.mtx", "hash"),
    ("pre2", "data/ocean/square/pre2.mtx", "hash"),
    ("rgg", "data/ocean/square/rgg_n_2_21_s0.mtx", "hash"),
    ("Hook", "data/ocean/square/Hook_1498.mtx", "hash"),
    ("cage15", "data/ocean/square/cage15.mtx", "hash"),
    ("Bump", "data/ocean/square/Bump_2911.mtx", "hash"),
]
BINS = [("pf", "./spgemm_test"), ("sm90", "./spgemm_test_sm90"), ("c61", "./spgemm_test_c61")]
RE_TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
RE_ACC = re.compile(r"\[hash-prof\] accumulate\s+([\d.]+)")
RE_CMP = re.compile(r"\[hash-prof\] compact\+sort\s+([\d.]+)")
RE_D2H = re.compile(r"\[hash-prof\] d2h\s+([\d.]+)")
RE_M3  = re.compile(r"\[mrg3-prof\] TOTAL\(GPU\)\s+([\d.]+)")
RE_NNZ = re.compile(r"Result C:.*nnz = (\d+)")

COLS = ["mat","bin","round","method","total","acc","cmp","d2h","m3","nnz"]
rows = {}
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT), fieldnames=COLS):
        if r["mat"] and r["mat"] != "mat":
            rows[(r["mat"], r["bin"], r["round"])] = r
os.makedirs(os.path.dirname(OUT), exist_ok=True)
newfile = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
fout = open(OUT, "a", newline="")
w = csv.writer(fout)
if newfile:
    w.writerow(COLS)

env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", WARMUP_N="3", USE_MEMPOOL="1",
           MP_HOST_MB="8192", D6EXP="0")
for name, path, method in MATS:
    for rnd in ("1", "2"):
        for bname, bpath in BINS:
            key = (name, bname, rnd)
            if key in rows:
                continue
            e = dict(env, METHOD=method)
            p = subprocess.run([bpath, path], cwd=REPO, env=e, capture_output=True,
                               text=True, timeout=1200)
            out = p.stdout + p.stderr
            if p.returncode != 0:
                print(f"[FAIL] {name}/{bname}/r{rnd} exit={p.returncode}", flush=True)
                sys.exit(2)
            nnz = RE_NNZ.search(out)
            tot = RE_TOT.search(out); acc = RE_ACC.search(out); cmp_ = RE_CMP.search(out)
            d2h = RE_D2H.search(out); m3 = RE_M3.search(out)
            row = [name, bname, rnd, method,
                   tot.group(1) if tot else "", acc.group(1) if acc else "",
                   cmp_.group(1) if cmp_ else "", d2h.group(1) if d2h else "",
                   m3.group(1) if m3 else "", nnz.group(1) if nnz else ""]
            w.writerow(row); fout.flush()
            # 值门1:nnz 三构建一致(同阵同轮),不一致即停
            rows[key] = dict(zip(["mat","bin","round","method","total","acc","cmp","d2h","m3","nnz"], row))
            others = [r["nnz"] for (m, b2, r2), r in rows.items() if m == name and r2 == rnd and r["nnz"]]
            if not all(o == row[9] for o in others):
                print(f"[NNZ MISMATCH] {name} r{rnd}: {others} vs {row[9]}", flush=True)
                sys.exit(3)
            print(f"[ok] {name}/{bname}/r{rnd} tot={row[4]} acc={row[5]} nnz={row[9]}", flush=True)
print("BATTERY DONE", flush=True)
