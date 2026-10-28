#!/usr/bin/env python3
"""P0 输面取证(study/59):confirm15 的 131 输阵逐个采 路由([adapt]裁定)×bins×相位 → results/loss_audit_1028.csv。
断点续跑(有值即跳);超时 600s;卡1。"""
import csv, os, re, subprocess, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "loss_audit_1028.csv")
COLS = ["mat", "ratio", "path", "route", "bins", "acc", "cmp", "tot", "nnz"]
rows = {}
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT), fieldnames=COLS):
        if r["mat"] and r["mat"] != "mat":
            rows[r["mat"]] = r
new = {r["mat"]: (float(r["adpt13_ms"]) if r["adpt13_ms"] else None,
                  float(r["ocn13_ms"]) if r["ocn13_ms"] else None)
       for r in csv.DictReader(open("compare/batch437/confirm15_428.csv"))}
losses = sorted(((a / o, m) for m, (a, o) in new.items() if a and o and a >= o), reverse=True)
print(f"输阵 {len(losses)} 个", flush=True)
fout = open(OUT, "a", newline="")
w = csv.writer(fout)
if not os.path.exists(OUT) or os.path.getsize(OUT) == 0:
    w.writerow(COLS)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", METHOD="adaptive", WARMUP_N="1",
           USE_MEMPOOL="1", MP_HOST_MB="8192", D6EXP="0")
for ratio, m in losses:
    if m in rows:
        continue
    p = None
    for d in ("data/ocean/square", "data/first100"):
        c = os.path.join(REPO, d, m + ".mtx")
        if os.path.exists(c):
            p = c
            break
    if not p:
        print(f"[MISS] {m} 无mtx", flush=True)
        w.writerow([m, f"{ratio:.3f}", "", "NOMTX", "", "", "", "", ""])
        fout.flush()
        continue
    try:
        r = subprocess.run(["./spgemm_test", p], cwd=REPO, env=env, capture_output=True,
                           text=True, timeout=600)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        w.writerow([m, f"{ratio:.3f}", p, "TIMEOUT", "", "", "", "", ""])
        fout.flush()
        continue
    route = re.findall(r"\[adapt\][^\n]*", out)
    bins = re.findall(r"bins: [0-9: ]+", out)
    acc = re.findall(r"accumulate\s+([\d.]+)", out)
    cmp_ = re.findall(r"compact\+sort\s+([\d.]+)", out)
    tot = re.findall(r"TOTAL\(GPU\)\s+([\d.]+)", out)
    nnz = re.findall(r"Result C:.*nnz = (\d+)", out)
    row = [m, f"{ratio:.3f}", p,
           (route[-1].strip()[:90] if route else "?"),
           (bins[-1].strip() if bins else ""),
           acc[-1] if acc else "", cmp_[-1] if cmp_ else "",
           tot[-1] if tot else "", nnz[-1] if nnz else ""]
    w.writerow(row)
    fout.flush()
    rows[m] = dict(zip(COLS, row))
    print(f"[ok] {m:<24} {ratio:.3f} {row[3][:60]}", flush=True)
print("AUDIT_DONE", flush=True)
