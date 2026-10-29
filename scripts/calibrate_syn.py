#!/usr/bin/env python3
"""合成阵标定:对 /tmp/syn_cal/ 的合成矩阵跑 3 变体,追加到 calibration_data.csv"""
import csv, os, re, subprocess
REPO = "/home/xueyizhou/spgemm-challenge"
SYN = "/tmp/syn_cal"
OUT = os.path.join(REPO, "results/calibration_data.csv")
COLS = ["mat","variant","co_ms","nnz","n","nnz_a","a_cols","avg_flop",
        "est_density","dup_ratio","bin9_pop","bin45_pop","bin0_pop","total_bins","route","bins_summary"]
VARIANTS = [("no_h69",{"HASH_H69":"0"}),("no_h4555",{"HASH_H45":"0","HASH_H55":"0"}),("no_all",{"HASH_H69":"0","HASH_H45":"0","HASH_H55":"0"})]
def run(path, venv):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", METHOD="adaptive", WARMUP_N="3", USE_MEMPOOL="1", MP_HOST_MB="8192", D6EXP="0")
    env.update(venv)
    try:
        r = subprocess.run(["./spgemm_test", path], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
        out = r.stdout + r.stderr
        tot = re.findall(r"TOTAL\(GPU\)\s+([\d.]+)", out)
        h = re.findall(r"\[hash-prof\] h2d\s+([\d.]+)", out)
        d = re.findall(r"\[hash-prof\] d2h\s+([\d.]+)", out)
        nnz = re.findall(r"Result C:.*nnz = (\d+)", out)
        bins = re.findall(r"bins: ([0-9: ]+)", out)
        co = float(tot[-1]) - float(h[-1]) - float(d[-1]) if tot and h and d else -1
        return co, int(nnz[-1]) if nnz else -1, bins[-1] if bins else ""
    except:
        return -1, -1, ""
done = set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)): done.add((r["mat"], r["variant"]))
need_h = not os.path.exists(OUT) or os.path.getsize(OUT)==0
fout = open(OUT,"a",newline=""); w = csv.writer(fout)
if need_h: w.writerow(COLS)
mats = sorted([f for f in os.listdir(SYN) if f.endswith(".mtx")])
print(f"合成阵 {len(mats)} × 3", flush=True)
for i,mf in enumerate(mats):
    mat = mf.replace(".mtx",""); path = os.path.join(SYN, mf)
    for vn, ve in VARIANTS:
        if (mat,vn) in done: continue
        co, nnz, bs = run(path, ve)
        b9=b45=b0=tb=0
        for tok in bs.split():
            if ":" in tok:
                k,v = tok.split(":"); v=int(v); tb+=v
                if int(k)==9: b9=v
                elif int(k) in (4,5): b45+=v
                elif int(k)==0: b0=v
        w.writerow([mat,vn,f"{co:.3f}",nnz,"","","","","","",b9,b45,b0,tb,"syn",bs[:60]])
        fout.flush()
    if (i+1)%10==0: print(f"[{i+1}/{len(mats)}]", flush=True)
print("SYN_DONE", flush=True)
