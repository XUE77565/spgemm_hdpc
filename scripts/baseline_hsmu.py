#!/usr/bin/env python3
"""基线电池(study/59 战役C2):HSMU-SpGEMM × 428 阵(AA 轨,mtx 直入)。
输出 compare/batch437/baseline_hsmu.csv(断点续跑);每阵 timeout 600s;
解析:相位时间(ms)与 cnnz。对照 = 我方 confirm16 ADPT16(compute-only)。
"""
import csv, os, re, subprocess
REPO = "/home/xueyizhou/spgemm-challenge"
OUT = os.path.join(REPO, "compare/batch437/baseline_hsmu.csv")
BIN = os.path.join(REPO, "external_sota/HSMU-SpGEMM/evaluation/script/test")
COLS = ["mat", "status", "total", "cnnz", "note"]
rows = {}
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT), fieldnames=COLS):
        if r["mat"] and r["mat"] != "mat":
            rows[r["mat"]] = r
mats = []
for src in ("compare/batch437/confirm16_428.csv",):
    for r in csv.DictReader(open(os.path.join(REPO, src))):
        m = r["mat"]
        if m and m != "mat" and m not in mats:
            mats.append(m)
fout = open(OUT, "a", newline="")
w = csv.writer(fout)
if not os.path.exists(OUT) or os.path.getsize(OUT) == 0:
    w.writerow(COLS)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="1")
done = 0
for m in mats:
    if m in rows:
        continue
    p = None
    for d in ("data/ocean/square", "data/first100"):
        c = os.path.join(REPO, d, m + ".mtx")
        if os.path.exists(c):
            p = c
            break
    if not p:
        w.writerow([m, "NOMTX", "", "", ""]); fout.flush(); continue
    try:
        r = subprocess.run([BIN, p], cwd=os.path.dirname(BIN), env=env,
                           capture_output=True, text=True, timeout=600)
        out = r.stdout + r.stderr
        cnnz = re.findall(r"cnnz = (\d+)", out)
        # 相位和(total 打印在小阵上舍入为 0,取相位和为口径;单位 ms)
        ph = [float(x) for x in re.findall(r"run time for \w+ is ([\d.]+)", out)]
        tot = sum(ph) if ph else -1
        w.writerow([m, "ok", f"{tot:.3f}", cnnz[-1] if cnnz else "", ""])
    except subprocess.TimeoutExpired:
        w.writerow([m, "TIMEOUT", "", "", ""])
    except Exception as e:
        w.writerow([m, f"ERR:{type(e).__name__}", "", "", ""])
    fout.flush()
    done += 1
    if done % 20 == 0:
        print(f"[{done}] 进度", flush=True)
print("HSMU_BATTERY_DONE", flush=True)
