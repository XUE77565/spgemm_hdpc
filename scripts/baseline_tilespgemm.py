#!/usr/bin/env python3
"""基线电池:TileSpGEMM × 428 阵(AA,`./test -d 0 <mtx>`;输出行 'new stir_spgemm_time = X ms')。
断点续跑;timeout 900;卡0(与电池v7卡1并行,注意外来RL在卡0——微阵可容)。"""
import csv, os, re, subprocess
REPO="/home/xueyizhou/spgemm-challenge"
OUT=os.path.join(REPO,"compare/batch437/baseline_tilespgemm.csv")
BIN=os.path.join(REPO,"external_sota/TileSpGEMM/test")
COLS=["mat","status","total_ms","note"]
rows={}
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT),fieldnames=COLS):
        if r["mat"] and r["mat"]!="mat": rows[r["mat"]]=r
mats=[]
for r in csv.DictReader(open(os.path.join(REPO,"compare/batch437/confirm16_428.csv"))):
    if r["mat"] and r["mat"]!="mat" and r["mat"] not in mats: mats.append(r["mat"])
fout=open(OUT,"a",newline=""); w=csv.writer(fout)
if not os.path.exists(OUT) or os.path.getsize(OUT)==0: w.writerow(COLS)
env=dict(os.environ, CUDA_VISIBLE_DEVICES="0")
done=0
for m in mats:
    if m in rows: continue
    p=None
    for d in ("data/ocean/square","data/first100"):
        c=os.path.join(REPO,d,m+".mtx")
        if os.path.exists(c): p=c; break
    if not p:
        w.writerow([m,"NOMTX","",""]); fout.flush(); continue
    try:
        r=subprocess.run([BIN,"-d","0",p],cwd=os.path.dirname(BIN),env=env,
                         capture_output=True,text=True,timeout=900)
        out=r.stdout+r.stderr
        t=re.findall(r"new stir_spgemm_time = ([\d.]+)",out)
        w.writerow([m,"ok" if t else "NOPARSE",t[-1] if t else "",""])
    except subprocess.TimeoutExpired:
        w.writerow([m,"TIMEOUT","",""])
    except Exception as e:
        w.writerow([m,f"ERR:{type(e).__name__}","",""])
    fout.flush(); done+=1
    if done%20==0: print(f"[{done}]",flush=True)
print("TILE_BATTERY_DONE",flush=True)
