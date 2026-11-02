#!/usr/bin/env python3
"""held-out 40 阵 AIROUTER A/B(泛化实验,预注册选样 results/heldout_router72.txt)
每阵配对跑 AIROUTER=0(手门默认) vs 1(门控AI),同卡同会话。
判读:①几阵触发AI决策(hash路径);②门控翻了几阵;③时间差;
④翻对阵单独做 ground-truth 消融(h45/h55/h69 on/off)复核。
"""
import csv, os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIST = os.path.join(REPO, "results", "heldout_router72.txt")
OUT = os.path.join(REPO, "results", "heldout_router_ab.csv")

def run_one(mat_path, airouter, gpu="0"):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, METHOD="adaptive",
               WARMUP_N="3", USE_MEMPOOL="1", MP_HOST_MB="8192", D6EXP="0",
               AIROUTER=str(airouter))
    try:
        r = subprocess.run(["./spgemm_test", mat_path], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
        out = r.stdout + r.stderr
        tot = re.findall(r"TOTAL\(GPU\)\s+([\d.]+)", out)
        h = re.findall(r"\[hash-prof\] h2d\s+([\d.]+)", out)
        d = re.findall(r"\[hash-prof\] d2h\s+([\d.]+)", out)
        m3h = re.findall(r"\[mrg3-prof\] h2d\s+([\d.]+)", out)
        m3d = re.findall(r"\[mrg3-prof\] d2h\s+([\d.]+)", out)
        ai = re.findall(r"AIROUTER:.*→ h45=(\d) h55=(\d) h69=(\d)", out)
        if tot and h and d:
            co = float(tot[-1]) - float(h[-1]) - float(d[-1])
        elif tot and m3h and m3d:
            co = float(tot[-1]) - float(m3h[-1]) - float(m3d[-1])
        else:
            co = float(tot[-1]) if tot else -1
        flags = ai[-1] if ai else None
        return co, flags
    except Exception:
        return -1, None

def main():
    gpu = sys.argv[1] if len(sys.argv) > 1 else "0"
    mats = [l.strip() for l in open(LIST) if l.strip()]
    print(f"held-out {len(mats)} 阵, GPU{gpu}, 配对 A/B", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mat", "t_ai0_ms", "t_ai1_ms", "ratio", "ai_flags", "note"])
        for i, mat in enumerate(mats):
            path = os.path.join(REPO, "data", "train_disp", mat + ".mtx")
            if not os.path.exists(path):
                w.writerow([mat, -1, -1, -1, "", "no_file"]); continue
            t0, f0 = run_one(path, 0, gpu)
            t1, f1 = run_one(path, 1, gpu)
            ratio = t1 / t0 if t0 > 0 and t1 > 0 else -1
            note = ""
            if f1 is None: note = "no_hash_path"       # 没走 hash(merge3/dense),AI 不触发
            elif f1 != ("1","1","1"): note = "AI_FLIPPED"  # 门控翻了默认
            w.writerow([mat, f"{t0:.3f}", f"{t1:.3f}", f"{ratio:.4f}",
                        "/".join(f1) if f1 else "", note])
            f.flush()
            mark = " ◄" if note == "AI_FLIPPED" else ""
            print(f"  [{i+1:2d}/{len(mats)}] {mat:<24} off={t0:>8.2f} on={t1:>8.2f} "
                  f"ratio={ratio:.4f} flags={f1}{mark}", flush=True)
    print("HELDOUT_AB_DONE", flush=True)

if __name__ == "__main__":
    main()
