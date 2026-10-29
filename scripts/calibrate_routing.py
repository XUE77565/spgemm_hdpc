#!/usr/bin/env python3
"""S4 标定台(study/60):变体 × 签名 → cost 交叉面 → 决策表数据。
对每阵:①跑各变体配置(H69/H45/H55 on/off)②提取签名特征 ③记录 compute-only。
输出:results/calibration_data.csv(断点续跑)。
用法:python3 scripts/calibrate_routing.py [mat_file] [gpu]
"""
import csv, os, re, subprocess, sys, json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "calibration_data.csv")
COLS = ["mat", "variant", "co_ms", "nnz", "n", "nnz_a", "a_cols", "avg_flop",
        "est_density", "dup_ratio", "bin9_pop", "bin45_pop", "bin0_pop",
         "total_bins", "route", "bins_summary"]

def run_one(mat_path, env_extra, gpu="1"):
    """跑一个配置,返回 (co_ms, nnz, [adapt行], [bins行])"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, METHOD="adaptive",
               WARMUP_N="3", USE_MEMPOOL="1", MP_HOST_MB="8192", D6EXP="0")
    env.update(env_extra)
    try:
        r = subprocess.run(["./spgemm_test", mat_path], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=600)
        out = r.stdout + r.stderr
        tot = re.findall(r"TOTAL\(GPU\)\s+([\d.]+)", out)
        h2d = re.findall(r"\[hash-prof\] h2d\s+([\d.]+)", out)
        d2h = re.findall(r"\[hash-prof\] d2h\s+([\d.]+)", out)
        nnz = re.findall(r"Result C:.*nnz = (\d+)", out)
        route = re.findall(r"\[adapt\][^\n]*→ (\w+)", out)
        bins = re.findall(r"bins: ([0-9: ]+)", out)
        m3h = re.findall(r"\[mrg3-prof\] h2d\s+([\d.]+)", out)
        m3d = re.findall(r"\[mrg3-prof\] d2h\s+([\d.]+)", out)
        if tot and h2d and d2h:
            co = float(tot[-1]) - float(h2d[-1]) - float(d2h[-1])
        elif tot and m3h and m3d:
            co = float(tot[-1]) - float(m3h[-1]) - float(m3d[-1])
        else:
            co = -1
        return (co, int(nnz[-1]) if nnz else -1,
                route[-1] if route else "?",
                bins[-1] if bins else "")
    except Exception as e:
        return (-1, -1, f"ERR:{type(e).__name__}", "")

def extract_features(bins_str):
    """从 bins 直方图提取签名特征"""
    bins = {}
    total = 0
    for tok in bins_str.split():
        if ":" in tok:
            k, v = tok.split(":")
            bins[int(k)] = int(v)
            total += int(v)
    bin9 = bins.get(9, 0)
    bin45 = bins.get(4, 0) + bins.get(5, 0)
    bin0 = bins.get(0, 0)
    return bin9, bin45, bin0, total, bins_str[:60]

def compute_matrix_features(mat_path):
    """从 mtx 头部提取基础特征(n, nnz)"""
    try:
        with open(mat_path) as f:
            line = f.readline()
            while line.startswith("%"):
                line = f.readline()
            parts = line.split()
            return int(parts[0]), int(parts[2])  # n, nnz
    except:
        return -1, -1

def main():
    mat_file = sys.argv[1] if len(sys.argv) > 1 else None
    gpu = sys.argv[2] if len(sys.argv) > 2 else "1"

    # 选择标定阵集:从 428 中选,覆盖不同人口
    all_mats = []
    for r in csv.DictReader(open(os.path.join(REPO, "compare/batch437/confirm16_428.csv"))):
        if r["mat"] and r["mat"] != "mat":
            all_mats.append(r["mat"])

    # 变体配置(核心路由轴)
    VARIANTS = [
        ("baseline", {}),           # 全默认(H69/H45/H55 全 on)
        ("no_h69", {"HASH_H69": "0"}),  # bin9 回 LHT2(14)
        ("no_h4555", {"HASH_H45": "0", "HASH_H55": "0"}),  # bin4/5 回 LHT2
        ("no_all", {"HASH_H69": "0", "HASH_H45": "0", "HASH_H55": "0"}),  # 全关
    ]

    # 断点续跑
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            done.add((r["mat"], r["variant"]))
    else:
        with open(OUT, "w", newline="") as f:
            csv.writer(f).writerow(COLS)

    fout = open(OUT, "a", newline="")
    w = csv.writer(fout)

    # 训练集:356 阵(72 held-out 排除,防训练集泄露)
    if mat_file:
        cal_mats = [l.strip() for l in open(mat_file) if l.strip()]
    else:
        train_file = os.path.join(REPO, "results", "train_356.txt")
        if os.path.exists(train_file):
            cal_mats = [l.strip() for l in open(train_file) if l.strip()]
        else:
            cal_mats = all_mats

    print(f"标定阵 {len(cal_mats)} 个 × {len(VARIANTS)} 变体 = {len(cal_mats)*len(VARIANTS)} 配置", flush=True)

    for idx, mat in enumerate(cal_mats):
        mat_path = None
        for d in ("data/ocean/square", "data/first100"):
            p = os.path.join(REPO, d, mat + ".mtx")
            if os.path.exists(p):
                mat_path = p
                break
        if not mat_path:
            continue

        n, nnz_a = compute_matrix_features(mat_path)

        for vname, venv in VARIANTS:
            if vname == "baseline":
                continue  # baseline 用 confirm16 已有数据
            if (mat, vname) in done:
                continue
            co, nnz, route, bins_str = run_one(mat_path, venv, gpu)
            bin9, bin45, bin0, tb, bs = extract_features(bins_str)
            # 估算特征(粗)
            avg_flop = "?"  # 需要跑一次才能拿到
            est_den = "?"
            dup = "?"
            w.writerow([mat, vname, f"{co:.3f}", nnz, n, nnz_a, "?",
                        avg_flop, est_den, dup, bin9, bin45, bin0, tb, route, bs])
            fout.flush()
            if co > 0:
                print(f"  [{idx+1}/{len(cal_mats)}] {mat:<22} {vname:<10} co={co:.2f} nnz={nnz}", flush=True)

    print("CALIBRATION_DONE", flush=True)

if __name__ == "__main__":
    main()
