#!/usr/bin/env python3
"""批次删除 A/B 配对电池(打扫战役 11-10):冻结二进制 vs 新构建,16 阵 × N 轮交错。
判据:①nnz 逐位同(不同即停)②逐阵中位比 + GM(微阵 CV~19%,看中位与显著性,不看点值)。
口径与 p1_battery 同源:METHOD=hash(adaptive 阵除外)/WARMUP_N=3/USE_MEMPOOL=1/
MP_HOST_MB=8192/D6EXP=0;co = TOTAL − h2d − d2h。
用法:ab_pair.py BIN_A BIN_B OUT_CSV [rounds] [gpu]
"""
import csv, math, os, re, subprocess, sys, statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
RE_TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
RE_H2D = re.compile(r"\[hash-prof\] h2d\s+([\d.]+)")
RE_D2H = re.compile(r"\[hash-prof\] d2h\s+([\d.]+)")
RE_M3  = re.compile(r"\[mrg3-prof\] TOTAL\(GPU\)\s+([\d.]+)")
RE_NNZ = re.compile(r"Result C:.*nnz = (\d+)")

def run_one(binary, mat_path, method, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, METHOD=method,
               WARMUP_N="3", USE_MEMPOOL="1", MP_HOST_MB="8192", D6EXP="0")
    try:
        r = subprocess.run([f"./{binary}", mat_path], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=600)
        out = r.stdout + r.stderr
        tot = RE_TOT.findall(out); h = RE_H2D.findall(out); d = RE_D2H.findall(out)
        m3 = RE_M3.findall(out); nnz = RE_NNZ.findall(out)
        if not nnz: return None
        co = None
        if m3: co = float(m3[-1])                      # merge3 路径:mrg3 口径
        elif tot and h and d: co = float(tot[-1]) - float(h[-1]) - float(d[-1])
        else: co = float(tot[-1]) if tot else None
        if co is None or co <= 0: return None
        return co, int(nnz[-1])
    except Exception:
        return None

def main():
    bin_a, bin_b, out = sys.argv[1], sys.argv[2], sys.argv[3]
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    gpu = sys.argv[5] if len(sys.argv) > 5 else "1"
    rows = []
    for name, path, method in MATS:
        for r in range(rounds):
            for label, b in (("A", bin_a), ("B", bin_b)):
                res = run_one(b, path, method, gpu)
                rows.append({"mat": name, "bin": label, "round": r,
                             "co": res[0] if res else -1, "nnz": res[1] if res else -1})
                print(f"{name:>12} {label} r{r} {'%.3f' % res[0] if res else 'FAIL'} nnz={res[1] if res else '-'}", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mat","bin","round","co","nnz"]); w.writeheader(); w.writerows(rows)
    # 判读
    by = {}
    for r in rows:
        if r["co"] > 0: by.setdefault((r["mat"], r["bin"]), []).append(r["co"])
    nnzmap = {}
    for r in rows:
        if r["nnz"] > 0: nnzmap.setdefault((r["mat"], r["bin"]), set()).add(r["nnz"])
    ratios = []; bad = []
    print("\n===== 判读(B/A 中位)=====")
    for name, _, _ in MATS:
        a, b = by.get((name,"A")), by.get((name,"B"))
        na, nb = nnzmap.get((name,"A")), nnzmap.get((name,"B"))
        nnz_bad = (na != nb) or na is None
        if not a or not b:
            bad.append(name); print(f"{name:>12}  FAIL(缺数据)"); continue
        ratio = statistics.median(b) / statistics.median(a)
        ratios.append(ratio)
        flag = "NNZ-MISMATCH!" if nnz_bad else ("Δ>5%" if abs(ratio-1) > 0.05 else "")
        print(f"{name:>12}  {ratio:.4f}  {flag}")
        if nnz_bad: bad.append(name)
    if ratios:
        gm = math.exp(sum(math.log(x) for x in ratios)/len(ratios))
        print(f"\nGM(B/A) = {gm:.4f} | 最差 = {max(ratios):.4f} | nnz 异常 = {len([x for x in bad])}")
        print("裁定:", "红-停" if bad or (gm > 1.02) else "绿-进下批(≤2% 带内)")

if __name__ == "__main__":
    main()
