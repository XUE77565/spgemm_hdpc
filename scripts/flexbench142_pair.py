#!/usr/bin/env python3
"""FlexSpGEMM 自家 142 测试集上的双雄对(数据就绪后一键跑):
每阵:我方 spgemm_test(标准 env,compute-only)+ 对方 src/test(其官方协议,time_tile 三相和),
nnz 双向对拍,GM/赢面按完成对口径。输出 compare/flexbench142/pair.csv。
前置:external_sota/FlexSpGEMM/data/test142/<name>.mtx 齐备(下载守护 tools/flex142_fetch.sh)。
用法:flexbench142_pair.py [gpu]
"""
import csv, math, os, subprocess, sys, re

REPO = "/home/xueyizhou/spgemm-challenge"
FLEX = os.path.join(REPO, "external_sota/FlexSpGEMM")
DIR = os.path.join(FLEX, "data/test142")
OUT = os.path.join(REPO, "compare/flexbench142/pair.csv")
TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
H2D = re.compile(r"\[hash-prof\] h2d\s+([\d.]+)")
D2H = re.compile(r"\[hash-prof\] d2h\s+([\d.]+)")
M3H = re.compile(r"\[mrg3-prof\] h2d\s+([\d.]+)")
M3D = re.compile(r"\[mrg3-prof\] d2h\s+([\d.]+)")
NNZ = re.compile(r"Result C:.*nnz = (\d+)")
FTILE = re.compile(r"time_tile.*")  # 从 CSV 取,不用 stdout

rows = list(csv.reader(open(os.path.join(FLEX, "data/mat-142.csv"))))
mats = [(r[1], os.path.join(DIR, r[1] + ".mtx")) for r in rows if len(r) >= 2]
mats = [(n, p) for n, p in mats if os.path.exists(p)]
print(f"就绪 {len(mats)}/142 阵", flush=True)
if len(mats) < 142:
    print("警告:数据未齐,只跑就绪部分(完成对口径)")

def run_ours(p, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, METHOD="adaptive", WARMUP_N="3",
               USE_MEMPOOL="1", MP_HOST_MB="8192", D6EXP="0")
    try:
        r = subprocess.run([os.path.join(REPO, "spgemm_test"), p], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None, None, "TIMEOUT"
    out = r.stdout + r.stderr
    t = TOT.findall(out); n = NNZ.findall(out)
    if not t or not n: return None, None, "NOPROF"
    h = H2D.findall(out); d = D2H.findall(out)
    h3 = M3H.findall(out); d3 = M3D.findall(out)
    hh = float(h[-1]) if h else (float(h3[-1]) if h3 else 0)
    dd = float(d[-1]) if d else (float(d3[-1]) if d3 else 0)
    return float(t[-1]) - hh - dd, int(n[-1]), ""

def run_flex(p, gpu):
    res_csv = os.path.join(FLEX, "result/results_tile.csv")
    off = os.path.getsize(res_csv) if os.path.exists(res_csv) else 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    try:
        r = subprocess.run(["./test", "-d", "0", "-aat", "0", "-tau", "0.7", "-csr2tile", "gpu", p],
                           cwd=os.path.join(FLEX, "src"), env=env, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None, None, "TIMEOUT"
    err = r.stdout + r.stderr
    if "CUDA error" in err or r.returncode != 0:
        return None, None, "CUDA-ERR"
    try:
        with open(res_csv) as f:
            f.seek(off); new = f.read().strip().splitlines()
        if not new: return None, None, "NOCSV"
        c = new[-1].split(",")
        return float(c[7]), int(c[5]), ""
    except Exception as e:
        return None, None, f"PARSE:{e}"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
f = open(OUT, "w", newline="")
w = csv.writer(f); w.writerow(["mat", "ours_ms", "flex_ms", "ratio", "ours_nnz", "flex_nnz", "nnz_match", "note"])
ratios = []
for i, (m, p) in enumerate(mats):
    o, on, oe = run_ours(p, "1")
    x, xn, xe = run_flex(p, "0")
    note = (oe or "") + "/" + (xe or "")
    match = "same" if (on and xn and on == xn) else ("diff!" if (on and xn) else "")
    r = (o / x) if (o and x) else None
    if r: ratios.append((r, m))
    w.writerow([m, f"{o:.3f}" if o else "", f"{x:.3f}" if x else "",
                f"{r:.4f}" if r else "", on or "", xn or "", match, note])
    f.flush()
    if (i + 1) % 10 == 0: print(f"[{i+1}/{len(mats)}] {m}: {'%.3f' % r if r else note}", flush=True)
f.close()
if ratios:
    gm = math.exp(sum(math.log(x) for x, _ in ratios) / len(ratios))
    print(f"\n===== Flex 自家 142 对决判读 =====")
    print(f"完成对 {len(ratios)} | GM(我/Flex) = {gm:.4f} | 我方赢 {sum(1 for x,_ in ratios if x<1)}/{len(ratios)}")
    print("最差5:", ", ".join(f"{m}={x:.2f}" for x, m in sorted(ratios, reverse=True)[:5]))
