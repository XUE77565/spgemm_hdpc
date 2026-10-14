#!/usr/bin/env python3
"""Phase 2 临时检查点(10-13,study/42 风险#1):337 × 3 臂,冻结二进制。

臂(全部 D6EXP=0 K7=0,METHOD=hash,WARMUP_N=1,compute-only 口径同 l0_collect):
  C      默认管线(含 DCEST 代码但 env 关)
  CD     默认 + DCEST=1(margin 默认 200;判废=DNF 计数,expected≈0)
  A      全 hash(DITER_MIN_FLOP=999999999)
输出:compare/p2_provisional/p2_337.csv + 逐阵进度行(stdout);小阵先跑(早出信号)。
GM/Δ 到末尾打印;断点续跑:--skip 已有结果的 mat。
用法:nohup env CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/p2_provisional_337.py > /tmp/p2.log 2>&1 &
"""
import csv, os, re, subprocess, sys, math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("SPGEMM_BIN", os.path.join(REPO, "spgemm_test_p2prov"))
DIR = os.path.join(REPO, "data", "ocean", "square")
OUT = os.path.join(REPO, "compare", "p2_provisional")
TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
H2D = re.compile(r"\[hash-prof\] h2d\s+([\d.]+)")
D2H = re.compile(r"\[hash-prof\] d2h\s+([\d.]+)")
ARMS = {
    "C":  {},
    "CD": {"DCEST": "1"},
    "A":  {"DITER_MIN_FLOP": "999999999"},
}
BASE = {"D6EXP": "0", "K7": "0"}

mats = [r["mat"] for r in csv.DictReader(open(os.path.join(REPO, "compare/l0_337/l0_337_results.csv")))]
# 小阵先跑(按文件体积升序):断点处已覆盖大多数阵,部分结果也有统计意义
mats.sort(key=lambda m: os.path.getsize(os.path.join(DIR, m + ".mtx")))
os.makedirs(OUT, exist_ok=True)
rows = {}
csv_path = os.path.join(OUT, "p2_337.csv")
if os.path.exists(csv_path):
    for r in csv.DictReader(open(csv_path)):
        rows[(r["mat"], r["arm"])] = (r["net_ms"], r["note"])
        rows[(r["mat"], r["arm"])] = (float(r["net_ms"]) if r["net_ms"] else None, r["note"])

def run(m, arm):
    env = dict(os.environ, METHOD="hash", WARMUP_N="1", USE_MEMPOOL="1", MP_HOST_MB="8192")
    env.update(BASE); env.update(ARMS[arm])
    try:
        r = subprocess.run([BIN, os.path.join(DIR, m + ".mtx")], capture_output=True, text=True, env=env, timeout=1500)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    t = TOT.findall(r.stderr)
    if not t:
        return None, "NOPROF"
    if "OVERFLOW" in r.stdout:
        return None, "DCEST_DNF"   # 判废(仅 CD 臂可能;provisional 按 DNF 记)
    h = H2D.findall(r.stderr); d = D2H.findall(r.stderr)
    return float(t[-1]) - (float(h[-1]) if h else 0) - (float(d[-1]) if d else 0), ""

done = 0
for i, m in enumerate(mats):
    for arm in ARMS:
        cur = rows.get((m, arm))
        if cur and cur[0] is not None:   # 有数值才跳过(NOPROF/TIMEOUT 必须重试)
            continue
        net, note = run(m, arm)
        rows[(m, arm)] = (net, note)
        done += 1
    print(f"[{i+1}/{len(mats)}] {m:<22} " + "  ".join(
        f"{arm}={rows.get((m,arm),('?','?'))[0] if rows.get((m,arm),('?','?'))[0] else rows.get((m,arm),('','?'))[1]}" for arm in ARMS), flush=True)
    # 每阵落盘(断点续跑)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mat", "arm", "net_ms", "note"])
        for (mm, aa), (net, note) in sorted(rows.items()):
            w.writerow([mm, aa, "" if net is None else f"{net:.3f}", note])

# 末尾:GM 汇总
def gm(arm):
    vals = [v for (m, a), (v, n) in rows.items() if a == arm and v and v > 0]
    return (math.exp(sum(math.log(v) for v in vals) / len(vals)), len(vals)) if vals else (None, 0)
print("\n=== 汇总(compute-only,单位 ms;GM 为我方绝对时间——与 ocean 的比值要另配对)===", flush=True)
for arm in ARMS:
    g, n = gm(arm)
    print(f"{arm}: GM={g:.3f} (n={n}, DNF/TO={sum(1 for (m,a),(v,nn) in rows.items() if a==arm and not v)})", flush=True)
# CD vs C / A vs C 的逐阵配对 Δ
for x in ("CD", "A"):
    pairs = [(rows[(m, "C")][0], rows[(m, x)][0]) for m in mats
             if (m, "C") in rows and (m, x) in rows and rows[(m, "C")][0] and rows[(m, x)][0]]
    if pairs:
        ratio = math.exp(sum(math.log(b / a) for a, b in pairs) / len(pairs))
        print(f"{x}/C 配对几何均值比 = {ratio:.4f}(n={len(pairs)})", flush=True)
print("P2PROV_DONE", flush=True)
