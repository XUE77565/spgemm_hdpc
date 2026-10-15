#!/usr/bin/env python3
"""437 权威批(10-15 发车,study/42 规格冻结):437=first100 ∪ ocean337(重叠 1 去重)。

列(同日同机同口径,WARMUP_N=3,安静 GPU0):
  C     我方默认管线(D6EXP=0 K7=0,DCEST env 关)
  CD    C + DCEST=1(margin 200;判废=DNF 计数)
  A     全 hash(DITER_MIN_FLOP=999999999)
  ADPT  adaptive dispatcher(hash vs merge3 路由,study/37)
  OCN   ocean 同日重测(convert→spgemm→stats.json,口径=其 bench_detail)
输出:compare/batch437/batch_437.csv(断点续跑:有数值即跳过)+ run.log(带日期)。
纪律:Result C/TOTAL 缺失=硬红旗(NOPROF 记录,续跑重试);小阵先跑(早出信号);
      本批 = L0/L1 训练与论文记分唯一来源(入 compare/README manifest)。
用法:nohup env CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/batch_437_authoritative.py \
      > compare/batch437/run.log 2>&1 &
"""
import csv, json, math, os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("SPGEMM_BIN", os.path.join(REPO, "spgemm_test"))
OUT = os.path.join(REPO, "compare", "batch437")
os.makedirs(OUT, exist_ok=True)
CSRTMP = "/home/xueyizhou/.claude/jobs/a3c873ae/tmp/oceancsr"
os.makedirs(CSRTMP, exist_ok=True)
os.makedirs(os.path.join(OUT, "prof"), exist_ok=True)
os.makedirs(os.path.join(OUT, "ocn_stats"), exist_ok=True)
TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
H2D = re.compile(r"\[hash-prof\] h2d\s+([\d.]+)")
D2H = re.compile(r"\[hash-prof\] d2h\s+([\d.]+)")
# adaptive 路径相位名不同 → 用其自身 TOTAL/h2d/d2h(名字同 hash-prof 前缀?防御两种)
BASE = {"D6EXP": "0", "K7": "0"}
ARMS = {
    "C":    {"METHOD": "hash"},
    "CD":   {"METHOD": "hash", "DCEST": "1"},
    "A":    {"METHOD": "hash", "DITER_MIN_FLOP": "999999999"},
    "ADPT": {"METHOD": "adaptive"},
}

def load_mats():
    # 10-15 定谳:data/first100 目录漂移(混入 8 杂文件/缺 8 个 can*,长方形 ash* 被
    # ocean 正确拒跑暴露)→ 权威名单 = July 表(methods_cmp_first100.csv)∩ 现有文件。
    mats = []
    o337 = [r["mat"] for r in csv.DictReader(open(os.path.join(REPO, "compare/l0_337/l0_337_results.csv")))]
    for m in o337:
        mats.append((m, os.path.join(REPO, "data", "ocean", "square", m + ".mtx")))
    seen = set(o337)
    july = [r["matrix"] for r in csv.DictReader(open(os.path.join(REPO, "compare/methods_cmp_first100.csv")))]
    fdir = os.path.join(REPO, "data", "first100")
    missing = []
    for m in july:
        if m in seen:
            continue
        p = os.path.join(fdir, m + ".mtx")
        if os.path.exists(p):
            mats.append((m, p))
        else:
            missing.append(m)
    if missing:
        print(f"# ⚠ 权威名单缺文件(待补,SuiteSparse 镜像被网络策略拒): {','.join(missing)}", flush=True)
    return mats

def ours(mtx, arm, m):
    env = dict(os.environ, WARMUP_N="3", USE_MEMPOOL="1", MP_HOST_MB="8192")
    env.update(BASE); env.update(ARMS[arm])
    try:
        r = subprocess.run([BIN, mtx], capture_output=True, text=True, env=env, timeout=1500)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    # 全相位 sidecar(口径后验重算的保险,10-15 用户拦车促成)
    try:
        with open(os.path.join(OUT, "prof", f"{m}__{arm}.log"), "w") as f:
            f.write(r.stderr)
    except OSError:
        pass
    t = TOT.findall(r.stderr)
    if not t:
        return None, "NOPROF"
    if "OVERFLOW" in r.stdout:
        return None, "OVERFLOW"
    h = H2D.findall(r.stderr); d = D2H.findall(r.stderr)
    return float(t[-1]) - (float(h[-1]) if h else 0) - (float(d[-1]) if d else 0), ""

def ocean(m, mtx):
    csr = os.path.join(CSRTMP, m + ".csr")
    stats = os.path.join(REPO, "ocean", "stats.json")     # 固定路径,cwd=ocean(实测)
    try:
        if not os.path.exists(csr):
            subprocess.run([os.path.join(REPO, "ocean", "convert"), mtx, csr],
                           capture_output=True, timeout=600, check=True)
        if os.path.exists(stats):
            os.remove(stats)
        subprocess.run(["./spgemm", csr, "config/bench_detail.json"],
                       cwd=os.path.join(REPO, "ocean"), capture_output=True, timeout=1500)
        if not os.path.exists(stats):
            return None, "NOPROF"
        import shutil
        shutil.copy(stats, os.path.join(OUT, "ocn_stats", f"{m}.json"))   # 全相位 sidecar
        os.remove(stats)
        return float(json.load(open(os.path.join(OUT, "ocn_stats", f"{m}.json")))["total_time"]), ""
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, f"ERR:{type(e).__name__}"

mats = load_mats()
mats.sort(key=lambda x: os.path.getsize(x[1]))     # 小阵先跑
csv_path = os.path.join(OUT, "batch_437.csv")
rows = {}
if os.path.exists(csv_path):
    for r in csv.DictReader(open(csv_path)):
        rows[(r["mat"], r["arm"])] = (float(r["ms"]) if r["ms"] else None, r["note"])

def save():
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mat", "arm", "ms", "note"])
        for (m, a), (v, n) in sorted(rows.items()):
            w.writerow([m, a, "" if v is None else f"{v:.3f}", n])

import datetime
print(f"# 437 权威批 启动 {datetime.date.today()} BIN={BIN} WARMUP_N=3", flush=True)
for i, (m, mtx) in enumerate(mats):
    for arm in list(ARMS) + ["OCN"]:
        cur = rows.get((m, arm))
        if cur and cur[0] is not None:
            continue
        if arm == "OCN":
            v, n = ocean(m, mtx)
        else:
            v, n = ours(mtx, arm, m)
        rows[(m, arm)] = (v, n)
    cells = "  ".join(f"{a}={rows.get((m,a),('?','?'))[0] if rows.get((m,a),('?','?'))[0] else rows.get((m,a),('','?'))[1]}"
                      for a in list(ARMS) + ["OCN"])
    print(f"[{i+1}/{len(mats)}] {m:<26} {cells}", flush=True)
    save()

def gm(arm):
    vals = [v for (m, a), (v, n) in rows.items() if a == arm and v and v > 0]
    return (math.exp(sum(math.log(v) for v in vals) / len(vals)), len(vals)) if vals else (None, 0)
print("\n=== 汇总(我方 compute-only ms / ocean 其口径 ms;GM 配对需另算比值)===", flush=True)
for arm in list(ARMS) + ["OCN"]:
    g, n = gm(arm)
    if g: print(f"{arm}: GM={g:.3f} (n={n})", flush=True)
# 与 ocean 的配对 GM(记分牌口径)
for arm in list(ARMS):
    pairs = [(rows[(m, arm)][0], rows[(m, "OCN")][0]) for m in {mm for (mm, a) in rows}
             if (m, arm) in rows and (m, "OCN") in rows and rows[(m, arm)][0] and rows[(m, "OCN")][0]]
    if pairs:
        ratio = math.exp(sum(math.log(a / b) for a, b in pairs) / len(pairs))
        print(f"{arm}/OCN 配对几何均值比 = {ratio:.4f} (n={len(pairs)})", flush=True)
print("BATCH437_DONE", flush=True)
