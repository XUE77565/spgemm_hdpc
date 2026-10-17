#!/usr/bin/env python3
"""held-out 泛化批(10-18):246 阵池(train_disp+synth_disp,与本轮全部拟合不相交)。

列:ADPT3(部署系统,min-of-3) / A(min-of-3,S3 参照) / OCN(ocean)。
目的:①路由规则在陌生人群 sanity ②S3 保持 ③min-of-N 协议试点(夜稳性)。
输出:compare/batch437/heldout_246.csv + run.log;断点续跑。
"""
import csv, json, math, os, re, subprocess, datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_test")
OUT = os.path.join(REPO, "compare", "batch437")
CSRTMP = "/home/xueyizhou/.claude/jobs/a3c873ae/tmp/oceancsr"
os.makedirs(CSRTMP, exist_ok=True)
TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
H2D = re.compile(r"\[(?:hash|mrg3)-prof\] h2d\s+([\d.]+)")
D2H = re.compile(r"\[(?:hash|mrg3)-prof\] d2h\s+([\d.]+)")

pool = []
for d, tag in (("train_disp", "real"), ("synth_disp", "synth")):
    p = os.path.join(REPO, "data", d)
    if not os.path.isdir(p): continue
    for f in os.listdir(p):
        if f.endswith(".mtx"):
            pool.append((f[:-4], os.path.join(p, f), tag))
pool.sort(key=lambda x: os.path.getsize(x[1]))
# 分层抽样:real 按 n 四分位各取 15,synth 取 10 → 70 阵
def n_of(path):
    try:
        line = next(l for l in open(path, encoding="utf-8", errors="ignore") if not l.startswith("%"))
        return int(line.split()[0])
    except Exception:
        return 0
real = [(m, p) for m, p, t in pool if t == "real"]
ns = sorted(n_of(p) for _, p in real)
qs = [ns[min(len(ns)-1, int(len(ns)*f))] for f in (0.25, 0.5, 0.75, 1.0)]
sel, used = [], set()
for qi in range(4):
    cnt = 0
    for m, p in real:
        if m in used: continue
        n = n_of(p)
        lo = qs[qi-1] if qi else 0
        if lo < n <= qs[qi]:
            sel.append((m, p)); used.add(m); cnt += 1
            if cnt >= 15: break
sel += [(m, p) for m, p, t in pool if t == "synth"][:10]
sel.sort(key=lambda x: os.path.getsize(x[1]))
print(f"# held-out 抽样 {len(sel)} 阵(real 60 分层 + synth 10) {datetime.datetime.now():%m-%d %H:%M}", flush=True)

csv_path = os.path.join(OUT, "heldout_246.csv")
rows = {}
if os.path.exists(csv_path):
    for r in csv.DictReader(open(csv_path)):
        rows[r["mat"]] = (float(r["adpt3"]) if r["adpt3"] else None,
                          float(r["a_ms"]) if r["a_ms"] else None,
                          float(r["ocn"]) if r["ocn"] else None, r["note"])

def run_ours(m, p, method, diter):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", USE_MEMPOOL="1", MP_HOST_MB="8192",
               D6EXP="0", K7="0", WARMUP_N="3", METHOD=method)
    if diter: env["DITER_MIN_FLOP"] = "999999999"
    best = None
    for _ in range(3):                      # min-of-3(协议试点)
        try:
            r = subprocess.run([BIN, p], capture_output=True, text=True, env=env, timeout=900)
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT"
        t = TOT.findall(r.stderr)
        if not t: return None, "NOPROF"
        h = H2D.findall(r.stderr); d = D2H.findall(r.stderr)
        v = float(t[-1]) - (float(h[-1]) if h else 0) - (float(d[-1]) if d else 0)
        best = v if best is None else min(best, v)
    return best, ""

def run_ocn(m, p):
    csr = os.path.join(CSRTMP, "ho_" + m + ".csr")
    stats = os.path.join(REPO, "ocean", "stats.json")
    try:
        if not os.path.exists(csr):
            subprocess.run([os.path.join(REPO, "ocean", "convert"), p, csr], capture_output=True, timeout=600, check=True)
        if os.path.exists(stats): os.remove(stats)
        subprocess.run(["./spgemm", csr, "config/bench_detail.json"],
                       cwd=os.path.join(REPO, "ocean"), capture_output=True, timeout=1200)
        if not os.path.exists(stats): return None
        return float(json.load(open(stats))["total_time"])
    except Exception:
        return None

for i, (m, p) in enumerate(sel):
    cur = rows.get(m, (None, None, None, ""))
    if cur[0] is None: cur = (run_ours(m, p, "adaptive", False)[0], *cur[1:])
    if cur[1] is None: cur = (cur[0], run_ours(m, p, "hash", True)[0], cur[2], "")
    if cur[2] is None: cur = (cur[0], cur[1], run_ocn(m, p), "")
    rows[m] = cur
    a, aa, o, nt = cur
    print(f"[{i+1}/{len(sel)}] {m:<24} ADPT3={a if a else nt} A={aa if aa else ''} OCN={o if o else ''}", flush=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mat", "adpt3", "a_ms", "ocn", "note"])
        for mm, (x, y, z, nn) in sorted(rows.items()):
            w.writerow([mm, "" if x is None else f"{x:.3f}", "" if y is None else f"{y:.3f}", "" if z is None else f"{z:.3f}", nn])

pairs = [(a, o) for a, _, o, _ in rows.values() if a and o]
if pairs:
    print(f"\nheld-out ADPT3/OCN GM = {math.exp(sum(math.log(a/o) for a,o in pairs)/len(pairs)):.4f} (n={len(pairs)})", flush=True)
print("HELDOUT_DONE", flush=True)
