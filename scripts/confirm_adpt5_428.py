#!/usr/bin/env python3
"""确认批五(10-21 夜,FLAT 转正夜):FLAT 翻默认+预门短路后的部署系统独立实测 + ocean 同夜同卡(GPU1)重测。

列:ADPT5 = METHOD=adaptive(新二进制,3ec2e88);OCN5 = ocean 同夜(convert+spgemm)。
口径:compute-only(WARMUP_N=3)/ocean total_time,同 batch437;同夜=同日配对纪律。
输出:compare/batch437/confirm5_428.csv + run.log;断点续跑(有值即跳)。
"""
import csv, json, math, os, re, shutil, subprocess, datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_test")
OUT = os.path.join(REPO, "compare", "batch437")
CSRTMP = "/home/xueyizhou/.claude/jobs/a3c873ae/tmp/oceancsr"
os.makedirs(CSRTMP, exist_ok=True)
TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
H2D = re.compile(r"\[hash-prof\] h2d\s+([\d.]+)")
D2H = re.compile(r"\[hash-prof\] d2h\s+([\d.]+)")
M3_H = re.compile(r"\[mrg3-prof\] h2d\s+([\d.]+)")
M3_D = re.compile(r"\[mrg3-prof\] d2h\s+([\d.]+)")

o337 = [r["mat"] for r in csv.DictReader(open(os.path.join(REPO, "compare/l0_337/l0_337_results.csv")))]
seen = set(o337)
july = [r["matrix"] for r in csv.DictReader(open(os.path.join(REPO, "compare/methods_cmp_first100.csv")))]
mats = [(m, os.path.join(REPO, "data", "ocean", "square", m + ".mtx")) for m in o337]
mats += [(m, os.path.join(REPO, "data", "first100", m + ".mtx")) for m in july
         if m not in seen and os.path.exists(os.path.join(REPO, "data/first100", m + ".mtx"))]
mats.sort(key=lambda x: os.path.getsize(x[1]))

csv_path = os.path.join(OUT, "confirm5_428.csv")
rows = {}
if os.path.exists(csv_path):
    for r in csv.DictReader(open(csv_path)):
        rows[r["mat"]] = (float(r["adpt5_ms"]) if r["adpt5_ms"] else None,
                          float(r["ocn5_ms"]) if r["ocn5_ms"] else None, r["note"])

def adpt2(m, p):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", WARMUP_N="3", USE_MEMPOOL="1",
               MP_HOST_MB="8192", D6EXP="0", K7="0", METHOD="adaptive")   # FLAT=新默认(翻默认验证)
    try:
        r = subprocess.run([BIN, p], capture_output=True, text=True, env=env, timeout=1500)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    t = TOT.findall(r.stderr)
    if not t: return None, "NOPROF"
    h = H2D.findall(r.stderr); d = D2H.findall(r.stderr)
    h3 = M3_H.findall(r.stderr); d3 = M3_D.findall(r.stderr)
    hh = float(h[-1]) if h else (float(h3[-1]) if h3 else 0)
    dd = float(d[-1]) if d else (float(d3[-1]) if d3 else 0)
    return float(t[-1]) - hh - dd, ""

def ocn(m, p):
    csr = os.path.join(CSRTMP, m + ".csr")
    stats = os.path.join(REPO, "ocean", "stats.json")
    try:
        if not os.path.exists(csr):
            subprocess.run([os.path.join(REPO, "ocean", "convert"), p, csr],
                           capture_output=True, timeout=600, check=True)
        if os.path.exists(stats): os.remove(stats)
        subprocess.run(["./spgemm", csr, "config/bench_detail.json"],
                       cwd=os.path.join(REPO, "ocean"), capture_output=True, timeout=1500,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="1"))
        if not os.path.exists(stats): return None, "NOPROF"
        return float(json.load(open(stats))["total_time"]), ""
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, f"ERR:{type(e).__name__}"

print(f"# 确认批启动 {datetime.datetime.now():%m-%d %H:%M} BIN@3ec2e88", flush=True)
for i, (m, p) in enumerate(mats):
    cur = rows.get(m, (None, None, ""))
    if cur[0] is None:
        cur = (adpt2(m, p)[0], cur[1], "")
    if cur[1] is None:
        o = ocn(m, p)
        cur = (cur[0], o[0], o[1] if o[0] is None else "")
    rows[m] = cur
    a, o, nt = cur
    print(f"[{i+1}/{len(mats)}] {m:<26} ADPT5={a if a else nt} OCN5={o if o else ''}", flush=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mat", "adpt5_ms", "ocn5_ms", "note"])
        for mm, (aa, oo, nn) in sorted(rows.items()):
            w.writerow([mm, "" if aa is None else f"{aa:.3f}", "" if oo is None else f"{oo:.3f}", nn])

pairs = [(a, o) for a, o, _ in rows.values() if a and o]
if pairs:
    gm = math.exp(sum(math.log(a / o) for a, o in pairs) / len(pairs))
    reg = max((rows[m][0] / oldA for m in rows if rows[m][0] and (oldA := None)), default=0)
    print(f"\nADPT5/OCN5 配对 GM = {gm:.4f} (n={len(pairs)})", flush=True)
print("CONFIRM_DONE", flush=True)
