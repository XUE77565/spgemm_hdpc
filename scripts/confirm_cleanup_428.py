#!/usr/bin/env python3
"""打扫战役收口 428 confirm(11-10):冻结记分二进制 spgemm_test(A,0.8054) vs spgemm_clean_b4(B,b1-b4累计删除)。
判据:nnz 逐位同(不同即停)+ GM(B/A) 认证 0.8054±噪声。同卡同会话配对(A/B 相邻交错)。
口径:compute-only(TOTAL−h2d−d2h,mrg3 侧同),标准 env 组;断点续跑(有值即跳)。
输出:compare/batch437/confirm_cleanup428.csv + 判读到 stdout。
"""
import csv, math, os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_A = os.path.join(REPO, "spgemm_test")
BIN_B = os.path.join(REPO, "spgemm_clean_b4")
TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
H2D = re.compile(r"\[hash-prof\] h2d\s+([\d.]+)")
D2H = re.compile(r"\[hash-prof\] d2h\s+([\d.]+)")
M3_H = re.compile(r"\[mrg3-prof\] h2d\s+([\d.]+)")
M3_D = re.compile(r"\[mrg3-prof\] d2h\s+([\d.]+)")
NNZ = re.compile(r"Result C:.*nnz = (\d+)")

o337 = [r["mat"] for r in csv.DictReader(open(os.path.join(REPO, "compare/l0_337/l0_337_results.csv")))]
seen = set(o337)
july = [r["matrix"] for r in csv.DictReader(open(os.path.join(REPO, "compare/methods_cmp_first100.csv")))]
mats = [(m, os.path.join(REPO, "data", "ocean", "square", m + ".mtx")) for m in o337]
mats += [(m, os.path.join(REPO, "data", "first100", m + ".mtx")) for m in july
         if m not in seen and os.path.exists(os.path.join(REPO, "data/first100", m + ".mtx"))]
mats.sort(key=lambda x: os.path.getsize(x[1]))
print(f"阵数 {len(mats)}", flush=True)

csv_path = os.path.join(REPO, "compare/batch437/confirm_cleanup428.csv")
rows = {}
if os.path.exists(csv_path):
    for r in csv.DictReader(open(csv_path)):
        if r.get("a_ms"): rows[r["mat"]] = r

def run(binpath, p):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", WARMUP_N="3", USE_MEMPOOL="1",
               MP_HOST_MB="8192", D6EXP="0", METHOD="adaptive")
    try:
        r = subprocess.run([binpath, p], capture_output=True, text=True, env=env, timeout=1500)
    except subprocess.TimeoutExpired:
        return None, None, "TIMEOUT"
    out = r.stdout + r.stderr
    t = TOT.findall(out)
    if not t: return None, None, "NOPROF"
    h = H2D.findall(out); d = D2H.findall(out)
    h3 = M3_H.findall(out); d3 = M3_D.findall(out)
    hh = float(h[-1]) if h else (float(h3[-1]) if h3 else 0)
    dd = float(d[-1]) if d else (float(d3[-1]) if d3 else 0)
    n = NNZ.findall(out)
    return float(t[-1]) - hh - dd, (int(n[-1]) if n else None), ""

fout = open(csv_path, "w", newline="")
w = csv.writer(fout); w.writerow(["mat","a_ms","b_ms","a_nnz","b_nnz","note"])
done = 0
for m, p in mats:
    if m in rows:
        w.writerow([m, rows[m].get("a_ms"), rows[m].get("b_ms"), rows[m].get("a_nnz"), rows[m].get("b_nnz"), rows[m].get("note","")]); fout.flush(); done += 1; continue
    a, na, ea = run(BIN_A, p)
    b, nb, eb = run(BIN_B, p)
    note = (ea or "") + "/" + (eb or "")
    nnz_bad = (na is not None and nb is not None and na != nb)
    if nnz_bad: note += "/NNZ-MISMATCH"
    w.writerow([m, f"{a:.3f}" if a else "", f"{b:.3f}" if b else "", na if na is not None else "", nb if nb is not None else "", note])
    fout.flush(); done += 1
    if done % 25 == 0: print(f"[{done}/{len(mats)}] {m}: A={a} B={b}{' ⚠️'+note if note else ''}", flush=True)
fout.close()

# 判读
rs = list(csv.DictReader(open(csv_path)))
ratios = []; bad = []
for r in rs:
    try:
        if r["a_ms"] and r["b_ms"] and float(r["a_ms"]) > 0 and float(r["b_ms"]) > 0:
            ratios.append((float(r["b_ms"]) / float(r["a_ms"]), r["mat"]))
    except: pass
    if "NNZ-MISMATCH" in (r.get("note") or ""): bad.append(r["mat"])
gm = math.exp(sum(math.log(x) for x, _ in ratios) / len(ratios)) if ratios else -1
worst = sorted(ratios, reverse=True)[:10]
print(f"\n===== 收口判读 =====")
print(f"有效配对 {len(ratios)} | GM(B/A) = {gm:.4f} | nnz 异常 {len(bad)} {bad[:5]}")
print("最差10:", ", ".join(f"{m}={v:.3f}" for v, m in worst))
print("裁定:", "红-停(nnz异)" if bad else ("认证通过:打扫零回归(GM带内)" if gm <= 1.02 and gm >= 0.95 else "超出±2%带,需人工判读"))
