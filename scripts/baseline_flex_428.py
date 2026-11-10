#!/usr/bin/env python3
"""FlexSpGEMM 基线 @我方 428 bench(11-10):external_sota/FlexSpGEMM(src/test,sm_90,cuda-12.8 构建)。
口径:对方 time_tile = csr2tile+symbolic+numeric(gettimeofday 端到端计算相,含格式转换);
对照我方 compute-only(TOTAL−h2d−d2h)→ Tab.1 备注相位差。
协议:对方官方 = 单发无 warmup(CHECK_RESULT=0 计时件);CUDA-700 时以 -csr2tile cpu 复试一次。
GPU=0(卡1 被收口 confirm 占用);超时 1500s;nnz 与 baseline_hsmu.cnnz 交叉核对。
输出:compare/batch437/baseline_flex.csv(断点续跑:有 ok 值即跳)。
"""
import csv, math, os, subprocess, sys

REPO = "/home/xueyizhou/spgemm-challenge"
FLEX = os.path.join(REPO, "external_sota/FlexSpGEMM")
SRC = os.path.join(FLEX, "src")
OUT = os.path.join(REPO, "compare/batch437/baseline_flex.csv")

o337 = [r["mat"] for r in csv.DictReader(open(os.path.join(REPO, "compare/l0_337/l0_337_results.csv")))]
seen = set(o337)
july = [r["matrix"] for r in csv.DictReader(open(os.path.join(REPO, "compare/methods_cmp_first100.csv")))]
mats = [(m, os.path.join(REPO, "data/ocean/square", m + ".mtx")) for m in o337]
mats += [(m, os.path.join(REPO, "data/first100", m + ".mtx")) for m in july
         if m not in seen and os.path.exists(os.path.join(REPO, "data/first100", m + ".mtx"))]
mats.sort(key=lambda x: os.path.getsize(x[1]))
hsmu_nnz = {r["mat"]: r.get("cnnz", "") for r in csv.DictReader(open(os.path.join(REPO, "compare/batch437/baseline_hsmu.csv")))}

done = {}
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        if r["status"] == "ok": done[r["mat"]] = r

def run_one(mat, path, backend):
    res_csv = os.path.join(FLEX, "result/results_tile.csv")
    step_csv = os.path.join(FLEX, "result/step_runtime.csv")
    off1 = os.path.getsize(res_csv) if os.path.exists(res_csv) else 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    try:
        p = subprocess.run(["./test", "-d", "0", "-aat", "0", "-tau", "0.7", "-csr2tile", backend, path],
                           cwd=SRC, env=env, capture_output=True, text=True, timeout=1500)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    err = (p.stdout + p.stderr)
    if "CUDA error" in err or p.returncode != 0:
        tag = "CUDA-ERR:" + (err.split("CUDA error")[1][:40].strip().replace("\n", " ") if "CUDA error" in err else str(p.returncode))
        return None, tag
    try:
        with open(res_csv) as f:
            f.seek(off1); new = f.read().strip().splitlines()
        if not new: return None, "NOCSV"
        cols = new[-1].split(",")
        nnzC = int(cols[5]); time_tile = float(cols[7])
        # step 相位(若 step_runtime.csv 有新行)
        phases = ""
        off2 = None
        step = os.path.join(FLEX, "result/step_runtime.csv")
        off2f = os.path.join(FLEX, ".step_off")
        off2 = int(open(off2f).read()) if os.path.exists(off2f) else (os.path.getsize(step) - 200 if os.path.exists(step) else 0)
        with open(step) as f:
            f.seek(max(0, os.path.getsize(step) - 200)); tail = f.read().strip().splitlines()
        if tail and tail[-1].split(",")[0] == path:
            c2 = tail[-1].split(",")
            if len(c2) >= 11: phases = f"csr2tile={c2[7]},sym={c2[8]},num={c2[9]},malloc={c2[10]}"
        return (time_tile, nnzC, phases), ""
    except Exception as e:
        return None, f"PARSE:{e}"

cols_w = ["mat", "status", "time_tile_ms", "nnzC", "nnz_hsmu", "nnz_match", "backend", "phases", "note"]
f = open(OUT, "w", newline="")
w = csv.DictWriter(f, fieldnames=cols_w); w.writeheader()
ok_n = fail_n = 0
for i, (m, p) in enumerate(mats):
    if m in done:
        w.writerow(done[m]); f.flush(); ok_n += 1; continue
    r, err = run_one(m, p, "gpu")
    backend = "gpu"
    if r is None and ("CUDA-ERR" in err or "TIMEOUT" in err):
        r2, err2 = run_one(m, p, "cpu")
        if r2 is not None: r, backend = r2, "cpu"; err = ""
        elif "PARSE" in err2: r, err = None, err2
    if r is None:
        w.writerow({"mat": m, "status": "fail", "note": err}); fail_n += 1
    else:
        time_tile, nnzC, phases = r
        hn = hsmu_nnz.get(m, "")
        match = ("same" if hn and hn.isdigit() and int(hn) == nnzC else ("diff!" if hn and hn.isdigit() else "n/a"))
        w.writerow({"mat": m, "status": "ok", "time_tile_ms": f"{time_tile:.3f}", "nnzC": nnzC,
                    "nnz_hsmu": hn, "nnz_match": match, "backend": backend, "phases": phases, "note": ""})
        ok_n += 1
    f.flush()
    if (i + 1) % 25 == 0: print(f"[{i+1}/{len(mats)}] {m}: {'ok' if r else err}", flush=True)
f.close()
rs = [float(r["time_tile_ms"]) for r in csv.DictReader(open(OUT)) if r["status"] == "ok"]
nnzbad = [r["mat"] for r in csv.DictReader(open(OUT)) if r.get("nnz_match") == "diff!"]
print(f"\n===== FlexSpGEMM@428 判读 =====")
print(f"ok {len(rs)} / fail {fail_n} | nnz 与 HSMU 交叉不一致 {len(nnzbad)} {nnzbad[:5]}")
if rs: print(f"time_tile 中位 {sorted(rs)[len(rs)//2]:.3f} ms | 最大 {max(rs):.1f} ms")
