#!/usr/bin/env python3
"""1c 计时验证(10-15 晚,study/42 GPU-gated 队列 #1):HEAD(1c 两刀)vs pre-1c(b014735)。

设计:ABBA 配对(每阵 head→pre→pre→head 消漂移),WARMUP_N=3(安静机器),
compute-only 口径同批脚本(TOTAL−h2d−d2h);靶=launch 税敏感的小阵。
判据:配对几何均值比 ≤1.01 = PASS(1c 声称中性偏正);>1.03 = FAIL(回滚评估)。
用法:env CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/validate_1c_timing.py
"""
import os, re, subprocess, math, statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAD = os.path.join(REPO, "spgemm_test")
PRE  = os.environ.get("PRE_BIN", "/home/xueyizhou/.claude/jobs/a3c873ae/tmp/pre1c/spgemm_test")
TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
H2D = re.compile(r"\[hash-prof\] h2d\s+([\d.]+)")
D2H = re.compile(r"\[hash-prof\] d2h\s+([\d.]+)")

# 靶阵:launch 税敏感=小阵。first100 取 8 小 + ocean337 取 6 小(体积升序)
def pick(d, n):
    fs = sorted((f for f in os.listdir(d) if f.endswith(".mtx")),
                key=lambda f: os.path.getsize(os.path.join(d, f)))
    return [os.path.join(d, f) for f in fs[:n]]
mats = pick(os.path.join(REPO, "data", "first100"), 8) + \
       pick(os.path.join(REPO, "data", "ocean", "square"), 6)

def run(binp, mtx):
    env = dict(os.environ, METHOD="hash", WARMUP_N="3", USE_MEMPOOL="1", MP_HOST_MB="8192",
               D6EXP="0", K7="0")
    try:
        r = subprocess.run([binp, mtx], capture_output=True, text=True, env=env, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    t = TOT.findall(r.stderr)
    if not t or "OVERFLOW" in r.stdout:
        return None
    h = H2D.findall(r.stderr); d = D2H.findall(r.stderr)
    return float(t[-1]) - (float(h[-1]) if h else 0) - (float(d[-1]) if d else 0)

ratios = []
print(f"{'matrix':<28}{'pre1c(ms)':>11}{'head(ms)':>11}{'ratio':>8}", flush=True)
for mtx in mats:
    name = os.path.basename(mtx)[:-4]
    seq = [run(HEAD, mtx), run(PRE, mtx), run(PRE, mtx), run(HEAD, mtx)]  # ABBA
    hs = [v for v in (seq[0], seq[3]) if v]; ps = [v for v in (seq[1], seq[2]) if v]
    if not hs or not ps:
        print(f"{name:<28}{'NOPROF/TIMEOUT':>30}", flush=True); continue
    h, p = min(hs), min(ps)          # min=净窗惯例
    r = h / p
    ratios.append(r)
    print(f"{name:<28}{p:>11.3f}{h:>11.3f}{r:>8.4f}", flush=True)

if ratios:
    g = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    verdict = "PASS(中性或更好)" if g <= 1.01 else ("FAIL(评估回滚)" if g > 1.03 else "BORDER(>1%:查明细)")
    print(f"\n配对几何均值比 head/pre1c = {g:.4f} (n={len(ratios)})  => {verdict}", flush=True)
print("V1C_DONE", flush=True)
