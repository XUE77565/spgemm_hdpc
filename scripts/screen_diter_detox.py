#!/usr/bin/env python3
"""1b' diter 解毒屏(10-12,study/39):L0 中毒阵(A 赢且 ratio≤0.95)上 DCEST on/off 电池。

三重产出:
  ① 1b 价值入账:on 相对 off 的 Δ%,及对 L0 表 A_ms 的"差距关闭率" gap% = (off−on)/(off−A)
  ② L0 特征/解析门标定:双臂相位表(dense_count vs dcest+dcest_fixup)+ diter 行数
  ③ margin 维度数据:on 臂的证书直方图行(直接落盘)
口径:屏级 = AB_RUNS(默认 3)×WARMUP_N=1 交替取中位,compute-only(=TOTAL−h2d−d2h);
正式数(×5+WARMUP_N 默认)留给 Phase 2。双臂 METHOD=hash(ab_env 的 adaptive 会在
小阵上偷跑 merge3 污染口径)。
用法:.venv/bin/python scripts/screen_diter_detox.py [mat ...](默认内置 20 阵清单)
"""
import os, re, subprocess, statistics, sys

BIN = os.environ.get("SPGEMM_BIN", "./spgemm_test")
DIR = "data/ocean/square"
RUNS = int(os.environ.get("AB_RUNS", "3"))
def _f(x):
    try:
        return float(x)
    except ValueError:
        return None   # DNF 等非数值
A_REF = {r["mat"]: _f(r["allhash_ms"]) for r in
         __import__("csv").DictReader(open("compare/l0_337/l0_337_results.csv"))}
A_REF = {k: v for k, v in A_REF.items() if v}
DEFAULT = ["Stanford_Berkeley", "wb-edu", "Ga41As41H72", "TSOPF_RS_b2383",
           "TSOPF_RS_b2383_c1", "Si41Ge41H72", "Ga19As19H42", "nd12k", "Ge99H100",
           "Ge87H76", "pkustk04", "crankseg_2", "Ga10As10H30", "Flan_1565",
           "crankseg_1", "SiO2", "pkustk14", "Si34H36", "a2nnsnsl", "c-71"]

def one(mtx, dcest):
    env = dict(os.environ, MP_HOST_MB="8192", USE_MEMPOOL="1", METHOD="hash",
               DBG="1", WARMUP_N="1", DCEST=str(dcest))
    try:
        r = subprocess.run([BIN, f"{DIR}/{mtx}.mtx"], capture_output=True, text=True, env=env, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    prof = re.findall(r"\[hash-prof\]\s+(\S+)\s+([0-9.]+)\s*ms", r.stderr)
    nnz = re.findall(r"Result C:.*?nnz\s*=\s*(\d+)", r.stdout)
    if not prof or not nnz:
        return None
    keys = [k for k, _ in prof]
    idx = len(keys) - 1 - keys[::-1].index("h2d") if "h2d" in keys else 0
    ph = {k: float(v) for (k, v) in list(prof)[idx:]}
    tot = ph.get("TOTAL(GPU)")
    if tot is None:
        return None
    comp = tot - ph.get("h2d", 0) - ph.get("d2h", 0)
    feat = {"dense_count": ph.get("dense_count", 0.0), "dcest": ph.get("dcest", 0.0),
            "dcest_fixup": ph.get("dcest_fixup", 0.0), "dense_direct": ph.get("dense_direct", 0.0)}
    diter = re.findall(r"diter bin=(\d+) 行", r.stderr)
    cert = re.findall(r"DCEST 证书: (.*)", r.stderr)
    return comp, int(nnz[-1]), feat, (diter[-1] if diter else "?"), (cert[-1] if cert else "")

def med(v):
    return statistics.median(v) if v else None

mats = sys.argv[1:] or DEFAULT
print(f"diter 解毒屏 ×{RUNS}(屏级口径:WARMUP_N=1,双臂 METHOD=hash)\n")
print(f"{'mat':<20}{'off(ms)':>9}{'on(ms)':>9}{'Δ%':>8}{'A_ref':>8}{'gap%':>7}  {'diter':>7}{'dc_ms':>7}{'链ms':>6}  nnz 证书直方图(末轮)")
wins = losses = 0
for mtx in mats:
    off, on, nz, f_off, f_on, dit, cert = [], [], set(), {}, {}, "?", ""
    for _ in range(RUNS):
        for d, acc, fs in ((0, off, f_off), (1, on, f_on)):
            res = one(mtx, d)
            if res:
                acc.append(res[0]); nz.add(res[1])
                for k, v in res[2].items():
                    fs.setdefault(k, []).append(v)
                if d:
                    dit, cert = res[3], res[4]
    mo, mn = med(off), med(on)
    if not (mo and mn and len(nz) == 1):
        print(f"{mtx:<20}  无效(nnz 不一致或缺相位:{sorted(nz)[:2]})")
        continue
    dp = (mn / mo - 1) * 100
    a = A_REF.get(mtx)
    gap = f"{(mo - mn) / (mo - a) * 100:6.1f}" if a and mo > a else "  n/a"
    dc = med(f_off.get("dense_count", [0]))
    ch = med(f_on.get("dcest", [0])) + med(f_on.get("dcest_fixup", [0]))
    wins += dp < -1; losses += dp > 1
    print(f"{mtx:<20}{mo:>9.2f}{mn:>9.2f}{dp:>7.1f}%{a if a else 0:>8.1f}{gap}%  {dit:>7}{dc:>7.2f}{ch:>6.2f}  {list(nz)[0]} {cert[:64]}")
print(f"\n净赢(<−1%):{wins}  净输(>+1%):{losses}  共 {len(mats)} 阵(屏级)")
