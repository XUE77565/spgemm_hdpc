#!/usr/bin/env python3
"""first100 净窗复测(2026-09-03 用户纠错驱动):
07-27 档案显示 dispatcher→merge3 净赢 Ocean 0.622×(81/19),但那是旧口径(墙钟/相位和)。
本脚本用现代口径(compute-only + nnz 红旗 + harness 双 expand)复测当前 v30-default 二进制。
用法:.venv/bin/python scripts/revalidate_first100.py [--limit N]"""
import os, sys, csv, re, subprocess, math, argparse

BIN = "./spgemm_test"
DIR = "data/first100"
REF = "compare/method_cmp_20260727_160909/methods_cmp.csv"
OUT = "compare/first100_revalidate.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

ref = {r["matrix"]: r for r in csv.DictReader(open(REF))}
def num(x):
    try: return float(x)
    except: return None

def one(mtx):
    env = dict(os.environ, USE_MEMPOOL="1", METHOD="adaptive", DBG="1", CUDA_VISIBLE_DEVICES=os.environ.get("F100_GPU","0"))
    try:
        r = subprocess.run([BIN, f"{DIR}/{mtx}.mtx"], capture_output=True, text=True, env=env, timeout=300)
    except subprocess.TimeoutExpired:
        return None, None, ""
    prof = re.findall(r"\[(?:hash|mrg3)-prof\]\s+(\S+)\s+([0-9.]+)\s*ms", r.stderr)
    nnz = re.findall(r"Result C:.*?nnz\s*=\s*(\d+)", r.stdout)
    if not prof or not nnz:
        return None, None, ""
    keys = [k for k, _ in prof]
    h2d_key = "h2d" if "h2d" in keys else keys[0]
    idx = len(keys) - 1 - keys[::-1].index(h2d_key)
    ph = {k: float(v) for (k, v) in list(prof)[idx:]}
    tot = ph.get("TOTAL(GPU)")
    if tot is None:
        return None, None, ""
    comp = tot - ph.get("h2d", 0) - ph.get("d2h", 0)
    ch = "[mrg3]" if "mrg3-prof" in r.stderr.split("TOTAL(GPU)")[-1][:2000] else ""
    return comp, int(nnz[-1]), ch

rows = []
for m, r in ref.items():
    if args.limit and len(rows) >= args.limit: break
    comp, nz, tag = one(m)
    ocn = num(r.get("Ocean")); old = num(r.get("Auto")); old_nz = num(r.get("cnnz"))
    red = (old_nz and nz and abs(nz - old_nz) / old_nz > 0.01)
    rows.append({"matrix": m, "n": r["n"], "comp_ms": comp if comp else "DNF",
                 "nnz": nz if nz else "", "red": "⚠" if red else "",
                 "ocean0727": ocn, "auto0727": old, "cnnz0727": r.get("cnnz", ""), "tag": tag})
    print(f"[{len(rows)}/{len(ref)}] {m:16} comp={comp if comp else 'DNF':>8} nnz={nz}{' ⚠RED' if red else ''} (0727 Auto={old})", flush=True)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

v = [(float(r["comp_ms"]), r["ocean0727"]) for r in rows if r["comp_ms"] != "DNF" and r["ocean0727"] and not r["red"]]
if v:
    w_ = sum(1 for a, o in v if a <= o)
    g = math.exp(sum(math.log(a / o) for a, o in v) / len(v))
    print(f"\n=== first100 复测:{len(v)} 有效,赢 Ocean {w_},geomean {g:.4f}×(07-27 档案:0.622×)===")
