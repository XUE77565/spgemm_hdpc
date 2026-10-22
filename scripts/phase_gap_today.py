#!/usr/bin/env python3
# tick5:今日钱图 = /tmp/phase_now/*.log(现役 A 臂,电池 env)逐相位中位 × ocean ocn_stats
import re, statistics, json, sys
from pathlib import Path

OCN = Path("/home/xueyizhou/spgemm-challenge/compare/batch437/ocn_stats")
MATS = ["CurlCurl_4","rgg_n_2_22_s0","af_shell10","af_1_k101","StocF-1465",
        "nlpkkt120","Bump_2911","Geo_1438","Serena","Hook_1498","cage15","nlpkkt80"]

# 我方相位 → ocean 对位(口径注:ocean numeric.hash≈accumulate+compact 内排序分工不同;
# epilogue≈compact+sort;analysis+estimation≈count_flop+mh+est_scan;prologue≈h2d)
PAT = {
    "h2d": r"\[hash-prof\] h2d\s+([\d.]+)",
    "count": r"\[hash-prof\] count_flop\s+([\d.]+)",
    "mh": r"\[hash-prof\] mh_construct\s+([\d.]+)",
    "mhmerge": r"\[hash-prof\] mh_merge\s+([\d.]+)",
    "estscan": r"\[hash-prof\] est_scan\s+([\d.]+)",
    "binning": r"\[hash-prof\] binning\s+([\d.]+)",
    "dcount": r"\[hash-prof\] dense_count\s+([\d.]+)",
    "acc": r"\[hash-prof\] accumulate\s+([\d.]+)",
    "retry": r"\[hash-prof\] retry\s+([\d.]+)",
    "cnnz": r"\[hash-prof\] cnnz_scan\s+([\d.]+)",
    "compact": r"\[hash-prof\] compact\+sort\s+([\d.]+)",
    "ddirect": r"\[hash-prof\] dense_direct\s+([\d.]+)",
}
re_flop = re.compile(r"total_flop=(\d+)")

def parse(mat):
    f = Path(f"/tmp/phase_now/{mat}.log")
    if not f.exists(): return None
    txt = f.read_text(errors="replace")
    out = {}
    for k, p in PAT.items():
        v = [float(x) for x in re.findall(p, txt)]
        if v: out[k] = statistics.median(v)
    mf = re_flop.findall(txt)
    out["flop"] = int(mf[-1]) if mf else 0
    return out

def ocn(mat):
    f = OCN / f"{mat}.json"
    if not f.exists(): return None
    d = json.loads(f.read_text())
    t = d["timing"]; num = t.get("numeric", {}); epi = t.get("epilogue", {})
    est = t.get("estimation", {}); ana = t.get("analysis", {})
    return dict(
        est=sum(v for v in est.values() if isinstance(v,(int,float))),
        ana=sum(v for v in ana.values() if isinstance(v,(int,float))),
        num_hash=num.get("hash",0), num_dense=num.get("dense",0), num_fat=num.get("fat",0),
        num_us=(num.get("usparse_main",0) or 0)+(num.get("usparse_outlier",0) or 0),
        epi=sum(v for v in epi.values() if isinstance(v,(int,float))),
    )

print(f"{'阵':16} {'acc':>7} {'ocn_numcore':>10} {'acc比':>6} | {'compact':>7} {'ocn_epi':>7} {'cmp比':>6} | {'count+est链':>10} {'ocn_est+ana':>10} {'比':>6} | {'flop(M)':>7} {'拍ns':>5}")
tot = {}
for m in MATS:
    p, o = parse(m), ocn(m)
    if not p or not o:
        print(f"{m:16}  缺数据 p={bool(p)} o={bool(o)}"); continue
    ours_est_chain = p.get("count",0)+p.get("mh",0)+p.get("mhmerge",0)+p.get("estscan",0)+p.get("binning",0)+p.get("dcount",0)
    ocn_est = o["est"]+o["ana"]
    ocn_numcore = o["num_hash"]+o["num_dense"]+o["num_fat"]+o["num_us"]
    compact = p.get("compact",0)+p.get("ddirect",0)
    flop = p["flop"]
    beat = p["acc"]*1e6/(flop/132) if flop else 0
    print(f"{m:16} {p.get('acc',0):>7.2f} {ocn_numcore:>10.2f} {p.get('acc',0)/ocn_numcore if ocn_numcore else 0:>6.2f} | "
          f"{compact:>7.2f} {o['epi']:>7.2f} {compact/o['epi'] if o['epi'] else 0:>6.2f} | "
          f"{ours_est_chain:>10.2f} {ocn_est:>10.2f} {ours_est_chain/ocn_est if ocn_est else 0:>6.2f} | "
          f"{flop/1e6:>7.0f} {beat:>5.1f}")
    for k in ("acc","compact","est"):
        pass
    tot[m] = (p, o)
# 差距汇总(我−ocean,ms):按相位块
print("\n差距汇总(我−ocean,正=我慢):")
print(f"{'阵':16} {'accΔ':>8} {'cmpΔ':>8} {'estΔ':>8} {'ΣΔ':>8}")
for m,(p,o) in tot.items():
    ocn_numcore = o["num_hash"]+o["num_dense"]+o["num_fat"]+o["num_us"]
    accd = p.get("acc",0)-ocn_numcore
    cmpd = (p.get("compact",0)+p.get("ddirect",0))-o["epi"]
    estd = (p.get("count",0)+p.get("mh",0)+p.get("mhmerge",0)+p.get("estscan",0)+p.get("binning",0)+p.get("dcount",0))-(o["est"]+o["ana"])
    print(f"{m:16} {accd:>8.2f} {cmpd:>8.2f} {estd:>8.2f} {accd+cmpd+estd:>8.2f}")