#!/usr/bin/env python3
# B线第一刀前置:重锚 bins2-6 每乘积拍数,排热 bin 优先级(loop tick1)
# 原料:batch437/prof/*__ADPT.log(bins/total_flop/分相位,10-15 采集=七刀前)
#       ocn_stats/*.json(numeric.hash 分项)、confirm14_428.csv(终态比值)
# 注意:prof 的 accumulate 是七刀前口径(含 T4a 可剥的 extract 税);
#       结构量(bins 分布/flop)刀不变,bin4-6 numeric 拍数对现役 hash_spa 核仍有效。
import csv, json, re, statistics, sys, os
from pathlib import Path

ROOT = Path("/home/xueyizhou/spgemm-challenge")
PROF = ROOT / "compare/batch437/prof"
OCN  = ROOT / "compare/batch437/ocn_stats"
CONF = ROOT / "compare/batch437/confirm14_428.csv"
SM   = 132          # H100 SXM SM 数
LADDER_MAX = 10     # HASH_LADDER_MAX_BIN;bin b ↔ ht = 32<<b

re_acc  = re.compile(r"\[hash-prof\] accumulate\s+([\d.]+) ms")
re_cs   = re.compile(r"\[hash-prof\] compact\+sort\s+([\d.]+) ms")
re_flop = re.compile(r"total_flop=(\d+)")
re_bins = re.compile(r"\[hash\] bins: (.*)")

def parse_prof(mat):
    f = PROF / f"{mat}__ADPT.log"
    if not f.exists(): return None
    txt = f.read_text(errors="replace")
    accs, css = [], []
    for line in txt.splitlines():
        m = re_acc.search(line)
        if m: accs.append(float(m.group(1)))
        m = re_cs.search(line)
        if m: css.append(float(m.group(1)))
    mf = re_flop.findall(txt)
    mb = re_bins.findall(txt)
    if not accs or not mf or not mb: return None
    flop = int(mf[-1])
    bins = {}
    for tok in mb[-1].split():
        b, c = tok.split(":")
        bins[int(b)] = int(c)
    return dict(acc=statistics.median(accs), acc_n=len(accs),
                cs=statistics.median(css) if css else None,
                flop=flop, bins=bins)

def parse_ocn(mat):
    f = OCN / f"{mat}.json"
    if not f.exists(): return None
    d = json.loads(f.read_text())
    t = d.get("timing", {})
    num = t.get("numeric", {})
    hash_ms = num.get("hash", 0.0) or 0.0
    fat_ms  = num.get("fat", 0.0) or 0.0
    dense_ms= num.get("dense", 0.0) or 0.0
    us_m    = num.get("usparse_main", 0.0) or 0.0   # wf=2:ESC 子组梯主核
    us_o    = num.get("usparse_outlier", 0.0) or 0.0
    bin_ms  = num.get("binning", 0.0) or 0.0
    epi = t.get("epilogue", {})
    epi_ms = sum(v for v in epi.values() if isinstance(v, (int, float)))
    flop = d.get("stats", {}).get("intermediate_product")
    wf = d.get("decision", {}).get("workflow")
    return dict(ocn_hash=hash_ms, ocn_fat=fat_ms, ocn_dense=dense_ms,
                ocn_us=us_m + us_o, ocn_bin=bin_ms,
                ocn_num_core=hash_ms+fat_ms+dense_ms+us_m+us_o,
                ocn_epi=epi_ms, ocn_flop=flop, wf=wf)

# 七刀对 accumulate 的已知折算(study/53/54;估算层,GPU 重锚前作排序参考)
def knife_effect(mat):
    m = mat.lower()
    if mat == "StocF-1465": return 0.53, "T4a −53%"
    if mat.startswith("rgg_n_2_22"): return 0.37, "T4a −37%"
    if mat == "ldoor": return 0.07, "T4a −7%"
    if m.startswith("para-"): return 0.23, "T4c −23%"
    if m.startswith(("atmosmod", "cnr-")): return 0.25, "T4c 族~−25%"
    if m.startswith("htc_336") or m.startswith("htc_"): return 0.42, "T4e −39~44%"
    if mat.startswith("exdata"): return 0.88, "T6-v7 18.0→2.2"
    return 0.0, ""

rows_out = []
conf = {r["mat"]: r for r in csv.DictReader(open(CONF))}
for mat, cr in conf.items():
    ratio = float(cr["adpt14_ms"]) / float(cr["ocn14_ms"]) if cr["ocn14_ms"] else None
    p = parse_prof(mat)
    o = parse_ocn(mat)
    if not p or not o or not p["flop"]: continue
    flop = p["flop"]
    if o["ocn_flop"] and abs(o["ocn_flop"] - flop) > max(1000, 1e-6*flop):
        flop_mismatch = True
    else:
        flop_mismatch = False
    ladder = {b: c for b, c in p["bins"].items() if 0 <= b <= LADDER_MAX}
    n_ladder = sum(ladder.values())
    allrows  = sum(p["bins"].values())
    hash_frac = n_ladder / allrows if allrows else 0
    # 行加权 bin 中心(梯内);b23/b45/b56 带拆分(行数)
    if n_ladder:
        wmean_bin = sum(b*c for b, c in ladder.items()) / n_ladder
    else:
        wmean_bin = -1
    b23 = sum(c for b, c in ladder.items() if 2 <= b <= 3)
    b45 = sum(c for b, c in ladder.items() if 4 <= b <= 5)
    b6p = sum(c for b, c in ladder.items() if 6 <= b <= LADDER_MAX)
    # 拍数:accumulate 折算 ns/乘积/SM(cyc@1.76GHz ≈ ×1.76)
    ocn_core = o["ocn_hash"] if o["ocn_hash"] > 0 else o["ocn_num_core"]
    beat_ns  = p["acc"] * 1e6 / (flop / SM)
    obeat_ns = ocn_core * 1e6 / (flop / SM) if ocn_core > 0 else None
    gap = p["acc"] - ocn_core
    keff, kname = knife_effect(mat)
    gap_now = gap * (1 - keff)
    rows_out.append(dict(
        mat=mat, ratio14=ratio, wf=o["wf"],
        acc=p["acc"], ocn_hash=o["ocn_hash"], ocn_core=round(ocn_core, 2),
        gap=gap, gap_now=round(gap_now, 2), knife=kname,
        cs=p["cs"], ocn_epi=o["ocn_epi"],
        flop=flop, flop_mismatch=flop_mismatch,
        hash_frac=round(hash_frac, 3),
        wmean_bin=round(wmean_bin, 2),
        b23=b23, b45=b45, b6p=b6p,
        beat_ns=round(beat_ns, 1),
        obeat_ns=round(obeat_ns, 1) if obeat_ns else None,
        beat_ratio=round(beat_ns/obeat_ns, 2) if obeat_ns else None,
    ))

rows_out.sort(key=lambda r: -(r["gap_now"] or 0))
out = ROOT / "compare/batch437/reanchor_bins.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader(); w.writerows(rows_out)

# ── 汇总:钱在哪个带(accumulate 正差距,按主导带聚合)──
def band(r):
    m = max([("b23", r["b23"]), ("b45", r["b45"]), ("b6p", r["b6p"])],
            key=lambda kv: kv[1])
    tot = r["b23"] + r["b45"] + r["b6p"]
    return m[0] if tot > 0 and m[1] / tot > 0.5 else "mixed"
agg = {}
for r in rows_out:
    if r["gap_now"] <= 0.05 or r["hash_frac"] < 0.3: continue
    k = band(r)
    a = agg.setdefault(k, dict(n=0, gap=0.0, mats=[]))
    a["n"] += 1; a["gap"] += r["gap_now"]; a["mats"].append((r["mat"], round(r["gap_now"],1), r["ratio14"], r["knife"]))
print(f"{'带':6} {'阵数':>4} {'accumulate正差距ms(七刀后估算)':>20}  top阵")
for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["gap"]):
    tops = sorted(a["mats"], key=lambda x: -x[1])[:6]
    print(f"{k:6} {a['n']:>4} {a['gap']:>20.1f}  " +
          ", ".join(f"{m}({g}ms,r={rt and round(rt,2)},{kn})" for m, g, rt, kn in tops))
print(f"\n总计正差距(七刀后估算){sum(a['gap'] for a in agg.values()):.0f} ms / {sum(a['n'] for a in agg.values())} 阵")
print("Top30(按 gap_now,七刀后估算):")
for r in rows_out[:30]:
    print(f"  {r['mat']:22} r14={r['ratio14'] and round(r['ratio14'],2)} wf={r['wf']} "
          f"acc={r['acc']:7.2f} ocn_core={r['ocn_core']:7.2f} gap_now={r['gap_now']:7.2f} "
          f"beat={r['beat_ns']:6.1f}/{r['obeat_ns'] and round(r['obeat_ns'])}ns ratio={r['beat_ratio']} "
          f"wm={r['wmean_bin']} 2-3:{r['b23']} 4-5:{r['b45']} 6+:{r['b6p']} hf={r['hash_frac']} {r['knife']}")
# 微输带 1.1-1.3 × 的人群分布
band111 = [r for r in rows_out if r["ratio14"] and 1.1 <= r["ratio14"] <= 1.3]
print(f"\n微输带(1.1-1.3×){len(band111)} 阵,其中 hash 人口(hf≥0.3)带分布:")
for k in ("b23", "b45", "b6p", "mixed"):
    sub = [r for r in band111 if band(r) == k and r["hash_frac"] >= 0.3]
    if sub:
        print(f"  {k:6} {len(sub):>3} 阵 gap_now Σ={sum(r['gap_now'] for r in sub):7.1f}ms  "
              + ", ".join(f"{r['mat']}({r['gap_now']})" for r in sorted(sub, key=lambda x: -x['gap_now'])[:5]))
mm = [r["mat"] for r in rows_out if r["flop_mismatch"]]
if mm: print(f"\n⚠ flop 口径不一致 {len(mm)} 阵(分析里剔除拍数比较):{mm[:10]}")
print(f"\n→ {out}")
