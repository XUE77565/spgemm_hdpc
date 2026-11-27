#!/usr/bin/env python3
"""ncu 原始指标的挑选汇总(第一版:指标名按子串挑,不认识的版本差异不报错)。

输入:--dir 下的 <阵>__<臂>.raw.csv,每个是 `ncu --import X.ncu-rep --csv --page raw` 的原样输出
      (宽表:首行指标名,次行可能是单位行,其后每个被剖析的 kernel 发射一行;表头之前可以夹别的行)。
输出:<前缀>_long.csv  mat,arm,launch,kernel,metric,unit,value(只含挑中的指标)
      <前缀>.txt       逐(阵,臂,核族)把挑中的指标按发射求均值,并排列出
用法:ncu_sum.py --dir DIR --out PREFIX
"""
import argparse, csv, glob, io, os, re, sys

PICK = [
    "gpu__time_duration", "sm__throughput", "gpu__compute_memory_throughput", "dram__throughput",
    "dram__bytes_read.sum", "dram__bytes_write.sum", "l1tex__t_sector_hit_rate", "lts__t_sector_hit_rate",
    "l1tex__throughput", "lts__throughput", "smsp__inst_executed.sum", "smsp__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak", "smsp__inst_executed_per_warp", "smsp__warp_issue_stalled",
    "smsp__warps_issue_stalled", "launch__occupancy_limit", "launch__grid_size", "launch__block_size",
    "launch__shared_mem", "launch__registers_per_thread", "sm__maximum_warps_per_active_cycle_pct",
]


def family(name):
    name = re.sub(r"^(static\s+)?void\s+", "", name.strip())
    m = re.match(r"[A-Za-z_][A-Za-z0-9_:]*", name)
    return m.group(0) if m else name[:40]


def to_float(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def read_raw(path):
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return None
    hi = -1
    for i, ln in enumerate(lines):
        if "Kernel Name" in ln and "ID" in ln:
            hi = i
            break
    if hi < 0:
        return None
    rd = list(csv.reader(io.StringIO("\n".join(lines[hi:]))))
    if len(rd) < 2:
        return None
    hdr = rd[0]
    try:
        ck = hdr.index("Kernel Name")
    except ValueError:
        return None
    body = rd[1:]
    units = [""] * len(hdr)
    if body and body[0] and not body[0][0].strip().isdigit():
        units = body[0] + [""] * (len(hdr) - len(body[0]))
        body = body[1:]
    cols = [j for j, h in enumerate(hdr) if any(p in h for p in PICK)]
    out = []
    for k, r in enumerate(body):
        if len(r) <= ck:
            continue
        for j in cols:
            if j < len(r) and r[j] != "":
                out.append((k, r[ck], hdr[j], units[j] if j < len(units) else "", r[j]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    outf = open(a.out + ".txt", "w", encoding="utf-8")

    def say(s=""):
        print(s)
        outf.write(s + "\n")

    files = sorted(glob.glob(os.path.join(a.dir, "*__*.raw.csv")))
    allrows = []
    bad = []
    for f in files:
        b = os.path.basename(f)[:-len(".raw.csv")]
        parts = b.split("__")
        if len(parts) != 2:
            continue
        rows = read_raw(f)
        if not rows:
            bad.append(b)
            continue
        for k, kern, metric, unit, val in rows:
            allrows.append((parts[0], parts[1], k, kern, metric, unit, val))
    say("# ncu_sum: 文件 %d 个,读出指标的 %d 个,读不出的: %s" % (len(files), len(files) - len(bad), " ".join(bad) or "无"))
    with open(a.out + "_long.csv", "w", encoding="utf-8", newline="") as g:
        w = csv.writer(g)
        w.writerow(["mat", "arm", "launch", "kernel", "metric", "unit", "value"])
        for r in allrows:
            w.writerow([r[0], r[1], r[2], r[3][:160], r[4], r[5], r[6]])
    if not allrows:
        say("# 没有可用数据(指标名可能与挑选表不符:原始 CSV 仍在,下一轮按实际表头改挑选表)")
        outf.close()
        return 0
    agg = {}      # (mat, family, metric) → arm → [values]
    units = {}
    arms = []
    for m, arm, k, kern, metric, unit, val in allrows:
        v = to_float(val)
        if v is None:
            continue
        agg.setdefault((m, family(kern), metric), {}).setdefault(arm, []).append(v)
        units[metric] = unit
        if arm not in arms:
            arms.append(arm)
    say("")
    say("===== 逐(阵,核族,指标):各臂在被剖析发射上的均值(发射数)=====")
    say("%-12s %-22s %-58s %-8s" % ("mat", "核族", "指标", "单位") + "".join(" %18s" % x for x in arms))
    for (m, fam, metric) in sorted(agg):
        line = "%-12s %-22s %-58s %-8s" % (m[:12], fam[:22], metric[:58], units.get(metric, "")[:8])
        for x in arms:
            v = agg[(m, fam, metric)].get(x)
            line += " %18s" % (("%.4g(%d)" % (sum(v) / len(v), len(v))) if v else "-")
        say(line)
    outf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
