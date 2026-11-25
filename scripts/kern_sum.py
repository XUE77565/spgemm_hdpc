#!/usr/bin/env python3
"""nsys 逐 kernel GPU 侧时长的汇总(设备侧时间戳,不吃主机 CPU 争用)。

输入:--dir 下的 <阵>__<臂>__<轮>.csv,每个是 `nsys stats --report cuda_gpu_kern_sum --format csv`
      (老版本报告名 gpukernsum)的原样输出;文件里表头之前可以夹别的行。
输出:
  <前缀>.txt          ① 逐阵:各臂 Σkernel 时长(各轮取中位)与对基线臂之比,另给两遍式相关核族
                        (hash_heavy_kernel + hash_bm2_kernel + hh_bm2_part_kernel)的同一比值;各臂 GM
                      ② 基线臂上逐核族的累计时长排行(P3:谁在吃时间)
                      ③ 基线臂上逐实例排行(前 N)与"被发射过的实例"清单(瘦身对账用)
  <前缀>_long.csv.gz  长表:mat,arm,rep,family,instances,total_ns,name
说明:一次进程 = WARMUP_N 次预热 + 1 次计时轮,这里的累计时长含预热轮;各臂 WARMUP_N 相同,比值不受影响。
      多流并行的核,时长按各自墙钟相加(会重复计入重叠区间),只用于臂间比与排行,不等于相位墙钟。
用法:kern_sum.py --dir DIR --out PREFIX [--base off] [--top 40]
"""
import argparse, csv, glob, gzip, io, math, os, re, statistics, sys

HH_FAM = ("hash_heavy_kernel", "hash_bm2_kernel", "hh_bm2_part_kernel")


def family(name):
    s = name.strip()
    s = re.sub(r"^(static\s+)?void\s+", "", s)
    m = re.match(r"[A-Za-z_][A-Za-z0-9_:]*", s)
    return m.group(0) if m else s[:40]


def read_one(path):
    """→ (rows=[(name, instances, total_ns)], 表头原文) ;读不出表返回 ([], None)"""
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return [], None
    hi = -1
    for i, ln in enumerate(lines):
        low = ln.lower()
        if "total time" in low and "name" in low:
            hi = i
            break
    if hi < 0:
        return [], None
    rd = list(csv.reader(io.StringIO("\n".join(lines[hi:]))))
    if not rd:
        return [], None
    hdr = [h.strip() for h in rd[0]]
    lowh = [h.lower() for h in hdr]

    def col(*keys):
        for k in keys:
            for j, h in enumerate(lowh):
                if h.startswith(k):
                    return j
        for k in keys:
            for j, h in enumerate(lowh):
                if k in h:
                    return j
        return -1

    c_tot, c_ins, c_name = col("total time"), col("instances", "count"), col("name")
    if c_tot < 0 or c_name < 0:
        return [], None
    scale = 1.0
    ht = lowh[c_tot]
    if "(us)" in ht or "(µs)" in ht:
        scale = 1e3
    elif "(ms)" in ht:
        scale = 1e6
    elif "(s)" in ht:
        scale = 1e9
    out = []
    for r in rd[1:]:
        if len(r) <= max(c_tot, c_name):
            continue
        try:
            tot = float(r[c_tot].replace(",", "")) * scale
        except ValueError:
            continue
        try:
            ins = int(float(r[c_ins].replace(",", ""))) if c_ins >= 0 else 0
        except ValueError:
            ins = 0
        out.append((r[c_name].strip(), ins, tot))
    return out, ",".join(hdr)


def med(v):
    return statistics.median(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="off")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    outf = open(a.out + ".txt", "w", encoding="utf-8")

    def say(s=""):
        print(s)
        outf.write(s + "\n")

    files = sorted(glob.glob(os.path.join(a.dir, "*__*__*.csv")))
    data = {}          # (mat, arm, rep) → [(name, ins, ns)]
    hdr_seen = None
    bad = []
    for f in files:
        b = os.path.basename(f)[:-4]
        parts = b.split("__")
        if len(parts) != 3:
            continue
        rows, hdr = read_one(f)
        if not rows:
            bad.append(b)
            continue
        hdr_seen = hdr_seen or hdr
        data[(parts[0], parts[1], parts[2])] = rows
    say("# kern_sum: 读到 %d 个文件,有表的 %d 个,读不出表的 %d 个" % (len(files), len(data), len(bad)))
    if hdr_seen:
        say("# nsys 表头: " + hdr_seen)
    if bad:
        say("# 读不出表: " + " ".join(bad[:30]) + (" …" if len(bad) > 30 else ""))
    if not data:
        say("# 没有可用数据")
        outf.close()
        return 0

    with gzip.open(a.out + "_long.csv.gz", "wt", encoding="utf-8", newline="") as g:
        w = csv.writer(g)
        w.writerow(["mat", "arm", "rep", "family", "instances", "total_ns", "name"])
        for (m, arm, rep), rows in sorted(data.items()):
            for name, ins, ns in rows:
                w.writerow([m, arm, rep, family(name), ins, "%.0f" % ns, name])

    mats, arms = [], []
    for (m, arm, rep) in sorted(data):
        if m not in mats:
            mats.append(m)
        if arm not in arms:
            arms.append(arm)
    if a.base in arms:
        arms.remove(a.base)
        arms.insert(0, a.base)
    base = arms[0]

    def tot(m, arm, fam=None):
        v = []
        for (mm, aa, rep), rows in data.items():
            if mm == m and aa == arm:
                v.append(sum(ns for name, ins, ns in rows if fam is None or family(name) in fam))
        return v

    say("")
    say("===== ① 逐阵:Σkernel GPU 时长(ms,各轮中位;含预热轮)与对 %s 之比;括号里是两遍式相关核族(%s)的比 =====" % (base, " + ".join(HH_FAM)))
    say("%-18s %10s %6s" % ("mat", base + "_ms", "轮间%") + "".join("  %22s" % (x + "/" + base + "(核族比)") for x in arms[1:]))
    logs = {x: [] for x in arms[1:]}
    logs_cov = {x: [] for x in arms[1:]}
    for m in mats:
        b = tot(m, base)
        if not b:
            say("%-18s  基线臂无数据" % m)
            continue
        bm = med(b)
        spread = 100.0 * (max(b) - min(b)) / bm if bm > 0 and len(b) > 1 else float("nan")
        bh = med(tot(m, base, HH_FAM))
        line = "%-18s %10.3f %6.1f" % (m, bm / 1e6, spread)
        for x in arms[1:]:
            v = tot(m, x)
            if not v or bm <= 0:
                line += "  %22s" % "-"
                continue
            r = med(v) / bm
            if not r > 0:
                line += "  %22s" % "-"
                continue
            xh = med(tot(m, x, HH_FAM))
            rh = (xh / bh) if bh > 0 else float("nan")
            took = any(family(name) == "hash_bm2_kernel" for (mm, aa, rep), rows in data.items() if mm == m and aa == x for name, ins, ns in rows)
            line += "  %22s" % ("%.4f (%.4f)%s" % (r, rh, "*" if took else " "))
            logs[x].append(math.log(r))
            if took:
                logs_cov[x].append(math.log(r))
        say(line)
    say("(* = 该臂在这阵上确有 hash_bm2_kernel 发射)")
    for x in arms[1:]:
        if logs[x]:
            say("GM_KERN(%s/%s) = %.4f | 阵数 %d | 最差 %.4f | 有两遍式发射的阵 %d 个,其 GM = %s" % (
                x, base, math.exp(sum(logs[x]) / len(logs[x])), len(logs[x]), math.exp(max(logs[x])),
                len(logs_cov[x]), ("%.4f" % math.exp(sum(logs_cov[x]) / len(logs_cov[x]))) if logs_cov[x] else "-"))

    # ② 核族排行(基线臂;各阵各轮取均值后跨阵相加)
    fam_ns, inst_ns, inst_mats, inst_cnt = {}, {}, {}, {}
    for m in mats:
        reps = [rows for (mm, aa, rep), rows in data.items() if mm == m and aa == base]
        if not reps:
            continue
        acc = {}
        for rows in reps:
            for name, ins, ns in rows:
                t = acc.setdefault(name, [0.0, 0])
                t[0] += ns / len(reps)
                t[1] += ins
        for name, (ns, ins) in acc.items():
            fam_ns[family(name)] = fam_ns.get(family(name), 0.0) + ns
            inst_ns[name] = inst_ns.get(name, 0.0) + ns
            inst_mats.setdefault(name, set()).add(m)
            inst_cnt[name] = inst_cnt.get(name, 0) + ins
    allns = sum(fam_ns.values())
    say("")
    say("===== ② 核族累计 GPU 时长排行(臂 %s,%d 阵相加;含预热轮)=====" % (base, len(mats)))
    say("%-34s %12s %7s %8s" % ("核族", "ms", "占比%", "实例数"))
    for fam, ns in sorted(fam_ns.items(), key=lambda kv: -kv[1]):
        ninst = sum(1 for n in inst_ns if family(n) == fam)
        say("%-34s %12.3f %7.2f %8d" % (fam[:34], ns / 1e6, 100.0 * ns / allns if allns > 0 else 0.0, ninst))

    say("")
    say("===== ③ 逐实例排行(臂 %s,前 %d;被发射过的实例共 %d 个)=====" % (base, a.top, len(inst_ns)))
    say("%12s %7s %5s %8s  %s" % ("ms", "占比%", "阵数", "发射次数", "实例"))
    for name, ns in sorted(inst_ns.items(), key=lambda kv: -kv[1])[:a.top]:
        say("%12.3f %7.2f %5d %8d  %s" % (ns / 1e6, 100.0 * ns / allns if allns > 0 else 0.0, len(inst_mats[name]), inst_cnt[name], name[:150]))
    # 其它臂多出来的实例(两遍式核等)
    extra = {}
    for (m, arm, rep), rows in data.items():
        if arm == base:
            continue
        for name, ins, ns in rows:
            if name not in inst_ns:
                extra.setdefault(family(name), set()).add(name)
    if extra:
        say("")
        say("# 仅在实验臂出现的实例数(按核族): " + ", ".join("%s=%d" % (k, len(v)) for k, v in sorted(extra.items())))
    outf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
