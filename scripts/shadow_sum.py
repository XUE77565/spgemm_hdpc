#!/usr/bin/env python3
"""全集影子 / 相位地图的判读(读 shadow428.py 的 PREFIX.csv 与 PREFIX_phases.csv)。

预先写定的判据(dialog/qa-009;电池口径在全集上的对应):
  测量分级:基线臂中位 co <1ms = 微级(不作单阵裁决)/ 1-10ms = 中 / ≥10ms = 稳。
  A/A 干净 = |GM(aa/off) − 1| ≤ 1%(全体与非微级各算一次,都要过)
             且 非微级阵里 aa/off 偏离超过 ±5% 的不超过 2%(这些阵逐个列出,记为"不裁决阵")。
  某臂绿   = A/A 干净 且 GM(臂/off) ≤ 1.02 且 没有"可裁决的"非微级阵 > 1.10 且 nnz 各臂逐阵相同
             (可裁决 = 不在"不裁决阵"里)。判据不过 → 整段作废,不挑段。
  签名     = 各臂每发的签名(门普查结果行的 md5)逐阵全部相同 → 行为保持在全集上成立。
相位地图(基线臂;各阵各相位取各轮中位):池 = accumulate / compact+sort / dense_direct / mh_merge+dense_count,
  池总量 = 各阵该池 ms 求和;"可回收" = 我方自己的池时间;列各池前 20 阵。
输出:<前缀>_sum.txt(全文)、<前缀>_permat.csv(逐阵:各臂中位 co、比值、分级、nnz/签名是否一致)、
      <前缀>_map.csv(相位地图长表:mat,arm,phase,ms 中位)。
用法:shadow_sum.py PREFIX [--base off] [--aa aa]
"""
import argparse, csv, math, statistics, sys

POOLS = [("accumulate", ("accumulate",)), ("compact+sort", ("compact+sort",)),
         ("dense_direct", ("dense_direct",)), ("mh_merge+dense_count", ("mh_merge", "dense_count"))]


def gm(v):
    v = [x for x in v if x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--base", default="off")
    ap.add_argument("--aa", default="aa")
    a = ap.parse_args()
    outf = open(a.prefix + "_sum.txt", "w", encoding="utf-8")

    def say(s=""):
        print(s)
        outf.write(s + "\n")

    co, nnz, sig, tier, order, arms, nfail = {}, {}, {}, {}, [], [], 0
    try:
        rd = csv.DictReader(open(a.prefix + ".csv", encoding="utf-8"))
        for r in rd:
            m, arm = r["mat"], r["arm"]
            if m not in order:
                order.append(m)
            if arm not in arms:
                arms.append(arm)
            tier[m] = r.get("tier", "")
            try:
                c = float(r["co"])
            except ValueError:
                c = -1.0
            if c <= 0:
                nfail += 1
                continue
            co.setdefault((m, arm), []).append(c)
            nnz.setdefault(m, set()).add(r["nnz"])
            sig.setdefault(m, set()).add(r.get("sig", ""))
    except OSError as e:
        say("# 读不到 %s.csv: %s" % (a.prefix, e))
        outf.close()
        return 0
    if a.base in arms:
        arms.remove(a.base)
        arms.insert(0, a.base)
    base = arms[0] if arms else a.base
    mats = [m for m in order if co.get((m, base))]
    say("# shadow_sum: 清单内 %d 阵,基线臂有数据的 %d 阵;失败发 %d;臂: %s" % (len(order), len(mats), nfail, " ".join(arms)))
    if not mats:
        outf.close()
        return 0
    bmed = {m: statistics.median(co[(m, base)]) for m in mats}
    micro = {m: bmed[m] < 1.0 for m in mats}
    cls = {m: ("微" if bmed[m] < 1.0 else ("中" if bmed[m] < 10.0 else "稳")) for m in mats}
    ratio = {}
    for arm in arms[1:]:
        for m in mats:
            v = co.get((m, arm))
            if v:
                ratio[(m, arm)] = statistics.median(v) / bmed[m]
    nnz_bad = [m for m in mats if len(nnz.get(m, ())) > 1]
    sig_set = [m for m in mats if any(s for s in sig.get(m, ()))]
    sig_bad = [m for m in sig_set if len(sig[m]) > 1]
    say("# 分级:微 %d / 中 %d / 稳 %d;nnz 各臂各轮不全同的阵 %d 个%s;签名不全同的阵 %d 个(有签名的 %d 阵)%s"
        % (sum(1 for m in mats if cls[m] == "微"), sum(1 for m in mats if cls[m] == "中"), sum(1 for m in mats if cls[m] == "稳"),
           len(nnz_bad), (":" + " ".join(nnz_bad[:20])) if nnz_bad else "", len(sig_bad), len(sig_set),
           (":" + " ".join(sig_bad[:20])) if sig_bad else ""))

    # ── A/A ──
    undecid = set()
    aa_ok = None
    if a.aa in arms:
        ra = [(m, ratio[(m, a.aa)]) for m in mats if (m, a.aa) in ratio]
        nm = [(m, r) for m, r in ra if not micro[m]]
        g_all, g_nm = gm([r for _, r in ra]), gm([r for _, r in nm])
        out5 = [(m, r) for m, r in nm if abs(r - 1.0) > 0.05]
        undecid = set(m for m, _ in out5)
        frac = len(out5) / len(nm) if nm else 0.0
        aa_ok = abs(g_all - 1.0) <= 0.01 and abs(g_nm - 1.0) <= 0.01 and frac <= 0.02
        say("")
        say("===== A/A 对照(%s / %s)=====" % (a.aa, base))
        say("AA_N %d(非微级 %d)" % (len(ra), len(nm)))
        say("AA_GM_ALL %.4f" % g_all)
        say("AA_GM_NONMICRO %.4f" % g_nm)
        say("AA_NONMICRO_OUT5 %d(占非微级 %.1f%%;上限 2%%)" % (len(out5), 100 * frac))
        say("AA_VERDICT %s" % ("clean" if aa_ok else "polluted"))
        if out5:
            say("不裁决阵(A/A 偏离超 ±5% 的非微级阵): " + " ".join("%s(%.3f)" % (m, r) for m, r in sorted(out5, key=lambda t: -abs(t[1] - 1))[:40]))
    else:
        say("# 没有 A/A 臂(%s),无法认证" % a.aa)

    # ── 各臂 ──
    for arm in arms[1:]:
        if arm == a.aa:
            continue
        rr = [(m, ratio[(m, arm)]) for m in mats if (m, arm) in ratio]
        if not rr:
            continue
        nm = [(m, r) for m, r in rr if not micro[m]]
        say("")
        say("===== 判读:%s / %s =====" % (arm, base))
        say("GM(%s/%s) = %.4f(%d 阵)| 非微级 GM = %.4f(%d 阵)| 微级 GM = %.4f | 各档(小/中/大文件)GM = %s"
            % (arm, base, gm([r for _, r in rr]), len(rr), gm([r for _, r in nm]), len(nm),
               gm([r for m, r in rr if micro[m]]),
               " / ".join("%.4f" % gm([r for m, r in rr if tier.get(m) == t]) for t in ("0", "1", "2"))))
        over = sorted([(m, r) for m, r in nm if r > 1.10], key=lambda t: -t[1])
        over_dec = [(m, r) for m, r in over if m not in undecid]
        say("非微级阵 >1.10:%d 个(其中可裁决的 %d 个)%s" % (len(over), len(over_dec),
            (":" + " ".join("%s(%.3f%s)" % (m, r, "" if m not in undecid else ",不裁决") for m, r in over[:20])) if over else ""))
        say("非微级阵 <0.95:%d 个;>1.02:%d 个" % (sum(1 for _, r in nm if r < 0.95), sum(1 for _, r in nm if r > 1.02)))
        srt = sorted(nm, key=lambda t: t[1])
        say("非微级最快 12: " + " ".join("%s(%.3f)" % t for t in srt[:12]))
        say("非微级最慢 12: " + " ".join("%s(%.3f)" % t for t in srt[-12:][::-1]))
        green = (aa_ok is True) and gm([r for _, r in rr]) <= 1.02 and not over_dec and not nnz_bad
        say("裁定: %s" % ("绿" if green else ("红" if aa_ok else "A/A 未过 → 整段作废(上面的数字只留档)")))

    with open(a.prefix + "_permat.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mat", "tier", "class", base + "_ms"] + [x + "/" + base for x in arms[1:]] + ["nnz_same", "sig_same"])
        for m in mats:
            w.writerow([m, tier.get(m, ""), cls[m], "%.4f" % bmed[m]] + [("%.4f" % ratio[(m, x)]) if (m, x) in ratio else "" for x in arms[1:]]
                       + [int(m not in nnz_bad), int(m not in sig_bad)])

    # ── 相位地图 ──
    ph = {}
    try:
        for r in csv.DictReader(open(a.prefix + "_phases.csv", encoding="utf-8")):
            try:
                ph.setdefault((r["mat"], r["arm"], r["phase"]), []).append(float(r["ms"]))
            except ValueError:
                pass
    except OSError:
        ph = {}
    if ph:
        pm = {k: statistics.median(v) for k, v in ph.items()}
        with open(a.prefix + "_map.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mat", "arm", "phase", "ms"])
            for (m, arm, p), v in sorted(pm.items()):
                w.writerow([m, arm, p, "%.4f" % v])
        phases = sorted({k[2] for k in pm})
        say("")
        say("===== 相位地图(各阵各相位取各轮中位,再跨阵求和;单位 ms)=====")
        say("%-18s" % "相位" + "".join(" %12s" % x for x in arms))
        for p in phases:
            say("%-18s" % p[:18] + "".join(" %12.2f" % sum(v for (m, arm, pp), v in pm.items() if arm == x and pp == p) for x in arms))
        say("")
        say("----- 池(臂 %s):池总量与前 20 阵 -----" % base)
        top20_sum = 0.0
        for pname, pls in POOLS:
            per = {}
            for (m, arm, p), v in pm.items():
                if arm == base and p in pls:
                    per[m] = per.get(m, 0.0) + v
            tot = sum(per.values())
            top = sorted(per.items(), key=lambda t: -t[1])[:20]
            top20_sum += sum(v for _, v in top)
            say("池 %-22s 总量 %10.2f ms;前 20 阵合计 %10.2f ms" % (pname, tot, sum(v for _, v in top)))
            say("    " + " ".join("%s(%.1f)" % (m, v) for m, v in top))
            for x in arms[1:]:
                if x == a.aa:
                    continue
                tx = sum(v for (m, arm, p), v in pm.items() if arm == x and p in pls)
                say("    臂 %-6s 该池总量 %10.2f ms(对 %s %+.2f ms)" % (x, tx, base, tx - tot))
        say("各池前 20 阵合计的总和(可回收 top20 口径): %.2f ms" % top20_sum)
    else:
        say("")
        say("# 没有相位数据(%s_phases.csv 为空或读不到)" % a.prefix)
    outf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
