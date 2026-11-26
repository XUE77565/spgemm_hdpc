#!/usr/bin/env python3
"""发射清单比对:两条臂在每个阵上的 kernel 发射清单(核名含模板实参 + 发射次数)是否逐项相同。

输入与 kern_sum.py 相同:--dir 下的 <阵>__<臂>__<轮>.csv(nsys cuda_gpu_kern_sum 的原样输出;Instances 列 = 发射次数)。
用途(dialog/qa-008 的验证注):基线源码件(臂 base)对本件开关全关(臂 off)——清单逐项相同 = 默认关闭时零侵入的直接证据;
也可比任意两条臂,看差异是否恰好是设计的那几项。
输出:LAUNCH_DIFF_MATS n(有差异的阵数)、LAUNCH_DIFF_VERDICT same|differ,随后逐阵列出差异项(核名:A 次数 → B 次数)。
用法:launch_diff.py --dir DIR --a base --b off [--rep 1]
"""
import argparse, glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kern_sum import read_one   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--rep", default="1")
    x = ap.parse_args()
    lists = {}
    for f in sorted(glob.glob(os.path.join(x.dir, "*__*__%s.csv" % x.rep))):
        parts = os.path.basename(f)[:-4].split("__")
        if len(parts) != 3 or parts[1] not in (x.a, x.b):
            continue
        rows, _ = read_one(f)
        d = {}
        for name, ins, ns in rows:
            d[name] = d.get(name, 0) + ins
        lists[(parts[0], parts[1])] = d
    mats = sorted({m for m, arm in lists if (m, x.a) in lists and (m, x.b) in lists})
    diffs = []
    for m in mats:
        da, db = lists[(m, x.a)], lists[(m, x.b)]
        items = [(n, da.get(n, 0), db.get(n, 0)) for n in sorted(set(da) | set(db)) if da.get(n, 0) != db.get(n, 0)]
        if items:
            diffs.append((m, items))
    print("LAUNCH_DIFF_N %d" % len(mats))
    print("LAUNCH_DIFF_MATS %d" % len(diffs))
    print("LAUNCH_DIFF_VERDICT %s" % ("nodata" if not mats else ("same" if not diffs else "differ")))
    tot = sum(sum(lists[(m, x.a)].values()) for m in mats)
    print("# 臂 %s 对臂 %s:比对 %d 阵;臂 %s 的发射总次数 %d(含预热轮)" % (x.a, x.b, len(mats), x.a, tot))
    for m, items in diffs:
        print("== %s" % m)
        for n, ca, cb in items:
            print("   %6d → %-6d %s" % (ca, cb, n[:150]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
