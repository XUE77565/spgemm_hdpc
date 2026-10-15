#!/usr/bin/env python3
"""L0 特征抽取 v1(study/46 A/E 组,10-16):428 权威阵 → compare/batch437/features.csv

特征(全部"选择前"可用=部署时廉价预扫可得):
  n, nnz, avg_row, max_row, skew(max/avg), row_len 熵, 基尼(排序公式 O(nlogn)),
  p90/p50, flop_proxy = nnz²/n
awk 流式逐文件数行 nnz(C 速度);python 汇总统计。断点续跑。
"""
import csv, os, subprocess, math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

o337 = [r["mat"] for r in csv.DictReader(open(os.path.join(REPO, "compare/l0_337/l0_337_results.csv")))]
seen = set(o337)
july = [r["matrix"] for r in csv.DictReader(open(os.path.join(REPO, "compare/methods_cmp_first100.csv")))]
mats = [(m, os.path.join(REPO, "data/ocean/square", m + ".mtx")) for m in o337]
mats += [(m, os.path.join(REPO, "data/first100", m + ".mtx")) for m in july
         if m not in seen and os.path.exists(os.path.join(REPO, "data/first100", m + ".mtx"))]

out_path = os.path.join(REPO, "compare/batch437/features.csv")
done = set()
if os.path.exists(out_path):
    done = {r["mat"] for r in csv.DictReader(open(out_path))}

f = open(out_path, "a", newline="")
w = csv.writer(f)
if not done:
    w.writerow(["mat","n","nnz","avg_row","max_row","skew","row_entropy","row_gini","p90p50","flop_proxy"])

# awk:跳过 % 注释,第一条非注释=尺寸头,其余按行号(第1列)计数;END 吐每行 nnz
AWK_CNT = "/^%/{next} !h{h=1;next} {c[$1]++} END{for(i in c) print c[i]}"
AWK_HDR = "/^%/{next} NF==3{print $1,$2,$3; exit}"

for m, path in mats:
    if m in done:
        continue
    try:
        hdr = subprocess.run(["awk", AWK_HDR, path], capture_output=True, text=True, timeout=120).stdout.split()
        if len(hdr) < 3:
            print("ERR", m, "no header", flush=True); continue
        n, _, nnz = int(hdr[0]), int(hdr[1]), int(hdr[2])
        r = subprocess.run(["awk", AWK_CNT, path], capture_output=True, text=True, timeout=900)
        lens = [int(x) for x in r.stdout.split()]
        if not lens:
            print("ERR", m, "no rows", flush=True); continue
        nr = len(lens)
        avg = nnz / nr
        mx = max(lens)
        sl = sorted(lens)
        p50 = sl[nr // 2]; p90 = sl[int(nr * 0.9)]
        from collections import Counter
        cnt = Counter(lens)
        ent = -sum((c / nr) * math.log(c / nr) for c in cnt.values())
        # 基尼(排序公式):G = 2*Σ(i*x_i)/(n*Σx) - (n+1)/n
        cum = sum((i + 1) * x for i, x in enumerate(sl))
        gini = 2 * cum / (nr * nnz) - (nr + 1) / nr
        w.writerow([m, n, nnz, round(avg, 3), mx, round(mx / avg, 3), round(ent, 3),
                    round(gini, 4), round(p90 / max(p50, 1), 3), round(nnz * nnz / n, 1)])
        f.flush()
        print("done", m, flush=True)
    except Exception as e:
        print("ERR", m, repr(e), flush=True)
print("FEATURES_DONE")
