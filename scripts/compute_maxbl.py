#!/usr/bin/env python3
"""D6 v4 特征补维:逐行 maxbl(行内最长 B 链长)CPU 补算。

语义与生产 reader 对齐:header symmetric 则镜像(同 k7_bench/matrix_utils);
maxbl[i] = max_{j∈row_i} (B_row_len[j])——管线 LLB(local_load_balance)同源量。
用法:.venv/bin/python scripts/compute_maxbl.py [mat ...](默认 24 标定阵)
输出:compare/d1_oracle/feat/<mat>.maxbl.tsv(两列:行号 maxbl)
"""
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "compare", "d1_oracle", "feat")
BIG = ["bloweybq", "pre2", "c-43", "Si10H16", "in-2004", "nd24k", "crankseg_2", "F2",
       "c-64", "pf2177", "HTC_336_9129", "333SP", "opt1", "af_shell4"]
F100 = ["bcsstk32", "bcsstk23", "bcsstm23", "bcspwr08", "bcspwr05", "bcsstm01",
        "1138_bus", "ash219", "bp_0", "arc130"]


def row_maxbl(path):
    with open(path, "r") as f:
        hdr = f.readline()
        sym = "symmetric" in hdr
        line = f.readline()
        while line and line[0] == "%":
            line = f.readline()
        M, K, NNZ = map(int, line.split())
        deg = [0] * (M + 1)
        entries = []
        for _ in range(NNZ):
            parts = f.readline().split()
            r, c = int(parts[0]) - 1, int(parts[1]) - 1
            entries.append((r, c))
            deg[r + 1] += 1
            if sym and c != r:
                deg[c + 1] += 1
    rp = [0] * (M + 1)
    for i in range(M):
        rp[i + 1] = rp[i] + deg[i + 1]
    blen = [rp[i + 1] - rp[i] for i in range(M)]          # B=A 链长(镜像后)
    cur = rp[:]
    rows_of = [[] for _ in range(M)]
    for r, c in entries:
        rows_of[r].append(c)
        if sym and c != r:
            rows_of[c].append(r)
    out = []
    for i in range(M):
        mb = 0
        for j in rows_of[i]:
            if blen[j] > mb:
                mb = blen[j]
        out.append(mb)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    mats = sys.argv[1:] or (BIG + F100)
    for m in mats:
        d = "data/first100" if m in F100 else "data/ocean/square"
        p = os.path.join(REPO, d, m + ".mtx")
        o = os.path.join(OUT, m + ".maxbl.tsv")
        if os.path.exists(o):
            print(f"{m}: 已有,跳过")
            continue
        if not os.path.exists(p):
            print(f"{m}: 矩阵缺失,跳过")
            continue
        mb = row_maxbl(p)
        with open(o, "w") as fo:
            for r, v in enumerate(mb):
                fo.write(f"{r} {v}\n")
        print(f"{m}: {len(mb)} 行 maxbl p50={sorted(mb)[len(mb)//2]} max={max(mb)}")


if __name__ == "__main__":
    main()
