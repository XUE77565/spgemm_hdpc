#!/usr/bin/env python3
"""行级对拍合成阵生成器:每组 g 行共享 F 列核心(重复度=F),组内位置 p 的行 est≈g−p
   → 天然铺满 est 梯;每 bin 内行形状均匀,双边跑同一批行,per-row 时间 = bin kernel 时长/行数。
用法:gen_rowband_synth.py <g> <F> <P_private> <n_groups> <out.mtx>"""
import sys, random

g, F, P, ngrp, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
random.seed(42)
entries = []   # (组内逻辑行, cols);先攒后乱序行号,打散组员索引 → span 爆开,免 BCNT 位图路由
col_base = 0
for m in range(ngrp):
    core = [col_base + i for i in range(F)]          # 组内共享核心 → 两两 overlap = F
    col_base += F
    for p in range(g):
        priv = [col_base + i for i in range(P)]      # 私有列(垫 a_len,无 R 贡献)
        col_base += P
        entries.append(sorted(set(core + priv)))
perm = list(range(len(entries)))
random.shuffle(perm)                                  # 行号乱序:组员散布全空间
rows = [(perm[i] + 1, entries[i]) for i in range(len(entries))]
rows.sort()

ncol = col_base
nnz = sum(len(c) for _, c in rows)
with open(out, "w") as f:
    f.write(f"%%MatrixMarket matrix coordinate real general\n{len(rows)} {ncol} {nnz}\n")
    for i, cols in rows:
        for c in cols:
            f.write(f"{i} {c+1} {random.random()+0.5}\n")
print(f"g={g} F={F} P={P} groups={ngrp}: {len(rows)} rows x {ncol} cols, nnz={nnz}")
