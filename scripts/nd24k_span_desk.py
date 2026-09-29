#!/usr/bin/env python3
"""nd24k 重行 span 特征 desk 检查(09-29 队首任务,零 GPU)。

目的:为"nd24k 重行 dense 位图同行对拍"提供行级几何事实。
reader 语义对齐:src/matrix_utils.cu 的 read_matrix_market —— symmetric 镜像展开成全矩阵。
输出:全阵行分布 + 重行(flop>32K)的 est/span/dup 分布与位图经济学量。
"""
import sys
import time
import numpy as np

MTX = "/home/xueyizhou/spgemm-challenge/data/ocean/square/nd24k.mtx"
HEAVY_FLOP = 32768          # §3.1e:nd24k 重行全部 flop>32K(中位 107K)
SMEM_BUDGET = 228 * 1024    # H100 每块 SMEM 上限(字节)

t0 = time.time()

# ---- 1. 解析 mtx(与 reader 同语义:symmetric 镜像展开) ----
with open(MTX) as f:
    banner = f.readline().strip()
    line = f.readline()
    while line[0] == '%':
        line = f.readline()
    M, N, nz_stored = map(int, line.split())
    is_sym = "symmetric" in banner.lower() or "hermitian" in banner.lower()
    rc = np.loadtxt(f, usecols=(0, 1), dtype=np.int64)

r = rc[:, 0] - 1
c = rc[:, 1] - 1
offd = r != c
src = np.concatenate([r, c[offd]])  # (r,c) 原条目 + 镜像 (c,r)
dst = np.concatenate([c, r[offd]])
print(f"[parse] banner={banner} M={M} N={N} stored_nz={nz_stored} "
      f"expanded_nz={len(src)} sym={is_sym} ({time.time()-t0:.1f}s)")

# ---- 2. CSR 构建 ----
cnt = np.bincount(src, minlength=M).astype(np.int64)
indptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
order = np.lexsort((dst, src))
indices = dst[order].astype(np.int64)
row_nnz = cnt

# ---- 3. 全阵行级 flop(向量化) ----
flat = row_nnz[indices]
csum = np.concatenate([[0], np.cumsum(flat)])
flop = csum[indptr[1:]] - csum[indptr[:-1]]
print(f"[csr] rows={M} nnz_full={row_nnz.sum()} "
      f"a_len med={np.median(row_nnz):.0f} q25={np.percentile(row_nnz,25):.0f} "
      f"q75={np.percentile(row_nnz,75):.0f} max={row_nnz.max()}")
print(f"[flop] all-row med={np.median(flop):.0f} q75={np.percentile(flop,75):.0f} "
      f"q95={np.percentile(flop,95):.0f} max={flop.max():.0f} "
      f"heavy(flop>{HEAVY_FLOP})={int((flop>HEAVY_FLOP).sum())} ({time.time()-t0:.1f}s)")

# ---- 4. 重行逐行:est(精确 distinct)/span/dup ----
heavy_idx = np.flatnonzero(flop > HEAVY_FLOP)
ests, spans, dups, alens, flops, nwin7277, nwin28k = [], [], [], [], [], [], []
W_OCEAN_MAX = 7277          # 彼 dense 档最大窗宽(DENSE_NUMERIC_BIN_SIZES 尾档)
W_VAL8 = SMEM_BUDGET // 8   # 我方窗宽上限(val8/列;先作上限估)
for i in heavy_idx:
    ks = indices[indptr[i]:indptr[i + 1]]
    segs = [indices[indptr[k]:indptr[k + 1]] for k in ks]
    cols = np.concatenate(segs)
    est = np.unique(cols).size
    span = int(cols.max() - cols.min()) + 1
    ests.append(est); spans.append(span)
    dups.append(flop[i] / est); alens.append(row_nnz[i]); flops.append(flop[i])
    nwin7277.append(-(-span // W_OCEAN_MAX))
    nwin28k.append(-(-span // W_VAL8))

ests = np.array(ests); spans = np.array(spans); dups = np.array(dups)
alens = np.array(alens); flops = np.array(flops)
nwin7277 = np.array(nwin7277); nwin28k = np.array(nwin28k)

def q(name, a, fmt="{:.0f}"):
    print(f"    {name:14s} med={fmt.format(np.median(a))} "
          f"q25={fmt.format(np.percentile(a,25))} q75={fmt.format(np.percentile(a,75))} "
          f"min={fmt.format(a.min())} max={fmt.format(a.max())}")

print(f"[heavy n={len(ests)}] ({time.time()-t0:.1f}s)")
q("a_len", alens)
q("flop", flops)
q("est(distinct)", ests)
q("span", spans)
q("dup=flop/est", dups, "{:.1f}")
q("span/n", spans / N, "{:.3f}")
q("est/span", ests / spans, "{:.3f}")
q("bitmap_bytes", spans / 8)
q("windows@7277", nwin7277)
q("windows@val8", nwin28k)

# ---- 5. 位图经济学(重行中位口径) ----
med_span = np.median(spans); med_est = np.median(ests); med_flop = np.median(flops)
print(f"[econ] med: span={med_span:.0f} est={med_est:.0f} flop={med_flop:.0f}")
print(f"    bitmap(row)=span/8={med_span/8:.0f}B vs hash槽=est*12B={med_est*12/1024:.1f}KB")
print(f"    in-span密度=est/span={med_est/med_span:.3f}; 全幅密度=est/n={med_est/N:.3f}")
print(f"    单窗塞下占比: span<=7277 {np.mean(spans<=W_OCEAN_MAX)*100:.1f}% | "
      f"span<={W_VAL8} {np.mean(spans<=W_VAL8)*100:.1f}% | span<=SMEM/17B"
      f"({SMEM_BUDGET//17}) {np.mean(spans<=SMEM_BUDGET//17)*100:.1f}%")
print(f"[done] {time.time()-t0:.1f}s")
