#!/usr/bin/env python3
# tick4 枚举形态取证①(纯磁盘):G-group 步进的负载失衡税静态模型
# 模型(hash_spa_kernel 主环):行=块;组 g ∈ [0,ng) 认领 k=rs+g 步 ng;组功 W_g=Σ_k ceil(len_k/G)
#   行税 = max_g W_g / (Σ_k ceil(len_k/G) / ng);天花板税 = Σceil/Σ(len/G)(组内 lane 圆整)
# 带近似:products∈(32,256]≈bin2-3(G=8),(256,2048]≈bin4-5(G=16),>2048≈bin6(G=32)
# 判据方向:税 与 reanchor beat_ratio(我/ocean)的相关性——b23 应显著高于 b45。
import sys, numpy as np
from scipy.io import mmread
import scipy.sparse as sp

def tax_for_row(blen, ks, G, ng):
    # ks: 该行 k 的 B 行长数组
    if len(ks) == 0: return None
    w = np.ceil(ks / G).astype(np.int64)          # 每 k 的组内迭代数(lane 并行后)
    total = w.sum()
    if total == 0: return None
    # 组功:按 residue 分桶(ng 组)
    gsum = np.zeros(ng, dtype=np.int64)
    np.add.at(gsum, np.arange(len(ks)) % ng, w)
    return gsum.max() / (total / ng), w.sum() / (ks.sum() / G) if ks.sum() else 1.0

def analyze(mtx):
    M = mmread(mtx).tocsr()
    n = M.shape[0]
    indptr, indices = M.indptr, M.indices
    blen = np.diff(indptr)                          # B=A 方阵:行 k 长
    a_len = blen.copy()
    rows = np.arange(n)
    # 每行 products = Σ_k blen[k]:向量化——repeat + reduceat
    counts = np.diff(indptr)
    if counts.sum() == 0: return None
    prod_per_k = blen[indices]                      # 每个 A 非零位的 B 行长
    csum = np.concatenate([[0], np.cumsum(prod_per_k)])
    row_prod = csum[indptr[1:]] - csum[indptr[:-1]]
    out = {}
    for band, lo, hi, G, blk in [("b23", 32, 256, 8, 64), ("b45", 256, 2048, 16, 256), ("b6", 2048, 1<<20, 32, 512)]:
        sel = np.where((row_prod > lo) & (row_prod <= hi) & (counts > 0))[0]
        if len(sel) < 1000: continue
        ng = blk // G
        taxes, ceilings = [], []
        wprod = row_prod[sel]
        # 大人口抽样上限 20 万行(速度)
        step = max(1, len(sel) // 200000)
        for i in sel[::step]:
            rs, re = indptr[i], indptr[i+1]
            r = tax_for_row(blen, prod_per_k[rs:re].astype(float), G, ng)
            if r:
                taxes.append(r[0]); ceilings.append(r[1])
        taxes = np.array(taxes); ceilings = np.array(ceilings)
        # 人口加权(按行 products)
        out[band] = dict(n=len(sel), tax_mean=float(taxes.mean()), tax_p90=float(np.percentile(taxes, 90)),
                         ceil_mean=float(ceilings.mean()),
                         prod_sum=int(row_prod[sel].sum()))
    return out

if __name__ == "__main__":
    mats = sys.argv[1:]
    beats = {"CurlCurl_4": 2.26, "rgg_n_2_22_s0": 1.88, "StocF-1465": 1.29, "af_shell10": 1.74,
             "af_1_k101": 1.7, "cage15": 1.24, "nlpkkt120": 1.52, "Bump_2911": 1.30,
             "Geo_1438": 1.33, "Serena": 1.27, "Hook_1498": 1.29, "nlpkkt80": 1.73}
    print(f"{'阵':18} {'带':4} {'行数':>9} {'prodΣ(M)':>9} {'组失衡税mean':>12} {'p90':>6} {'ceil税':>7} beat")
    for m in mats:
        r = analyze(f"data/ocean/square/{m}.mtx")
        if not r: continue
        b = beats.get(m, "?")
        for band, d in r.items():
            print(f"{m:18} {band:4} {d['n']:>9} {d['prod_sum']/1e6:>9.1f} {d['tax_mean']:>12.3f} {d['tax_p90']:>6.3f} {d['ceil_mean']:>7.3f} {b}")