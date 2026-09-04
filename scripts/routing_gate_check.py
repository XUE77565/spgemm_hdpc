#!/usr/bin/env python3
"""docs/76 §22 收口校准:Ocean 双维档位比较能否分离 in-2004(赢)与 c-64(输)。

Ocean 判据(Analysis.cuh L248-264):dense_bin(span) ≤ hash_bin(est) 才进 dense。
连续形式 ≈ span×8.125B ≤ ht×12B(同一 SMEM 预算梯 {128K/32,/16,/8,/4,/2} 上比较)。
数据源:DUMP_ROUTING=1 落盘的 est/span/flop/bkid 四数组(路由后快照)。
"""
import sys, struct
import numpy as np

SMEM_TOTAL = 128 * 1024
OCC = [32, 16, 8, 4, 2, 1]                    # dense_tier_occ(§22d 六档)
DENSE_MAX = [SMEM_TOTAL // o // 9 - 4 for o in OCC]     # {451,906,1816,3636,7277,14543}
HASH_SLOT_MAX = [SMEM_TOTAL // o // 12 for o in OCC]    # {341,682,1365,2730,5461,10922}
TIER_SPAN_MAX = DENSE_MAX[-1]

def ht_of(e, doubling_thr=4096, cap=8192):
    """本行 hash 表槽(2× 松弛 ≤4096;pow2 梯;cap 后 heavy)"""
    t = 2 * e if e <= doubling_thr else e
    ht = 32
    while ht < t and ht < cap:
        ht <<= 1
    return ht                                  # =cap 表示顶格(超 = heavy 全局表)

def dense_tier(span):
    for t, m in enumerate(DENSE_MAX):
        if span <= m: return t
    return len(DENSE_MAX)                      # 5 = 超档,不可 dense

def hash_tier(ht):
    for t, m in enumerate(HASH_SLOT_MAX):
        if ht <= m: return t
    return len(HASH_SLOT_MAX)                  # 5 = 表超预算梯(heavy 最贵档)

def load(path):
    with open(path, "rb") as f:
        n, bin_diter = struct.unpack("ii", f.read(8))
        arrs = [np.frombuffer(f.read(4 * n), dtype=np.int32) for _ in range(4)]
    return n, bin_diter, arrs  # est, span, flop, bkid

def pct(a, qs=(10, 25, 50, 75, 90)):
    if len(a) == 0: return "-"
    return " ".join(f"p{q}={np.percentile(a, q):.0f}" for q in qs)

for path in sys.argv[1:]:
    n, BIN_DITER, (est, span, flop, bkid) = load(path)
    cur = bkid == BIN_DITER
    print(f"\n===== {path} =====  rows={n}  cur_diter={cur.sum()}")
    e, s, f = est[cur].astype(np.int64), span[cur].astype(np.int64), flop[cur].astype(np.int64)
    if cur.sum() == 0:
        print("  无 diter 行"); continue
    dup = np.where(e > 0, f / np.maximum(e, 1), 0)
    print(f"  当前 diter 行: est {pct(e)}  span {pct(s)}  dup {pct(dup, (25,50,75))}")
    # Ocean 双维判据(对我们表梯:ht 含 2× 松弛;heavy = 最贵档)
    hts = np.array([ht_of(int(x)) for x in e])
    d_t = np.array([dense_tier(int(x)) for x in s])
    h_t = np.array([hash_tier(int(x)) for x in hts])
    ok = d_t <= h_t
    print(f"  Ocean 档位比较通过:{ok.sum()}/{cur.sum()} ({100*ok.mean():.1f}%)")
    print(f"    通过行: est {pct(e[ok])}  span {pct(s[ok])}  dup {pct(dup[ok], (25,50,75))}")
    if (~ok).any():
        bad = ~ok
        print(f"    拦下行(回 hash): est {pct(e[bad])}  span {pct(s[bad])}  dup {pct(dup[bad], (25,50,75))}")
        print(f"      拦下行 est/flop 占当前 diter 的 {100*e[bad].sum()/max(e.sum(),1):.0f}%/{100*f[bad].sum()/max(f.sum(),1):.0f}%")
    # 反向:hash 档行里 Ocean 会送 dense 的(潜在新增赢面)
    hs = (bkid != BIN_DITER) & (est > 0) & (span > 0)
    he, hs_, hf = est[hs].astype(np.int64), span[hs].astype(np.int64), flop[hs].astype(np.int64)
    hh = np.array([ht_of(int(x)) for x in he])
    d2 = np.array([dense_tier(int(x)) for x in hs_])
    h2 = np.array([hash_tier(int(x)) for x in hh])
    add = (d2 <= h2) & (hf >= 2048) & (he >= 1024)   # 加上现行 flop/est 保险门
    print(f"  hash 行翻 dense 候选(Ocean 门+现行保险门):{add.sum()} 行"
          f"  est {pct(he[add]) if add.any() else ''}")
    # 档位分布(数值侧 per-tier 拆分的依据)
    print(f"  dense 档位分布(当前行): {[(t, int((d_t==t).sum())) for t in range(6) if (d_t==t).any()]}")
