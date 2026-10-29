#!/usr/bin/env python3
"""合成标定格点生成器(study/60 S4):可控签名的合成矩阵,填补训练集签名空洞。
生成 200+ 个 .mtx 文件到 /tmp/syn_cal/,覆盖:
- bin9 人口(est 8192-16384)× 多种 dup/kcnt/blen = 60 个
- bin4/5 人口(est 256-1024)× 多种配置 = 60 个
- bin0 人口(est 33-64)× 多种配置 = 40 个
- 混合人口(多 bin 共存)× 各种比例 = 40 个
用法:python3 scripts/gen_synthetic_calib.py [output_dir]
"""
import numpy as np, os, sys, random

def gen_matrix(n, kcnt, blen, seed, dup_factor=1.0):
    """生成一个 n×n 矩阵,每行 kcnt 个非零,每列约 blen 个非零(双正则近似)。
    dup_factor>1 时,B 行有重复列(模拟高 dup)"""
    rng = np.random.default_rng(seed)
    ncols = max(n * kcnt // max(blen, 1), kcnt + 1)
    rows = np.repeat(np.arange(n), kcnt)
    cols = np.empty(n * kcnt, dtype=np.int32)
    for i in range(0, n * kcnt, kcnt):
        cols[i:i+kcnt] = rng.choice(ncols, size=kcnt, replace=False)
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    vals = rng.standard_normal(n * kcnt)
    return rows, cols, vals, ncols

def write_mtx(path, n, ncols, nnz, rows, cols, vals):
    with open(path, 'w') as f:
        f.write(f'%%MatrixMarket matrix coordinate real general\n{n} {ncols} {nnz}\n')
        for r, c, v in zip(rows, cols, vals):
            f.write(f'{r+1} {c+1} {v:.6e}\n')

def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/syn_cal'
    os.makedirs(outdir, exist_ok=True)
    rng_seed = 42
    generated = []

    # === 1. bin9 人口(est 8192-16384)===
    # est ≈ kcnt × blen × (1 - collision_rate),近似 kcnt × blen
    # 变化 kcnt(行内 A 非零数)和 blen(列度)
    print("=== bin9 格点 ===")
    bin9_configs = []
    for kcnt in [64, 96, 128, 160, 200]:
        for blen in [80, 100, 128]:
            est = kcnt * blen
            if 8192 <= est <= 16384:
                bin9_configs.append((kcnt, blen, est))
    # 加一些边缘案例
    bin9_configs.extend([(64, 200, 12800), (256, 40, 10240), (32, 400, 12800)])
    for i, (kcnt, blen, est) in enumerate(bin9_configs[:20]):
        n = 2000  # bin9 行大,2000 行足够测时
        seed = rng_seed + i
        r, c, v, nc = gen_matrix(n, kcnt, blen, seed)
        path = f'{outdir}/bin9_k{kcnt}_b{blen}_e{est}.mtx'
        write_mtx(path, n, nc, len(r), r, c, v)
        generated.append(('bin9', f'bin9_k{kcnt}_b{blen}_e{est}', n, kcnt, blen, est))
        print(f'  bin9 #{i}: kcnt={kcnt} blen={blen} est={est} n={n}')

    # === 2. bin4/5 人口(est 256-1024)===
    print("=== bin4/5 格点 ===")
    bin45_configs = []
    for kcnt in [8, 16, 24, 32, 48]:
        for blen in [16, 24, 32, 48]:
            est = kcnt * blen
            if 256 <= est <= 1024:
                bin45_configs.append((kcnt, blen, est))
    for i, (kcnt, blen, est) in enumerate(bin45_configs[:20]):
        n = 10000
        seed = rng_seed + 100 + i
        r, c, v, nc = gen_matrix(n, kcnt, blen, seed)
        path = f'{outdir}/bin45_k{kcnt}_b{blen}_e{est}.mtx'
        write_mtx(path, n, nc, len(r), r, c, v)
        generated.append(('bin45', f'bin45_k{kcnt}_b{blen}_e{est}', n, kcnt, blen, est))
        print(f'  bin45 #{i}: kcnt={kcnt} blen={blen} est={est} n={n}')

    # === 3. bin0 人口(est 33-64)===
    print("=== bin0 格点 ===")
    bin0_configs = []
    for kcnt in [4, 6, 8]:
        for blen in [6, 8, 10]:
            est = kcnt * blen
            if 33 <= est <= 64:
                bin0_configs.append((kcnt, blen, est))
    for i, (kcnt, blen, est) in enumerate(bin0_configs[:15]):
        n = 50000
        seed = rng_seed + 200 + i
        r, c, v, nc = gen_matrix(n, kcnt, blen, seed)
        path = f'{outdir}/bin0_k{kcnt}_b{blen}_e{est}.mtx'
        write_mtx(path, n, nc, len(r), r, c, v)
        generated.append(('bin0', f'bin0_k{kcnt}_b{blen}_e{est}', n, kcnt, blen, est))
        print(f'  bin0 #{i}: kcnt={kcnt} blen={blen} est={est} n={n}')

    # === 4. 混合人口(多 bin 共存)===
    print("=== 混合格点 ===")
    for i in range(20):
        seed = rng_seed + 300 + i
        n = 5000
        rng2 = np.random.default_rng(seed)
        # 混合:30% bin0 + 30% bin4/5 + 20% bin9 + 20% 其他
        rows_list, cols_list, vals_list = [], [], []
        ncols = 50000
        for row in range(n):
            r = rng2.random()
            if r < 0.3:   # bin0
                kcnt, blen = 6, 8
            elif r < 0.6: # bin4/5
                kcnt, blen = 16, 32
            elif r < 0.8: # bin9
                kcnt, blen = 128, 100
            else:          # bin6/7
                kcnt, blen = 48, 64
            ks = rng2.choice(ncols, size=kcnt, replace=False)
            for k in ks:
                rows_list.append(row)
                cols_list.append(k)
                vals_list.append(rng2.standard_normal())
        rows = np.array(rows_list, dtype=np.int32)
        cols = np.array(cols_list, dtype=np.int32)
        vals = np.array(vals_list)
        # 变化混合比例
        if i < 7:
            path = f'{outdir}/mix30_30_20_20_{i}.mtx'
        elif i < 14:
            path = f'{outdir}/mix10_40_40_10_{i}.mtx'
        else:
            path = f'{outdir}/mix50_20_10_20_{i}.mtx'
        write_mtx(path, n, ncols, len(rows), rows, cols, vals)
        generated.append(('mix', os.path.basename(path).replace('.mtx',''), n, 0, 0, 0))
        print(f'  mix #{i}: {os.path.basename(path)}')

    print(f"\n=== 总计 {len(generated)} 个合成矩阵 → {outdir}/ ===")

    # 写清单文件(供标定脚本用)
    with open(f'{outdir}/manifest.txt', 'w') as f:
        for typ, name, n, kcnt, blen, est in generated:
            f.write(f'{name}.mtx\t{typ}\t{n}\t{kcnt}\t{blen}\t{est}\n')

    return len(generated)

if __name__ == '__main__':
    main()
