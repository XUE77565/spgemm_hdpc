#!/usr/bin/env python3
"""verify_sorted_truth.py — 输出矩阵三层真值比对(正确性铁律工具)

用法: python3 verify_sorted_truth.py <A.mtx> <C_dump.mtx> [--tol 1e-9]

三层判据:
  L1 结构: 每行列集合与真值一致(浮点完全抵消的显式零差异→软判 benign)
  L2 序:   行内列序(检查器盲区主靶——nnz 对但乱序)
  L3 值:   配对后数值相对误差

A 口径: mmread 已自动镜像 symmetric;若返回仍非对称(旧后端)才手动展开。
C 按文件序逐行解析,不排序——保序才能看出乱序。
"""
import sys
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix


def load_A_full(path):
    m = mmread(path).tocsr().astype(np.float64)
    with open(path) as f:
        header = f.readline()
    if "symmetric" in header:
        d = (m - m.T)
        if d.nnz and abs(d.data).max() > 0:   # 旧后端未镜像 → 手动展开
            diag = m.diagonal()
            m = (m + m.T - csr_matrix(np.diag(diag))).tocsr()
    return m


def parse_C_preserve_order(path):
    """手动解析 coordinate MM,保留文件序(不排序、不去重)。"""
    with open(path) as f:
        f.readline()
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        r, c, nnz = (int(x) for x in line.split())
        rows = [[] for _ in range(r)]
        for line in f:
            if not line.strip() or line.startswith("%"):
                continue
            p = line.split()
            rows[int(p[0]) - 1].append((int(p[1]) - 1, float(p[2])))
    return r, c, rows


def main():
    a_path, c_path = sys.argv[1], sys.argv[2]
    tol = 1e-9
    if "--tol" in sys.argv:
        tol = float(sys.argv[sys.argv.index("--tol") + 1])

    A = load_A_full(a_path)
    A.sum_duplicates()
    T = (A @ A).tocsr()
    T.sort_indices()

    n, _, c_rows = parse_C_preserve_order(c_path)
    assert n == T.shape[0], f"n mismatch {n} vs {T.shape[0]}"

    L1_bad = L2_bad = L3_bad = benign = 0
    max_rel = 0.0
    examples = []
    total_c = 0

    for i in range(n):
        t_sl = slice(T.indptr[i], T.indptr[i + 1])
        t_cols = T.indices[t_sl].tolist()
        t_dat = dict(zip(T.indices[t_sl].tolist(), T.data[t_sl].tolist()))
        c = c_rows[i]
        total_c += len(c)
        c_cols = [x[0] for x in c]

        # L1 结构(集合级;显式零抵消 → 软判)
        if sorted(c_cols) != t_cols:
            eps = 1e-12 * (max((abs(v) for _, v in c), default=1.0)
                           + max((abs(v) for v in t_dat.values()), default=1.0) + 1e-300)
            cz = sorted(j for j, v in c if abs(v) > eps)
            tz = sorted(j for j, v in t_dat.items() if abs(v) > eps)
            if cz == tz:
                benign += 1
            else:
                L1_bad += 1
                if len(examples) < 3:
                    only_c = set(c_cols) - set(t_cols)
                    only_t = set(t_cols) - set(c_cols)
                    examples.append((i, "L1-集合", f"dump独有{list(only_c)[:4]} 真值独有{list(only_t)[:4]}"))
                continue

        # L2 序:文件序必须 == 自身排序(行内列序)
        if c_cols != sorted(c_cols):
            L2_bad += 1
            if len(examples) < 3:
                k = next(x for x in range(len(c_cols)) if c_cols[x] != sorted(c_cols)[x])
                examples.append((i, "L2-乱序", f"pos{k}: 文件序{c_cols[k:k+3]} 应为{sorted(c_cols)[k:k+3]}"))

        # L3 值(与序无关,dict 配对)
        row_bad = False
        for j, v in c:
            ref = t_dat.get(j)
            if ref is None:
                continue
            rel = abs(v - ref) / max(abs(ref), 1e-300)
            max_rel = max(max_rel, rel)
            if rel > tol:
                row_bad = True
                if len(examples) < 5:
                    examples.append((i, "L3-数值", f"col{j}: got {v:.6e} want {ref:.6e} rel={rel:.2e}"))
                break
        if row_bad:
            L3_bad += 1

    total_t = T.nnz
    print(f"== {c_path}")
    print(f"nnz: truth={total_t} dump={total_c}")
    print(f"L1 集合错行: {L1_bad}   (零抵消软判 benign: {benign})")
    print(f"L2 乱序行:   {L2_bad}")
    print(f"L3 数值错行: {L3_bad}   全局最大相对误差: {max_rel:.3e}")
    for e in examples:
        print("  例:", e)
    verdict = "PASS" if (L1_bad == 0 and L2_bad == 0 and L3_bad == 0) else "FAIL"
    print(f"判定: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
