#!/usr/bin/env python3
"""l3_diag.py — 值级 L3 标红的归因诊断(只补证据;不改 verify_sorted_truth.py 的判据)

用法:
  python3 scripts/l3_diag.py <A.mtx> <臂名=dump.mtx> [<臂名=dump.mtx> ...] [--tol 1e-9] [--list 40] [--csv out.csv]
                             [--full-csv all.csv.gz] [--rows-csv rows.csv.gz]
  --csv       并排清单里列出的那几条(≤ --list)
  --full-csv  任一臂官方口径判错的**全部**条目(审计用;.gz 结尾则压缩)。条目可达百万级,文件大,留在作业目录
  --rows-csv  按行聚合的全量摘要:每个官方判错的行、每臂一组(错条目数、最大 rel、最大 nerr、最大 |差|),体积小,可进 OUT_DIR

对每个转储(同一 A 的 C=A·A,MatrixMarket coordinate)与 scipy 真值 T=A@A 逐条目比,给两种口径:
  官方口径  rel  = |got−want| / |want|            verify_sorted_truth.py 的 L3(rel>tol 记错;真值里没有的条目不判)
  归一口径  nerr = |got−want| / (|A|·|A|)_ij      分母 = 该输出项全部乘积的绝对值之和 Σ|a_ik·b_kj|
归一口径的出处:k 项浮点求和,任意求和次序的误差界 ≤ (k−1)·u·Σ|x_i|(u=1.1e-16,Higham《Accuracy and
Stability of Numerical Algorithms》§4.2)。GPU 原子累加与 scipy 的 CSR 序累加只是求和次序不同,
所以两者之差的 nerr 应在 k·u 量级(k=该项乘积数,通常 ≤1e4 → nerr ≲ 1e-12);漏加/多加一项乘积的真错,
nerr 在 1/k 量级(≥1e-4)。相消严重的条目 |want| ≪ Σ|乘积|,rel 会被放大而 nerr 不会。

输出(每臂):条目数、行主序是否成立、官方口径错的条目数/行数与最大 rel、nerr 的全体最大值、
官方错条目里 nerr 的中位/99%/最大、nerr>1e-12 与 >1e-10 的条目数;以及机读行
  NERR_MAX <臂名> <值>     OFFICIAL_BAD_ROWS <臂名> <行数>
随后把"任一臂官方口径判错"的条目并排列出(最多 --list 条,按各臂最大 rel 降序):
  行 列(0 基) want Σ|乘积| | 各臂 got / |差| / rel / nerr
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_sorted_truth import load_A_full   # noqa: E402  与官方三判同一个 A 口径(对称镜像规则)


def csr_keys(M):
    M = M.tocsr()
    M.sort_indices()
    n, ncol = M.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(M.indptr))
    return rows * np.int64(ncol) + M.indices.astype(np.int64), np.asarray(M.data, dtype=np.float64)


def load_dump(path):
    with open(path) as f:
        f.readline()
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        r, c, nnz = (int(x) for x in line.split())
        arr = np.loadtxt(f, dtype=np.float64, ndmin=2)
    if arr.size == 0:
        arr = np.zeros((0, 3))
    i = arr[:, 0].astype(np.int64) - 1
    j = arr[:, 1].astype(np.int64) - 1
    return r, c, nnz, i, j, np.ascontiguousarray(arr[:, 2])


def lookup(keys_sorted, data, q):
    """q 中每个键在升序 keys_sorted 里的值;不存在记 0.0。返回 (值, 命中掩码)。"""
    if keys_sorted.size == 0:
        return np.zeros(q.shape, dtype=np.float64), np.zeros(q.shape, dtype=bool)
    idx = np.minimum(np.searchsorted(keys_sorted, q), keys_sorted.size - 1)
    hit = keys_sorted[idx] == q
    return np.where(hit, data[idx], 0.0), hit


def main():
    args = sys.argv[1:]
    tol, nlist, csv_path, pos = 1e-9, 40, None, []
    full_path, rows_path = None, None
    k = 0
    while k < len(args):
        if args[k] == "--tol":
            tol = float(args[k + 1]); k += 2
        elif args[k] == "--list":
            nlist = int(args[k + 1]); k += 2
        elif args[k] == "--csv":
            csv_path = args[k + 1]; k += 2
        elif args[k] == "--full-csv":
            full_path = args[k + 1]; k += 2
        elif args[k] == "--rows-csv":
            rows_path = args[k + 1]; k += 2
        else:
            pos.append(args[k]); k += 1
    if len(pos) < 2:
        print(__doc__)
        return 2
    a_path = pos[0]
    dumps = [p.split("=", 1) for p in pos[1:]]

    A = load_A_full(a_path)
    A.sum_duplicates()
    T = (A @ A).tocsr()
    Aabs = abs(A)
    S = (Aabs @ Aabs).tocsr()
    ncol = np.int64(T.shape[1])
    tk, td = csr_keys(T)
    sk, sd = csr_keys(S)
    print("# l3_diag: A=%s  n=%d  truth_nnz=%d  abs_nnz=%d  tol=%g" % (a_path, T.shape[0], tk.size, sk.size, tol))

    arms = []            # (name, 升序键, 值, bad 掩码, rel, nerr)
    for name, path in dumps:
        r, c, nnz, di, dj, dv = load_dump(path)
        dk = di * ncol + dj
        order_ok = bool(np.all(dk[1:] > dk[:-1])) if dk.size > 1 else True
        if not order_ok:                                   # 乱序/重键时排序后再比(序本身由官方 L2 与 gate 负责)
            o = np.argsort(dk, kind="stable")
            dk, dv = dk[o], dv[o]
        want, hit = lookup(tk, td, dk)
        scale, _ = lookup(sk, sd, dk)
        diff = np.abs(dv - want)
        rel = diff / np.maximum(np.abs(want), 1e-300)
        bad = hit & (rel > tol)
        with np.errstate(divide="ignore", invalid="ignore"):
            nerr = np.where(scale > 0, diff / np.where(scale > 0, scale, 1.0), np.where(diff > 0, np.inf, 0.0))
        _, hit_rev = lookup(dk, dv, tk)
        bad_rows = int(np.unique(dk[bad] // ncol).size)
        nb = int(np.count_nonzero(bad))
        nmax = float(nerr.max()) if nerr.size else 0.0
        print("")
        print("== %s: %s" % (name, path))
        print("条目=%d(头部声明 %d) 行主序且行内严格升序=%s 真值里没有的条目=%d 真值有而转储缺=%d"
              % (dk.size, nnz, order_ok, int(np.count_nonzero(~hit)), int(np.count_nonzero(~hit_rev))))
        print("官方口径(rel>%g): 错条目 %d / 错行 %d;最大 rel = %.3e" % (tol, nb, bad_rows, float(rel[hit].max()) if hit.any() else 0.0))
        if nb:
            nb_err = nerr[bad]
            print("归一口径 nerr=|差|/Σ|乘积|: 全体最大 = %.3e;官方错条目里 中位 = %.3e  99%% = %.3e  最大 = %.3e"
                  % (nmax, float(np.median(nb_err)), float(np.quantile(nb_err, 0.99)), float(nb_err.max())))
        else:
            print("归一口径 nerr=|差|/Σ|乘积|: 全体最大 = %.3e" % nmax)
        print("归一口径: nerr>1e-12 的条目 = %d;nerr>1e-10 的条目 = %d"
              % (int(np.count_nonzero(nerr > 1e-12)), int(np.count_nonzero(nerr > 1e-10))))
        print("NERR_MAX %s %.6e" % (name, nmax))
        print("OFFICIAL_BAD_ROWS %s %d" % (name, bad_rows))
        arms.append((name, dk, dv, bad, rel, nerr))

    # ── 并排清单:任一臂官方口径判错的条目 ──
    if not arms:
        return 0
    ukeys = np.unique(np.concatenate([a[1][a[3]] for a in arms]))
    if ukeys.size == 0:
        print("\n(没有任何臂在官方口径下判错,清单为空)")
        return 0
    want_u, _ = lookup(tk, td, ukeys)
    scale_u, _ = lookup(sk, sd, ukeys)
    cols = []
    worst = np.zeros(ukeys.size)
    for name, dk, dv, bad, rel, nerr in arms:
        got, hit = lookup(dk, dv, ukeys)
        d = np.abs(got - want_u)
        rl = d / np.maximum(np.abs(want_u), 1e-300)
        ne = np.where(scale_u > 0, d / np.where(scale_u > 0, scale_u, 1.0), np.inf)
        cols.append((name, got, d, rl, ne, hit))
        worst = np.maximum(worst, np.where(hit, rl, 0.0))
    sel = np.argsort(-worst, kind="stable")[:nlist] if ukeys.size > nlist else np.arange(ukeys.size)
    sel = sel[np.argsort(ukeys[sel], kind="stable")]       # 输出按 (行,列) 排
    print("\n===== 任一臂官方口径判错的条目:共 %d 条,列出 %d 条(行/列 0 基,与 verify_sorted_truth 的例子同口径)=====" % (ukeys.size, sel.size))
    head = "%8s %8s %14s %12s" % ("row", "col", "want", "Σ|乘积|")
    for name, *_ in cols:
        head += " | %14s %10s %10s %10s" % (name + ":got", "|差|", "rel", "nerr")
    print(head)
    lines = []
    for x in sel:
        row, col = int(ukeys[x] // ncol), int(ukeys[x] % ncol)
        ln = "%8d %8d %14.6e %12.4e" % (row, col, want_u[x], scale_u[x])
        rec = [row, col, repr(float(want_u[x])), repr(float(scale_u[x]))]
        for name, got, d, rl, ne, hit in cols:
            ln += " | %14.6e %10.3e %10.3e %10.3e" % (got[x], d[x], rl[x], ne[x])
            rec += [repr(float(got[x])), "%.6e" % d[x], "%.6e" % rl[x], "%.6e" % ne[x]]
        print(ln)
        lines.append(rec)
    if csv_path:
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            hdr = ["row", "col", "want", "abs_sum"]
            for name, *_ in cols:
                hdr += [name + "_got", name + "_absdiff", name + "_rel", name + "_nerr"]
            w.writerow(hdr)
            w.writerows(lines)
    if full_path or rows_path:
        import gzip

        def op(path):
            return gzip.open(path, "wt", newline="") if path.endswith(".gz") else open(path, "w", newline="")

        names = [c[0] for c in cols]
        if full_path:                                      # 全量条目级清单(审计用)
            with op(full_path) as f:
                f.write("row,col,want,abs_sum" + "".join(",%s_got,%s_absdiff,%s_rel,%s_nerr" % (n, n, n, n) for n in names) + "\n")
                rows_u = ukeys // ncol
                cols_u = ukeys % ncol
                CH = 200000
                for a0 in range(0, ukeys.size, CH):
                    a1 = min(a0 + CH, ukeys.size)
                    buf = []
                    for x in range(a0, a1):
                        ln = "%d,%d,%.17g,%.6e" % (rows_u[x], cols_u[x], want_u[x], scale_u[x])
                        for name, got, d, rl, ne, hit in cols:
                            ln += ",%.17g,%.3e,%.3e,%.3e" % (got[x], d[x], rl[x], ne[x])
                        buf.append(ln)
                    f.write("\n".join(buf) + "\n")
            print("全量条目清单: %s(%d 条)" % (full_path, ukeys.size))
        if rows_path:                                      # 按行聚合(每个判错行一行)
            rows_u = ukeys // ncol
            urow, inv = np.unique(rows_u, return_inverse=True)
            agg = []
            for name, got, d, rl, ne, hit in cols:
                isbad = hit & (rl > tol)
                cnt = np.bincount(inv, weights=isbad.astype(np.float64), minlength=urow.size).astype(np.int64)
                mrel = np.zeros(urow.size); np.maximum.at(mrel, inv, np.where(isbad, rl, 0.0))
                mne = np.zeros(urow.size); np.maximum.at(mne, inv, np.where(isbad, ne, 0.0))
                mdf = np.zeros(urow.size); np.maximum.at(mdf, inv, np.where(isbad, d, 0.0))
                agg.append((cnt, mrel, mne, mdf))
            with op(rows_path) as f:
                f.write("row" + "".join(",%s_bad_entries,%s_max_rel,%s_max_nerr,%s_max_absdiff" % (n, n, n, n) for n in names) + "\n")
                for x in range(urow.size):
                    ln = "%d" % urow[x]
                    for cnt, mrel, mne, mdf in agg:
                        ln += ",%d,%.3e,%.3e,%.3e" % (cnt[x], mrel[x], mne[x], mdf[x])
                    f.write(ln + "\n")
            print("按行聚合清单: %s(%d 行)" % (rows_path, urow.size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
