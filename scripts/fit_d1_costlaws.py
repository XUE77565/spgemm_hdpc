#!/usr/bin/env python3
"""D2 成本律拟合(study/34:四武器解析式系数最小二乘 + argmin 一致率)。

成本律形态(study/34 §L1,物理先验):
  C_hash  ≈ α·ht + β·flop + c        (ht=表槽位数=32<<bkid_hash,占用物理轴)
  C_dense ≈ γ·span + δ·flop + c      (窗扫 + 逐乘积原子)
  C_k7    ≈ ε·span + ζ·flop + c      (位图清/扫 + 标记/写)
  C_merge ≈ η·a_len·log2(a_len) + θ·flop + c   (归并比较链)

口径:truth 表(D1 产物)cycles=clock64 周期;仅拟合各武器已测行(域内);
hash 行剔除 ovf(estimator 失败人口)——build_d1_truth 已滤。
一致率 = argmin(预测) vs oracle best(实测 argmin),仅对"≥2 武器有测度"的行计
(单武器行无判别信息);另报 oracle-gap = ΣC_选路/ΣC_最优(时间加权,论文 C3 头条口径)。
用法:.venv/bin/python scripts/fit_d1_costlaws.py [--min-feat 2] [mat ...]
"""
import argparse, math, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUTH = os.path.join(REPO, "compare", "d1_oracle", "truth")
BIG6 = ["bloweybq", "pre2", "c-43", "Si10H16", "in-2004", "nd24k"]
F100 = ["bcsstk32", "bcsstk23", "bcsstm23", "bcspwr08", "bcspwr05", "bcsstm01"]

# 武器 → (特征构造, 列名)。ht 从 truth 的 bkid_def 不可靠(那是默认路由 bin)——
# 用 est 重建表槽:连续 est 时代 target=α%×est(150),再上取 2 的幂,帽 HASH_CAP。
HASH_ALPHA, HASH_CAP = 1.5, 65536


def ht_slots(est):
    if est <= 0:
        return 32
    t = int(est * HASH_ALPHA)
    s = 32
    while s < t and s < HASH_CAP:
        s <<= 1
    return s


PB2_W = 5400          # dense 窗宽(structural_params.cuh:半SM 114KB÷21B,docs/67 §6 单源化)
TIER_SPAN_MAX = 7277  # 档位梯最大档(DENSE_TIER_SPAN_MAX)


def feats(row):
    r, a_len, flop, est, span, d, dup, bkid, ch, cd, ck, cm = row[:12]
    W = math.ceil(span / PB2_W) if span > 0 else 1
    return {
        "hash":  [float(ht_slots(est)), float(flop), float(a_len), 1.0],
        "dense": [float(W), float(flop), 1.0],          # docs/67 形态:窗数+乘积(子律按 dclass 分)
        "k7":    [float(span), float(flop), 1.0],
        "merge": [a_len * max(1.0, math.log2(max(a_len, 2))), float(flop), 1.0],
    }


def dclass_pred(row):
    """dense 子律预测:档内(≤7277)=2(dsn 梯);档外 avgB=flop/a_len≥64→cursor(1) 否则 search(0)。"""
    _, a_len, flop, est, span = row[:5]
    if span <= TIER_SPAN_MAX:
        return 2
    return 1 if (a_len > 0 and flop / a_len >= 64) else 0


def load(mats):
    rows = []
    for m in mats:
        p = os.path.join(TRUTH, m + ".truth.tsv")
        if not os.path.exists(p):
            continue
        for ln in open(p).readlines()[1:]:
            f = ln.split()
            rows.append((m,) + tuple(float(x) for x in f[:12]) + (f[12], float(f[13])))
    return rows


def fit(rows, key, cost_col, dclass=None):
    """武器 key 的最小二乘(dense 按 dclass 子律):返回 (coef, R², n)。"""
    X, y = [], []
    for t in rows:
        cyc = t[9 + ["hash", "dense", "k7", "mrg3"].index(cost_col)]
        if cyc > 0 and (dclass is None or t[14] == dclass):
            X.append(feats(t[1:])[key])
            y.append(cyc)
    if len(X) < 100:
        return None, 0.0, len(X)
    X, y = np.array(X), np.array(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return coef, 1 - ss_res / ss_tot if ss_tot else 0.0, len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat", nargs="*", default=BIG6 + F100)
    ap.add_argument("--json", default="", help="可选:系数落盘 json(D3 部署原料)")
    a = ap.parse_args()
    rows = load(a.mat)
    if not rows:
        print("无 truth 数据;先跑 collect_d1_oracle.py + build_d1_truth.py")
        sys.exit(1)
    print(f"truth 行总数:{len(rows)}(阵 {'/'.join(a.mat)})\n")

    laws, cols = {}, {"hash": "hash", "dense": "dense", "k7": "k7", "mrg3": "merge"}
    names = {"hash": ["α(ht)", "β(flop)", "κ(a_len)", "c"],
             "dense": ["γ(W)", "δ(flop)", "c"],
             "k7": ["ε(span)", "ζ(flop)", "c"],
             "mrg3": ["η(al·log)", "θ(flop)", "c"]}
    for w, key in cols.items():
        if w == "dense":
            laws[w] = {}
            for dc, dnm in [(0, "search"), (1, "cursor"), (2, "dsn梯")]:
                coef, r2, n = fit(rows, key, w, dclass=dc)
                if coef is None:
                    print(f"C_dense[{dnm}] 样本不足(n={n}),跳过")
                    continue
                laws[w][dc] = coef.tolist()
                print(f"C_dense[{dnm}] n={n:<8} R²={r2:.3f}   " +
                      "  ".join(f"{nm}={v:.1f}" for nm, v in zip(names[w], coef)))
        else:
            coef, r2, n = fit(rows, key, w)
            if coef is None:
                print(f"C_{w:<6} 样本不足(n={n}),跳过")
                continue
            laws[w] = coef.tolist()
            print(f"C_{w:<6} n={n:<8} R²={r2:.3f}   " +
                  "  ".join(f"{nm}={v:.1f}" for nm, v in zip(names[w], coef)))

    # argmin 一致率(≥2 武器有测度的行)+ oracle-gap
    n_judge = n_agree = 0
    gap_num = gap_den = 0.0
    per_mat = {}
    for t in rows:
        cycs = {"hash": t[9], "dense": t[10], "k7": t[11], "mrg3": t[12]}
        have = {k: v for k, v in cycs.items() if v > 0 and k in laws}
        if len(have) < 2:
            continue
        oracle = min(have, key=have.get)
        f = feats(t[1:])
        pred = {}
        for k in have:
            if k == "dense":
                dcp = dclass_pred(t[1:])
                coef_d = laws["dense"].get(dcp) or next(iter(laws["dense"].values()))
                pred[k] = float(np.dot(coef_d, f["dense"]))
            else:
                pred[k] = float(np.dot(laws[k], f[cols[k]]))
        pick = min(pred, key=pred.get)
        n_judge += 1
        n_agree += (pick == oracle)
        gap_num += have[pick]
        gap_den += have[oracle]
        m = per_mat.setdefault(t[0], [0, 0])
        m[0] += (pick == oracle)
        m[1] += 1
    if n_judge:
        print(f"\nargmin 一致率:{n_agree}/{n_judge} = {n_agree/n_judge:.1%}(D2 判据 >85%)")
        print(f"oracle-gap(时间加权 ΣC_选路/ΣC_最优) = {gap_num/gap_den:.4f}(D5 头条口径,目标 <1.05)")
        for m, (ag, tot) in sorted(per_mat.items()):
            print(f"  {m:<10} {ag}/{tot} = {ag/tot:.1%}")
    if a.json and laws:
        import json
        with open(a.json, "w") as fj:
            json.dump({"laws": laws, "hash_alpha": HASH_ALPHA, "hash_cap": HASH_CAP,
                       "n_rows": len(rows)}, fj, indent=1)
        print(f"系数 → {a.json}")


if __name__ == "__main__":
    main()
