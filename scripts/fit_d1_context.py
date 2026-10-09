#!/usr/bin/env python3
"""D2 v4:语境缩放检验(study/35 §七 残症假设终审;向量化版)。

假设:每行成本 = 阵级语境系数 λ_m × 全局成本律。两层:
  上限(in-sample ALS):每阵 λ 已知的天花板;
  泛化(LOO):λ̂ = 其余阵 λ 对 dispatch 可算特征(log rows/log Σflop/log mean_est)
  的对数线性回归;λ̂=1(无语境)对照。
合法域同 v3b(search 幽灵律出律;dense 需 avgB≥64)。拟合用每阵 ≤40 万行抽样
(系数稳定),一致率评估用全量行。
"""
import argparse, math, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from fit_d1_costlaws import load, ht_slots

PB2_W, TIER_SPAN_MAX = 5400, 7277
WEAPONS = ["hash", "dense", "k7", "mrg3"]
FIT_CAP = 400_000
rng = np.random.default_rng(42)


def prep(mats):
    """真值表 → 每阵 numpy 块:特征、实测 cyc、合法/测度掩码、阵级 ctx。"""
    rows = load(mats)
    by = {}
    for t in rows:
        by.setdefault(t[0], []).append(t)
    out = {}
    for m, rs in by.items():
        n = len(rs)
        al = np.array([t[2] for t in rs]); fl = np.array([t[3] for t in rs], dtype=float)
        es = np.array([t[4] for t in rs]); sp = np.array([t[5] for t in rs], dtype=float)
        dclass = np.where(sp <= TIER_SPAN_MAX, 2, np.where(fl / np.maximum(al, 1) >= 64, 1, 0))
        ht = np.array([ht_slots(e) for e in es], dtype=float)
        W = np.ceil(sp / PB2_W)
        cyc = np.stack([np.array([t[9] for t in rs], dtype=float),
                        np.array([t[10] for t in rs], dtype=float),
                        np.array([t[11] for t in rs], dtype=float),
                        np.array([t[12] for t in rs], dtype=float)], 1)   # hash,dense,k7,mrg3
        meas = cyc > 0
        legal = meas.copy()
        legal[:, 1] &= (dclass != 0)          # dense:search 幽灵态非法
        feats = {"hash": np.stack([ht, fl, al, np.ones(n)], 1),
                 "dense": np.stack([W, fl, np.ones(n)], 1),
                 "k7": np.stack([sp, fl, np.ones(n)], 1),
                 "mrg3": np.stack([al * np.maximum(1.0, np.log2(np.maximum(al, 2))), fl, np.ones(n)], 1)}
        # ctx v2 严谨口径:R 臂 DUMP_ROUTING 全行(truth 行子集=train/serve skew,D3② A/B 教训)
        import struct as _st
        _rp = os.path.join(REPO, "compare", "d1_oracle", f"{m}.R1.routing.bin")
        if os.path.exists(_rp):
            with open(_rp, "rb") as _f:
                _rn, _ = _st.unpack("2i", _f.read(8))
                _f.seek(8)
                _es = _st.unpack(f"{_rn}i", _f.read(_rn * 4))
                _f.seek(8 + 3 * _rn * 4)
                _fl = _st.unpack(f"{_rn}i", _f.read(_rn * 4))
            ctx = [math.log(_rn), math.log(max(1, sum(_fl))), math.log(max(1.0, sum(_es) / max(1, _rn)))]
        else:
            ctx = [math.log(n), math.log(max(1.0, fl.sum())), math.log(max(1.0, es.mean()))]
        # 拟合抽样(每武器独立掩码,一致率评估用全量)
        fitmask = {}
        for w, col in zip(WEAPONS, range(4)):
            idx = np.where(legal[:, col])[0]
            if len(idx) > FIT_CAP:
                idx = rng.choice(idx, FIT_CAP, replace=False)
            fitmask[w] = idx
        out[m] = dict(feats=feats, cyc=cyc, legal=legal, ctx=ctx, fitmask=fitmask, n=n,
                      rid=np.array([t[1] for t in rs], dtype=np.int64),
                      raw=dict(al=al.astype(float), fl=fl, es=es.astype(float), sp=sp))
    return out


def als(data, iters=10):
    coef, lam = {w: None for w in WEAPONS}, {m: {w: 1.0 for w in WEAPONS} for m in data}
    for _ in range(iters):
        for w in WEAPONS:
            Xs, ys = [], []
            for m, d in data.items():
                idx = d["fitmask"][w]
                if not len(idx):
                    continue
                Xs.append(d["feats"][w][idx] * lam[m][w])
                ys.append(d["cyc"][idx, WEAPONS.index(w)])
            if sum(len(x) for x in Xs) < 100:
                continue
            X = np.concatenate(Xs); y = np.concatenate(ys)
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            coef[w] = c
        for m, d in data.items():
            for w in WEAPONS:
                if coef[w] is None:
                    continue
                idx = d["fitmask"][w]
                if not len(idx):
                    continue
                xv = d["feats"][w][idx] @ coef[w]
                yv = d["cyc"][idx, WEAPONS.index(w)]
                den = float((xv * xv).sum())
                if den > 0:
                    lam[m][w] = max(0.05, float((xv * yv).sum()) / den)
    return coef, lam


def evaluate(data, coef, lam, mats=None):
    agree = tot = 0
    gn = gd = 0.0
    per = {}
    for m, d in data.items():
        if mats and m not in mats:
            continue
        with np.errstate(invalid="ignore"):
            pred = np.stack([lam[m][w] * (d["feats"][w] @ coef[w]) if coef[w] is not None
                             else np.full(d["n"], np.nan) for w in WEAPONS], 1)
        cand = d["legal"] & np.isfinite(pred)
        nm = cand.sum(1)
        sel = nm >= 2
        if not sel.any():
            continue
        p2, c2 = pred[sel], d["cyc"][sel]
        cand2 = cand[sel]
        p2 = np.where(cand2, p2, np.inf)
        c2 = np.where(cand2, c2, np.inf)
        pick = p2.argmin(1)
        oracle = c2.argmin(1)
        ok = pick == oracle
        agree += int(ok.sum()); tot += int(sel.sum())
        gn += float(np.take_along_axis(c2, pick[:, None], 1).sum())
        gd += float(np.take_along_axis(c2, oracle[:, None], 1).sum())
        per[m] = (int(ok.sum()), int(sel.sum()))
    return agree, tot, (gn / gd if gd else float("inf")), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat", nargs="*",
                    default=["bloweybq", "pre2", "c-43", "Si10H16", "in-2004", "nd24k",
                             "crankseg_2", "F2", "c-64", "bcsstk32", "bcsstk23", "bcsstm23",
                             "bcspwr08", "bcspwr05", "bcsstm01", "1138_bus", "ash219"])
    a = ap.parse_args()
    data = prep(a.mat)
    mats = sorted(data)

    coef, lam = als(data)
    print("== λ(每阵×每武器,ALS in-sample)==")
    for w in WEAPONS:
        if coef[w] is not None:
            print(f"  λ[{w}]: " + " ".join(f"{m[:8]}={lam[m][w]:.2f}" for m in mats))
    ag, tot, gap, per = evaluate(data, coef, lam)
    print(f"\n== v4 上限(λ 已知):一致率 {ag}/{tot} = {ag/tot:.1%}  oracle-gap {gap:.4f} ==")
    for m in mats:
        if m in per:
            print(f"    {m:<10} {per[m][0]}/{per[m][1]} = {per[m][0]/per[m][1]:.1%}")

    print("\n== v4 LOO(λ̂ 回归 vs λ̂=1)==")
    agg = {"reg": [0, 0], "one": [0, 0]}
    rows_reg = {}
    for hold in mats:
        sub = {m: d for m, d in data.items() if m != hold}
        coef_h, lam_h = als(sub, iters=8)
        A = np.array([d["ctx"] + [1.0] for d in sub.values()])
        lh, lo = {}, {}
        for w in WEAPONS:
            if coef_h[w] is None:
                lh[w] = lo[w] = 1.0
                continue
            y = np.log(np.clip(np.array([lam_h[m][w] for m in sub]), 0.05, 20))
            b, *_ = np.linalg.lstsq(A, y, rcond=None)
            lh[w] = math.exp(float(np.dot(np.array(data[hold]["ctx"] + [1.0]), b)))
            lo[w] = 1.0
        for tag, lm in (("reg", lh), ("one", lo)):
            ag, tot, gap, per = evaluate(data, coef_h, {hold: lm}, mats=[hold])
            agg[tag][0] += ag; agg[tag][1] += tot
            if tag == "reg":
                rows_reg[hold] = per.get(hold, (0, 0))
    for tag, nm2 in (("reg", "λ̂ 回归"), ("one", "λ̂=1")):
        print(f"  {nm2}: {agg[tag][0]}/{agg[tag][1]} = {agg[tag][0]/max(1,agg[tag][1]):.1%}")
    print("  逐阵(回归版):")
    for m in mats:
        r = rows_reg.get(m)
        if r and r[1]:
            print(f"    {m:<10} {r[0]/r[1]:.1%}  n={r[1]}")


if __name__ == "__main__":
    main()
