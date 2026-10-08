#!/usr/bin/env python3
"""D6 v5 前置:顽固阵错误解剖——混淆对(预测 X 实际 Y)+错行特征切片。

对每个顽固阵:LOO 式训练(其余阵),输出:
  · 混淆矩阵(4×4):主错源=哪个回归器在哪儿撒谎
  · 错行 vs 对行的特征中位对比(定位判别缺失维)
  · 错行的代价倍数分布(错得多贵)
用法:.venv/bin/python scripts/d6_error_anatomy.py [mat ...](默认顽固六阵)
"""
import argparse, os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from fit_d1_costlaws import ht_slots
from fit_d1_context import prep, WEAPONS
from d6_learned_dispatch import feats_matrix

STUBBORN = ["bcsstk32", "pre2", "Si10H16", "opt1", "HTC_336_9129", "c-43"]
ALL = STUBBORN + ["bloweybq", "in-2004", "nd24k", "crankseg_2", "F2", "c-64",
                  "333SP", "af_shell4", "bcsstk23", "bcsstm23", "bcspwr08", "bcspwr05",
                  "bcsstm01", "1138_bus", "ash219", "bp_0", "arc130"]
rng = np.random.default_rng(42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat", nargs="*", default=STUBBORN)
    ap.add_argument("--bandcap", type=int, default=60000)
    ap.add_argument("--iters", type=int, default=150)
    a = ap.parse_args()
    mats_all = sorted(set(ALL))
    data = prep(mats_all)
    for m, d in data.items():
        d["X"] = feats_matrix(d, mat=m)
        d["trainmask"] = {}
        from d6_learned_dispatch import stratified_mask
        for w in WEAPONS:
            d["trainmask"][w] = stratified_mask(d, w, a.bandcap)

    for hold in a.mat:
        models = {}
        for w in WEAPONS:
            Xs, ys = [], []
            for m, d in data.items():
                if m == hold:
                    continue
                idx = d["trainmask"][w]
                if len(idx):
                    Xs.append(d["X"][idx]); ys.append(d["cyc"][idx, WEAPONS.index(w)])
            X = np.concatenate(Xs); y = np.concatenate(ys)
            mdl = HistGradientBoostingRegressor(max_iter=a.iters, learning_rate=0.1,
                                                 early_stopping=True, validation_fraction=0.1,
                                                 random_state=0)
            mdl.fit(X, y)
            models[w] = mdl
        d = data[hold]
        pred = np.stack([models[w].predict(d["X"]) for w in WEAPONS], 1)
        cand = d["legal"] & np.isfinite(pred)
        sel = cand.sum(1) >= 2
        p2 = np.where(cand, pred, np.inf)[sel]
        c2 = np.where(cand, d["cyc"], np.inf)[sel]
        pick = p2.argmin(1); oracle = c2.argmin(1)
        ok = pick == oracle
        r = d["raw"]; rid = d["rid"][sel]
        Wnames = WEAPONS
        conf = {}
        for pi, oi in zip(pick[~ok], oracle[~ok]):
            conf[(Wnames[pi], Wnames[oi])] = conf.get((Wnames[pi], Wnames[oi]), 0) + 1
        top = sorted(conf.items(), key=lambda x: -x[1])[:3]
        # 错行代价倍数 + 特征对比
        cost_ratio = np.take_along_axis(c2, pick[:, None], 1).ravel() / np.maximum(
            np.take_along_axis(c2, oracle[:, None], 1).ravel(), 1)
        print(f"\n== {hold}: 判行 {sel.sum()} 错 {int((~ok).sum())} ({(~ok).mean():.1%}) ==")
        print("  混淆对(预测→实际): " + "  ".join(f"{p}→{o}:{n}" for (p, o), n in top))
        print(f"  错行代价倍数 p50={np.median(cost_ratio[~ok]):.2f} p90={np.percentile(cost_ratio[~ok],90):.2f}"
              f"  Σ浪费={cost_ratio[~ok].sum()-len(cost_ratio[~ok]):.0f} 行倍")
        al, fl, es, sp = r["al"][sel], r["fl"][sel], r["es"][sel], r["sp"][sel]
        mb = feats_matrix(d, mat=hold)[sel][:, 18]
        for nm, arr in [("a_len", al), ("flop", fl), ("est", es), ("span", sp), ("maxbl", mb)]:
            e, g = arr[~ok], arr[ok]
            if len(e) and len(g):
                print(f"    {nm:<6} 错行 p50={np.median(e):.0f} vs 对行 p50={np.median(g):.0f}")


if __name__ == "__main__":
    main()
