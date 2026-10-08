#!/usr/bin/env python3
"""D6 学习型逐行派发(study/34 v1.1;v4 λ 证伪后的主力路线)。

形态:HistGradientBoosting 回归器 ×4 武器(行特征+阵级语境 → 该武器逐行周期),
argmin(合法域内)为派发决策;leave-one-matrix-out 为泛化口径(=新阵部署场景)。
特征(全 dispatch 时可算,无答案泄漏:d/dup 列禁入):
  a_len, flop, est, span, ht(表槽), W(窗数), log 族, 阵级 ctx(log rows/log Σflop/log mean_est)
合法域同 v3b/v4:search 态 dense 非候选;未测武器不评分(与分析律同基准)。
指标:LOO argmin 一致率 + oracle-gap;对照=同数据 v3b 解析律。
用法:.venv/bin/python scripts/d6_learned_dispatch.py [--iters 200] [--cap 400000]
"""
import argparse, math, os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from fit_d1_costlaws import load, ht_slots
from fit_d1_context import prep, evaluate, WEAPONS

FEATNAMES = ["a_len", "flop", "est", "span", "ht", "W",
             "log_al", "log_fl", "log_es", "log_sp", "ctx_rows", "ctx_sumfl", "ctx_meanes"]


def feats_matrix(d):
    """一行的 13 维特征矩阵(n×13)。"""
    r = d["raw"]
    al, fl, es, sp = r["al"], r["fl"], r["es"], r["sp"]
    ht, W = d["feats"]["hash"][:, 0], d["feats"]["dense"][:, 0]
    ctx = np.array(d["ctx"])
    n = len(al)
    return np.stack([al, fl, es, sp, ht, W,
                     np.log1p(al), np.log1p(fl), np.log1p(np.maximum(es, 1)),
                     np.log1p(sp), np.full(n, ctx[0]), np.full(n, ctx[1]), np.full(n, ctx[2])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat", nargs="*",
                    default=["bloweybq", "pre2", "c-43", "Si10H16", "in-2004", "nd24k",
                             "crankseg_2", "F2", "c-64", "bcsstk32", "bcsstk23", "bcsstm23",
                             "bcspwr08", "bcspwr05", "bcsstm01", "1138_bus", "ash219"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--cap", type=int, default=400000)
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    data = prep(a.mat)
    for m, d in data.items():
        d["X"] = feats_matrix(d)
        d["trainmask"] = {}
        for w in WEAPONS:
            idx = np.where(d["legal"][:, WEAPONS.index(w)])[0]
            if len(idx) > a.cap:
                idx = rng.choice(idx, a.cap, replace=False)
            d["trainmask"][w] = idx
    mats = sorted(data)
    print(f"特征({len(FEATNAMES)} 维):{','.join(FEATNAMES)}  HGB iters={a.iters} 训练帽={a.cap}/阵/武器\n")

    tot_ag = tot_n = 0
    gn = gd = 0.0
    per = {}
    for hold in mats:
        Xtr, ytr = {w: [] for w in WEAPONS}, {w: [] for w in WEAPONS}
        for m, d in data.items():
            if m == hold:
                continue
            for w in WEAPONS:
                idx = d["trainmask"][w]
                if len(idx):
                    Xtr[w].append(d["X"][idx])
                    ytr[w].append(d["cyc"][idx, WEAPONS.index(w)])
        models = {}
        for w in WEAPONS:
            X = np.concatenate(Xtr[w]); y = np.concatenate(ytr[w])
            if len(X) < 200:
                models[w] = None
                continue
            mdl = HistGradientBoostingRegressor(max_iter=a.iters, learning_rate=0.1,
                                                 early_stopping=True, validation_fraction=0.1,
                                                 random_state=0, verbose=0)
            mdl.fit(X, y)
            models[w] = mdl
        # 评估 hold:预测即"伪律"接口——借用 evaluate:lam=1,feats 用预测值替换
        d = data[hold]
        pred = np.stack([models[w].predict(d["X"]) if models[w] is not None
                         else np.full(d["n"], np.nan) for w in WEAPONS], 1)
        cand = d["legal"] & np.isfinite(pred)
        sel = cand.sum(1) >= 2
        p2 = np.where(cand, pred, np.inf)[sel]
        c2 = np.where(cand, d["cyc"], np.inf)[sel]
        pick = p2.argmin(1); oracle = c2.argmin(1)
        ok = pick == oracle
        ag, n = int(ok.sum()), int(sel.sum())
        tot_ag += ag; tot_n += n
        gn += float(np.take_along_axis(c2, pick[:, None], 1).sum())
        gd += float(np.take_along_axis(c2, oracle[:, None], 1).sum())
        per[hold] = (ag, n)
        print(f"  [LOO] {hold:<10} {ag}/{n} = {ag/max(1,n):.1%}", flush=True)
    print(f"\n== D6 LOO 汇总:一致率 {tot_ag}/{tot_n} = {tot_ag/max(1,tot_n):.1%}  "
          f"oracle-gap {gn/gd:.4f} ==")
    print("(对照:v3b 解析律池化 45.6%/gap 6.66;v4 λ 上限 in-sample 57.6%/1.095)")


if __name__ == "__main__":
    main()
