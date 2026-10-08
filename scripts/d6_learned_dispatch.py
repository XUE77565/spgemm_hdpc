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
             "log_al", "log_fl", "log_es", "log_sp", "ctx_rows", "ctx_sumfl", "ctx_meanes",
             "dup_est", "dens", "fl_per_sp",
             "avgB", "log_avgB", "maxbl", "log_maxbl"]   # v4:avgB=flop/a_len(docs/66 §8 主轴)+maxbl(LLB 同源)


def load_maxbl(m):
    """全矩阵 maxbl 数组(按矩阵行号);truth 行由 rid 对位取值。"""
    p = os.path.join(REPO, "compare", "d1_oracle", "feat", m + ".maxbl.tsv")
    if not os.path.exists(p):
        return None
    mb = {}
    mx = 0
    for ln in open(p):
        r, v = ln.split()
        ri = int(r)
        mb[ri] = float(v)
        if ri > mx:
            mx = ri
    arr = np.zeros(mx + 1)
    for ri, v in mb.items():
        arr[ri] = v
    return arr


def feats_matrix(d, mat=None):
    """一行的特征矩阵(n×20)。比值特征=三武器单价分岔的物理主轴,dispatch 时可算,无答案泄漏。"""
    r = d["raw"]
    al, fl, es, sp = r["al"], r["fl"], r["es"], r["sp"]
    ht, W = d["feats"]["hash"][:, 0], d["feats"]["dense"][:, 0]
    ctx = np.array(d["ctx"])
    n = len(al)
    dup = fl / np.maximum(es, 1)                      # dup 估计(估计器口径)
    dens = es / np.maximum(sp, 1)                     # 密度(近密=pre2 型标志)
    fps = fl / np.maximum(sp, 1)
    avgB = fl / np.maximum(al, 1)                     # 平均 B 链长(cursor/search 模式主轴)
    mb_full = load_maxbl(mat) if mat else None
    mb = mb_full[d["rid"]] if mb_full is not None else np.zeros(n)   # 最长 B 链(LLB 同源,rid 对位)
    return np.stack([al, fl, es, sp, ht, W,
                     np.log1p(al), np.log1p(fl), np.log1p(np.maximum(es, 1)),
                     np.log1p(sp), np.full(n, ctx[0]), np.full(n, ctx[1]), np.full(n, ctx[2]),
                     dup, dens, fps,
                     avgB, np.log1p(avgB), mb, np.log1p(mb)], 1)


def stratified_mask(d, w, bandcap):
    """v3 分层抽样帽:est 五带 [0,256)/[256,1K)/[1K,4K)/[4K,16K)/[16K,∞) 各限帽——
    治 333SP 型微行洪水(微行带单阵最多贡献 bandcap 行,重行/边际带保底同权)。"""
    import numpy as _np
    idx_all = _np.where(d["legal"][:, WEAPONS.index(w)])[0]
    if len(idx_all) <= bandcap * 5:
        return idx_all
    es = d["raw"]["es"][idx_all]
    bands = _np.searchsorted([256, 1024, 4096, 16384], es)
    keep = []
    for b in range(5):
        sub = idx_all[bands == b]
        if len(sub) > bandcap:
            sub = sub[:bandcap]     # 顺序取(行序=矩阵行序,无偏)
        keep.append(sub)
    return _np.concatenate(keep) if keep else idx_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat", nargs="*",
                    default=["bloweybq", "pre2", "c-43", "Si10H16", "in-2004", "nd24k",
                             "crankseg_2", "F2", "c-64", "bcsstk32", "bcsstk23", "bcsstm23",
                             "bcspwr08", "bcspwr05", "bcsstm01", "1138_bus", "ash219",
                             "pf2177", "HTC_336_9129", "333SP", "opt1", "af_shell4",   # 10-08 补师
                             "bp_0", "arc130"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--cap", type=int, default=400000)
    ap.add_argument("--bandcap", type=int, default=0,
                    help="v3 分层帽:每阵×武器×est带 行数帽(0=关,用旧 per-matrix 帽)")
    ap.add_argument("--exclude", default="",
                    help="消融:排除阵名(逗号分隔,如 333SP)")
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    ex = set(x for x in a.exclude.split(",") if x)
    data = prep([m for m in a.mat if m not in ex])
    for m, d in data.items():
        d["X"] = feats_matrix(d, mat=m)
        d["trainmask"] = {}
        for w in WEAPONS:
            if a.bandcap > 0:
                idx = stratified_mask(d, w, a.bandcap)
            else:
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
