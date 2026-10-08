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
             "avgB", "log_avgB", "maxbl", "log_maxbl",
             "collision"]   # v5:flop−est(k7 fixup 尾/原子争用物理量)


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
    coll = np.maximum(fl - es, 0)                     # 碰撞数估计(k7 fixup 尾;dense 原子争用)
    return np.stack([al, fl, es, sp, ht, W,
                     np.log1p(al), np.log1p(fl), np.log1p(np.maximum(es, 1)),
                     np.log1p(sp), np.full(n, ctx[0]), np.full(n, ctx[1]), np.full(n, ctx[2]),
                     dup, dens, fps,
                     avgB, np.log1p(avgB), mb, np.log1p(mb), coll], 1)


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
    ap.add_argument("--mode", choices=["reg", "argmax", "ens"], default="reg",
                    help="v5:reg=4 回归器 argmin;argmax=分类器直选;ens=双模型+分歧仲裁(D2 收口)")
    ap.add_argument("--seed", type=int, default=0, help="HGB random_state(刀锋阵多种子中位用)")
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
    # ens 仲裁规则账本:rule → [对, 判行, Σ选, Σ优];另记三路口径(mrg3 出局)
    RULES = ["reg", "argmax", "margin0.7", "gap1.2"]
    ens_acc = {r: [0, 0, 0.0, 0.0] for r in RULES}
    ens3_acc = {r: [0, 0, 0.0, 0.0] for r in RULES}
    ens_per = {}
    for hold in mats:
        if a.mode == "ens":
            # ── 双模型同折训练 ──
            Xtr, ytr = {w: [] for w in WEAPONS}, {w: [] for w in WEAPONS}
            Xc, yc = [], []
            for m, d0 in data.items():
                if m == hold:
                    continue
                for w in WEAPONS:
                    idx = d0["trainmask"][w]
                    if len(idx):
                        Xtr[w].append(d0["X"][idx]); ytr[w].append(d0["cyc"][idx, WEAPONS.index(w)])
                c2t = np.where(d0["legal"], d0["cyc"], np.inf)
                st = (np.isfinite(c2t).sum(1) >= 2) & (c2t.min(1) < np.inf)
                if st.any():
                    Xc.append(d0["X"][st]); yc.append(c2t[st].argmin(1))
            models = {}
            for w in WEAPONS:
                X = np.concatenate(Xtr[w]); y = np.concatenate(ytr[w])
                mdl = HistGradientBoostingRegressor(max_iter=a.iters, learning_rate=0.1,
                                                     early_stopping=True, validation_fraction=0.1,
                                                     random_state=a.seed, verbose=0)
                mdl.fit(X, y)
                models[w] = mdl
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(max_iter=a.iters, learning_rate=0.1,
                                                  early_stopping=True, validation_fraction=0.1,
                                                  random_state=a.seed)
            clf.fit(np.concatenate(Xc), np.concatenate(yc))
            d = data[hold]
            pr = np.stack([models[w].predict(d["X"]) for w in WEAPONS], 1)   # 成本
            pb = np.zeros((d["n"], 4)); pb[:] = np.nan
            pb[:, clf.classes_] = clf.predict_proba(d["X"])                   # 概率
            cand = d["legal"] & np.isfinite(pr) & np.isfinite(pb)
            sel = cand.sum(1) >= 2
            cs, ps, bs = cand[sel], pr[sel], pb[sel]
            c2 = np.where(cs, d["cyc"][sel], np.inf)
            oracle = c2.argmin(1)
            pick_reg = np.where(cs, ps, np.inf).argmin(1)
            pick_am = np.where(cs, bs, -np.inf).argmax(1)
            # 分歧行上的两个仲裁信号
            pmax = np.where(cs, bs, -np.inf).max(1)                            # argmax 置信
            pr_sort = np.sort(np.where(cs, ps, np.inf), 1)
            near = (pr_sort[:, 1] <= 1.2 * np.maximum(pr_sort[:, 0], 1e-9))    # reg 认为前二接近
            disagree = pick_reg != pick_am
            picks = {"reg": pick_reg, "argmax": pick_am}
            picks["margin0.7"] = np.where(disagree & (pmax >= 0.7), pick_am, pick_reg)
            picks["gap1.2"] = np.where(disagree & near, pick_am, pick_reg)
            ag_rate = 1 - disagree.mean()
            line = f"  [ens] {hold:<12} 判行 {int(sel.sum())} 双模一致 {ag_rate:.0%} | "
            for r in RULES:
                pk = picks[r]
                ok = pk == oracle
                s_ = float(np.take_along_axis(c2, pk[:, None], 1).sum())
                o_ = float(np.take_along_axis(c2, oracle[:, None], 1).sum())
                A = ens_acc[r]; A[0] += int(ok.sum()); A[1] += int(sel.sum()); A[2] += s_; A[3] += o_
                line += f"{r}:{ok.mean():.0%} "
                ens_per.setdefault((hold, r), int(ok.sum()))
            # 三路口径(mrg3 出局——架构修正后的诚实门值)
            cs3 = cs.copy(); cs3[:, 3] = False
            i3 = np.where(cs3.sum(1) >= 2)[0]          # sel 子集内的行号
            if len(i3):
                c3 = np.where(cs3, d["cyc"][sel], np.inf)[i3]
                oracle3 = c3.argmin(1)
                ps3 = np.where(cs3, ps, np.inf)[i3]
                bs3 = np.where(cs3, bs, -np.inf)[i3]
                pick_reg3 = ps3.argmin(1); pick_am3 = bs3.argmax(1)
                dis3 = pick_reg3 != pick_am3
                pm3 = bs3.max(1)
                psrt3 = np.sort(ps3, 1)
                near3 = psrt3[:, 1] <= 1.2 * np.maximum(psrt3[:, 0], 1e-9)
                p3 = {"reg": pick_reg3, "argmax": pick_am3}
                p3["margin0.7"] = np.where(dis3 & (pm3 >= 0.7), pick_am3, pick_reg3)
                p3["gap1.2"] = np.where(dis3 & near3, pick_am3, pick_reg3)
                for r in RULES:
                    ok3 = p3[r] == oracle3
                    s3_ = float(np.take_along_axis(c3, p3[r][:, None], 1).sum())
                    o3_ = float(np.take_along_axis(c3, oracle3[:, None], 1).sum())
                    A = ens3_acc[r]; A[0] += int(ok3.sum()); A[1] += len(i3); A[2] += s3_; A[3] += o3_
            t3s = f"{(pick_reg3==oracle3).mean():.0%}" if len(i3) else "n/a"
            print(line + "| 三路reg:" + t3s, flush=True)
            continue
        if a.mode == "reg":
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
            d = data[hold]
            pred = np.stack([models[w].predict(d["X"]) if models[w] is not None
                             else np.full(d["n"], np.nan) for w in WEAPONS], 1)
        else:   # argmax:多类分类器(训练行=≥2 候选,y=实测 argmin)
            from sklearn.ensemble import HistGradientBoostingClassifier
            Xs, ys = [], []
            for m, d in data.items():
                if m == hold:
                    continue
                c2 = np.where(d["legal"], d["cyc"], np.inf)
                sel = (np.isfinite(c2).sum(1) >= 2) & (c2.min(1) < np.inf)
                if sel.any():
                    Xs.append(d["X"][sel]); ys.append(c2[sel].argmin(1))
            clf = HistGradientBoostingClassifier(max_iter=a.iters, learning_rate=0.1,
                                                  early_stopping=True, validation_fraction=0.1,
                                                  random_state=0)
            clf.fit(np.concatenate(Xs), np.concatenate(ys))
            d = data[hold]
            pr = np.zeros((d["n"], 4))
            pr[:] = np.nan
            pr[:, clf.classes_] = clf.predict_proba(d["X"])
            pred = pr   # 概率当"负成本"用:越大越优
        cand = d["legal"] & np.isfinite(pred)
        if a.mode == "argmax":   # cand 语义翻转:概率高者优
            pred = -pred
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
    if a.mode == "ens":
        print("\n== D2 收口:集成仲裁规则汇总(四路口径含 mrg3 | 三路口径=架构修正后门值)==")
        for r in RULES:
            A, B = ens_acc[r], ens3_acc[r]
            print(f"  {r:<10} 四路 {A[0]}/{A[1]} = {A[0]/max(1,A[1]):.1%}  gap {A[2]/max(1,A[3]):.4f}"
                  f"   |   三路 {B[0]}/{B[1]} = {B[0]/max(1,B[1]):.1%}  gap {B[2]/max(1,B[3]):.4f}")
        return
    print(f"\n== D6 LOO 汇总:一致率 {tot_ag}/{tot_n} = {tot_ag/max(1,tot_n):.1%}  "
          f"oracle-gap {gn/gd:.4f} ==")
    print("(对照:v3b 解析律池化 45.6%/gap 6.66;v4 λ 上限 in-sample 57.6%/1.095)")


if __name__ == "__main__":
    main()
