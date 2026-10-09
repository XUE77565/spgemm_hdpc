#!/usr/bin/env python3
"""D3①:部署版教师训练(全数据 margin0.7 全家)+ 学生蒸馏素材导出。

教师 = 4×HGB 回归(逐武器周期)+ 1×HGB 分类(argmin 直选),全 26 阵训练
(LOO 是考试口径;部署吃全部数据)。margin0.7 仲裁规则内嵌导出。
输出 compare/d1_oracle/deploy/teacher.joblib + 学生训练素材(逐行特征+教师决策)。
用法:.venv/bin/python scripts/d3_train_teacher.py [--seed 0] [--iters 300]
"""
import argparse, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from fit_d1_context import prep, WEAPONS
from d6_learned_dispatch import feats_matrix, stratified_mask, FEATNAMES

MATS = ["bloweybq", "pre2", "c-43", "Si10H16", "in-2004", "nd24k", "crankseg_2", "F2", "c-64",
        "bcsstk32", "bcsstk23", "bcsstm23", "bcspwr08", "bcspwr05", "bcsstm01", "1138_bus", "ash219",
        "pf2177", "HTC_336_9129", "333SP", "opt1", "af_shell4", "bp_0", "arc130", "bcsstk30", "bcsstk31",
        "af_2_k101", "crankseg_1", "rajat20", "wb-edu", "3Dspectralwave2", "Ga41As41H72",   # v2 拓广 10
        "TSOPF_FS_b39_c7", "mult_dcop_03", "Ge99H100", "c-58",
        "c-71", "msc23052", "exdata_1", "rajat25", "msc10848", "hugetrace-00020",          # v2 拓广 +14(50 阵)
        "bundle1", "web-BerkStan", "cage13", "M6", "para-6", "offshore", "std1_Jac3", "email-Enron"]   # 第 50 阵(小)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--bandcap", type=int, default=60000)
    a = ap.parse_args()
    os.makedirs(os.path.join(REPO, "compare", "d1_oracle", "deploy"), exist_ok=True)

    data = prep(MATS)
    Xr = {w: [] for w in WEAPONS}
    yr = {w: [] for w in WEAPONS}
    Xc, yc = [], []
    Xall = []
    for m, d in data.items():
        d["X"] = feats_matrix(d, mat=m)
        Xall.append(d["X"])
        for w in WEAPONS:
            idx = stratified_mask(d, w, a.bandcap)
            if len(idx):
                Xr[w].append(d["X"][idx]); yr[w].append(d["cyc"][idx, WEAPONS.index(w)])
        c2t = np.where(d["legal"], d["cyc"], np.inf)
        st = (np.isfinite(c2t).sum(1) >= 2) & (c2t.min(1) < np.inf)
        if st.any():
            Xc.append(d["X"][st]); yc.append(c2t[st].argmin(1))

    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    teacher = {"featnames": FEATNAMES, "weapons": WEAPONS, "seed": a.seed, "mats": MATS,
               "rule": "margin0.7"}
    for w in WEAPONS:
        mdl = HistGradientBoostingRegressor(max_iter=a.iters, learning_rate=0.08,
                                            early_stopping=True, validation_fraction=0.1,
                                            random_state=a.seed)
        mdl.fit(np.concatenate(Xr[w]), np.concatenate(yr[w]))
        teacher[f"reg_{w}"] = mdl
        print(f"reg_{w} trained (n={sum(len(x) for x in Xr[w])})", flush=True)
    clf = HistGradientBoostingClassifier(max_iter=a.iters, learning_rate=0.08,
                                          early_stopping=True, validation_fraction=0.1,
                                          random_state=a.seed)
    clf.fit(np.concatenate(Xc), np.concatenate(yc))
    teacher["clf"] = clf
    print(f"clf trained (n={sum(len(x) for x in Xc)}, classes={clf.classes_})", flush=True)

    out = os.path.join(REPO, "compare", "d1_oracle", "deploy", "teacher.joblib")
    joblib.dump(teacher, out)
    print(f"teacher → {out}")

    # 学生素材:全行特征 + 教师逐行决策(margin0.7)导出
    picks = []
    for m, d in data.items():
        pr = np.stack([teacher[f"reg_{w}"].predict(d["X"]) for w in WEAPONS], 1)
        pb = np.full((d["n"], 4), np.nan)
        pb[:, teacher["clf"].classes_] = teacher["clf"].predict_proba(d["X"])
        cand = d["legal"] & np.isfinite(pr) & np.isfinite(pb)
        pick_reg = np.where(cand, pr, np.inf).argmin(1)
        pick_am = np.where(cand, pb, -np.inf).argmax(1)
        pmax = np.where(cand, pb, -np.inf).max(1)
        pick = np.where((pick_reg != pick_am) & (pmax >= 0.7), pick_am, pick_reg)
        pick[cand.sum(1) == 0] = -1
        picks.append(np.column_stack([np.full(d["n"], MATS.index(m)), d["rid"], pick]))
    P = np.concatenate(picks)
    np.savetxt(os.path.join(REPO, "compare", "d1_oracle", "deploy", "teacher_picks.tsv"),
               P, fmt="%d", header="mat_idx rid teacher_pick", comments="")
    X = np.concatenate(Xall)
    np.save(os.path.join(REPO, "compare", "d1_oracle", "deploy", "X_all.npy"), X.astype(np.float32))
    print(f"student material: X {X.shape} + picks {P.shape}")


if __name__ == "__main__":
    main()
