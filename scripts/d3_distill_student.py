#!/usr/bin/env python3
"""D3①:学生蒸馏与保真度考核(部署形态竞标)。

标签语义(D3 v1 部署边界):
  争夺区 = bkid_def ∈ {0-10 hash 梯, 13 diter, 15 k7} → 学教师的三路裁决(hash/dense/k7)
  其余(micro/heavy/largest/sspan/esc/batch)→ 3 = passthrough(原静态 bin 放行)
学生:
  A. DecisionTreeClassifier depth=8(511 节点,C 数组直译,kernel 内 ~8 次比较/行)
  B. MLPClassifier 21→16→4(标准缩放;~400 FLOPs/行)
考核:GroupKFold 按阵 5 折,学生 vs 教师逐行一致率;验收线 ≥95%(全折)。
胜者导出 deploy/student_export.(h|cuh)。
用法:.venv/bin/python scripts/d3_distill_student.py
"""
import os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from fit_d1_costlaws import load
from d3_train_teacher import MATS
from d6_learned_dispatch import FEATNAMES

DEPLOY = os.path.join(REPO, "compare", "d1_oracle", "deploy")
CONTESTED = set(range(0, 11)) | {13, 15}


def mat_ncols():
    """各阵方阵阶数(读 mtx 头;L0 全稠密规则用 est/n)。"""
    import re
    nc = {}
    for m in MATS:
        d = "data/first100" if m in ("bcsstk32", "bcsstk23", "bcsstm23", "bcspwr08", "bcspwr05",
                                     "bcsstm01", "1138_bus", "ash219", "bp_0", "arc130",
                                     "bcsstk30", "bcsstk31") else "data/ocean/square"
        with open(os.path.join(REPO, d, m + ".mtx")) as f:
            line = f.readline()
            while line and (line[0] == "%" or line.startswith("%%")):
                line = f.readline()
            nc[m] = int(line.split()[1])
    return nc


def build_labels():
    """部署语义标签:教师模型的**无条件特征决策**(候选=特征合法性,三路+passthrough)。
    教师 pick 文件的测量掩码口径不用于蒸馏(同类特征跨阵矛盾标签的病根)。"""
    import joblib
    X = np.load(os.path.join(DEPLOY, "X_all.npy"))
    picks = np.loadtxt(os.path.join(DEPLOY, "teacher_picks.tsv"), skiprows=1, dtype=np.int64)
    mat_idx, rid = picks[:, 0], picks[:, 1]
    bdef = {}
    for t in load(MATS):
        bdef[(t[0], int(t[1]))] = int(t[8])
    contested = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        contested[i] = bdef.get((MATS[mat_idx[i]], int(rid[i])), -1) in CONTESTED
    tch = joblib.load(os.path.join(DEPLOY, "teacher.joblib"))
    W = tch["weapons"]
    pr3 = np.stack([tch[f"reg_{w}"].predict(X) for w in ("hash", "dense", "k7")], 1)
    pb = np.full((len(X), 4), np.nan)
    pb[:, tch["clf"].classes_] = tch["clf"].predict_proba(X)
    # 特征合法性(与路由核谓词同源):dense=档内或 avgB≥64;k7=a_len∈[1,32]∧span≤32768∧est≥256∧collision≤512∧flop≤1.67est∧16flop≥span∧16est≥span
    al, fl, es, sp = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    avgB, coll, dup = X[:, 16], X[:, 20], X[:, 13]
    dense_ok = (sp <= 7277) | (avgB >= 64)
    k7_ok = ((al >= 1) & (al <= 32) & (sp > 0) & (sp <= 32768) & (es >= 256)
             & (coll <= 512) & (dup <= 1.67) & (16 * fl >= sp) & (16 * es >= sp))
    cand = np.column_stack([np.ones(len(X), bool), dense_ok, k7_ok])   # hash 恒候选
    pr_masked = np.where(cand, pr3, np.inf)
    pb_masked = np.where(cand, pb[:, :3], -np.inf)                    # 三路概率(mrg3 出局)
    pick_reg = pr_masked.argmin(1)
    pick_am = pb_masked.argmax(1)
    pmax = pb_masked.max(1)
    y = np.where((pick_reg != pick_am) & (pmax >= 0.7), pick_am, pick_reg)
    y[~cand.any(1)] = 0
    # L0 全稠密规则(D3① 病理处方):est/n≥0.9 = C 全稠密 = 位图主场,k7 谓词合法即 k7
    # (dense 回归器此域无训练样本=瞎外推,bloweybq 病理;k7 实测 4× 快于矩阵累积器)
    NC = mat_ncols()
    nc_arr = np.array([NC[MATS[m]] for m in mat_idx], dtype=float)
    # 架构分界(D3① v2):确定性规则走代码、不进学生——
    #   ① micro bin 放行(现有路由结构)② L0 全稠密规则(est/n≥0.9∧k7_ok→k7)
    # 学生只学【争夺区 ∧ 非规则行】的三路 {hash,dense,k7};规则行从训练/考核中剔除
    l0_dense = (X[:, 2] / nc_arr >= 0.9) & k7_ok
    keep = contested & ~l0_dense & (y != 3)
    return X[keep], y[keep], mat_idx[keep], l0_dense


def export_tree(dt, path):
    """sklearn 树 → C 头文件(constant 数组:feature/threshold/child左/child右/leaf 类)。"""
    t = dt.tree_
    n = t.node_count

    def leaf_class(i):
        return int(np.argmax(t.value[i][0]))
    feat, thr, cl, cr, leaf = [], [], [], [], []
    for i in range(n):
        if t.children_left[i] == -1:
            feat.append(0); thr.append(0.0); cl.append(-1); cr.append(-1); leaf.append(leaf_class(i))
        else:
            feat.append(int(t.feature[i])); thr.append(float(t.threshold[i]))
            cl.append(int(t.children_left[i])); cr.append(int(t.children_right[i])); leaf.append(-1)
    with open(path, "w") as f:
        f.write("// D3 学生导出:DecisionTree depth-8(margin0.7 教师蒸馏)\n")
        f.write(f"// 特征序({len(FEATNAMES)}):{','.join(FEATNAMES)}\n")
        f.write(f"#define STU_NNODES {n}\n")
        f.write("__device__ constant int stu_feat[STU_NNODES] = {" + ",".join(map(str, feat)) + "};\n")
        f.write("__device__ constant float stu_thr[STU_NNODES] = {" + ",".join(f"{v:.6f}f" for v in thr) + "};\n")
        f.write("__device__ constant int stu_cl[STU_NNODES] = {" + ",".join(map(str, cl)) + "};\n")
        f.write("__device__ constant int stu_cr[STU_NNODES] = {" + ",".join(map(str, cr)) + "};\n")
        f.write("__device__ constant int stu_leaf[STU_NNODES] = {" + ",".join(map(str, leaf)) + "};\n")
        f.write("// 推理:node=0; while(stu_leaf[node]<0) node = x[stu_feat[node]]<=stu_thr[node]?stu_cl[node]:stu_cr[node]; pick=stu_leaf[node];\n")
    print(f"tree exported → {path} ({n} nodes)")


def main():
    X, y, g, l0 = build_labels()
    print(f"学生素材(争夺区三路,规则行已剔除){X.shape},标签分布: "
          + str({int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})
          + f";L0 规则行={int(l0.sum())}")
    from sklearn.model_selection import GroupKFold
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    gk = GroupKFold(n_splits=5)
    res = {f"tree_d{d}": [] for d in (8, 10, 12)}
    res["gbt4x6"] = []
    for k, (tr, te) in enumerate(gk.split(X, y, groups=g)):
        line = f"fold{k}:"
        for d in (8, 10, 12):
            dt = DecisionTreeClassifier(max_depth=d, random_state=0)
            dt.fit(X[tr], y[tr])
            a = (dt.predict(X[te]) == y[te]).mean()
            res[f"tree_d{d}"].append(a)
            line += f"  d{d}={a:.4f}"
        gb = GradientBoostingClassifier(n_estimators=4, max_depth=6, learning_rate=0.3,
                                        random_state=0)
        gb.fit(X[tr], y[tr])
        a_g = (gb.predict(X[te]) == y[te]).mean()
        res["gbt4x6"].append(a_g)
        line += f"  gbt4x6={a_g:.4f}"
        print(line, flush=True)
    best_d, best_worst = 8, 0.0
    for nm, v in res.items():
        worst = min(v)
        print(f"{nm}: 5 折保真度 中位 {np.median(v):.4f} 最差 {worst:.4f}  判 {'✅≥0.95' if worst >= 0.95 else '❌<0.95'}")
        if worst > best_worst and nm.startswith("tree_d"):
            best_d, best_worst = int(nm.split("_d")[1]), worst
    dt = DecisionTreeClassifier(max_depth=best_d, random_state=0).fit(X, y)
    export_tree(dt, os.path.join(DEPLOY, "student_export.cuh"))
    print(f"胜者:tree_d{best_d}(全量重训+导出,最差折保真度 {best_worst:.4f})")


if __name__ == "__main__":
    main()
