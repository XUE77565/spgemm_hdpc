#!/usr/bin/env python3
"""L0 选择器训练 v1(10-16,study/46):特征→臂选择,验收=ocean 自证协议四指标。

选择集 = {C, CD, A, ADPT}(ADPT 兼任 merge3 路由代表;批437 唯一数据源);
模型 = GBT(浅)+ 逻辑回归基线;验证 = 矩阵级 5 折 CV + 跨种群(337↔first100);
硬门 = 无阵 vs A(最佳单臂)回归 >1.10;目标 = 选择器 GM 逼近 oracle(≤1.5%)。
"""
import csv, math, os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATS = ["n", "nnz", "avg_row", "max_row", "skew", "row_entropy", "row_gini", "p90p50", "flop_proxy"]
LOGF = ["n", "nnz", "avg_row", "max_row", "flop_proxy"]
ARMS = ["C", "CD", "A", "ADPT"]

def load():
    fx = {r["mat"]: r for r in csv.DictReader(open(os.path.join(REPO, "compare/batch437/features.csv")))}
    tx = {}
    for r in csv.DictReader(open(os.path.join(REPO, "compare/batch437/batch_437.csv"))):
        tx.setdefault(r["mat"], {})[r["arm"]] = float(r["ms"]) if r["ms"] else None
    o337 = [r["mat"] for r in csv.DictReader(open(os.path.join(REPO, "compare/l0_337/l0_337_results.csv")))]
    seen = set(o337)
    july = [r["matrix"] for r in csv.DictReader(open(os.path.join(REPO, "compare/methods_cmp_first100.csv")))]
    f100 = [m for m in july if m not in seen and os.path.exists(os.path.join(REPO, "data/first100", m + ".mtx"))]
    data = []
    for pop, mats in (("337", o337), ("f100", f100)):
        for m in mats:
            if m not in fx or m not in tx: continue
            row = [float(fx[m][k]) for k in FEATS]
            t = tx[m]
            if not all(t.get(a) for a in ARMS): continue
            data.append((m, pop, row, {a: t[a] for a in ARMS}, t.get("OCN")))
    return data

def X(d):  return np.array([logify(r[2]) for r in d])
def logify(row):
    v = list(row)
    for i, k in enumerate(FEATS):
        if k in LOGF: v[i] = math.log10(v[i] + 1e-9)
    return v

def evaluate(data, pred_arm, tag):
    """ocean 自证协议四指标 + GM(指标1/2 条件版:只统计差距≥5%存在的阵,ocean 原口径)"""
    chosen_t, next_t, reg = [], [], []
    ratios_ocn = []
    for (m, pop, feats, t, ocn), pa in zip(data, pred_arm):
        best_alt = min(t[a] for a in ARMS if a != pa)
        chosen_t.append(t[pa]); next_t.append(best_alt)
        if ocn: ratios_ocn.append(t[pa] / ocn)
        reg.append(t[pa] / t["A"])
    chosen_t, next_t = np.array(chosen_t), np.array(next_t)
    gap = np.maximum(next_t / chosen_t, chosen_t / next_t)
    big = gap >= 1.05                                             # 差距显著子集
    m1 = np.mean(next_t[big] / chosen_t[big] >= 1.05) if big.any() else 1.0
    m2 = np.mean(chosen_t[~big] / next_t[~big] >= 1.05) if (~big).any() else 0.0
    m3 = np.max(reg)                                             # vs A 回归硬门
    gm = math.exp(np.mean(np.log(ratios_ocn))) if ratios_ocn else None
    # oracle 参照
    ogm = math.exp(np.mean([math.log(min(t[a] for a in ARMS) / ocn) for (_, _, _, t, ocn) in data if ocn]))
    print(f"[{tag}] 指标1 选中获益≥1.05×: {m1*100:.1f}% | 指标2 反向: {m2*100:.1f}% | "
          f"指标3 vs A 最大回归: {m3:.3f} | 指标4 选择器/OCN GM={gm:.4f} (oracle={ogm:.4f}, 差距 {(gm/ogm-1)*100:+.2f}%)")
    return gm, ogm

def run_cv(data, make_model, tag, guard=0.5):
    """guard:置信低于此值→保守回退 A(三层哲学:学习层不确定→退安全默认)"""
    y = [int(np.argmin([t[a] for a in ARMS])) for (_, _, _, t, _) in data]
    pred = [None] * len(data)
    for tr, te in KFold(5, shuffle=True, random_state=42).split(range(len(data))):
        mdl = make_model()
        mdl.fit(X([data[i] for i in tr]), [y[i] for i in tr])
        P = mdl.predict_proba(X([data[i] for i in te]))
        cls = mdl.classes_
        for j, i in enumerate(te):
            k = int(np.argmax(P[j]))
            pred[i] = cls[k] if P[j, k] >= guard else 3   # 3 = ARMS.index('A')
    pred_arm = [ARMS[int(i)] for i in pred]
    acc = accuracy_score(y, pred)
    print(f"[{tag}] CV 臂预测准确率 {acc*100:.1f}%")
    evaluate(data, pred_arm, tag)

def cross_pop(data, tag, guard=0.5):
    tr = [d for d in data if d[1] == "337"]; te = [d for d in data if d[1] == "f100"]
    y = [int(np.argmin([t[a] for a in ARMS])) for (_, _, _, t, _) in tr]
    mdl = HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=42)
    mdl.fit(X(tr), y)
    P = mdl.predict_proba(X(te)); cls = mdl.classes_
    pred = [cls[int(np.argmax(P[j]))] if P[j, int(np.argmax(P[j]))] >= guard else "A" for j in range(len(te))]
    evaluate(te, pred, tag + "(337→f100,guard)")

def main():
    data = load()
    print(f"样本 n={len(data)}(337:{sum(1 for d in data if d[1]=='337')} / f100:{sum(1 for d in data if d[1]=='f100')})\n")
    # 基线:恒 A / 恒 ADPT
    evaluate(data, ["A"] * len(data), "恒A基线")
    evaluate(data, ["ADPT"] * len(data), "恒ADPT基线")
    print()
    run_cv(data, lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=42), "GBT-CV")
    run_cv(data, lambda: LogisticRegression(max_iter=2000, C=0.5), "LR-CV")
    cross_pop(data, "跨种群")

if __name__ == "__main__":
    main()
