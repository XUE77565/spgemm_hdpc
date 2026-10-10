#!/usr/bin/env python3
"""L0 训练(study/37 第二步):矩阵级配置选择分类器。

标签:l0_data.csv 中每阵 net_ms 最小的配置(A/B/C/D)
特征:阵级统计量(从 R 臂 DUMP_ROUTING 全行提取:log rows/log Σflop/log mean est/
      est 分位数/maxbl 均值/dup 均值/span 均值/各 bin 人口占比)
模型:GBT(46 样本 4 类,特征 ~12 维;LOO 验证)
输出:compare/l0_calibration/l0_model.joblib + LOO 准确率
"""
import csv, math, os, struct, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from fit_d1_costlaws import load
from d3_train_teacher import MATS

CAL = os.path.join(REPO, "compare", "l0_calibration")

CONFIGS = ["A_allhash", "B_allhash_k7", "C_default", "D_default_k7"]


def load_labels():
    from collections import defaultdict
    data = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(CAL, "l0_data.csv"))):
        if r["net_ms"] != "DNF":
            data[r["mat"]][r["config"]] = float(r["net_ms"])
    labels = {}
    for m, d in data.items():
        if len(d) >= 2:
            labels[m] = min(d, key=d.get)
    return labels


def extract_features(mat):
    """从 R 臂 DUMP_ROUTING 提取阵级统计量(12 维)。"""
    p = os.path.join(REPO, "compare", "d1_oracle", f"{mat}.R1.routing.bin")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        rows, _ = struct.unpack("2i", f.read(8))
        est = np.array(struct.unpack(f"{rows}i", f.read(rows * 4)), dtype=float)
        span = np.array(struct.unpack(f"{rows}i", f.read(rows * 4)), dtype=float)
        flop = np.array(struct.unpack(f"{rows}i", f.read(rows * 4)), dtype=float)
        bkid = np.array(struct.unpack(f"{rows}i", f.read(rows * 4)))
        alen = np.array(struct.unpack(f"{rows}i", f.read(rows * 4)), dtype=float)
    # maxbl 近似:avgB(=flop/alen 的均值/中位)
    avgB = flop / np.maximum(alen, 1)
    dup = flop / np.maximum(est, 1)
    from collections import Counter
    bc = Counter(bkid.tolist())
    def frac(*bids):
        return sum(bc.get(b, 0) for b in bids) / rows
    return [
        math.log(rows), math.log(max(1, flop.sum())), math.log(max(1, est.mean())),
        math.log(max(1, est.std() + 1)), math.log(max(1, span.mean())),
        math.log(max(1, np.median(avgB) + 1)), math.log(max(1, np.median(dup) + 1)),
        frac(0,1,2,3,4,5,6,7,8,9,10,12),   # hash 族占比
        frac(13), frac(15),                    # diter/k7 占比
        frac(21,22,23,19,14,20,11),           # micro 族占比
        math.log(max(1, flop.mean())),
    ]


def main():
    labels = load_labels()
    mats, X, y = [], [], []
    cfg_idx = {c: i for i, c in enumerate(CONFIGS)}
    for m in sorted(labels):
        feat = extract_features(m)
        if feat is None:
            continue
        mats.append(m); X.append(feat); y.append(cfg_idx[labels[m]])
    X, y = np.array(X), np.array(y)
    print(f"训练集:{len(mats)} 阵,{X.shape[1]} 维特征")
    print(f"标签分布:{ {CONFIGS[i]: int((y==i).sum()) for i in range(4)} }")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import LeaveOneOut
    # LOO 验证
    correct = 0
    loo_preds = {}
    for tr, te in LeaveOneOut().split(X):
        clf = HistGradientBoostingClassifier(max_iter=50, random_state=0)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])[0]
        loo_preds[mats[te[0]]] = (CONFIGS[pred], CONFIGS[y[te[0]]])
        if pred == y[te[0]]:
            correct += 1
    print(f"\nLOO 配置选择准确率: {correct}/{len(mats)} = {correct/len(mats):.1%}")
    print("\n错判清单(预测→实际):")
    for m, (p, t) in loo_preds.items():
        if p != t:
            print(f"  {m:<18} {p:<16} → {t}")
    # 全量重训存模型
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=0)
    clf.fit(X, y)
    import joblib
    joblib.dump({"model": clf, "mats": mats, "configs": CONFIGS, "features": "R1_routing_stats12"},
                os.path.join(CAL, "l0_model.joblib"))
    print(f"\n模型 → {CAL}/l0_model.joblib")


if __name__ == "__main__":
    main()
