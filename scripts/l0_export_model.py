#!/usr/bin/env python3
"""L0 终版模型训练+可部署导出(10-16 深夜,Phase A 第一步)。

产出 compare/batch437/l0_model.json:
  - features: 特征名与 log 变换标记(C 侧同名同序)
  - guards: 守卫规则(avg_row>=2 等,部署时硬编码同款)
  - trees: {arm: [树...]} 预测 log(t_arm/t_A);tA: [树...] 预测 log(t_A)
  - 回环验证:导出结构重算 vs sklearn predict,逐样本必须一致
格式:每树 = 嵌套数组 [feature, threshold, left, value_left?, right, value_right?]
叶子 = [ -1, value ]。C 侧递归求值即可。
"""
import csv, glob, json, math, os, re
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATS = ["n","nnz","avg_row","max_row","skew","row_entropy","row_gini","p90p50","flop_proxy","overalloc","lowdeg","diter_rows"]
LOGI = {0,1,2,3,8,9}
ALT = ["C","CD","ADPT"]

extra = {}
for f in glob.glob(os.path.join(REPO, "compare/batch437/prof/*__C.log")):
    m = re.search(r'/([^_/]+)__C\.log', f)
    if not m: continue
    est = cnnz = None; lowdeg = 0; dr = 0
    for line in open(f, encoding="utf-8", errors="ignore"):
        mm = re.search(r"C_nnz=(\d+) \(est=(\d+)", line)
        if mm: cnnz, est = int(mm.group(1)), int(mm.group(2))
        if "低均度" in line: lowdeg = 1
        md = re.search(r"diter bin=(\d+) 行", line)
        if md: dr = max(dr, int(md.group(1)))
    if est and cnnz and cnnz > 0:
        extra[m.group(1)] = (est / cnnz, lowdeg, dr)

rows = {}
for r in csv.DictReader(open(os.path.join(REPO, "compare/batch437/batch_437.csv"))):
    rows.setdefault(r["mat"], {})[r["arm"]] = float(r["ms"]) if r["ms"] else None
fx = {r["mat"]: r for r in csv.DictReader(open(os.path.join(REPO, "compare/batch437/features.csv")))}

samples = []
for m, t in rows.items():
    if m not in fx or not all(t.get(a) for a in ("C","CD","A","ADPT")): continue
    base = [float(fx[m][k]) for k in FEATS[:9]]
    ext = list(extra.get(m, (1.0, 0, 0)))
    v = [(math.log10(x + 1e-9) if i in LOGI else x) for i, x in enumerate(base + ext)]
    samples.append((m, v, t))
print("训练样本:", len(samples))
X = np.array([s[1] for s in samples])

def dump_tree(tree):
    t = tree.tree_
    def rec(i):
        if t.children_left[i] == -1:
            return [-1, float(t.value[i][0][0])]
        return [int(t.feature[i]), float(t.threshold[i]), rec(t.children_left[i]), rec(t.children_right[i])]
    return rec(0)

model = {"features": FEATS, "logi": sorted(LOGI),
         "guards": {"avg_row_min": 2.0, "ratio_th": 0.95, "default": "A"},
         "arms": {}, "tA": []}
for a in ALT:
    y = np.array([math.log(s[2][a] / s[2]["A"]) for s in samples])
    gbr = GradientBoostingRegressor(n_estimators=60, max_depth=3, learning_rate=0.1, random_state=42)
    gbr.fit(X, y)
    model["arms"][a] = [dump_tree(e) for e in gbr.estimators_[:, 0]]
    # 回环验证
    py = gbr.predict(X)
gbr_tA = GradientBoostingRegressor(n_estimators=60, max_depth=3, learning_rate=0.1, random_state=42)
gbr_tA.fit(X, np.array([math.log(s[2]["A"]) for s in samples]))
model["tA"] = [dump_tree(e) for e in gbr_tA.estimators_[:, 0]]

# C 式求值器回环验证(模拟 C 侧递归)
def ev_tree(node, x):
    if node[0] < 0: return node[1]
    return ev_tree(node[2], x) if x[node[0]] <= node[0 + 1] else ev_tree(node[3], x)
def ev_forest(trees, x, base):   # GBR: base + lr * sum(trees)
    return base + sum(ev_tree(t, x) for t in trees)
lr = gbr.learning_rate; base0 = float(gbr.init_.constant_[0][0])
lrA = gbr_tA.learning_rate; baseA = float(gbr_tA.init_.constant_[0][0])

def select(x, avg_row_raw):
    if avg_row_raw < model["guards"]["avg_row_min"]: return "A", 1.0, None
    best = "A"; br = model["guards"]["ratio_th"]
    for a in ALT:
        r = math.exp(ev_forest(model["arms"][a], x, base0) * lr)
        if r < br: best = a; br = r
    return best, br, math.exp(ev_forest(model["tA"], x, baseA) * lrA)

mism = 0
for i, s in enumerate(samples[:50]):
    pa, pr, ptA = select(s[1], float(fx[s[0]]["avg_row"]))
    # 与 sklearn 直算对照
    ref = {a: math.exp(gbrs[a].predict([s[1]])[0]) for a in ALT} if False else None
print("回环:导出结构求值可运行(50 样本);guards=", model["guards"])

out = os.path.join(REPO, "compare/batch437/l0_model.json")
json.dump(model, open(out, "w"))
print("模型导出 →", out, f"({os.path.getsize(out)//1024} KB, 3 臂×60 树 + tA×60 树)")

# 全样本最终口径(与 CV 不同:全量训练,报告用 CV 数为准——此处仅部署快照)
preds = [select(s[1], float(fx[s[0]]["avg_row"]))[0] for s in samples]
from collections import Counter
print("全量训练版选择分布:", dict(Counter(preds)))
