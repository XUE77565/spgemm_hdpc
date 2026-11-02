#!/usr/bin/env python3
"""
真·漂移实验(双二进制):编译层内核变化 → 静态路由作废 → 在线模型恢复

漂移源 = 我们自己的双 TU 资产(真内核变化,非模拟):
  旧世界 = spgemm_test_sm90(hash_heavy 原生 sm_90 编译 = 判死刑时代,最优=H45 off)
  新世界 = spgemm_test_c61  (hash_heavy compute_61 编译 = 复活时代,最优=H45 on)

三阶段:
  Phase 1: 旧世界逐阵测 H45 on/off → 旧世界最优路由 + 模型在旧世界在线学习
  Phase 2: 新世界逐阵测 H45 on/off → 新世界最优(= 漂移后的 ground truth)
  Phase 3: 模型(带着旧世界先验)流入新世界观测流,在线 SGD 更新
           → 路由准确率恢复曲线:早期错(旧先验)→ 后期对(已适应)

对照臂:
  static: 旧世界标定的静态表(= if-else/离线查表在内核变化后的下场)
  online: 我们的在线模型
  always_on / always_off: 平凡臂
"""
import csv, json, math, os, re, subprocess, sys
import numpy as np
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# bin45 人口的快速阵(今日消融数据取,按耗时升序)
DRIFT_MATS = [
    "patents", "af_shell1", "af_shell3", "af_shell4", "af_shell2",
    "kkt_power", "Ga10As10H30", "para-6", "para-9", "para-7",
    "para-8", "para-5", "Fault_639", "Ga19As19H42",
]

def run_one(binary, mat_path, h45, gpu="0"):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, METHOD="adaptive",
               WARMUP_N="3", USE_MEMPOOL="1", MP_HOST_MB="8192", D6EXP="0",
               **({"HASH_H45": "1", "HASH_H55": "1"} if h45 else {"HASH_H45": "0", "HASH_H55": "0"}))
    try:
        r = subprocess.run([f"./{binary}", mat_path], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
        out = r.stdout + r.stderr
        tot = re.findall(r"TOTAL\(GPU\)\s+([\d.]+)", out)
        h = re.findall(r"\[hash-prof\] h2d\s+([\d.]+)", out)
        d = re.findall(r"\[hash-prof\] d2h\s+([\d.]+)", out)
        bins = re.findall(r"bins: ([0-9: ]+)", out)
        if tot and h and d:
            co = float(tot[-1]) - float(h[-1]) - float(d[-1])
            return co, (bins[-1] if bins else "")
        return -1, ""
    except Exception:
        return -1, ""

def get_path(mat):
    for d in ("data/ocean/square", "data/first100"):
        p = os.path.join(REPO, d, mat + ".mtx")
        if os.path.exists(p): return p
    return None

# ---- 特征(与 ai_cost_model 同构,降维到本实验相关的 10 维) ----
def compute_features(bins_str, n_est):
    bins = {}; total = 0
    for tok in bins_str.split():
        if ':' in tok:
            p = tok.split(':')
            try: bins[int(p[0])] = int(p[1]); total += int(p[1])
            except: pass
    if total == 0: return None
    mid = {0:5,1:6,2:7,3:8,4:9,5:10,6:11,7:12,8:13,9:14}
    ev = [mid[b] for b,c in bins.items() if b in mid for _ in range(min(c,500))]
    if not ev: ev=[5]
    bin45 = (bins.get(4,0)+bins.get(5,0))/total
    return np.array([
        math.log2(max(total,1)), float(np.median(ev)),
        float(np.percentile(ev,75)-np.percentile(ev,25)),
        bin45, bins.get(9,0)/total, bins.get(0,0)/total,
        math.log2(max(total*5,1)), math.log2(max(total,1))*float(np.median(ev)),
        math.log2(max(total*5,1))*float(np.median(ev)), 1.0,
    ])

# ---- 物理先验(log2 ms): hash_heavy vs LHT 的表清理+插入+提取+带宽 ----
def phys_cost(x, option):
    n = 2**min(x[0], 25); est = 2**min(x[1], 14)
    products = min(n*est, 1e10)
    if option == "hash_heavy":   # 素数表 + warp 私有,几何 2048/256
        clear = n*2048*2/256/(132*1.7e9)*1000
        insert = products*2/450e9*1000
        extract = n*est**2/32/(132*1.7e9)*1000
        bw = min(n*5*12*est, 1e12)/3.35e12*1000
    else:                        # LHT 梯,表更大提取更慢
        clear = n*1024*2/32/(132*1.7e9)*1000*1.1
        insert = products*2/450e9*1000*1.15
        extract = n*est**2/32/(132*1.7e9)*1000*1.3
        bw = min(n*5*12*est, 1e12)/3.35e12*1000
    return math.log2(max(clear+insert+extract+bw, 0.001))

class OnlineRouter:
    """物理先验 + 线性残差,在线 SGD(与 ai_cost_model.UnifiedRouter 同构)"""
    def __init__(self, nf, lr=0.01):
        self.nf, self.lr = nf, lr
        self.W = {"hash_heavy": np.zeros(nf), "lht": np.zeros(nf)}
        self.b = {"hash_heavy": 0.0, "lht": 0.0}
    def predict(self, x, opt): return phys_cost(x, opt) + float(np.dot(self.W[opt], x)) + self.b[opt]
    def select(self, x): return "hash_heavy" if self.predict(x,"hash_heavy") <= self.predict(x,"lht") else "lht"
    def update(self, x, opt, actual_log_ms):
        err = np.clip(actual_log_ms - self.predict(x, opt), -5, 5)
        self.W[opt] += self.lr*(err*x - 0.01*self.W[opt])
        self.b[opt] += self.lr*err
        return abs(err)

def main():
    gpu = sys.argv[1] if len(sys.argv) > 1 else "0"
    router = OnlineRouter(10)

    # ===== Phase 1: 旧世界(sm90 二进制) =====
    print("="*64)
    print("Phase 1: 旧世界 spgemm_test_sm90 (hash_heavy=死刑时代)")
    print("="*64)
    old_world = {}
    feats = {}
    for i, mat in enumerate(DRIFT_MATS):
        path = get_path(mat)
        if not path: continue
        t_on,  bs1 = run_one("spgemm_test_sm90", path, True,  gpu)
        t_off, bs2 = run_one("spgemm_test_sm90", path, False, gpu)
        if t_on <= 0 or t_off <= 0:
            print(f"  {mat:<18} 测量失败 on={t_on} off={t_off},跳过"); continue
        best = "hash_heavy" if t_on < t_off else "lht"
        old_world[mat] = {"on": t_on, "off": t_off, "best": best}
        x = compute_features(bs1 or bs2, 0)
        if x is not None: feats[mat] = x
        # 模型在旧世界在线学习(多轮把先验吃进去)
        if x is not None:
            for _ in range(20):
                router.update(x, "hash_heavy", math.log2(t_on))
                router.update(x, "lht", math.log2(t_off))
        print(f"  [{i+1:2d}] {mat:<18} on={t_on:>7.2f} off={t_off:>7.2f} → 旧最优={best}")

    valid = [m for m in old_world if m in feats]
    old_off_count = sum(1 for m in valid if old_world[m]["best"] == "lht")
    print(f"\n旧世界定谳: {old_off_count}/{len(valid)} 阵最优=H45关 "
          f"({'预期漂移前提成立' if old_off_count > len(valid)/2 else '⚠️ 漂移前提弱,检查阵集'})")

    # ===== Phase 2: 新世界(c61 二进制)—— 真内核变化 =====
    print("\n" + "="*64)
    print("Phase 2: 新世界 spgemm_test_c61 (hash_heavy=复活时代) ← 编译层内核变化")
    print("="*64)
    new_world = {}
    for i, mat in enumerate(valid):
        path = get_path(mat)
        t_on,  bs1 = run_one("spgemm_test_c61", path, True,  gpu)
        t_off, bs2 = run_one("spgemm_test_c61", path, False, gpu)
        if t_on <= 0 or t_off <= 0: continue
        best = "hash_heavy" if t_on < t_off else "lht"
        new_world[mat] = {"on": t_on, "off": t_off, "best": best}
        print(f"  [{i+1:2d}] {mat:<18} on={t_on:>7.2f} off={t_off:>7.2f} → 新最优={best}")

    both = [m for m in new_world if m in old_world]
    flips = sum(1 for m in both if old_world[m]["best"] != new_world[m]["best"])
    print(f"\n漂移实测定谳: {flips}/{len(both)} 阵最优路由翻面 "
          f"(翻面阵上,旧世界静态表的答案在新世界是错的)")

    # ===== Phase 3: 在线恢复(真模型,真观测流) =====
    print("\n" + "="*64)
    print("Phase 3: 在线适应(旧世界先验 + 新世界观测流)")
    print("="*64)
    router2 = OnlineRouter(10, lr=0.02)
    # 复制旧世界训练好的先验
    router2.W = {k: v.copy() for k, v in router.W.items()}
    router2.b = dict(router.b)

    stream = both  # 按名单顺序流入
    decisions = []
    for i, mat in enumerate(stream):
        x = feats[mat]
        choice = router2.select(x)                 # 决策(先验,再看观测)
        truth = new_world[mat]["best"]
        correct = (choice == truth)
        # 观测到来:该阵两个选项的新世界实测 → 监督式两选项更新
        router2.update(x, "hash_heavy", math.log2(new_world[mat]["on"]))
        router2.update(x, "lht",       math.log2(new_world[mat]["off"]))
        decisions.append(correct)
        acc = 100*sum(decisions)/len(decisions)
        if i < 8 or (i+1) % 4 == 0:
            print(f"  [{i+1:2d}] {mat:<18} 模型选={choice:<11} 真最优={truth:<11} "
                  f"{'✓' if correct else '✗'} 累计准确率={acc:.0f}%")

    early = decisions[:5]; late = decisions[8:] if len(decisions) > 8 else decisions[5:]
    print(f"\n恢复曲线: 前5决策准确率 {100*sum(early)/len(early):.0f}% → "
          f"第9+决策准确率 {100*sum(late)/len(late):.0f}%")

    # ===== 对照臂汇总 =====
    static_correct = sum(1 for m in both if old_world[m]["best"] == new_world[m]["best"])
    always_on  = sum(1 for m in both if new_world[m]["best"] == "hash_heavy")
    always_off = sum(1 for m in both if new_world[m]["best"] == "lht")
    online_final = sum(decisions)
    n = len(both)

    print("\n" + "="*64)
    print(f"四臂对比(新世界 {n} 阵, 漂移翻面 {flips} 阵)")
    print("="*64)
    print(f"  静态表(旧世界标定,=if-else 内核变化后): {static_correct}/{n} = {100*static_correct/n:.0f}%")
    print(f"  在线模型(本实验):                      {online_final}/{n} = {100*online_final/n:.0f}%")
    print(f"  always H45-on:                          {always_on}/{n} = {100*always_on/n:.0f}%")
    print(f"  always H45-off:                         {always_off}/{n} = {100*always_off/n:.0f}%")

    out = {
        "old_world": old_world, "new_world": new_world,
        "flips": flips, "n": n,
        "arms": {"static": 100*static_correct/n, "online": 100*online_final/n,
                 "always_on": 100*always_on/n, "always_off": 100*always_off/n},
        "recovery_curve": decisions,
    }
    with open(os.path.join(REPO, "results", "drift_dual_experiment.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n保存 → results/drift_dual_experiment.json")

if __name__ == "__main__":
    main()
