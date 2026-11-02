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

    # ===== Phase 1: 时代A = sm90 二进制(10-27,hash_heavy 死刑时代) =====
    print("="*64)
    print("Phase 1: 时代A spgemm_test_sm90 (10-27, hash_heavy=死刑时代)")
    print("="*64)
    era_a = {}
    feats = {}
    for i, mat in enumerate(DRIFT_MATS):
        path = get_path(mat)
        if not path: continue
        t_on,  bs1 = run_one("spgemm_test_sm90", path, True,  gpu)
        t_off, bs2 = run_one("spgemm_test_sm90", path, False, gpu)
        if t_on <= 0 or t_off <= 0:
            print(f"  {mat:<18} 测量失败 on={t_on} off={t_off},跳过"); continue
        best = "hash_heavy" if t_on < t_off else "lht"
        era_a[mat] = {"on": t_on, "off": t_off, "best": best}
        x = compute_features(bs1 or bs2, 0)
        if x is not None: feats[mat] = x
        # 模型在旧世界在线学习(多轮把先验吃进去)
        if x is not None:
            for _ in range(20):
                router.update(x, "hash_heavy", math.log2(t_on))
                router.update(x, "lht", math.log2(t_off))
        print(f"  [{i+1:2d}] {mat:<18} on={t_on:>7.2f} off={t_off:>7.2f} → A最优={best}")

    valid = [m for m in era_a if m in feats]

    # ===== Phase 2: 时代B = c61 二进制(10-27,编译层单变量) =====
    print("\n" + "="*64)
    print("Phase 2: 时代B spgemm_test_c61 (10-27, 编译层单变量)")
    print("="*64)
    era_b = {}
    for i, mat in enumerate(valid):
        path = get_path(mat)
        t_on,  _ = run_one("spgemm_test_c61", path, True,  gpu)
        t_off, _ = run_one("spgemm_test_c61", path, False, gpu)
        if t_on <= 0 or t_off <= 0: continue
        best = "hash_heavy" if t_on < t_off else "lht"
        era_b[mat] = {"on": t_on, "off": t_off, "best": best}
        print(f"  [{i+1:2d}] {mat:<18} on={t_on:>7.2f} off={t_off:>7.2f} → B最优={best}")

    # ===== Phase 2c: 时代C = 当前生产二进制(10-30,刀链落地后) =====
    # 漂移主轴 = A/B → C:白天的消融数据显示 C 上 HH 14/14 全赢,
    # 但按测量纪律(同会话同卡配对)C 必须在本会话 GPU 上重测
    print("\n" + "="*64)
    print("Phase 2c: 时代C spgemm_test (10-30 生产, 刀链落地) ← 真内核演化")
    print("="*64)
    era_c = {}
    for i, mat in enumerate(valid):
        path = get_path(mat)
        t_on,  _ = run_one("spgemm_test", path, True,  gpu)
        t_off, _ = run_one("spgemm_test", path, False, gpu)
        if t_on <= 0 or t_off <= 0: continue
        best = "hash_heavy" if t_on < t_off else "lht"
        era_c[mat] = {"on": t_on, "off": t_off, "best": best}
        print(f"  [{i+1:2d}] {mat:<18} on={t_on:>7.2f} off={t_off:>7.2f} → C最优={best}")

    both = [m for m in era_c if m in era_a]
    n = len(both)
    flips_ac = sum(1 for m in both if era_a[m]["best"] != era_c[m]["best"])
    print(f"\n漂移实测定谳: 时代A→C 翻面 {flips_ac}/{n} 阵 "
          f"(旧世界标定的'真相'在新世界是错的)")

    # ===== Phase 3: 在线恢复(时代A先验 + 时代C观测流) =====
    # 修复(11-02 深夜):特征标准化缺失 → 交互特征量级~170,先验训练后 W·x 爆炸,
    # 两选项 err 双双饱和 clip=±5,更新同步化,相对差冻结(全 lr 7% 不动的根因)
    print("\n" + "="*64)
    print("Phase 3: 在线适应(时代A先验 + 时代C观测流, 特征标准化版)")
    print("="*64)
    X = np.array([feats[m] for m in valid if m in feats])
    fmu, fsd = X.mean(axis=0), X.std(axis=0) + 1e-8
    sfeats = {m: (feats[m]-fmu)/fsd for m in feats}

    router3 = OnlineRouter(10, lr=0.05)
    for _ in range(20):   # 时代A先验(标准化特征上重建)
        for m in valid:
            if m in sfeats:
                router3.update(sfeats[m], "hash_heavy", math.log2(era_a[m]["on"]))
                router3.update(sfeats[m], "lht", math.log2(era_a[m]["off"]))
    router3.lr = 0.1

    decisions = []
    stream_reject = []
    PASSES = 3            # 生产语义:负载里每阵会被多次计算,每pass=一轮全阵观测
    for p in range(PASSES):
        for mat in both:
            x = sfeats[mat]
            choice = router3.select(x)
            truth = era_c[mat]["best"]
            correct = (choice == truth)
            decisions.append(correct)
            t_choice = era_c[mat]["on"] if choice == "hash_heavy" else era_c[mat]["off"]
            t_best = min(era_c[mat]["on"], era_c[mat]["off"])
            stream_reject.append(t_choice / t_best)
            router3.update(x, "hash_heavy", math.log2(era_c[mat]["on"]))
            router3.update(x, "lht",       math.log2(era_c[mat]["off"]))
    for p in range(PASSES):
        seg = decisions[p*n:(p+1)*n]
        print(f"  pass {p+1}: 准确率 {100*sum(seg)/len(seg):.0f}%")
    final_seg = decisions[-n:]
    early = decisions[:5]; late = decisions[-n:]

    # ===== 双时代四臂(论文主表:哪个臂在两个时代都跟得住最优) =====
    def arm_acc(pick, era):
        return sum(1 for m in both if pick(m) == era[m]["best"])
    static_pick = lambda m: era_a[m]["best"]            # 旧世界标定的静态表
    hh_pick     = lambda m: "hash_heavy"
    lht_pick    = lambda m: "lht"

    # 时间后悔(时代C):各臂总时间 / oracle 总时间
    oracle_t = sum(min(era_c[m]["on"], era_c[m]["off"]) for m in both)
    def arm_time(pick):
        return sum((era_c[m]["on"] if pick(m)=="hash_heavy" else era_c[m]["off"]) for m in both) / oracle_t
    online_time = sum(stream_reject) / len(stream_reject)

    print("\n" + "="*64)
    print(f"双时代四臂对比(A={n}阵 → C={n}阵, A→C翻面 {flips_ac})")
    print("="*64)
    print(f"{'臂':<22} {'A时代准确率':>10} {'C时代准确率':>10} {'C时间后悔':>10}")
    for name, pick in [("静态表(旧世界标定)", static_pick),
                       ("always hash_heavy",  hh_pick),
                       ("always lht",         lht_pick)]:
        a = 100*arm_acc(pick, era_a)/n; c = 100*arm_acc(pick, era_c)/n
        t = arm_time(pick)
        print(f"{name:<24} {a:>9.0f}% {c:>9.0f}% {t:>9.3f}x")
    online_a = None  # 在线模型在A时代=训练中,准确率意义不同,不列入
    print(f"{'在线模型(本实验)':<24} {'(训练中)':>10} {100*sum(final_seg)/n:>9.0f}% {online_time:>9.3f}x")
    print(f"\n恢复曲线: 前5决策 {100*sum(early)/len(early):.0f}% → 末轮pass {100*sum(late)/len(late):.0f}%")

    out = {
        "era_a_sm90_1027": era_a, "era_b_c61_1027": era_b, "era_c_prod_1030": era_c,
        "flips_a_to_c": flips_ac, "n": n,
        "arms_dual_era": {
            "static":  {"A": 100*arm_acc(static_pick, era_a)/n, "C": 100*arm_acc(static_pick, era_c)/n, "C_time": arm_time(static_pick)},
            "always_hh": {"A": 100*arm_acc(hh_pick, era_a)/n, "C": 100*arm_acc(hh_pick, era_c)/n, "C_time": arm_time(hh_pick)},
            "always_lht": {"A": 100*arm_acc(lht_pick, era_a)/n, "C": 100*arm_acc(lht_pick, era_c)/n, "C_time": arm_time(lht_pick)},
            "online": {"C": 100*sum(final_seg)/n, "C_time": online_time, "C_time_finalpass": sum(stream_reject[-n:])/n},
        },
        "recovery_curve": decisions,
    }
    with open(os.path.join(REPO, "results", "drift_dual_experiment.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n保存 → results/drift_dual_experiment.json")

if __name__ == "__main__":
    main()
