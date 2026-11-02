#!/usr/bin/env python3
"""
AI Router 统一成本模型 v3: 物理 + 学习 + 多路由点
训练后导出 C 代码 + 漂移实验框架
"""
import csv, json, math, os, sys
import numpy as np
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(REPO, "results/ai_calibration_full.csv")
CONF_FILE = os.path.join(REPO, "compare/batch437/confirm16_428.csv")
OUTPUT_H = os.path.join(REPO, "src/ai_router.h")
OUTPUT_JSON = os.path.join(REPO, "results/ai_cost_model.json")

# ===== 硬件常数 (H100) =====
HW = {
    'SM_COUNT': 132,
    'SM_CLOCK': 1.7e9,
    'HBM_BW': 3.35e12,       # bytes/s
    'SMEM_ATOMIC': 450e9,     # ops/s (approximate)
    'SMEM_LIMIT': 228*1024,   # bytes per SM
    'LANE_WRITE': 1,          # 1 write/cycle/lane
}

# ===== 特征 =====
FEATURES = [
    'log_n', 'log_nnz', 'log_density', 'log_avg_row',
    'est_median', 'est_iqr',
    'bin0_frac', 'bin45_frac', 'bin67_frac', 'bin9_frac',
    'has_bin9', 'has_bin45',
    # 交互项(物理模型需要的)
    'log_n_x_est', 'log_nnz_x_est',
]

def compute_features(bins_str, n, nnz_a, A_cols):
    bins = {}; total = 0
    for tok in bins_str.split():
        if ':' in tok:
            try:
                p = tok.split(':'); bins[int(p[0])] = int(p[1]); total += int(p[1])
            except: pass
    if total == 0 or n <= 0: return None

    bin0 = bins.get(0,0)/total
    bin45 = (bins.get(4,0)+bins.get(5,0))/total
    bin9 = bins.get(9,0)/total
    bin67 = (bins.get(6,0)+bins.get(7,0))/total
    density = max(nnz_a/(n*A_cols), 1e-12) if A_cols > 0 else 1e-12
    avg_row = max(nnz_a/n, 1) if n > 0 else 1

    mid = {0:5,1:6,2:7,3:8,4:9,5:10,6:11,7:12,8:13,9:14}
    ev = [mid[b] for b,c in bins.items() if b in mid for _ in range(min(c,500))]
    if not ev: ev=[5]

    est_med = float(np.median(ev))
    log_n = math.log2(max(n,1))
    log_nnz = math.log2(max(nnz_a,1))

    return np.array([
        log_n, log_nnz,
        math.log2(density) if density > 0 else -20,
        math.log2(avg_row),
        est_med,
        float(np.percentile(ev,75)-np.percentile(ev,25)),
        bin0, bin45, bin67, bin9,
        1.0 if bin9 > 0.01 else 0.0,
        1.0 if bin45 > 0.01 else 0.0,
        log_n * est_med,
        log_nnz * est_med,
    ])

# ===== 物理成本模型 =====
def phys_cost_hash_heavy(x, ht_size, blk_size):
    """hash_heavy 物理成本(log ms)"""
    n = 2**min(x[0], 25); nnz = 2**min(x[1], 30)
    est = 2**min(x[4], 14); avg_row = max(2**x[3], 1)
    products = min(n * est, 1e10)

    # 表清理: n行 × ht_size×2槽 / blk_size线程 / SM并行
    clear = n * ht_size * 2 / blk_size / (HW['SM_COUNT'] * HW['SM_CLOCK']) * 1000
    # hash 插入: products × 2原子 / 原子吞吐
    insert = products * 2 / HW['SMEM_ATOMIC'] * 1000
    # 提取: O(est²/32)/行
    extract = n * est**2 / 32 / (HW['SM_COUNT'] * HW['SM_CLOCK']) * 1000
    # B行读取
    bytes_read = min(nnz * 12 * est / avg_row, 1e12)
    bw = bytes_read / HW['HBM_BW'] * 1000

    total = clear + insert + extract + bw
    return math.log2(max(total, 0.001))

def phys_cost_lht(x, log_ht):
    """LHT 物理成本"""
    n = 2**min(x[0], 25); nnz = 2**min(x[1], 30)
    est = 2**min(x[4], 14); avg_row = max(2**x[3], 1)
    ht = 2**log_ht
    products = min(n * est, 1e10)

    clear = n * ht * 2 / 32 / (HW['SM_COUNT'] * HW['SM_CLOCK']) * 1000
    insert = products * 2 / HW['SMEM_ATOMIC'] * 1000
    extract = n * est**2 / 32 / (HW['SM_COUNT'] * HW['SM_CLOCK']) * 1000
    bytes_read = min(nnz * 12 * est / avg_row, 1e12)
    bw = bytes_read / HW['HBM_BW'] * 1000

    total = clear * 1.1 + insert * 1.15 + extract * 1.3 + bw  # LHT 更慢
    return math.log2(max(total, 0.001))

# ===== 变体成本函数映射 =====
def get_phys_cost(x, routing_point, option_label):
    """给定特征和路由点+选项 → 物理成本(log ms)"""
    if routing_point == "bin45_kernel":
        if option_label == "hash_heavy":
            return phys_cost_hash_heavy(x, 2048, 256)
        else:  # lht
            return phys_cost_lht(x, 10)
    elif routing_point == "bin9_kernel":
        if option_label == "heavy_16253":
            return phys_cost_hash_heavy(x, 16253, 1024)
        else:  # lht14
            return phys_cost_lht(x, 14)
    elif routing_point == "compact_sorted":
        # sorted_copy vs unsorted_csort
        base = phys_cost_hash_heavy(x, 2048, 256)
        if option_label == "sorted_copy":
            return base + 0.1  # copy is cheap
        else:
            return base + 2.5  # csort is expensive
    elif routing_point == "est_expand":
        base = phys_cost_hash_heavy(x, 2048, 256)
        if option_label == "tight_115":
            return base  # tight = less retry
        else:
            return base + 0.5  # loose = more memory but less retry
    else:
        # 其他路由点用默认物理模型
        return phys_cost_hash_heavy(x, 2048, 256)

# ===== AI Router 类 =====
class UnifiedRouter:
    """统一成本模型: 每个路由点一个线性模型"""

    def __init__(self, n_features, lr=0.005):
        self.n_features = n_features
        self.lr = lr
        # 每个路由点 → 每个选项 → (W, b)
        self.models = {}  # {routing_point: {option_label: {'W': vec, 'b': float}}}
        self.feature_mean = None
        self.feature_std = None
        self.update_count = 0
        self.training_log = []

    def _get_model(self, rp, opt):
        key = f"{rp}/{opt}"
        if key not in self.models:
            self.models[key] = {
                'W': np.zeros(self.n_features),
                'b': 0.0,
                'rp': rp, 'opt': opt,
                'n_updates': 0, 'avg_err': 0.0
            }
        return self.models[key]

    def select(self, x, rp, options, phys_costs):
        """选择最优选项"""
        if self.feature_mean is not None:
            xp = (x - self.feature_mean) / (self.feature_std + 1e-8)
        else:
            xp = x

        best_cost = 1e30; best_opt = options[0]
        for opt in options:
            m = self._get_model(rp, opt)
            cost = phys_costs.get(opt, 1e6) + np.dot(m['W'], xp) + m['b']
            if cost < best_cost:
                best_cost = cost; best_opt = opt
        return best_opt

    def update(self, x, rp, opt, phys_cost, actual_log_ms):
        """在线更新"""
        if self.feature_mean is not None:
            xp = (x - self.feature_mean) / (self.feature_std + 1e-8)
        else:
            xp = x

        m = self._get_model(rp, opt)
        pred = phys_cost + np.dot(m['W'], xp) + m['b']
        err = np.clip(actual_log_ms - pred, -5, 5)

        m['W'] += self.lr * (err * xp - 0.01 * m['W'])
        m['b'] += self.lr * err
        m['n_updates'] += 1
        m['avg_err'] = 0.95 * m['avg_err'] + 0.05 * abs(err)

        self.update_count += 1
        return abs(err)

    def fit_scaler(self, X_list):
        X = np.array(X_list)
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-8

    def export_analysis(self):
        """导出模型分析(可解释性)"""
        analysis = {}
        for key, m in self.models.items():
            analysis[key] = {
                'n_updates': m['n_updates'],
                'avg_error': m['avg_err'],
                'bias': m['b'],
                'top_features': sorted(
                    zip(FEATURES, m['W'].tolist()),
                    key=lambda x: -abs(x[1])
                )[:5]
            }
        return analysis

    def export_c(self):
        """导出 C 代码"""
        lines = []
        lines.append("// AI Router v3: Unified Cost Model (physics + learned)")
        lines.append(f"// Total updates: {self.update_count}")
        lines.append(f"#define AI_NF {self.n_features}")
        lines.append(f"#define AI_NR {len(self.models)}")
        lines.append("")
        # Feature scaler
        if self.feature_mean is not None:
            lines.append(f"static const float ai_mean[{self.n_features}] = {{{','.join(f'{v:.4f}f' for v in self.feature_mean)}}};")
            lines.append(f"static const float ai_std[{self.n_features}] = {{{','.join(f'{v:.4f}f' for v in self.feature_std)}}};")
        else:
            lines.append(f"static const float ai_mean[{self.n_features}] = {{0}};")
            lines.append(f"static const float ai_std[{self.n_features}] = {{1}};")
        lines.append("")
        # Models (routing_point/option → W, b)
        lines.append(f"typedef struct {{ const char* rp; const char* opt; float W[{self.n_features}]; float b; }} ai_model_t;")
        lines.append(f"static const ai_model_t ai_models[{len(self.models)}] = {{")
        for key, m in sorted(self.models.items()):
            rp, opt = key.split('/')
            w_str = ','.join(f'{w:.4f}f' for w in m['W'])
            lines.append(f'  {{"{rp}","{opt}",{{{w_str}}},{m["b"]:.4f}f}},')
        lines.append("};")
        lines.append("")
        lines.append("""
static inline float ai_predict(int model_idx, const float* feat) {
    float xp[AI_NF];
    for (int i=0;i<AI_NF;i++) xp[i]=(feat[i]-ai_mean[i])/ai_std[i];
    const ai_model_t* m = &ai_models[model_idx];
    float c = m->b;
    for (int i=0;i<AI_NF;i++) c += m->W[i]*xp[i];
    return c;
}
""")
        return "\n".join(lines)

# ===== 主流程 =====
def train():
    print("=" * 60)
    print("AI Router v3: 统一成本模型训练")
    print("=" * 60)

    # 加载数据
    data = []
    if os.path.exists(DATA_FILE):
        for r in csv.DictReader(open(DATA_FILE)):
            if r['mat'] == 'mat': continue
            try:
                co = float(r['co_ms'])
                if co <= 0: continue
                data.append({
                    'mat': r['mat'],
                    'routing_point': r['routing_point'],
                    'option': r['option'],
                    'log_time': math.log2(co),
                })
            except: pass

    # 补充已有的标定数据
    cal_file = os.path.join(REPO, "results/calibration_data.csv")
    if os.path.exists(cal_file):
        for r in csv.DictReader(open(cal_file)):
            if r['mat'] == 'mat': continue
            try:
                co = float(r['co_ms'])
                if co <= 0: continue
                rp = r['variant']  # baseline/no_h69/no_h4555/no_all
                opt = r['variant']
                data.append({
                    'mat': r['mat'], 'routing_point': 'variant_combo',
                    'option': opt, 'log_time': math.log2(co),
                })
            except: pass

    print(f"数据点: {len(data)}")
    if len(data) < 20:
        print("数据不足,等消融跑完再训练")
        return None

    # 提取特征(用已有 bins 数据)
    bins_data = {}
    for r in csv.DictReader(open(cal_file if os.path.exists(cal_file) else '/dev/null')):
        if r['mat'] != 'mat' and r.get('bins_summary'):
            bins_data[r['mat']] = r['bins_summary']

    # 构建训练样本
    samples = []
    for d in data:
        bs = bins_data.get(d['mat'], '')
        # 估计 n 和 nnz(从 bins)
        bins = {}
        total = 0
        for tok in bs.split():
            if ':' in tok:
                try:
                    p = tok.split(':'); bins[int(p[0])] = int(p[1]); total += int(p[1])
                except: pass
        n = max(total, 1000)
        nnz_a = max(total * 5, 1000)  # 粗估
        feat = compute_features(bs, n, nnz_a, n)
        if feat is None: continue
        phys = get_phys_cost(feat, d['routing_point'], d['option'])
        samples.append({
            'feat': feat, 'rp': d['routing_point'],
            'opt': d['option'], 'phys': phys, 'log_t': d['log_time'],
            'mat': d['mat']
        })

    print(f"有效样本: {len(samples)}")

    # 路由点统计
    rp_counts = defaultdict(set)
    for s in samples:
        rp_counts[s['rp']].add(s['mat'])
    print("\n路由点覆盖:")
    for rp, mats in sorted(rp_counts.items()):
        print(f"  {rp}: {len(mats)} 阵")

    # 训练
    router = UnifiedRouter(len(FEATURES))
    router.fit_scaler([s['feat'] for s in samples])

    # 按路由点分组训练
    rp_groups = defaultdict(list)
    for s in samples:
        rp_groups[(s['rp'], s['opt'])].append(s)

    for epoch in range(30):
        total_err = 0
        order = np.random.permutation(len(samples))
        for i in order:
            s = samples[i]
            options = [x['opt'] for x in samples if x['rp'] == s['rp'] and x['mat'] == s['mat']]
            if not options: options = [s['opt']]
            phys_costs = {opt: get_phys_cost(s['feat'], s['rp'], opt) for opt in set(options)}
            chosen = router.select(s['feat'], s['rp'], options, phys_costs)
            total_err += router.update(s['feat'], s['rp'], chosen, phys_costs[chosen], s['log_t'])

        if (epoch+1) % 10 == 0:
            print(f"  epoch {epoch+1}: avg_err = {total_err/len(samples):.4f}")

    # 评估: 对每个 (mat, rp), 模型选的 vs 实际最优
    correct = 0; total = 0; suboptimal_costs = []
    for rp in set(s['rp'] for s in samples):
        rp_samples = defaultdict(dict)
        for s in samples:
            if s['rp'] == rp:
                rp_samples[s['mat']][s['opt']] = s['log_t']

        for mat, opts in rp_samples.items():
            if len(opts) < 2: continue
            # 获取特征
            mat_sample = next(s for s in samples if s['mat']==mat and s['rp']==rp)
            options = list(opts.keys())
            phys_costs = {opt: get_phys_cost(mat_sample['feat'], rp, opt) for opt in options}
            model_choice = router.select(mat_sample['feat'], rp, options, phys_costs)
            actual_best = min(opts, key=opts.get)

            total += 1
            if model_choice == actual_best:
                correct += 1
            else:
                # 计算选错的代价
                ratio = 2**(opts[model_choice] - opts[actual_best])
                suboptimal_costs.append(ratio)

    if total > 0:
        print(f"\n路由选择准确率: {correct}/{total} = {100*correct/total:.1f}%")
        if suboptimal_costs:
            avg_subopt = sum(suboptimal_costs)/len(suboptimal_costs)
            print(f"选错时代价: 平均 {avg_subopt:.2f}× (越接近 1 越好)")

    # 导出
    analysis = router.export_analysis()
    with open(OUTPUT_JSON, 'w') as f:
        json.dump({
            'weights': {k: {'W': m['W'].tolist(), 'b': m['b'],
                          'n_updates': m['n_updates'], 'avg_err': m['avg_err']}
                       for k, m in router.models.items()},
            'feature_names': FEATURES,
            'feature_mean': router.feature_mean.tolist() if router.feature_mean is not None else None,
            'feature_std': router.feature_std.tolist() if router.feature_std is not None else None,
            'total_updates': router.update_count,
            'analysis': analysis,
        }, f, indent=2)

    c_code = router.export_c()
    with open(OUTPUT_H, 'w') as f:
        f.write(c_code)

    print(f"\n导出: {OUTPUT_H}")
    print(f"模型分析: {OUTPUT_JSON}")

    # 打印可解释性摘要
    print("\n===== 模型可解释性(每个路由点的前 3 重要特征) =====")
    for key, info in sorted(analysis.items()):
        if info['n_updates'] < 5: continue
        print(f"\n  {key} ({info['n_updates']} 次更新, 平均误差 {info['avg_err']:.3f}):")
        for fname, weight in info['top_features'][:3]:
            direction = "↑" if weight > 0 else "↓"
            print(f"    {fname}: {weight:+.3f} {direction}")

    return router

if __name__ == "__main__":
    train()
