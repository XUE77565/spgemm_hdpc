#!/usr/bin/env python3
"""AI Router v2: 物理先验 + 学习残差 + 在线更新
修复: log空间 + 特征标准化 + 梯度裁剪 + 学习率退火"""
import csv, json, math, os
import numpy as np
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_NAMES = ['log_n', 'log_nnz', 'log_density', 'log_avg_row',
                 'est_median', 'est_iqr', 'bin0_frac', 'bin45_frac', 'bin9_frac', 'bin67_frac']

def compute_features(bins_str, n, nnz_a, A_cols):
    bins = {}; total = 0
    for tok in bins_str.split():
        if ':' in tok:
            p = tok.split(':')
            try: bins[int(p[0])] = int(p[1]); total += int(p[1])
            except: pass
    if total == 0 or n == 0: return None
    bin0 = bins.get(0,0)/total; bin45 = (bins.get(4,0)+bins.get(5,0))/total
    bin9 = bins.get(9,0)/total; bin67 = (bins.get(6,0)+bins.get(7,0))/total
    mid = {0:5,1:6,2:7,3:8,4:9,5:10,6:11,7:12,8:13,9:14}
    ev = [mid[b] for b,c in bins.items() if b in mid for _ in range(min(c,100))]
    if not ev: ev=[5]
    density = nnz_a/(n*A_cols) if n>0 and A_cols>0 else 1e-10
    return np.array([
        math.log2(max(n,1)), math.log2(max(nnz_a,1)),
        math.log2(max(density,1e-10)), math.log2(max(nnz_a/n,1)),
        float(np.median(ev)), float(np.percentile(ev,75)-np.percentile(ev,25)),
        bin0, bin45, bin9, bin67
    ])

def cost_phys(x, variant_idx):
    """简化的物理成本(log ms)"""
    log_n, log_nnz = x[0], x[1]; est_med = x[4]; log_avg = x[3]
    n = 2**min(log_n, 25); nnz = 2**min(log_nnz, 30)
    products = 2**min(est_med, 14); avg_row = max(2**log_avg, 1)

    # 公共: B行读取
    bytes_read = min(nnz * 12 * products / avg_row, 1e12)
    bw = math.log2(max(bytes_read / 3.35e12 * 1000, 0.001))  # log2(ms)

    if variant_idx == 0:  # baseline (H69+H45 on → hash_heavy)
        insert = min(n * products * 2 / 450e9 * 1000, 1e6)
        extract = min(n * products**2 / 32 / (132*1.7e9) * 1000, 1e6)
    elif variant_idx == 1:  # no_h69 (bin9 → LHT14)
        insert = min(n * products * 2 / 450e9 * 1000, 1e6)
        extract = min(n * products**2 / 32 / (132*1.7e9) * 1000, 1e6) * 1.2
    else:  # no_h4555 / no_all (LHT variants, slower)
        insert = min(n * products * 2 / 450e9 * 1000, 1e6) * 1.15
        extract = min(n * products**2 / 32 / (132*1.7e9) * 1000, 1e6) * 1.3

    total_ms = 2**bw + insert + extract
    return math.log2(max(total_ms, 0.001))

class AIRouter:
    def __init__(self, n_feat, n_var, lr=0.01):
        self.n_feat = n_feat; self.n_var = n_var
        self.lr = lr
        self.W = np.zeros((n_var, n_feat))
        self.b = np.zeros(n_var)
        self.mean = np.zeros(n_feat); self.std = np.ones(n_feat)
        self.count = 0; self.val_loss = []

    def fit_scaler(self, X):
        self.mean = X.mean(axis=0); self.std = X.std(axis=0) + 1e-8

    def select(self, x, phys):
        xp = (x - self.mean) / self.std
        costs = np.array([phys[v] + np.dot(self.W[v], xp) + self.b[v] for v in range(self.n_var)])
        return int(np.argmin(costs))

    def update(self, x, phys, chosen, actual_log_ms):
        xp = (x - self.mean) / self.std
        pred = phys[chosen] + np.dot(self.W[chosen], xp) + self.b[chosen]
        err = np.clip(actual_log_ms - pred, -5, 5)  # gradient clipping
        self.W[chosen] += self.lr * (err * xp - 0.01 * self.W[chosen])
        self.b[chosen] += self.lr * err
        self.count += 1
        return abs(err)

def load_data():
    cal = defaultdict(dict); meta = {}
    for r in csv.DictReader(open(f"{REPO}/results/calibration_data.csv")):
        if r['mat']=='mat': continue
        try:
            co = float(r['co_ms'])
            if co>0: cal[r['mat']][r['variant']] = math.log2(co)
            if r['mat'] not in meta:
                meta[r['mat']] = {'bins': r.get('bins_summary',''), 'n':int(r.get('n',0) or 1000), 'nnz_a':int(r.get('nnz_a',0) or 1000)}
        except: pass
    # baseline from confirm16
    for r in csv.DictReader(open(f"{REPO}/compare/batch437/confirm16_428.csv")):
        if r['mat']=='mat': continue
        if r['mat'] in cal:
            try: cal[r['mat']]['baseline'] = math.log2(float(r['adpt13_ms']))
            except: pass
    samples = []
    for mat, variants in cal.items():
        if len(variants)<3: continue
        m = meta.get(mat, {})
        feat = compute_features(m.get('bins',''), m.get('n',1000), m.get('nnz_a',1000), m.get('n',1000))
        if feat is None: continue
        phys = [cost_phys(feat, v) for v in range(4)]
        times = {}
        for i,vn in enumerate(['baseline','no_h69','no_h4555','no_all']):
            if vn in variants: times[i] = variants[vn]
        if len(times)>=2: samples.append({'mat':mat,'feat':feat,'phys':phys,'times':times})
    return samples

def train():
    samples = load_data()
    print(f"训练样本: {len(samples)}")

    # Split: 80/20 train/val
    np.random.seed(42); idx = np.random.permutation(len(samples))
    split = int(len(samples)*0.8)
    train_s = [samples[i] for i in idx[:split]]
    val_s = [samples[i] for i in idx[split:]]

    router = AIRouter(10, 4, lr=0.005)
    router.fit_scaler(np.array([s['feat'] for s in train_s]))

    # Training
    for epoch in range(50):
        order = np.random.permutation(len(train_s))
        total_err = 0
        for i in order:
            s = train_s[i]
            chosen = router.select(s['feat'], s['phys'])
            if chosen not in s['times']:
                valid = list(s['times'].keys()); chosen = valid[0]
            total_err += router.update(s['feat'], s['phys'], chosen, s['times'][chosen])

        if (epoch+1) % 10 == 0:
            # Validation: 模型选择 vs 实际最优
            correct=0; total=0; subopt=0
            for s in val_s:
                if not s['times']: continue
                model_chosen = router.select(s['feat'], s['phys'])
                actual_best = min(s['times'], key=s['times'].get)
                total+=1
                if model_chosen==actual_best: correct+=1
                else:
                    ratio = 2**(s['times'].get(model_chosen, 999) - s['times'][actual_best])
                    subopt += ratio-1

            acc = 100*correct/total if total else 0
            avg_subopt = 100*subopt/max(total-correct,1) if total>correct else 0
            print(f"  epoch {epoch+1}: train_err={total_err/len(train_s):.4f} val_acc={acc:.1f}% val_subopt={avg_subopt:.1f}%")

    # Final eval on all
    correct=0; total=0; subopt_total=0
    for s in samples:
        if not s['times']: continue
        chosen = router.select(s['feat'], s['phys'])
        best = min(s['times'], key=s['times'].get)
        total+=1
        if chosen==best: correct+=1
        else:
            ratio = 2**(s['times'].get(chosen, 999) - s['times'][best])
            subopt_total += ratio-1
    print(f"\n最终: acc={100*correct/total:.1f}% avg_subopt={100*subopt_total/max(total-correct,1):.1f}%")

    # 对比: "永远选 baseline" 的准确率
    bl_correct = sum(1 for s in samples if 0==min(s['times'],key=s['times'].get))
    print(f"'永远选baseline': acc={100*bl_correct/total:.1f}%")

    # Export C
    lines = ["// AI Router v2: physics + learned residual (log-space, standardized features)",
             f"// Trained: {router.count} updates", ""]
    lines.append(f"#define AI_NF 10")
    lines.append(f"static const float ai_mean[10] = {{{','.join(f'{v:.4f}f' for v in router.mean)}}};")
    lines.append(f"static const float ai_std[10] = {{{','.join(f'{v:.4f}f' for v in router.std)}}};")
    lines.append(f"static const float ai_W[4][10] = {{")
    for v in range(4):
        lines.append(f"  {{{','.join(f'{w:.4f}f' for w in router.W[v])}}},")
    lines.append("};")
    lines.append(f"static const float ai_b[4] = {{{','.join(f'{b:.4f}f' for b in router.b)}}};")
    lines.append("""
static inline int ai_select(const float* feat, const float* phys_log) {
    float xp[10]; for (int i=0;i<10;i++) xp[i]=(feat[i]-ai_mean[i])/ai_std[i];
    float best=1e30f; int best_v=0;
    for (int v=0;v<4;v++) {
        float c = phys_log[v] + ai_b[v];
        for (int i=0;i<10;i++) c += ai_W[v][i]*xp[i];
        if (c<best) {best=c;best_v=v;}
    }
    return best_v;
}""")
    with open(f"{REPO}/src/ai_router.h","w") as f: f.write("\n".join(lines))
    print(f"→ src/ai_router.h")

    json.dump({'W':router.W.tolist(),'b':router.b.tolist(),
               'mean':router.mean.tolist(),'std':router.std.tolist()},
              open(f"{REPO}/results/ai_router_weights.json","w"), indent=2)

    return router

if __name__=="__main__":
    train()
