# AI 路由设计:物理先验 + 上下文 Bandit + 在线自适应

> 目标:不是 if-else,不是查表,是一个**从数据中学习成本模型、在线自适应、
> 内核变化时零人工重标定**的路由系统。

## 一、为什么现有方法都不够

| 方法 | 问题 |
|---|---|
| 手拍 if-else | 没有出处,内核变了全部作废 |
| 查表(我们 S4 做的) | 3 个离散特征 → 27 格 → 70% 说"全开" = 没信息量 |
| FlexSpGEMM 的离线 ML | 黑盒,训练后不再更新,内核变化时需重训 |
| 纯 RL | 样本效率低,探索代价大 |

**核心问题**:现有方法都是"离线学一次,永远用"——但内核在变(我们两天加了
5 把刀),离线模型立刻过期。

## 二、我们的方案:物理先验 + 上下文 Bandit

### 2.1 核心思想

对于每个内核变体 v,学习一个**成本函数**:

```
cost(v | x) = cost_phys(v | x) × exp( w_v · φ(x) )
              ↑                      ↑
         物理闭式先验            学习的残差修正
```

- **x** = 矩阵/人口特征向量(连续,不离散化)
- **cost_phys** = 物理模型(占用 × 带宽 × 原子吞吐 → 闭式)
- **w_v · φ(x)** = 线性模型,在线更新,捕捉物理模型的偏差
- **决策** = argmin_v cost(v | x)

### 2.2 为什么物理先验很重要

纯 ML(如 FlexSpGEMM 的 LightGBM)需要大量数据训练,且在新人群上外推差。
物理先验提供了**结构化知识**:

```python
# 例: hash 内核的成本模型(闭式)
def cost_phys_hash(features, table_size, block_size):
    # 占用: 表在 smem 中能驻留几个块
    smem_per_block = table_size * 12  # col(4B) + val(8B)
    occupancy = min(SMEM_LIMIT / smem_per_block, REG_LIMIT)
    
    # 带宽: 读 B 行数据(必读)
    bytes_to_read = features['total_flop'] * 12  # col + val
    bandwidth_time = bytes_to_read / HBM_BW
    
    # 原子: hash 插入的 CAS + atomicAdd
    atomic_ops = features['total_flop'] * 2
    atomic_time = atomic_ops / SM_ATOMIC_THROUGHPUT / occupancy
    
    # 表初始化
    clear_time = table_size * 2 / (block_size * LANE_WRITE_THROUGHPUT) * num_rows
    
    return bandwidth_time + atomic_time + clear_time
```

**物理模型的作用**:即使没有训练数据,也能给出合理预测(先验)。
学习模型只需要修正物理模型的**系统性偏差**(如实际冲突率比假设高、
实际原子吞吐比理论低等)。

### 2.3 在线学习:每次运行都是一次训练

```python
# 生产运行时(每次矩阵计算后):
for each bin_population:
    x = compute_features(bin_population)      # 特征提取(已有,零开销)
    v_chosen = argmin cost(v | x)            # 选最优变体
    t_actual = measure_time()                # 运行并计时
    
    # 在线更新(SGD 一步):
    error = log(t_actual) - log(cost(v_chosen | x))
    w[v_chosen] += learning_rate * error * φ(x)
    
    # 不确定性估计(贝叶斯):
    sigma[v_chosen] *= exp(-decay)  # 随观测增多,不确定性降低
```

**关键性质**:
1. **零人工**:内核变了,下一次运行自动开始修正,不需要重跑标定
2. **不确定性感知**:对没见过的特征,不确定性大 → 倾向探索
3. **后悔值界**:可以证明与最优路由的累积差距(Regret)是 O(√T log V) 的

### 2.4 特征向量(连续,不离散化)

```python
features = {
    # 人口级(bin 内统计量)
    'est_median': ...,    # est 中位数
    'est_p25': ...,       # est 25 分位(人口多样性)
    'est_p75': ...,       # est 75 分位
    'maxbl_median': ...,  # B 行长中位数
    'dup_ratio': ...,     # flop/est(重复率)
    'bin_frac': ...,      # 此 bin 占总行数比例
    
    # 阵级
    'n': ...,             # 总行数
    'density': ...,       # nnz/n²
    'skew': ...,          # max_row/avg_row
}
```

**约 10 个连续特征**——不是 3 个离散档。

### 2.5 变体空间(不只是开关)

每个 bin 可选的完整变体(不是 on/off,是**选择哪个路径**):

```python
# bin4/5 的变体空间(4 选 1)
variants_bin45 = [
    {"kernel": "hash_heavy", "compact": "copy_warp"},   # 有序直写
    {"kernel": "hash_heavy", "compact": "csort"},       # 无序+排序
    {"kernel": "LHT2",       "compact": "csort"},       # LHT 梯
    {"kernel": "runtime",    "compact": "csort"},       # 运行时表长
]

# bin9 的变体空间(2 选 1)  
variants_bin9 = [
    {"kernel": "hash_heavy_16253"},  # 素数表
    {"kernel": "LHT2_14"},           # pow2 16384
]

# compact 的变体空间(独立路由)
variants_compact = ["FLAT", "csort", "counting_sort", "copy_warp"]
```

### 2.6 与 FlexSpGEMM 的差异

| 维度 | FlexSpGEMM | 我们 |
|---|---|---|
| 学习方式 | 离线训练(LightGBM/LLM) | **物理先验 + 在线更新** |
| 内核变化适应 | 需重训 | **自动适应(bandit 更新)** |
| 可解释性 | 黑盒 | **物理模型可解释 + 学习偏差可分析** |
| 不确定性 | 无 | **贝叶斯估计,知道何时不确定** |
| 理论保证 | 无 | **Regret 界 O(√T log V)** |
| 特征 | tile 特征 | **人口统计量(连续)** |
| 决策 | tile 大小 + 硬件路径 | **内核变体 + compact 策略** |

## 三、实现路线

### 第一阶段:特征工程 + 物理模型(1 天)

1. 提取每个 (matrix, bin) 对的特征向量(约 10 维连续)
2. 写 hash_heavy / LHT2 / batched 的物理成本模型(闭式)
3. 在现有标定数据上验证:物理模型预测 vs 实测的相关性

### 第二阶段:离线训练 + 验证(1 天)

1. 在标定数据上拟合残差模型 w_v(ridge 回归即可)
2. 在 held-out 72 阵上验证:模型选的变体 vs 手拍门的差距
3. 对比:物理模型 / 纯学习 / 物理+学习(消融)

### 第三阶段:在线更新 + 漂移演示(1 天)

1. 实现在线 SGD 更新(每次运行后 w ← w + η · error · φ)
2. **漂移实验**:故意切换内核版本 → 展示模型自动适应
3. 可视化:决策边界随在线更新变化

### 第四阶段:生产集成 + 评估(1 天)

1. 部署到 hash_product 调度器(host 侧,C++ 实现)
2. 428 阵全量评估
3. 与手拍门 / 查表 / 离线 ML 三臂对比

## 四、论文叙事

```
§5 Signature Routing → §5 Neural-Informed Adaptive Routing

贡献:
  1. 物理先验成本模型(闭式,可解释)
  2. 在线 bandit 更新(内核变化自动适应,Regret 界)
  3. 连续特征 + 完整变体空间(不是离散查表)
  4. 三臂消融(物理 / 学习 / 物理+学习)
  5. 决策漂移演示(内核变化 → 模型自动恢复最优)
```

**一句话**:不是"用 AI 替代 if-else",是"用物理先验 + 在线学习构建一个
内核变化时零人工维护的自适应路由系统"。
