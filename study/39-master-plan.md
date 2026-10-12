# study/39:终极计划——管线减重 + L0 部署 + L1/L2 突破尝试(10-12)

> 用户指示:"L1 L2 能够做出来是一个很大的突破"
> 起点:337 GM 1.2740(恒C)/ 1.1958(oracle)/ 赢 28% 输 64%
> **10-12 晚更新:1b 已落地定谳(DCEST V3,study/40,commit fc49855)**

## 全景:四阶段递进

```
Phase 1: 管线减重(修基础设施)
  ├── 1a. 双估计器(HLL+MinHash, L0 自适应)     → 估计器 9.5→~1ms   ← 下一刀
  ├── 1b. 消灭 dense_count ✅(10-12 DCEST V3)
  │     机制完成+守门闭环;电池 nd24k −14.8/in-2004 −10.4/Ga3As3H12 −4.1;
  │     c-64 +14/pre2 +11.9(=L0 择阵依据);margin 实测 200(证书零触发);
  │     L2 溢出证书提前落地;组合门 D5H/K7 全绿
  └── 1c. 相位融合(17→8)                        → 减 launch 间隙
        (+顺手刀:DCEST 链的两个 D2H 同步点合并,~0.2-0.5ms/阵)
  1b'. diter 解毒屏(小,先于 1a 执行亦可):149 中毒阵里挑 diter 重的 20-30 阵
        DCEST on/off 电池 → ①给 1b 价值入账(GM 级预估)②给 L0 攒真实特征
  
Phase 2: L0 部署(确认收益)
  ├── 2a. 恒A 为默认 + 选择性 k7/dense
  ├── 2b. 337 验证(新决策维度 ×3:DCEST 开关[dense_count vs 链费]、
  │       DCEST_MARGIN[125-200 按证书直方图]、估计器选择[1a 产物])
  └── 预期 GM: ~1.15-1.20

Phase 3: L1/L2 突破尝试(核心创新)
  ├── 3a. 探针校准:一发默认路由 → 测各武器 phase_share
  ├── 3b. L1 v2 训练:标签 = C_kernel + C_phase_share(系统级)
  ├── 3c. L2 证书:k7 count 证书(已有)+ ✅ dense 溢出证书(DCEST V3 已实战:
  │       actual>alloc→判废→merge3 回退,判据可算无阈值)
  └── 3d. 部署 + 337 验证
  → 成功 = 论文头条突破 / 失败 = 更精确的否定结果

Phase 4: 论文
  ├── 无论 L1 成败都写
  ├── 成功 = "首个学习型逐行 SpGEMM 调度器 + 证书兜底"
  └── 失败 = "我们精确定位了逐行调度的物理极限并刻画了根因"
```

## Phase 1: 管线减重(修好 L1 的地基)

### 1a. 双估计器(自适应估计选择)← **下一刀(10-12 晚更新)**

> 开场动作(半小时,先于写码):DDUMP_ROW 取证 in-2004 那 2 行结构性欠估
> (actual=raw×1.5-2,证书直方图左尾)——弄清它们是什么形态(超长链?skew?
> sketch 空 partition?),既是估计器盲区的标本,也是 HLL/MinHash 分界的输入证据。

```cpp
// L0 判断(矩阵级):
if (estimated_avg_dup < 1.5) {
    estimator = HLL;      // ~0.5ms, est≈flop 精度足够
} else {
    estimator = MinHash;  // ~5ms, 精度防 retry(值回票价)
}
```

- 物理出处:P(retry)×cost(retry) vs cost(MinHash)−cost(HLL) 的平衡点
- 创新点:**自适应估计选择**(ocean 一刀切 HLL,我们按矩阵特征选)
- HLL kernel:参考 ocean 原理,自己写(不是 copy)

### 1b. 消灭 dense_count

> **10-12 已落地 = DCEST V3(study/40)**,实测修正了本节两个纸面预期:
> ①margin 不是越小越好——它时间中立(只放大 allocation 间距),按"溢出证书零触发"实测
>   定 200(MinHash 左尾到 0.5,in-2004 2 行结构性欠估);证书+判废回退是 <2.16 margin
>   的安全前提,已从 3c 提前落地。
> ②数值核行内连续直写+自报精确 row_nnz → V2 的 fixup/sentinel/shift 全是死代码或带竞态,
>   V3 = 双 buffer 紧凑拷贝(仅 dense 行,k7/D5H 挪到链后写 exact 基址)。
> 电池:nd24k −14.8% / in-2004 −10.4% / Ga3As3H12 −4.1%;c-64 +14%(dense_count 本来
> 便宜)/ pre2 +11.9%(4 行巨鲸)——赢亏判据(dense_count vs 链费)是 L0 特征维度。

```cpp
// 旧路径:est(近似) → dense_count(精确,O(flop)) → dense_direct(写)
// 新路径(V3):alloc = (est==flop ? est : est/expand) × margin → dense_direct(gapped 直写)
//   → 证书(actual>alloc → 判废回退)→ 二次 scan → 双 buffer 紧凑拷贝 → k7/D5H 写 exact 基址
```

- 效果:dense 启动税从 82ms→≈0 → 更多阵受益于 diter(L0 择阵)
- 与 hash 的 retry 机制对称(hash 也是 est+retry;dense 溢出走整阵判废,cursor 核已有
  rank≥cap 截断同语义)

### 1c. 相位融合

```
旧(17 相位): count_flop → mh_construct → mh_merge → est_scan → binning
             → accumulate → retry → cnnz_scan → compact+sort → d2h

新(~8 相位): [估计+路由融合核] → accumulate → [retry+compact 融合核] → d2h
             (如果 k7/dense 被激活,各加一个相位)
```

- 关键:HLL 的更新可以融合进 count_flop(同一个核顺便做)
- binning 可以融合进 est_scan(算完 est 直接分 bin)
- cnnz_scan + compact 可以融合(一个核做两件事)

## Phase 3: L1/L2 突破尝试(核心)

### 为什么这次可能成功

| 之前失败的原因 | Phase 1 修复后 | 残余风险 |
|---|---|---|
| dense_count 爆炸 | ✅ 已消灭 | 低 |
| 估计器开销太大 | ✅ HLL 换装 | 低 |
| 标签不含集体效应 | ✅ 新标签含 phase_share | **中** |
| 占用效应 | ⚠️ 部分缓解(减相位) | 中 |
| 发射几何变化 | ⚠️ 相位少了,变化小了 | 低 |

### 3a. 探针校准(一次运行)

```python
# 第一次运行(默认路由=恒A):
#   测量: dense_count_time, compact_sort_time, retry_time
#   计算: phase_share[dense] = dense_count_time / num_dense_rows
#          phase_share[hash] = (compact+retry) / num_hash_rows
#          phase_share[k7] = 0
```

### 3b. L1 v2 训练

```python
# 标签 = C_kernel(row, w) + phase_share[w]  # 系统级成本
# 
# 特征 = 21 维(已有:a_len, flop, est, span, ht, W, logs, ctx, ratios, maxbl, collision)
# 模型 = GBT 回归(每武器一个)
# 验证 = LOO 按阵留一

# 关键区别 vs 旧 L1:
#   旧: 标签 = C_kernel(不含 phase)
#   新: 标签 = C_kernel + phase_share(含集体效应的摊账)
```

### 3c. L2 证书

```cpp
// k7 证书(已有):flop - distinct ≤ FIX_MAX → 安全
// dense 证书(新增):est × margin ≥ 实际写入 → 安全(溢出则 retry)
// hash 证书:表不满 → 安全(溢出则 retry,已有)
```

### 3d. 成功判据

| 指标 | 门槛 | 含义 |
|---|---|---|
| L1 vs 恒A | **<1.00**(净赢) | L1 有部署价值 |
| L1 vs oracle | **>0.90**(捕获 90% 天花板) | L1 接近最优 |
| 无阵回归 | **>1.10 的阵 = 0** | L2 证书有效 |

## 预期数字(完整路径)

| 里程碑 | GM | 来源 |
|---|---|---|
| 当前(恒C) | 1.2740 | 已测 |
| +Phase 1(管线减重) | ~1.20-1.22 | 估计器 −8ms + dense_count 归零 |
| +Phase 2(L0 部署) | ~1.15-1.20 | 配置选择优化 |
| +Phase 3(L1 成功) | **~1.10-1.15** | 逐行调度精细优化 |
| +E3(重带) | ~1.05-1.10 | 集群分片表(远期) |

## 论文叙事(四层贡献)

```
1. 否定结果(已证):逐行 kernel 时间 ≠ 逐行系统成本(集体效应)
2. 管线减重(Phase 1):自适应估计选择 + dense_count 消灭 = 系统级改善
3. L0 架构(已证):矩阵级配置选择 → 6.1% GM 改善(337 验证)
4. L1+L2(Phase 3,尝试中):系统级标签训练 + 证书兜底
   → 成功 = 首个可部署学习型逐行 SpGEMM 调度器
   → 失败 = 精确刻画物理极限(集体效应不可完全建模)
```

## 时间表(10-12 晚更新)

| 时段 | 任务 | 产出 |
|---|---|---|
| ✅ 10-12 | Phase 1b DCEST V3 | 机制+守门+电池定谳(fc49855) |
| 半天 | 1b'. diter 解毒屏(20-30 阵电池) | 1b 的 GM 级预估 + L0 特征数据 |
| 1-2 天 | Phase 1a 双估计器(开场:DDUMP_ROW 取证欠估行) | 估计器 9.5→~1ms |
| 2-3 天 | Phase 1c 相位融合(含 D2H 同步合并) | 17→8 相位 |
| 3 天 | Phase 2 L0 重跑 337(新维度:DCEST/margin/估计器) | GM ~1.15-1.20 |
| 1 周 | Phase 3 L1/L2(3c 的 dense 证书已提前落地) | L1 成败定谳 |
| 5+ | Phase 4 论文 | 投稿稿 |
| 3 | Phase 3a+3b(探针 + L1 v2 训练) | L1 模型 |
| 4 | Phase 3c+3d(L2 证书 + 部署验证) | L1 成败定谳 |
| 5+ | Phase 4(论文) | 投稿稿 |
