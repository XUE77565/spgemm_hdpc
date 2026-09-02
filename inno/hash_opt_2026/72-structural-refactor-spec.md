# docs/72 结构化重构规格(用户钦定最高优先,09-02:"不要魔法数字,全面学 Ocean 参数")

**原则**:发表级系统 = 每个常数可从硬件/算法推导(Ocean 模式);拟合门全部退役为实验脚手架。

## 结构常数表(全部推导,零拟合)

```
SMEM_BUDGET[] = {48K, 100K, 200K}(H100 opt-in 档;硬件手册量)
dense 每列字节 = 8(val)+1/8(bitmap)= 8.125B → 档位 dense_cols_max[i] = SMEM_BUDGET[i]/8.125 − 32
  (Ocean: DENSE_BIN = SMEM/(8+1)−32 同式)
hash 每槽字节 = 4(key)+8(val)= 12B → hash_slots_max[i] = SMEM_BUDGET[i]/12
  bin 梯 = {32,64,...}×12B ≤ 预算(2 的幂由 SMEM 对齐推导,非拟合)
路由:dense ⟺ span×8.125B ≤ ht×12B(数组不比表贵)∧ span×8.125B ≤ 档位预算
  (= Ocean 双梯,在成本∝字节的内核上才成立 → 前提 = Phase 1)
残余系数(≤5,各有据):HLL/MinHash 误差补偿式、expand=重试期望式、ESC≤warp(结构)
```

## Phase 1:dense 内核现代化(前提)

新 hash_dense_mod_kernel(替代 window/count/direct 三件):
- 窗 = span(≤档位),超档 span 才多窗(窗宽=档位,不再是固定 5400)
- SMEM = val[span] + bitmap[span/32] + prefix[span/32](发射用)
- symbolic/numeric 可拆(位图 count→精确偏移→直写)或单遍(值+位图同窗)——按 Phase 1 实测定
- 发射 = 字级并行 popcount 前缀(~span/32 项)→ 升序位直写;废除 warp0 串行 5400 前缀
- 吞并 BCNT(位图 count 成为内建)、窗宽地板病(Ga41)、串行前缀税(全阵)、c-73 的统一 max-span
  绑架(按档分 launch,同档行同 SMEM)

## Phase 2:路由换结构比较(compute_bucket_kernel 重写)

结构梯替全部拟合门;保留 ultra(≤16=ESC 位)/batched 小行(结构:warp 表预算)。

## Phase 3:验收

nnz 对 scipy + 电池(Ga 族/c-73/c-64/in-2004/crankseg/守卫)≥ v28 + 全量 refresh 对照
+ **代码审查:grep 无裸数字阈值**。
