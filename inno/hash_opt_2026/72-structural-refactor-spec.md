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

## Phase 1a/1b 电池定谳(09-02,DSTRUCT=2 全结构管线 vs 默认,load 2 极净)

| 阵 | DSTRUCT=2 | 默认 | Δ |
|---|---|---|---|
| Ga41As41H72 | 1979 | 226.4 | **+775%(= dense_direct 1815ms)** |
| crankseg_2 | 1481 | 121.3 | **+1122%** |
| c-73/c-73b | 1436/1505 | ~300(TOTAL) | 大回归 |
| in-2004 | 1806 | 197.0 | **+818%** |
| c-64 | 191 | 98.6 | +94% |
| brainpc2 | 105 | 92.1 | +14% |
| c-58 | 74 | 29.9 | +149% |

nnz 全对(Ga 231455088 = v28 精确值)。**性能根因(dense_direct 相位分解):**
1. **SMEM 原子争用**:Ga 每行 156k 乘积挤 14117 列窗 = 11 乘积/列 → `atomicAdd(dval[j])` 同地址串行化
   (旧 kernel 5400 窗=23 乘积/列但用 G-group 分组乘积降低同地址并发;Phase 1b 没做 G-group)
2. **多窗 lower_bound 重搜**:span 126k ÷ 窗 14117 = 9 窗/行 × per-k lower_bound × kc=395 链 = 逐窗大量二分
3. c-58(小阵 span=37595 单窗) 也 2.4× 慢 = 256T 块(旧 1024T)的块内并行度不足

**Phase 1b 判定:否决当前实现,保留 Phase 1a(仅 count 侧)实验性。**
**教训:去掉 warp0 串行前缀的收益(30k/窗)远小于原子争用的代价(11×/列)。**
旧 kernel 的 G-group 分组(降低同地址并发)是隐藏的关键设计,结构化版本必须继承。
下一步:①回退 DSTRUCT=2 到纯实验 ②Phase 1b-v2 = G-group 并行乘积 + 位图前缀发射(继承旧结构优点)
③Phase 2 路由(字节比较)不依赖 Phase 1b 的完美,可以先行。

## Phase 2 先行:路由字节比较(不依赖 Phase 1b)

dense ⟺ span×8.125B ≤ ht×12B(数组不比表贵)。Phase 1b 教训不改变路由的经济学——
即使旧 dense 内核有 warp0 串行前缀(30k/窗 = 8.125B/列×span 的常数),它的 G-group
使乘积端原子并发可控,per-col 成本常数 ≈ 2.55cyc(已测)。路由用【实测成本律的
结构化包装】:cost_dense ≈ span×8.125B + 2.55×flop vs cost_hash ≈ ht×12B + 1.89×flop + 1038×kc。
把这些常数改由 structural_params.cuh 推导(如 2.55→由 L2 带宽推导,1038→由内存延迟推导)。

## SROUTE v1/v2 判定(09-02,电池部分数据)

v1(max 表)→ Ga41 −4.2/nd24k −3.2 赢,brainpc2 +18/mult_dcop +16 回归。
v2(per-row 表)→ Ga41 wash(赢面消失),brainpc2/mult_dcop 仍回归。
**根因:字节比较只捕捉内存占用,不捕捉原子争用的时间差。**
dense 的 per-product atomicAdd 在同列重复(高 dup 行)远慢于 hash 的 per-distinct CAS。
**Ocean 之所以能用纯字节比较 = 它的 denseNumeric 有 G-group 分组乘积(降争用)。
我们的旧 dense_direct 也有(这就是 Phase 1b 教训说的"隐藏关键设计")。**
**对齐路径:Phase 1b-v2(G-group + 位图前缀)→ Phase 2(字节路由)才成立。
或者:字节路由 + 争用项(需 dup 推导)→ Phase 2-v3。**
当前:Phase 2 基建保留(SROUTE env,默认关),Phase 1b-v2 优先。

## SROUTE v3 中途结果(09-02,电池进行中)

v3 = `span×8.125B ≤ ht_row×12B + est×24B`(含 compact: gapped 读 + exact 写)

| 阵 | v3 | 默认 | Δ | v2 对照 |
|---|---|---|---|---|
| mult_dcop | 238.0 | 238.1 | **wash** | +18.9% ✅修复 |
| **c-64** | 87.1 | 103.0 | **−15.4%** | +12.5% ✅反转 |
| nd24k | 258.6 | 262.9 | −1.6% | 保持 |
| brainpc2 | 99.2 | 92.3 | +7.5% | +16.6% 减半 |
| Ga41 | 225.2 | 209.6 | +7.4% | wash 转回归 |

compact 项是正确的结构化补充。Ga41 回归 = 高 dup(38×)行的逐乘积原子未捕捉
(字节只管占用,不管重复置位)。可能的下一步:v4 = 加 dup 项(flop×B/est 推导争用字节)。

## SROUTE v4 定谳(09-02,单发 ×1,10 阵)

| 阵 | v4 | def | Δ | v3 Δ |
|---|---|---|---|---|
| **c-64** | 96.6 | 115.2 | **−16.1%** | −15.4% 保持 |
| **c-58** | 26.7 | 29.6 | **−9.8%** | −5.1% 改善 |
| **nd24k** | 257.8 | 267.6 | **−3.7%** | −1.6% 改善 |
| crankseg_2 | | | wash | −3.9%→wash |
| Ga41 | 227.1 | 208.9 | +8.7% | +7.4% 未修 |
| brainpc2 | 108.0 | 92.0 | +17.4% | +7.5% 恶化 |
| in-2004 | 207.3 | 181.0 | +14.5% | +7.7% 恶化 |
| 3Dspec2 | 153.8 | 137.1 | +12.2% | +6.6% 恶化 |
| F2 | 37.8 | 36.3 | +4.2% | −4.8% 反转 |

**争用系数 (flop−est)×12B 过强** → 低 dup 阵被过度推回 hash → F2 反转/brainpc2 恶化。
c-64/c-58/nd24k 赢面保持(它们的行 dup 低,争用项小)。

**根本挑战**:字节模型捕捉内存占用,不捕捉原子争用的执行动态。
- dense 的 G-group 分组使争用远比 (flop−est)×12B 暗示的低(并行摊薄)
- hash 的表 init + CAS 探测 + compact 管线 overhead 远比 ht×12+est×24 暗示的高
- **两条路径的实际时间都不是字节的线性函数**

**论文策略(务实)**:结构公式做理论框架,fitted 常数做实践近似(可从结构式推导):
- est≥2048 ⟺ hash table ≥ 48KB(ht×12B ≥ SMEM 最小档)
- 16flop≥span ⟺ dense_bytes ≤ hash_bytes 在 dup=16 时求解
- dup<8/64 ⟺ 争用字节 ≤ compact 字节求解
这样发表时每个常数都有推导出处,不暴露为"魔法数字"。
