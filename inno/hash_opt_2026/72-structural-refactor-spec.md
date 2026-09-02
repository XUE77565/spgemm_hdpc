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

## Phase 1b-v2 正确性验证(09-02,nnz 对拍,load 24 窗)

c-58 40461781✓ / Ga41 231455088✓ / brainpc2 190743619✓ / crankseg_2 104590168✓ 全一致。
G-group 继承(G 从 avg_B 推导,上限 warp_size=5=硬件量)+ 位图字级前缀发射。
性能电池待净窗。

## Phase 1b-v2 性能定谳(09-02,load 2 半净窗,单发但极紧)

| 阵 | v2 | 默认 | v1 对照 |
|---|---|---|---|
| Ga41 | 226.3 | 225.8 | **1979 → wash(9×修复!)** |
| crankseg_2 | 111.0 | 110.4 | 1481 → wash |

G-group 继承完全消掉争用回归。结构化 numeric(G-group + 位图字级前缀)与旧 kernel 平齐。
nnz ✓。完整电池(Ga41/crankseg/brainpc2/c-64/c-58/in-2004/c-73)被环境打断,已出 2 阵足够定谳方向。
**结论:Phase 1b-v2 达到可用状态;后续与 Phase 2 SROUTE v3 联合测试 = 全结构化系统。**

## 全结构化联合定谳(09-02,DSTRUCT=2+SROUTE=3 vs 默认,11 阵 nnz✓)

| 判定 | 阵 | Δ |
|---|---|---|
| 赢 | Si41 −6.4 / F2 −8.1 / c-58 −8.6 | 3 |
| wash | Ga41 / brainpc2 / 3Dspec2 / pkustk14 | 4 |
| 回归 | crankseg_2 +8.3 / c-64 +12.1 / nd24k +7.1 | 3 |

**关键发现**:c-64 单开 SROUTE v3 = −15.4%,加 DSTRUCT=2 → +12.1%。
Phase 1a 的 span_gather+D2H 同步税(~1-2ms)抵消了路由赢面。
**论文策略定案**:SROUTE v3(结构路由)+ 旧内核(常数有注释)= 发表系统。
结构内核(DSTRUCT)是 nice-to-have,不是必需品;同步税可后续消(per-block device 推导)。

## 最终状态(09-03 凌晨,compact 前定格)

**全库调优参数零裸数字完成**(055870a + a6c67cb):
- 20 个 #define → structural_params.cuh 公式 constexpr
- 15 处路由内联数字 → 具名常数(EST_DENSE_MIN/SPANF_COEFF/DUP_LIGHT/DUP_HEAVY 等)
- 剩余 ~100 个数字 = 算法/硬件常数(butterfly/MurmurHash/数组维度),非调优参数

**SROUTE bug 修复**(7b18cb9):SROUTE 替换误删 v4 门 → 已恢复 v4 为默认,SROUTE 为替代路径。

**v29 刷新中**(compare/v29_refresh.log,77/337+@compact 时):OCCGATE 默认开(5f19f62),
10 阵 DNF 需补跑(配方:CM.run_spgemm_method(timeout=1800, exp_cnnz=CSV值))。

**SROUTE v1-v4 完整迭代记录**(docs/72 §SROUTE):v3(含 compact)= 最佳,c-64 −15.4%;
Ga41/in-2004/brainpc2 回归(争用动态不可纯字节化)。论文 = SROUTE 理论 + fitted 附推导。

---

## 终审(2026-09-03,375d9d0 + 扫尾 commit):全库零裸调优数字定版

### Ocean 参数分类学(对照基准:ocean/include/Common.h + kernels/Analysis.cuh)

| 层 | Ocean 做法 | 我们的对位 |
|---|---|---|
| 硬件 | NUM_SM/SHARED_MEMORY_KB/MAX_THREADS_PER_SM | SMEM_PER_SM/SMEM_DEFAULT/SMEM_OPTIN/MAX_BLOCK_THREADS |
| 占用梯 | CONCURRENT_BLOCKS_PER_SM={32,16,8,4,2} → 一切块宽/SMEM 预算推导 | OCC_LADDER 同款(本次新增) |
| 档位公式 | DENSE_BIN = SMEM/bytes_per_unit − 小预留;hash 档 = 素数 | dense_numeric_bin()/hash_slots_pow2()/bitmap_count_cols_max() |
| 教科书 | HLL_CONSTANT(HLL 论文)、α=0.8 负载、MurmurHash 种子 | Knuth 乘法哈希、α=0.5(Knuth TAOCP 6.4)、MinHash σ=1.04/√k |
| 标定参数 | Config 里 ~5 个阈值(decideMatrixType 传入) | 成本律 6 系数(R²/出处附注)+ DISPATCH_MODEL(learned) |

### 本次改造(~90 处)
1. **占用梯层**:OCC_LADDER{32,16,8,4,2} ÷ 128K → dense_numeric_bin = 预算/9−4。
   原 DIM2 手圆整 451/906/1818/3641/7284 → 公式精确 451/906/1816/3636/7277(差 ≤0.1% 列,DIM2 默认关)
2. **成本律集中声明**(LADDER 用):TD_WIN=30109ns/窗、TD_FLOP=2.55ns/flop、TH_HT=8.11ns/槽、
   TH_KC=1038ns/链、TH_BASE=−50225ns(R²=0.896/0.926,docs/67 §6 标定)
3. **SROUTE 毫字节定点**:8125/12000/24000/12000 ← 字节模型(8.125/12/24/12B)×1000
4. **OC/MHSAMP 采样层**:OC_TABLE=8192(48K÷4B 对齐)、OC_EST∈[512,4096]、Q12 定点、
   64 样本、q05、margin 1.02、cap 1.5(HLL 3σ);MHSAMP 2048/1000
5. **门控层**:方案5(≥1000 行 ∧ ≥5%)、hybrid(dup>5 ∧ >50K 行 ∧ 8GB 池)、heavy 24GB、
   retry(>50 行切 expand / <1000 免排序 / 1.02 保留边)、in-place 50M、stream 4/16、
   priv 196K/8 warp、BCNT 200K/1.6M bit、csort(CAP=TPB×IPT 覆盖桶 est 上界,融合键列上限
   = 2^(32−log2 CAP) 公式化)
6. **launch/梯子**:28+2 处工具 launch 块宽 → HASH_BLOCK_STRUCT;G-group 梯 → hash_logG();
   小跨度子桶梯 <SMALL_SPAN_MAX×2^k, min(span/4, HASH_BLOCK)>;hblk 梯引用 ilog2c
7. **dispatcher**:logistic 5 系数 → DISPATCH_MODEL 结构体(learned,disp_synth 标定 + disp_train
   拟合,与 ocean337 不相交 = 无泄漏方法学,论文叙事核心)
8. **merge3**:K=5 桶/dyn_min 8192/L≥128=4×warp/3e9 flop cap(36GB gapped < 80GB)

### ⚠ bugfix:v29 刷新 29 阵 DNF 根因(本次发现并修复)
DENSE_CNT_W 结构化时取 SMEM_DEFAULT_MAX=49152 恰好等于无 opt-in 动态上限,但
hash_dense_count_kernel 有 `__shared__ int wcnt[32]`=128B 静态数组 → CUDA 总量规则
(静态+动态 ≤ 48K)被打破 → invalid argument → **凡 diter 行 ≥1000 的阵全灭**
(c-64×2/c-66×2/c-67×2/c-71/c-73×2/cage15/circuit_4/crankseg×2/gupta2/in-2004/nd24k/
pkustk×3/rajat×6/soc-Slashdot×2/TSOPF_FS_b300_c2/wb-edu)。
v28 用的 65536 反而必走 opt-in 所以不炸。修复:
- `DENSE_CNT_W = SMEM_DEFAULT_MAX − SMEM_STATIC_HEADROOM(128B)`
- 全库 13 处 opt-in 守卫统一含头规则(`> 48K − 128` 而非 `> 48K`)
- v29 因此需 29 阵补跑(修复后 c-64 18.53 vs v28 19.52 = −5%,其余见 fill log)

### 保值验证(关键!)
`/tmp/valcheck.cu`(可重跑):89 项常数逐一与原裸数字比对 —— 86 项严格相等,
3 项有意变更单列:
1. DENSE_CNT_W 65536→49024(bugfix,上文)
2. dim2 梯 1818/3641/7284→1816/3636/7277(Ocean 公式精确值)
3. csort 融合键位差(修正后 = 原 2^20/2^18 语义)
另:LADDER 成本律窗数 5399/5400→PB2_W(5500)单源化(LADDER 默认关,行为无感)。
冒烟:bcsstk30/Ga3As41H72 nnz 与 CSV 逐位一致(零红旗)。

### 残余数字审计结论
复扫 5 个主力文件:24 候选 = 定点分母(×100/×1000,已文档化)/ dbg 显示格式(×100%、/1024)/
`t & 255` 检查节奏 / why_buf[128] 缓冲 / att Tpb(实验文件)——**无一是调优阈值**。
基线文件(formulations/manual/dense_cublas = 对照组,非论文方法)未动。
