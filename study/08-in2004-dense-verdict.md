# in-2004(dense 场)对拍记分牌 + dense 家族首解剖(2026-09-11 深夜)

> 一句话:**v37 后系统 54.1 vs 38.7ms = 1.40×(v34 时 1.58×)**;差距构成 = 累加家族 −10.4ms(大半是 whale 行路由差)+ 排序/复制 −3.9ms − est 我们赢 +2.55ms。dense kernel 首次并排:他们无独立 count pass,我们的 whale 行走"SMEM 大表 hash + 事后全局重试",他们走"denseIter 一把吞"。

数据:compare/kernel_in2004/(Ocean 20 迭代 + 我方 v37 单跑,双 nsys-rep)。口径 = 全 GPU 内核和(含全部小核;手递手旧文 32.7ms 是前 13 大核之和,同口径全和为 38.7)。

## 1. 系统总记分牌(v37 后)

| 场 | 我方 | Ocean | 比值 | v34 时 |
|---|---|---|---|---|
| F2(hash-vs-hash) | 10.59 | 7.93 | **1.34×** | 1.48× |
| in-2004(dense 混合) | 54.09 | 38.70 | **1.40×** | 1.58× |

## 2. hash kernel:能对齐的档

小行档死平:我 bin3(201,693 行@64T)5.6 ns/row ↔ 彼 bin331(609,479 行@64T)5.6 ns/row。
中重档行集错位大(我 est 梯落点 ≠ 彼 HLL 梯落点),ns/row 跨侧比较口径脏;F2 干净场定谳(study/06)不变:**本体赢 5%,bin5 −17%/bin7 −34% 两处真回归待查**。

## 3. dense 家族首解剖(in-2004)

**我方**(v37,单跑):
| kernel | 行数 | 时长 | ns/row |
|---|---|---|---|
| hash_dense_count(count pass) | 23,212 | 8.15ms | 351 |
| hash_dense_direct(直写) | 10,403 | 3.88ms | 373 |
| dense_struct_numeric(4 launch) | ~12.8k | 3.23ms | — |
| (whale 行走 hash bin8/9) | 6,835 | 4.12ms | 445~1631 |
| (溢出重试 hash_global+CUB) | 9,666 | 6.29ms | 651 |

**Ocean**(每迭代):
| kernel | 行数 | 时长 | ns/row |
|---|---|---|---|
| denseNumeric(5 变体) | 63,987 | 4.28ms | 67 |
| denseNumericIter | 2,395 | 6.71ms | 2,780(单行极重) |
| denseNumericSubWarpDispatcher | 276,195 | 0.93ms | 3.2(小行) |
| copyOutput(每 bin 分段拷) | — | 6.84ms | 他们自己的税 |

**结构性发现**:
1. **他们没有独立 count pass**——我方 hash_dense_count 8.15ms 是纯相位税(手递手旧文 count 8.7 vs 4.7 的当代版)
2. **whale 行走法相反**:我把 est≤16384 的重行留 SMEM hash(bin8/9 共 6,835 行花 4.1ms)+ 9,666 行欠估走全局表重试(6.3ms);他们 est>5449 直接 denseIter(2,275 行 6.3ms 吞掉最重行)——**同款"溢出税"在 in-2004 放大到 6.3ms(F2 是 1.96ms)**,knife#2 的证据又厚一层
3. 他们的 copyOutput 6.84ms(18% 时长)是他们自己的结构税——可用作论文对照点

## 4. 相位总账(v37 vs Ocean,in-2004)

| 相位 | 我方 | Ocean | Δ |
|---|---|---|---|
| est/count | 3.07(count 0.41+mh 2.66) | 5.62(hll 5.09+analysis 1.11−0.58 重叠) | **我赢 2.55** |
| 累加家族 | 32.2 | 21.8 | 输 10.4(路由差异为主) |
| 排序/复制 | 15.1 | 11.2 | 输 3.9 |

排序细目:in-2004 的 1.38M 列 > csort_fused 列宽限(CAP4096→20 位→1.05M),bin7+ 只能走非融合 hash_compact_sort(bin7 133ns/row vs 彼 sortOutputDyn 97);**他们有 sortOutputDyn 动态变体正好兜这个场景**——下一刀候选(或把 fused 键改 64 位)。

## 5. 对下一刀的指向(与 study/07 §6 合并)

1. **溢出/whale 统一处理**(est 分档安全系数 + 重试载体换 SMEM dense 位图):F2 1.96 + in-2004 6.3 + bin8/9 的 4.1 ≈ 全人群最大单靶,预期两位数 ms 级赢面
2. in-2004 专有:dense count pass 8.15ms 能否并入 direct(他们 zero-count-pass 的结构)
3. sortOutputDyn 式动态排序兜底(宽列阵 fused 失效场景)
4. F2 bin5/bin7 两档 hash 内因(不变)
