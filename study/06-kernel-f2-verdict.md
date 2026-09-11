# F2 kernel-vs-Ocean 定谳(2026-09-11,GPU1,nsys per-launch 归一化;09-11 晚修订 v2)

> 结论一句话:**hash 累加 kernel 本体不输(flop 加权总量赢 5%,主档赢 1.20×);真差距全在系统税——排序配置 ~2.6×、溢出重试载体差 6×、est 管线 0.17ms**。11.7 vs 7.9ms 的 3.8ms 缺口归因完毕。
>
> **v2 修订**:昨日把 Ocean `denseNumericIterKernel<10,false,true>`(487 块,0.33ms/迭代)误记为"dense 路由";实为 `SpGEMM.cuh:1570` 的 **overflow 重试发射**(`<<<h_num_overflow_rows>>>`)。Ocean 也有溢出重试,只是载体与安全系数不同(§4)。

数据:compare/kernel_F2/(双 nsys-rep + kern CSV + f2_routing.bin 即 DUMP_ROUTING 的 est/span/flop/bkid/a_len 五数组)。

## 1. 方法:per-launch 行对齐

两边 launch 语义已从代码确认:**均一块一行**(我方 `hash_spa_kernel<0><<<n, hblk>>>` n=档内行数,spgemm_kernel_hash.cu:4122;Ocean `hashNumericKernel<..><<<h_num_outlier_rows, BLOCK_SIZES[bin]>>>`,Wrappers.cuh:315-345)。F2 行集两边几乎同批:

| 我方档(ht=32<<bi) | 行数 | Ocean bin(表容) | 行数 |
|---|---|---|---|
| bin4 ht512 | 1156 | bin331 | 1400 |
| bin5 ht1024 | 23683 | bin673 | 23638 |
| bin6 ht2048 | 42217 | bin1327 | 41223 |
| bin7 ht4096 | 4437 | bin2719 | 5220 |
| bin8 ht8192 | 12 | bin5449 | 24 |
| 溢出重试 | 1319 | 溢出重试(denseNumericIter) | 487 |

F2:71505²,nnz 5.29M,flop 4.93 亿,C_nnz 39.6M。我方 est 为 2 幂量化(中位 1024),两套档口切同一批行完全重合 → 每档 flop 质量可精确共享。bin6 占 63% flop。

## 2. hash 累加 kernel 定谳(逐档,flop 加权)

| 档(我 ht / 彼容量) | flopΣ | 我方 µs | ns/Kflop | 彼方 µs | ns/Kflop | 我/彼 |
|---|---|---|---|---|---|---|
| 512 / 331 | 2.9M | 94.0 | 32.4 | 246.3 | 84.9 | **0.38 赢 2.6×** |
| 1024 / 673 | 109.0M | 1753.5 | 16.1 | 1501.2 | 13.8 | 1.17 输 17% |
| 2048 / 1327(**主档**) | 311.9M | 3270.3 | 10.5 | 3931.4 | 12.6 | **0.83 赢 1.20×** |
| 4096 / 2719 | 68.8M | 819.6 | 11.9 | 613.3 | 8.9 | 1.34 输 34% |
| 8192 / 5449 | 0.3M | 98.5 | — | 55.3 | — | 12/24 行噪声 |

- **总量:6.04 vs 6.35ms(12.3 vs 12.9 ns/Kflop)= 我赢 5%**——"hash kernel 不如"定谳为否
- 剩两个内核层真靶:**ht1024 档输 17%**(23.7k 行)、**ht4096 档输 34%**(4.4k 行)
- 配置差异:彼 block 梯随档 64→128→256→512→1024;我 256/256/512/512/512(小档块过大、重档块过小双向都吃亏,是两输点的头号嫌疑)
- 彼 SMEM 占用/块:331×12B=4KB … 5449×12B=64KB;我 ht×12B=6…96KB(2 幂梯比素数梯浪费 ~30% SMEM)

## 3. 排序/输出:~2.6× 差的机理(已定位)

- 我方 `csort_fused<512,8>`:CAP=4096,512 线程排 ~1300 键(bin6),2.5 键/线程,TempStorage 超大 → occupancy 低;bin6+7+8 合计 **3.00ms**
- 彼方 `sortOutputFused<16,96,21>`:96T×16IPT CAP=1536 紧贴 1327 表,`__launch_bounds__(96,10)`;小行(≤阈值)免排序直拷;三 bin 合计 **0.72ms**;copyOutput 0.19
- 同为 cub::BlockRadixSort 一行一块、src_pos 搭车高位同一招——**差在配置不在算法**:主档每行 62.6 vs 12.3ns(5.1×)
- **工程靶:csort 按档重配 TPB/IPT(如 128T×16IPT CAP2048 给 bin6),预期 3.0→~1ms**

## 4. 溢出重试:两边都有,载体与安全系数不同(v2 核心)

**Ocean 侧(零"可见"重试的真相)**:
- est 安全链:`hllMerge` 里 `est = HLL×1.5~2.0`(经验系数,Utils.h:46:precision5→2.0 / 6→1.5)**再向上吸附到 bin 容量**(HLL.cuh:228-232:`temp = HASH_NUMERIC_BIN_SIZES[b]`)→ 表容量恒 ≥1.5×HLL 估计
- 溢出条件变成"真 distinct 超 1.5×HLL 还要吃掉梯级几何余量"→ F2 上仅 487/71505 = **0.68%** 行溢出
- 重试载体 = `denseNumericIterKernel`(SMEM 位图累积,无表容量上限;`d_global_mem_buffer_2` 兜底),**0.33ms/迭代,690ns/行**
- 代价:tmp 超配 ~2.15×(Σ容量 85.4M 槽 vs 真 C_nnz 39.6M)

**我方侧**:
- est 链:MinHash × **1.15** + 吸附 2 幂 ht → 尾部欠估行多:F2 上 **1319 行 = 1.8%**(是彼 2.7 倍);tmp 超配 1.63×(Σest 64.8M,更省)
- 重试载体 = `hash_global`(全局内存 hash 表 + CUB 分段排序),**1.96ms = 1486ns/行**(hash_global 1.44 + SegmentedRadixSort 0.50 + compact 0.02)
- 逐行对比:1486 vs 690 ns/行 = **载体差 2.2×;再乘行数差 2.7× = 该相位总差 6×**

**结论:不是"他们没有溢出",是他们把溢出买断了**——安全系数花在 est 链(×1.5 起跳)而不是事后重试,重试载体又是免容量的 dense 位图。我方反着选:tmp 省 25%,重试端付 6×。

## 5. 总账核对(v2)

| 相位 | 我方 ms | Ocean ms | Δ |
|---|---|---|---|
| est 管线 | 0.51(mh_merge 0.40 大头) | 0.34(hllMerge 0.17+analysis 0.10) | +0.17 |
| hash 累加本体 | 6.04 | 6.35 | **−0.31(我赢)** |
| 溢出重试 | 1.96(1319 行) | 0.33(487 行) | +1.63 |
| 排序/输出 | 3.23(csort 3.0+warp 0.22) | 0.91(sortFused 0.72+copy 0.19) | +2.32 |
| **合计** | **11.74** | **7.93** | **+3.81 ✓** |

## 6. 测量条件记录(今日 16:01 采集)

- GPU1 空闲全静;CPU co-tenant 负载 load~29/32(xuejing 仿真,9/05-9/10 起跑至今)——**GPU kernel 时长为设备端时间戳,不受 CPU 负载影响**;受影响的只有 wall-clock/发射间隙,本次口径不取
- 复现:`nsys export --type sqlite` 后查 CUPTI_ACTIVITY_KIND_KERNEL(gridX + 时长);BINS_DBG=1 取档行数;DUMP_ROUTING 取 flop 权重

## 7. 工程靶排序(F2 单场,待 c-64/in-2004 复证)

1. **csort 按档重配**(配置刀,出处 = CAP 冗余 2×+occupancy):预期 −2ms
2. **溢出安全系数/载体**(把 ×1.15 提到分档自适应,或重试改 SMEM dense 位图):预期 −1ms
3. **block 梯细化**(ht1024/ht4096 两输点的头号嫌疑,彼 5 级梯 vs 我 2 级):预期 −0.3~0.5ms
4. est 管线 mh_merge 0.40→~0.15(小头)
