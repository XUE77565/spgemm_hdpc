# spec91:重 est 行瘦身(探测环指令节食 + 固定成本削减)

> 靶子(§3.1c/e/f 定谳):hash 八阵唯一输阵 nd24k 1.484 + in-2004 est>1K 带弱势。
> 两型输因已分案:**in-2004 est1-5K=固定截距税 16.6K cyc;nd24k 重乘积=边际税 0.8 cyc/fl**。
> 本卷=①probe_hist 编译期化(hygiene,已落地)+②截距分解测量+③指令节食刀。

## 1. 证据台账(09-23 desk 复核,全部复现)

### 1.1 同行集拟合复现(38,111 行 est 1-5K,rt8×rc join)

| 侧 | 斜率 cyc/fl | 截距 cyc |
|---|---|---|
| 我方(A+ 窗 rt8) | 4.91 | **42,208** |
| ocean(rc) | 4.21 | **25,655** |
| 差 | +0.70 | **+16,553** |

est 1-5K 行结构:**a_len 中位 24(p75=25,超均匀),flop/a_len≈183**(少 k×长 B 行);
理想乘积功 ≈ 24×183×5cyc ≈ 22K cyc,截距 42K = 固定成本>边际功 —— 瘦固定成本即赢。

### 1.2 口径勘误(重大):ocean ROWCYCLES 窗【含 extract】

AccumulatorHash.cuh 600-660:彼 rc0=kernel 入口,写点在 compactAndSort/compactOnly
(就地排序+写回)**之后** —— 彼窗 = 清表+探测+**提取排序写回**全包。
我方 A+(RT_NOEXTRACT)排 extract。**§3.1 的 16.6K 截距差是"我方不带 extract vs 彼方带
extract"仍输 = 真探测侧差距更大,口径偏我仍输=保守判读不变,但归因要重算。**
→ 待办:rt8 是否真 A+ 需实证(同阵 NOEXTRACT on/off 双跑对截距之差=extract 份额)。

### 1.3 结构对拍(ocean 源码,AccumulatorHash/Hashmap.cuh)

| 维度 | 我方 | ocean | 判 |
|---|---|---|---|
| 表尺寸 | next_pow2(2×est),est3000→8192 | 素数梯 {331,673,1327,2719,5449,16253} LF0.8,est3000→5449 | **彼小 1.5×** |
| 清表 | col(-1)+val(0) 12B/槽 | ids+values 同 12B/槽 | 等价(量差随表尺寸) |
| SMEM/块 | 96KB→2 块/SM | 64KB→**3 块/SM** | 彼多 1 块驻留(停顿换指令的资本) |
| 每乘积流 | hash mul+and→read→cmp→CAS/Add | hash mul11+%→read→cmp→CAS/Add+occupancy Add | 近同构;彼多 occupancy 原子 |
| 窗内栅栏 | 2(清表后+探测环后) | 2(init 后+探测环后) | 等价 —— **"18 栅栏税"归因作废**(18=全 kernel 含 extract 段) |
| 溢出 | 探测链长≥ht 检测(retry kernel) | nearFull→GMEM 中途换表 | 彼窗内自愈 vs 我方二核 |
| G 选择 | hash_logG 梯(G=32@ht8192) | localLoadBalance 同构 | 等价 |

### 1.4 §3.1c "栅栏税" 结论修正

18 个 __syncthreads 是全 kernel 计数;A+ 窗内仅 2 个,且彼方同 2 个。
**截距差真嫌疑重排**:①表尺寸 1.5×(清表+SMEM 驻留+探测 cache 局部性)②G=32 对
a_len=24 行的组空转(32 组只 24 k→8 组闲置+组长尾=末栅栏等待计入 thread0 时钟)
③探测环 per-k 固定开销×24k ④(rt8 若非 A+)我方 extract+3 栅栏独担。

## 1.6 净窗终版对拍(09-23 15:2x-15:3x,双卡全净 14MiB,双侧同窗重测)——**斜率翻案+截距稳健**

| est 带 | n | 中位比 | 胜率 | 判 |
|---|---|---|---|---|
| <64 | 500 | 0.996 | 50.4% | wash |
| 64-255 | 300,205 | **0.711** | 79.2% | 胜 |
| 256-1K | 195,213 | **0.344** | 99.5% | 碾压 |
| 1-4K | 37,425 | **1.206** | 25.4% | 输(截距主导) |
| 4-16K | 822 | **1.900** | 3.8% | **重输(自旋争用)** |
| **全阵** | 534,165 | **0.497** | **82.7%** | 大胜 |

净窗拟合(est1-5K):**我 slope 2.26/int 41,668;彼 slope 4.15/int 25,780**
- **斜率翻案**:13:09 rt8 的 slope 4.91 = 污染/CNT_FUSE 虚高;净窗真实边际 **2.26 vs 4.15
  = 我方每乘积更便宜**(B16+Knuth+read-first 全兑现)——"砍动态指令"靶心修正为"砍截距+砍争用"
- **截距稳健**:42.2K/39.8K/41.7K 三测一致;16K 差 = est1-4K 带 1.206 输因全部
- **4-16K 带 1.90 拟合外推不达**(外推应 0.8 赢,实测重输)= 大表 SMEM F64 自旋的**动态争用税**
  (热槽重试环,fit 静态斜率测不到)——刀 E 靶心的独立实证
- ocean rc 生成法:ROWCYCLES=path ./ocean/spgemm /tmp/oc_in-2004.csr ocean/config/bench.json

## 1.5 SASS 级终局破案(09-23 15:3x,desk,双侧二进制反汇编)——**边际税 100% 定位**

我方 hash_spa<0,1,0> 命中路径(/tmp/sass91_h010.txt 0x2040-0x20a0):
```
LDS.64 R8,[sh_val] ; DADD R10,v+R8 ; ATOMS.CAST.SPIN.64 [sh_val],R8,R10
ISETP×2 ; @!P0 BRA retry   ← H100 SMEM 无原生 F64 ADD → ptxas CAS 自旋模拟,5-6 条/乘积
```
ocean 2719 档(HYBRID=false,est≤2.7K):**同款 CAST.SPIN**(0x0f90-0x0fe0)——中档双方
等价,我方 B16 还省 1 条 B 载荷(1×LDG.128 vs 2×LDG),静态指令数近似打平。
**ocean 16253 大档(HYBRID=true,est≥~5K/nd24k 带)**:
```
0x1540: REDG.E.ADD.F64.RN.STRONG.GPU desc[..], R12   ← 原生 fire-and-forget 全局 F64 归约
```
**1 条零等待,值区在 GMEM**;探测/claim 仍在 SMEM ids。
→ **0.8 cyc/fl 边际税 = 我方 F64 累加的 CAS 自旋模拟 vs 彼原生 REDG,定谳**。
彼 personality(ncu:多停顿少指令)同源:REDG 不阻塞发射 → 指令流精简,停顿全在
long_scoreboard(GMEM 延迟)+mio_throttle(STM 原子)——用停顿换指令的正交易。
彼 GMEM 值区还带来:SMEM/块减半(ids-only)→大表档驻留更松;16253 档才装得下大行。

## 2. 刀口设计(按证据强度排序)

### 刀 E(升头号,SASS 铁证):大表档 GMEM 值区(VAL_GMEM)
- 形态:ht≥阈值(默认 4096)的 bin 发射 `<MODE,1,0,VAL_GMEM=1>` 实例:
  val_arr = gmem 块切片(blockIdx.x×ht),探测/claim 仍在 SMEM sh_col;
  累加 `atomicAdd(&val_arr[slot],v)` 返回值弃用 → ptxas 出 REDG.E.ADD.F64(1 条);
  块内自清零(STG 批量)替代 SMEM 清 val;extract 从 gmem 读值(并行扫描,GMEM 延迟可叠)
- 连带红利:SMEM/块 ht×12B→ht×4B(ht16384:196KB→64KB)——**HASH_CAP 硬帽的根因
  消失**(wb-edu OVERFLOW 行后续可入 hash 车道,另立刀);寄存器态简化(自旋循环状态消除)
- 风险:GMEM 值区每乘积 8B L2 RMW 流量(nd24k ~2.3G 乘积=19GB L2 原子流量,fire-and-forget
  吞吐高,RED 单址排队在 L2 侧不阻塞 warp);零初始化正确性(块自清零+栅栏)
- 验收:nd24k 中位比 1.484→<1.1;值门三件套+memcheck 全绿;小表带(甜点区)零回退
  (VAL_GMEM 只在大表 bin,甜点带路径 bit-for-bit 不变)

### 刀 A:表尺寸贴 est(去掉 2× 余量的 pow2 上取整)
- 现状 ht=next_pow2(2×est):est3000→8192(用 37%),彼 5449(LF0.55)
- 方案:hash 表改素数/紧尺寸(modulo hash)或 pow2 但 1×余量+LF 监测→溢出才 retry
- 预期:清表 -33%、SMEM -32KB/块(96→64KB=3 块/SM 驻留对齐彼)、cache 局部性↑
- 风险:紧表=探测链变长(REXP 教训:无 RH 时紧表全 +56%)→ **必须搭配 LF 监测+溢出
  retry 路径已有**(RETRY_SMEM 车);紧表+RH(spec90)才是完全体,本刀=先摘 pow2 浪费
- 验收:in-2004 est1-5K 带截距 42.2K→<30K;nd24k 中位比 1.484→<1.2;八阵全绿不回退

### 刀 B:a_len 稀疏行的小 G 自适应(G 梯吃 a_len)
- hash_logG 只看 ht;est 大但 a_len=24 时 G=32→num_groups=32>24=组空转
- 方案:G=min(hash_logG(ht), next_pow2(a_len 上界));或 num_groups=min(1024/G,a_len)
  后按 k 重均分(组跨 k 领取)——后者已有 stride 结构天然支持,改起点划分即可
- 验收:in-2004 est 带胜率 79.8%→>85%;c-64/crankseg 不动(彼 a_len 大)

### 刀 C:探测环指令节食(nd24k 边际税)
- 靶:每乘积 2-4 槽纯税(§3.1f):MODE!=1 分支已在探测命中路径(编译期应已消,验 SASS)
  /B 双流装载 vs B16 交织(HASH_BI 对重乘积阵开没开?查默认)/probes 计数器环内自增
- 手段:SASS 级对照我方探测环 vs 彼(等长乘积流),逐指令数差;B16 对 nd24k 强制开电池
- 验收:nd24k 边际 3.2→<2.6 cyc/fl

### 刀 D(已落地):probe_hist 编译期化
- 模板参 PROBE_HIST_ON,生产实例热路径计数指令整体消除;仪器实例独立实例化
- 值门三阵(truth17 值级+有序)跑批中;A/B=old91 vs 新 binary 配对屏

## 3. 测量计划(门后执行)

1. **窗口 A/B**:in-2004 HASH_ROWTIME 默认 vs NOEXTRACT=1 双跑→截距差=extract 份额
   (判 rt8 口径归属+截距差归因修正)
2. **ncu source-level**(in-2004 accumulate):inst_executed 按源码行归属→清表/G 选择/
   探测环/末栅栏等待分解;nd24k 同法对照彼 SASS 流
3. 刀 B 先行(纯 kernel 内部,零路由涉入,G 划分改动小);刀 A 次之(动表尺寸=动分箱
   接口,牵 est 语义,须全阵守门)

## 4. 红线复述
- ①阶段只动 kernel 内部;G 划分/表尺寸=kernel 内部参数,不动 binning 行归属
- 紧表=R1 教训前线:值级三件套+memcheck+大阵(wb-edu OVERFLOW 前科)全绿才许屏
- Xid 哨兵:每轮测量后 journalctl 查 spgemm 进程名
