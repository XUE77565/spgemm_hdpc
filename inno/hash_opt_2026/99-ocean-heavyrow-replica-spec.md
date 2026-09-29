# spec99:ocean 重行同构形态复刻(立项草案,开工须用户确认)

> 背景:③重行两级转的第二级(study/26)。第一级(nd24k dense 位图对拍)经济屏若判负,则本 spec
> 为队首交付物。**性质=对照形态复刻,非创新**(用户 09-25 已听方案未否);目标=追平+一遍制微赢。
> 状态:**草案——等 study/26 §5 经济屏数字定稿+用户确认后才开工**。

## 1. 立项依据(事实层)

- 重行输因定谳=固定截距税(每行开张成本 +16K~54K cyc;乘积端斜率我方占优 1.16 vs 1.67,
  总账 01 §终·终版);形态内四族捷径+聚合族全判死,结构税定谳。
- ocean hashNumeric 重行形态的实测优势构件(81 号/inno/91 源码对照定谳):
  ①**素数常量表**(mod-on-const=magic-div 4 条/乘积;2719/5449/16253 档)②**512T×4 块/SM
  组织**(32KB SMEM/块 → 4 工位驻留 → 独立行进度多 → 延迟遮蔽 68%)③id*11+素数模探测
  ④per-issue 高停顿高发射密度型(LDS 链长但独立指令多)。
- 我方对偶物全部判死记录:素数紧表(寄存器墙挡死,§3.1m)/512T 松表(SMEM 墙)/REG 瘦身
  (2048 线程帽)——**单构件移植全撞墙,只有整套同构复刻才可能解锁**(对称消融 §3.1al 的
  反向实证:形态=六层常量网局部最优)。

## 2. 复刻设计(同构要件清单)

1. **编译期素数表族**:HT_P 模板参已有(spec91 刀资产,含 magic-div+8B 对齐修复)——
   2719/5449/16253 三档实例化。
2. **512T 块组织**:4 块/SM 驻留 = 每 SM 4 条独立重行;表 2719×12B≈32KB/块。
3. **探测=id×11+素数模**(彼同构;跳步 3)——替换我方 Knuth×2654435761(保留原路径 env 门)。
4. **极简栅栏结构**:单相位(init+收尾 2 栅栏),无 MODE 分发/upper_tri 运行时分支
   (编译期模板化)——直打截距。
5. **值区**:彼 REDG 值外置(HYBRID 16253 档)=大表端容量 workaround;我方先行 SMEM 内
   (刀E 判死记录在案,不重试 VG 值区)。
6. 发射:独立重行 kernel(不与现 hash_spa 共存路径),binning 只送 est≥8K 档行
   (nd24k/in-2004 重带人口;判据=行特征非阵身份,①合规)。

## 3. 验收(先门后屏)

- 值门:truth17 值级+行内有序+memcheck 全绿(新 kernel 三件套,零豁免)。
- 性能门:nd24k/in-2004 重带(est 1-16K)逐行 vs 彼 hashNumeric 同行,**截距差收敛到 ±20%
  以内**=追平;系统级 nd24k 相位和 vs ocean ≤1.05;甜点带(256-1K)零回退。
- 一遍制微赢:我方 est 连续化(免彼 HLL 两遍估计的余量税)在重行的兑现=论文③叙事点。

## 4. 风险与红线

- 复刻≠copy:只吸收机理(常量网组织),代码自研——铁律 [[idea-adoption-not-copy]]。
- 六层常量网连锁(SMEM 预算/并发/dense 档/USPARSE)在我方代码的对应物须逐一核对
  (§3.1al ocean 侧教训直接适用)。
- GPU 六条全程;任何 Xid/illegal 即熔断。
- 若经济屏判"dense 位图有戏"则本 spec 转冷档。

## §5 文件级实施蓝图(09-29 第 4 发,读码产出;彼方源=ocean/kernels/AccumulatorHash.cuh+Hashmap.cuh)

### 5.1 彼方组织机理五要素(hashNumericKernel 512-660 实读)

1. **占用驱动块几何**:`__launch_bounds__(1<<LOG_NTHR, 2048>>LOG_NTHR)`——线程×块积=2048/SM;
   档↔块配对:**2719↔512T×4 块**(32.6KB/块×4=130KB)/5449↔1024T×2(65.5KB×2=131KB)/
   16253→HYBRID(值外置 gmem,554 行)。
2. **localLoadBalance 组自适应**(570-580):每行按 (a_len, products, max_B) 算
   log_nthr_per_a_elem∈[2,LOG_NTHR]——group=2^x 线程共走一条 A 元素的 B 行;
   num_groups=块宽/group 条 A 元素同时在飞。**我方对偶物=G 帽 32(免跨 warp),彼可到整块**。
3. **极简主环**(582-606):双层 group-stride 乘积循环,**零栅栏零分支**(无 MODE/upper_tri/
   read-first 检查);探测=hmap[col](id*11%SIZE 素数 magic-div 线性走)→atomicAdd 直加。
4. **栅栏总数 ~3**:init(565)+HYBRID 缓冲(553)+收尾(608)——对照我方 18 个(§3.1c)。
5. **溢出=飞行中切表**(nearFull 早退+行重排 610-628);**提取=kernel 内 compactAndSort**
   (阈值 NUM_HASH_INPLACE_SORT_THRESHOLD 内就地排序,否则 compactOnly 全局排)——
   ROWCYCLES 含提取的口径来源。

### 5.2 我方复刻件清单(全部自研,机理吸收)

| 件 | 我方资产/落点 | 决策 |
|---|---|---|
| kernel 名 | `hash_heavy_kernel<LOG_NTHR,HT_P>` 新增于 spgemm_kernel_hash.cu(独立,不改 hash_spa) | 三档实例:2719/5449/16253 |
| 素数表+magic-div | **HT_P 模板参在库**(1680/1966/1981:Knuth+magic-div+线性走+wraparound) | 直接复用 |
| 8B 对齐 | 偶填充修复在库(1746) | 直接复用(素数奇表必需) |
| opt-in 属性 | 4492-4496 已有 5449/16253 模式 | 照抄到新 kernel 实例 |
| 探测哈希 | 我 Knuth×2654435761 vs 彼 id*11(同走 magic-div) | **A/B 实验定**(Knuth 雪崩好/同指令量级) |
| 组自适应 | 新写 localLoadBalance 同构(输入=a_len/products/max_B) | 核心新代码;先 [2,9] 域 |
| 块几何 | `__launch_bounds__(1<<L, 2048>>L)` 同式 | 2719 档 512T×4 起步 |
| 值区 | SMEM 内(16253 档 HYBRID 外置为后手,刀E 前科不先做) | — |
| 提取 | **我方 extract2/CSORT 资产 kernel 内融合**(计数排序优于彼就地排序) | 我方优势叠加点#1 |
| 一遍制 | 我方 est 连续化(表贴合:真 distinct 档映射,治 1.8× 超配) | 优势叠加点#2 |
| 发射/门 | accumulate 相位 SPA_LAUNCH 处(5285 同款)按 est 档分流;行特征门 est≥8K(esc/LARGEST 先例) | 甜点带零回退=验收 |

### 5.3 实施序(先门后屏,GPU 六条)

1. 刀1=skeleton:三档实例+组自适应主环+我方提取融合;**sanitizer(/usr/local/cuda/bin)全绿**
2. 刀2=值门三件套:F2/in-2004/nd24k truth17 值级+行内有序+nnz 对账
3. 刀3=配对屏:nd24k accumulate 80.2ms 基线 ×3 对拍 → 靶 ≤60ms;逐行 ≤1.10;截距 ±20%
4. 刀4=守门:全默认八阵屏(甜点带零回退)+in-2004 重带复核;A/B=Knuth vs id*11
5. 刀5=进阶:组自适应域扩展(crankseg 型宽 a_len)+16253 HYBRID 后手

### 5.4 六层常量网风险映射(ocean 侧教训→我方对应物)

| ocean 层 | 我方对应物 | 状态 |
|---|---|---|
| SMEM 预算 | opt-in MaxDynamicSharedMemorySize | 4493 已有模式 |
| 并发/块数 | launch_bounds minBlocks | 新 kernel 自带,不碰 hash_spa |
| dense 档联动 | DENSE_TIER/DSTRUCT 独立 | 新 kernel 不接 dense 梯,无连锁 |
| USPARSE/upper | ESC/ultra 路径独立 | 无连锁 |
| SMEM 48K 档 | 同 opt-in | 同上 |
| 模板膨胀 | 实例数=3 档×1 mode(overflow 走彼式早退,不加实例) | 受控 |

### 5.5 不做什么(边界)

- 不动 hash_spa 任何行为(甜点带零回退的保证=物理隔离)
- 不动现有路由/binning 逻辑,只加"est≥门→新 kernel"一处分流(行特征,ESC 同款)
- 不做 VG 值外置(刀E 2.1× 前科)、不做 WM/归并(冻结)、不做 CNT_FUSE(冻结)
