# study/44:ocean 消融情报 + Du 2022b 划界 + 基准碎片化实测(10-15)

> 来源:读文献副产品(全部一手)——ocean arXiv 全文 §5.3(HTML 已本地解析)、
> Du 2022b 摘要(arXiv:2207.13848)、HSMU 仓库 338matrix_list.csv 交叉比对。
> 用途:①对手 attack map ②三层裁定叙事的第三方证据 ③查新补条 ④L0 验收指标模板。

## 一、ocean 消融定谳(§5.3,337 方阵,A100)

**工作流分流(他们自己的分析步决定)**:
```
337 = 163 精确 symbolic(48%) + 148 HLL 估计(44%) + 26 上界估计(8%)
```

**增量消融(V1 基线 → V4 全开,对其选定子集的均值)**:
| 版本 | 加什么 | 增益 | 备注 |
|---|---|---|---|
| V2 | E 估计工作流 | **1.30×**(0.95-2.77) | 主增益来源;road_usa 上界法还顺带改善 binning |
| V3 | AS 辅助 kernel | 1.03×(max 2.19 on exdata_1) | 整体温和;exdata_1 类高压缩比 bitmap 更新流量是靶 |
| V4 | HA 混合累积器 | 1.06× | — |
| 总 | V4/V1 | 1.25× | GFLOPS 50.8→63.7 |

**选择器质量自证**:163 个被选 symbolic 的阵里只有 10.8% 换估计能 ≥1.05×;
174 个被选估计的阵里 92% 保持比回退 symbolic ≥1.05×。

### 对我们的三层含义

1. **外部验证(最有价值)**:连 ocean 的单层分析步都把 48% 的阵送回精确 symbolic——
   **"估计是门控 niche 不是默认"与我们 CD 定谳(裸开 +1.9% 劣化,门控=69 阵主场)
   完全同构**。写论文时引他们自己的数字给三层裁定叙事背书;
2. **attack map 更新**:他们的优势主体=E(估计工作流)。在 163 个 symbolic 阵上
   他们跑的是**常规两遍管线**——那是我们七武器行路由的主咬战场;
   他们的 AS/HA 合计只值 ~1.09×,说明 kernel 级我们并不落后主结构;
3. **L0 验收指标模板**(照抄他们的自证协议):我们的 L0 选择器验收 =
   "被选 CD 的阵中 ≥1.05× 获益比例 / 被选 C 的阵中反向获益比例",写进 L0 规格。

## 二、Du 2022b(Xie 组,arXiv:2207.13848)——查新补条

**一手摘要**:采样 FLOP 与 NNZ → 预测**压缩比** → 每行输出 = 该行 FLOP ÷ 压缩比;
精度 1.56% 均值(worst 25%),对比"现有采样法参考实现"8.12%(worst 156%)。

**定位**:
- 概念上与我们的 dup 维度(Σflop/Σest)同族——**压缩比 = 1/dup**;必引,划界入
  report-1021/03-novelty-sweep.md;
- **其软肋恰是我们的活证据**:全局/少样本压缩比对**行异质**矩阵会崩——
  in-2004 两行结构性欠估(actual=raw×1.5-2,study/40)就是全局比率失效的实例。
  我们的逐行 minhash + margin 状态机 + 溢出证书是针对性的解;
- **可借(机理不借形)**:压缩比可以由 mh_merge 顺手产出(Σflop/Σest 已有),
  作为 L0 特征几乎免费——记入 L0 特征工程清单。

## 三、基准碎片化实测(10-15,一手 CSV 交叉)

```
first100 ∩ ocean337 = 1 阵 | ocean337 ∩ HSMU338 = 0 阵 | first100 ∩ HSMU338 = 0 阵
```

**证伪**:今晨"HSMU 338 → ocean 337 传承"假说(数量相近纯属巧合,两集完全不相交;
HSMU 集含大量编号名合成阵)。**修正后的事实更强**:领域不止双基准不相交,
是**三套战场两两互斥、每篇论文自选基准**——统一评测缺失是领域级问题,
我们的 437 并集(双方基准之并)正面对此。已同步修正 01/02/slides 三处。

> 方法论记录:本次"推断→一手数据→证伪→修正"在一小时内闭环,
> 是"结论稳定纪律+一手数据制度"的又一次实操。

## 五、可借机理清单(10-15,按"只借机理不抄形态"铁律筛)

| # | 机理 | 出处 | 插进我们哪里 | 价值/成本 | 状态 |
|---|---|---|---|---|---|
| **1** | **hash 索引/值分离放置**:索引留 smem(32bit 原子便宜),**值放 gmem 用原生 FP64 RED**(fire-and-forget,不需要返回值→编译器出 RED 单指令;gmem 原子延迟仅 2-4× smem) | **ocean §3.3(一手)** | **与 17 连败对账完成(10-15)**:我方"全局值"判死人群=**高 dup 自旋弹**(dup 23-61,同座 L2 串行化,gmem 不救);ocean 赢的人群=**容量界**行(表超 smem 容量,moderate dup)——**互补边界,非矛盾**。插进点=dense 梯子中带(span 1818-7277,输 1.15×)即此缺口;且仅"最大 hash 档"用 hybrid(他们自己也只这么用) | 中~高/中(需 GPU) | **补边界实验**:中带行 hybrid-hash vs D2 重写,二择一 |
| **1b** | **fire-and-forget 纪律审计**:凡值累加不需要返回值处,确保编译出 RED 而非带回值的 ATOM——SASS 一查便知 | ocean Listing 1(机理泛化) | **全部 kernel 的 gmem atomicAdd 路径**(不只 hash) | 中/**零 GPU** | **✅ 已执行(10-15,见 §六)**:主流热路径已全 REDG;残留 4 点=merge3 numeric×2+hash_spa_batched×3 实例,补丁待 GPU 守门 |
| 2 | 压缩比估计(采样 K 行→比率→每行 FLOP÷比率) | Du 2022b | 轻 hash 形态的廉价估计器候选(替代全行 minhash);dup 已是免费 L0 特征 | 中/低 | 并入轻 hash 设计+L0 特征规格 |
| 3 | 选择器自证协议(被选臂获益 ≥1.05× 比例双向) | ocean §5.3 | L0/L1 验收指标(本文档 §一之 3) | 中/零 | 已记入 |
| 4 | density representation 粗网格密度 sketch(如 16×16 计数) | MatNet(IA-SpGEMM) | L0/L1 特征:搭现有一遍 scan 顺手 256 计数器 | 中/低 | 并入 L0 特征规格 |
| 5 | 采样复杂度理论(同等误差更少样本) | Amossen 2014 | mh_merge 采样数/margin 数学;轻 hash 验票设计 | 中/读文 | 待读(GPU-free) |
| 6 | 等大 rank 窗/tiling 几何 | MMSpGEMM/TileSpGEMM | dense 梯子中带 D2 重写候选(1.15× 输于行间交错) | 中/高 | D2 设计时评估 |
| 7 | 矩阵重排+聚类分块(预处理形态) | Islam 2025 | 大阵局部性;A·A 需排回(2 额外遍),只对大阵可能值 | 低~中/高 | 记债务表,GM 投影后再议 |

**执行纪律**:每项入默认前必须 转正批+守门三件套;#1 需先核对 17 连败清单是否已含"值放 gmem"(若无→这是新机理;若有→读当时判负原因再议)。

## 六、SASS fire-and-forget 审计执行记录(10-15,#1b 完成)

**方法**:10 行最小复现定谳工具链行为 → `cuobjdump -sass spgemm_test`(62 万行)逐 kernel 统计。
**助记符陷阱(sm_90/CUDA 12.0)**:全局原子是 `ATOMG`/`REDG` 而非 `ATOM`/`RED`,
第一版正则漏 REDG——**审计脚本先被审计**(教训:数 0 之前先证明模式能数到 1)。

**复现定谳**:nvcc 12.0 对弃返回值的 `atomicAdd(double*)` **默认出 `REDG.E.ADD.F64`**
(fire-and-forget),用返回值才出 `ATOMG`——编译器无锅,是源码上下文问题。

**主 binary 审计结果**:
- ✅ **主流热路径全部已 REDG**:hash_spa_kernel 主累积(16×)、dense_largest(4×)、
  k7_numeric(3×)、hash_global/retry2/hkv/bsearch(各 2×)+ 非F64 REDG 106 处;
- ⚠️ **残留 ATOM 4 点**(返回值已弃但编译成 `ATOM.E.ADD.F64.RN.STRONG.GPU P0, RZ, ...`
  ——目标寄存器 RZ=弃值,但仍付 L2 往返;nvcc 未转 RED 疑因 break/分支上下文):
  1. `bucket_merge_flop_kernel`(merge3 numeric **内积热循环**,src/spgemm_merge.cu:739)
  2. `bucket_merge_kernel`(同族,spgemm_merge.cu:839)
  3. `hash_spa_batched_kernel` ×3 模板实例(spgemm_kernel_hash.cu,batched t_val 全局加)

**影响评估(诚实)**:merge3 numeric 是逐乘积原子加,ATOM→RED 省 L2 响应 flit,
受争用时可观——但 merge3 主场=first100 小阵,**GM 投影小(估 <0.5%)**,属"白捡但小"。
**处置**:20 行 inline PTX `red.global.add.f64` 补丁,**编译开关默认关**,
GPU 释放后守门三件套(值级+memcheck)+first100 小电池验证再议转正。
(EXCH 类 atomicExch 无 RED 等价,弃值也只能 ATOMG,不动。)

## 四、行动清单(衍生)

- [x] 三处文档修正(01/02/slides 观察③)
- [x] 本情报档(study/44)
- [ ] 03-novelty-sweep 补 Du 2022b 条目(本文档 §二即素材)
- [ ] L0 特征工程规格书:dup/压缩比、margin 证书直方图、binning 副产品(熵/基尼)、
      est/flop 行分布统计——等 437 权威批落地即训(验收协议=§一之 3 模板)
- [ ] 论文骨架 §三层裁定:引 ocean §5.3 数字作外部验证
