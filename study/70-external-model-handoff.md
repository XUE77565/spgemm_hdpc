# 交接文档:外部模型 kernel 优化与路由设计任务书(study/70)

> **对象**:将直接在本仓库工作的外部 AI 模型。本档 = 它的完整工作说明书。
> **大框架(锁死)**:设计新 kernel + 优化旧 kernel + 学习 SuiteSparse 分类并路由。
> 两条工作线:①kernel 加速(优化/新创);②路由设计与包装(类→kernel 指派表)。
> **创新点处置授权**:小创新点(见 §三)允许小的破坏与重构——但四大领衔创新点
> 的核心形态不得破坏,重构后须过完整验证链。
> 前置阅读:study/69(三线战役)/study/68(bug 修复史)/study/67(路由总计划)。

---

## 一、项目一句话与当前状态

GPU SpGEMM(C = A·A 和 A·Aᵀ),NVIDIA H100,CUDA C++。核心思想:**不同形状的
行用不同 kernel**——按行人口分派到多种累加形态。

**记分底座:confirm19 GM = 0.9085 vs ocean**(427 阵,H100,compute-only 口径)。
五线总账:带病 0.8137(退役)/HH=0 0.9278(正确地板)/Gv4.1 0.9289/Gv4.2 0.9251/
Gv5a 0.9085(现役)。冲刺目标 0.8x。

**三线战场(study/69,当前二进制实测)**:
| 战线 | 可回收 | 状态 |
|---|---|---|
| accumulate 本体 | 179ms | 未开刀(b45 墙,最难) |
| compact+sort | 148ms | 刀一 Gv5a 已收割大半(ldoor −33%/cage15 −19%) |
| dense_direct | 116ms | 刀二 dfia 在跑(dflag-in-atomic) |

## 二、目录与关键源码

```
spgemm-challenge/
├── src/
│   ├── spgemm_kernel_hash.cu       ~8000 行 ★kernel 主战场(整理重点)
│   │   内核分区(按功能段):
│   │   · hash_spa 梯(各种表长/块宽的累积核)—— ~82 个内核
│   │   · hash_heavy(重行素数表核 + Gv4 位图直写 + Gv5a 秩排序融合)
│   │   · dense_direct/dense_window(窗口直寻核,T4e/T6 档位)
│   │   · ESC 族(esc_rank_kernel,flop 分档子组核)
│   │   · csort 族(counting sort 载体/融合提取)
│   │   · 路由段:bin 分派 kernel(~3300-3550 行)、门控链
│   │   · dispatch 段:hash_accum_c61(~4700 行起)+ compact 循环(~7500 行起)
│   ├── spgemm_merge.cu             merge3(列域分桶归并)
│   ├── spgemm_adaptive.cu          L1 dispatcher(logit 矩阵级选路)
│   ├── structural_params.cuh       编译期常量(表长/档位/阈值/DUP_LIGHT 等)
│   ├── routing_table.h             bin 路由查表(自动重出)
│   └── ai_router.h                 AI Router v3 权重(opt-in)
├── data/                            .mtx 测试阵(不入库)
│   ├── ocean/square/               337 个 + first100/ 100 个 = 428 主账
├── scripts/                        ~170 个(confirm/ab_pair/verify 等)
│   ├── ab_pair.py                  16 阵守门电池(判读含测量分级)
│   ├── verify_sorted_truth.py      三层真值验证(L1集合/L2序/L3数值)
│   └── confirm_adpt19_428.py       最新 confirm 脚本
├── tools/                           测量台 + 微复现器
│   ├── hh_repro2.cu                Gv4 位图验证器(1925 用例)
│   ├── hh_repro3.cu                Gv5a 秩排序验证器(606 用例)
├── study/                           战役档案(全部决策出处,01-69)
├── Makefile                         per-file 双 TU(见下)
└── spgemm_test                      认证二进制(GM 0.9085,勿覆盖!)
```

**关键构建细节**:
- per-file 双 TU:hash.cu 编两遍(`-DHASH_TB=90` 走 sm_90,`-DHASH_TB=61` 走
  compute_61),大表实例在 c61 侧快 9%(刀F);
- `make` 会覆盖根目录 spgemm_test → **改代码后一律 `make TARGET=别名`**;
- Makefile 无头文件依赖追踪:改 `.h` 后必须 `touch` 对应 `.cu`;
- `make` 自动跑符号审计(check_disjoint + check_optin),必须通过。

## 三、创新点体系(优化目标,不是禁区)

### 四大领衔创新点(核心形态不可破坏)

| # | 创新点 | 载体 | 查新状态 |
|---|---|---|---|
| ① | **列域位图有序直写** + hash_heavy | Gv4(位图前缀 rank)+ Gv5a(秩排序融合) | 三家源码均后置排序,核内直写无先例 |
| ② | **无偏 MinHash 定容** | 估计器四层契约(校准/封顶/膨胀记账/溢出重算) | 与 ocean HLL 独立谱系 |
| ③ | **LLB 行内粒度自适应** | dense_direct 内 local_load_balance 组宽 | ≠ spECK 块级调度 |
| ④ | **merge3 列域分桶** | spgemm_merge.cu | GPU SpGEMM 领域无人用键域 |

**对这四条:可以优化其实现、调整其参数、扩展其适用面,但核心机制(位图前缀/
min-of-min 合并/行内动态组宽/有效列域切分)不得移除。每次改动走完整验证链。**

### 小创新点(授权破坏与重构)

以下各项**允许小的破坏与重构**(标注"可重构"),重构后须过 §六验证链:
- K7 双遍位图核(opt-in 归档,竞争力门未画,复活需行级判据);
- CS2 两级提取的交叉点参数(d=384);
- csort 族的 CAP 配置与梯子;
- ESC8/16/32/64/128 子组核的分界;
- T6 窗数门/T4c span 门/T4e LLB 门的具体阈值;
- BATCH_S/BATCH2 的批处理条件;
- dense 路由的 dim2/ladder/v4 门(门参数可调,方向可改);
- Gv4 门普查的绝对帽/比例帽参数;
- **任何新的行级/矩阵级门**——可加可删可改,但须带出处(§六纪律 D)。

## 四、工作线一:kernel 优化与新设计

### 现状与弹药(按优先级)

1. **accumulate 墙(179ms 池,最大未开战场)**:
   - b45 战役定谳:彼 symbolic 1.6cyc/乘积 vs 我方单遍+原子——瘦内核差距;
   - spec99 两遍式资产在(一遍便宜计数+一遍数值,每遍比 CAS 便宜);
   - 靶阵:c-big(41ms)/Bump(29ms)/Hook(14ms)/delaunay 族/dielFilter;
   - **这是"真正的 kernel 内部优化"**——内环机制级,不是路由级。

2. **dense 线(116ms 池,刀二在跑)**:
   - dfia(dflag-in-atomic)已实施,c-64 −11% 值级 PASS,电池定净账中;
   - 方向:同址 SMEM atomicAdd 争用消解(dup>4 人口的每位置一次 flag 写);
   - 后续:per-thread 私分+块内归约(更强的争用消解);
   - SB 的 dense_direct 227ms = 62.7M 乘积的同址争用,对照 hash 快 27%。

3. **compact 线(刀一已收割,残余)**:
   - bin7-9 的 fused 未扩(v2 候选:Gv5a 秩排序扩到 bin7-9 的表);
   - pre2 +5.9%(带内,重行载体回归税)。

### kernel 优化的纪律

- **微复现器先行**:新 kernel 逻辑 → standalone 验证器(如 hh_repro2/3.cu)
  → 全绿才进生产;
- **单因判别**:每次改动一个变量,判别实验分离嫌疑(刀一的五轮迭代是范例);
- **机制对账**:载体的零 per-row 屏障优势、位偶的 log² 屏障税——人口规模
  必须和机制对账(K7 三次死刑的核心教训);
- **编译维度**:同一份源码 compute_61 vs sm_90 差 9%——改动后双 TU 都要测。

## 五、工作线二:路由设计与包装(类→kernel 指派表)

### 终态架构(study/67 用户钦定)

```
三层分离:
  类体系(稳定)= 行结构的数据事实(F/U/S 三轴分类,§四特征体系)
  菜单(半稳定)= 五家族 kernel 清单(新 kernel = 注册+粗标定)
  指派表(易变)= 每类×每家族的实测赢家(内核演化即重标)
```

**核心原则**:
1. **竞争力门 ≠ 资格门**(K7 三度死刑定谳):门必须判"这行在这个 kernel 下
   vs 现任 kernel 下谁快",不是"这行能不能进这个 kernel";
2. **类数 ≤ kernel 数**(消灭假选择:同指派的类合并);
3. **指派表可重盖、类体系不许动**(人口漂移:静态表 100%→14% 的教训)。

### 当前路由架构(五级,study/66)

```
入口 spgemm_test matrix.mtx
  │ L1 矩阵级:spgemm_adaptive.cu(logit → hash/merge3)
  │ L2 人口级:est→bin 0-10,h_cnt 普查,Gv4 门控普查(位图门)
  │ L3 行级:CS2 d门/cursor-search/DSTRUCT 档位/LLB/T4c/T6/dup 门
  │ L4 编译级:per-file 双 TU(大表走 c61)
  │ L5 时间级:RTABLE 查表 / AI Router(opt-in)
```

### 路由侧的待做(study/67 四步)

1. **语料库扩建**(先行的燃料):UF 镜像爬全方阵(~1500-2500 新训练阵),
   排除测试账(428/72/flex142 逐一核对),单阵 512MB 封顶,零 GPU;
2. **分类学习**:行级特征(flop/est/dup/span/avgB/maxlog)→ "几类矩阵×几类行"
   交叉分类学,规律蒸馏配图;
3. **指派表 v1**:扩建池全量标定(每类×每家族实测)→ 影子 428 → 四臂 → confirm;
4. **竞争力门**:逐行判据从实测算子提炼(不是从矩阵级粗特征猜)。

### 路由侧可用的现成基建

- `routing_table.h`(菜单/决策表分离,新 kernel = 注册+粗标定+表自动重出);
- `ai_router.h`(AI Router v3,opt-in);
- 行签名(est/maxbl/dup/span 4-8B)= 在线类探测器;
- `DUMP_ROUTING` env(路由决策落盘,校准数据源);
- 分类学习模板:`d3_distill_student.py`(特征序 FEATNAMES 与 kernel 内一致)。

## 六、纪律(违反任何一条 = 红线)

1. **正确性铁律**:先正确后性能。每刀守门三件套 = 真值级对比(verify_sorted_
   truth.py 三判:集合/序/数值)+ 行内有序检查 + nnz 交叉。**nnz 对≠输出对**
   (v1.3 的教训:电池绿但列序错,值级门拦截)。
2. **测量分级**(memory/measurement-class-discipline):中位 <1ms 不作单阵裁决
   (只进 GM 聚合),1-10ms ≥5× 交错,≥10ms 3× 中位;交错发射(A/B/A/B)硬要求。
3. **GPU 纪律**:报错即停;微阵 10× 交错中位;同日同卡配对;单任务。
4. **认证二进制 spgemm_test 勿 make 覆盖**:改代码走 `make TARGET=别名`;
   过门(值级+电池双绿)才 commit → confirm 全量 → make 重建。
5. **红不 commit**:电池红/值级红 = 停下写明原因,不硬推。
6. **禁"跑一批补一门"**:新门必须带推导/成本出处(物理交叉点或判别实验)。
7. **测试账纯净**:data/ocean/square/ + data/first100/ = 428 主账,
   **永不参与训练/标定**;flex142 = 133 阵测试账;训练用扩建池(独立目录)。
8. **WRITE_MTX 转储即清**:results/matrices/ 批跑后必须清空(曾两次写满磁盘)。
9. **地图与二进制同代**:性能分析用当前二进制跑,不用旧 prof 日志。

## 七、验证协议(每刀的标准流程)

```
1. 微复现器(如适用)→ 全绿
2. make WRITE_MTX=1 TARGET=别名 → 构建零错 + 审计双绿
3. 值级:代表阵 dump + verify_sorted_truth.py 三判(L1=L2=0)
4. 电池:python3 scripts/ab_pair.py spgemm_test 别名 results/ab_x.csv 3 0
   → 判读自动分级(微带标注);GM ≤1.02 且无阵 >1.10(微带 10× 口径)= 绿
5. 绿 → git commit → make 重建正式件 → confirm 脚本(428+ocean 配对)
6. confirm 出新 GM → 入档 study/69 → 汇报
```

## 八、当前在飞(接手即处理的悬案)

- **刀二 dfia 电池**(/tmp/ab_dfia.log,16×3):c-64 单测 −11% 值级 PASS,
  但电池中 c-64 读数翻转(1.104,单测 0.887)——同址原子争用的时序敏感性,
  全账终判待出。绿 → 值级四阵 → commit;红 → 收缩或解剖;
- **pre2 +5.9%** 带内(刀一重行载体回归税,观察);
- **bin7-9 fused 未扩**(Gv5a v2 候选);
- **新仓库**:git@github.com:XUE77565/spgemm_hdpc.git(remote 名 hdpc),
  paper 分支 = 主线。

## 九、论文映射(创新点→论文槽)

| 论文槽 | 来源 | 状态 |
|---|---|---|
| §1/§2(SOTA 弱点) | 四基线 448 账 + Flex/TileSpGEMM 数值缺陷取证(study/68 §四) | 已齐 |
| §3+fig2(分类学+规律) | 路由四步的②(依赖语料库扩建) | **主缺口** |
| §4(五家族) | 四大创新点 + Gv5a(study/64 v4.1 + study/69) | 句式已定 |
| §5+fig3(指派表路由) | 路由四步的④ | 依赖② |
| §6(评测) | 五线消融 + 四基线 + 负结果群 | 已齐 |

**汇报材料**:design-story-merged.html(六循环叙事版,合并 GPT 图);
study/68 §九-3(K7 终案)/§十(测量纪律)。

---
*版本:11-23 v1。关联:study/69(战役书)/study/68(bug 史)/study/67(路由计划)/
study/65(旧交接,含更多构建细节)/study/64(创新点查新封版)。*
