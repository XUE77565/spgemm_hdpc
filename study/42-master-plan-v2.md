# study/42:终极计划 v2——三形态管线家族 + 三层裁定 + 证书兜底(10-13)

> 取代 study/39 成为当前任务书(39 保留作历史)。
> 两个晚上的定谳支撑:DCEST V3 落地(fc49855)+ 解毒屏 18/20 净赢 + SiO2 最优解翻面
> + 1a 病根 ncu 确诊 + 架构三层化讨论(用户全程参与定型)。
> 起点:337 GM 恒C 1.2740 / 恒A 1.2228 / oracle 1.1958(**被税扭曲的管线上测的**)。

## 一、架构总图(本计划的灵魂)

### 1.1 两种税,两层管(架构的第一性原理)

```
行级税(这行算多贵)→ 七武器管(ESC/批量/hash_spa/heavy/cursor/dense梯子/k7)
阵级税(仪式值不值)→ 形态三选一管(L0)
"小阵"和"大阵小行"是两个人群:前者税/总量 30-50%,后者 ~3%
```

### 1.2 管线家族(三形态,L0 按阵选)

| 形态 | 相位 | 主场 | 发动机 | 状态 |
|---|---|---|---|---|
| merge3 | 3 | 小阵/straggler(first100 0.614×,59 阵背书) | 行归并 | ✅ 在跑(adaptive) |
| 轻 hash | 4-5 | 中小阵 ∧ dup≈1(验票:avg_product≤64+规模门) | hash_spa 减配 | ❌ 待建(FASTPATH 复活为**有门**分支) |
| 全形态 | 17→8→5 | 大阵/复杂路由(kernel 级 0.952 的本钱) | 七武器 | 🔄 1c 进行中 |

### 1.3 调度三层(能解析的解析,该状态机的状态机,物理不可约的才学习)

```
L0(矩阵级):
  ① 方法级:hash vs merge3(学习型 DISPATCH_MODEL,已有)+ 溢出/判废安全网→merge3
  ② 形态级:三选一(上表)
  ③ 配置级:DCEST 开关=【解析门】α·Σflop_diter vs β·Σnnz_dense+γ·A_rows+δ
     (系数电池标定,binning 后原地判定零预跑)/ DCEST_MARGIN=【证书直方图状态机】
     (首轮 200,按阵收敛 125-150;ADAPT_EXPAND 同款哲学)/ 估计器选择
L1(行级,Phase 3):GBT 系统级标签(C_kernel + phase_share)——D6 翻车教训后重试
L2(证书):dense 溢出证书【已实战】/ k7 count 证书【已有】/ hash retry【已有】
  → 判废敢判死刑,因为 merge3 永远活着兜底
```

### 1.4 哲学升级(从 ocean 学到的,用我们的形态)

- 估计先行 + 运行时自纠(ocean DNA)→ 我们 = 估计 + **证书**(多一层安全,雷史决定)
- 该不估就不估:avg_product≤64 免估计闸门(8 月已有,4622 行)
- 全形态下限 ~5(dense 梯子 227KB/1024T 几何没法共存;再压=放弃路由=成为 ocean)

## 二、分阶段计划

### Phase 1:管线减重(全形态 17→8,修 L0 的地基)

| 刀 | 内容 | 状态 |
|---|---|---|
| 1b DCEST | dense_count 消灭(证书+紧凑拷贝+margin 200) | ✅ fc49855;屏 18/20 净赢,wb-edu −45.3% |
| 1a-1' | **mh_merge 行并行 block**(4warp/块;ncu 确诊占用 10.4% vs 形态上限 50%;预期 2.6→<1ms) | 🔜 下一刀(手术刀级:只动发射几何) |
| 1a-2' | construct 写侧砍到 K partition(4× 写流量,顺手) | 🔜 与 1a-1' 同刀 |
| 1c-safe | 相邻融合 17→8(分析段/计算段/善后段各自合并) | 📋 |
| 1c-aggressive | 8→5:持久核 grid.sync + 轻武器巨型核 | ⏸ **Phase 2 之后**(拿着重跑数据再动) |
| 轻 hash | 4-5 相位分支(FASTPATH 复活,进门验票) | ⏸ Phase 2 之后 |

### Phase 2:L0 重跑 337(把天花板从推测变成数字)

- 动机已实证:**SiO2 on 超过全 hash A(gap 214.8%)——修管线改变最优调度**,旧
  oracle 1.1958 低估了真实天花板;
- 新决策维度:DCEST 解析门 / margin 状态机 / (Phase 1 后的)管线形态;
- 预期 GM ~1.15-1.20;**这轮数据决定 1c-aggressive 和轻 hash 各值多少**。

### Phase 3:L1/L2 突破(用户钦定核心)

- 3a 探针校准 phase_share → 3b L1 v2 系统级标签 → 3c 证书补全(dense 已实战)
  → 3d 部署验证(判据:L1 vs 恒A <1.0 净赢 / vs oracle >0.90 / 无阵回归 >1.10);
- 成功 = 论文头条;失败 = 精确否定(462 万行数据已备)。

### Phase 4:论文(SC/PPoPP)

```
1. 否定结果(已证):逐行 kernel 时间 ≠ 系统成本(集体效应)
2. 管线家族 + 形态调度(实证中):dispatch 不只选 kernel,还选 pipeline form
3. 三层裁定(设计+部分实证):解析门/状态机/学习型,各有出处不调参
4. 证书兜底(已实战):估计先行 + 判废回退,dense/k7/hash 三路全有
```

## 三、时间表

| 时段 | 任务 | 产出 |
|---|---|---|
| ✅ 10-12 | DCEST V3 + 解毒屏 + 1a 确诊 + 架构定型 | 本文档 |
| 1 天 | 1a-1'/1a-2'(merge 行并行 + construct 砍写)+ 守门 | c-64 估计器税 1.9→<1ms |
| 2-3 天 | 1c-safe(17→8) | 全形态定稿 |
| 3 天 | **Phase 2:L0 重跑 337** | 真实天花板数字 |
| 2-3 天 | 1c-aggressive(8→5)+ 轻 hash(视 Phase 2 数据裁剪) | 形态家族齐 |
| 1 周 | Phase 3 L1/L2 | 成败定谳 |
| 5+ | Phase 4 论文 | 投稿稿 |

## 四、铁律清单(本计划全程有效)

1. 先正确后性能:每刀守门三件套(值级/有序/memcheck);空对照三防(strings 验
   binary/dump mtime/make -B);
2. 不改 kernel 形态(DCEST 全程零 kernel 改动的示范);不偷生态;
3. 先修管线再跑 L0(Phase 2 用 8 相位干净管线);
4. 结论稳定:宣布"X 缺失/是问题"前先对账(1a 闸门勘误的教训);
5. 结论分三档(定谳/实证/推测),跨版本同日配对;
6. GPU 纪律:sanitizer 先行、报错即停、Xid 哨兵、测量不并发。

## 五、关键文件索引

| 内容 | 文件 |
|---|---|
| 本任务书 | study/42-master-plan-v2.md |
| DCEST 全档+屏+雷史 | study/40-dcest-v3.md / study/40-detox-screen-results.txt |
| 1a 设计+ncu 确诊 | study/41-dual-estimator.md |
| 上一版任务书 | study/39-master-plan.md(历史) |
| 小白总览 | docs/83-ultimate-status-1012.md |
| L0 架构 | study/37-l0-system-dispatch.md |
