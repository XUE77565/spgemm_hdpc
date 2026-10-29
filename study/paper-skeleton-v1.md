# 论文骨架 v1(10-29 定稿)

> 标题方向:"Compile-Aware Morphology Routing for SpGEMM on Modern GPUs"
> (或多维路由/组合管理/签名路由——待定)
> 目标会场:PPoPP / SC / PACT(系统论文)

## 摘要(3-4 句)
SpGEMM kernel 的性能不仅取决于算法选择,还取决于编译层决策和人口级路由。
我们发现同一 kernel 在不同 PTX 层级下性能差异达 9-52%(且影响正确性),
提出 per-file 编译分层 + 签名路由 + 算法刀链的组合系统,
在 428 阵评测中净赢最强竞品 19.5%(GM 0.8054),
并证明固定几何变更无效(只有逐行自适应路由才有价值)。

## 1 Introduction(1 页)
- SpGEMM 的重要性 + 现有选择方法的局限(整阵选/ML 预测)
- **问题定义:决策漂移**(内核/编译变化使路由决策失效)
- **我们的贡献**(4 点):
  1. 编译层表征(PTX 孤峰/非对称受益/正确性条件性)
  2. per-file 三层架构(TU-61 c61 tagged / TU-90 sm90 / 三检体系)
  3. 算法刀链(hash_heavy 复活 + 有序直写 + bin9 素数档)
  4. 签名路由验证(est 维度实锤 + 固定几何无效 + 三连负证明核近极限)

## 2 Background & Related Work(0.5 页)
- SpGEMM 算法:hash-based, sort-based, dense accumulation
- 选择方法:IA-SpGEMM(2019 逐阵选), Wang2025(TACO), FlexSpGEMM(2026 ML+TC)
- GPU 编译:PTX/SASS 关系, nvcc 编译管线
- **定位:粒度(行/bin)× 判据(物理交叉)× 维度(编译层)× 漂移处理 = 四点正交**

## 3 Compile-Layer Discovery(1.5 页)★ 核心发现
### 3.1 The PTX-Level Performance Cliff
- compute_61 孤峰(61/70/80 三档扫描:61 是唯一甜点)
- 非对称受益(他们核 +52%, 我核 +9%)
- **正确性条件性**(第六例:web-Google LHT8 smem 越界,同一 flags 不同 TU)
### 3.2 Per-File Three-Layer Architecture
- TU-61(compute_61)/ TU-90(sm_90)/ 门控层
- TB_TAG 模板参(防弱合并)
- 三检体系:符号弱合并 + opt-in 覆盖 + 基线零漂
### 3.3 Methodology
- 链接层探针三定谳(弱合并 = 静默错 codegen)
- 死码定律(加休眠代码改变活路径)→ 方差定标修正(RL 噪声)
- 编译混沌与布局轮盘(微阵 ±15%)

## 4 Algorithmic Knives(1 页)
### 4.1 hash_heavy Revival(编译条件性死刑翻案)
- sm_90 下 +77% 死刑 → c61 下 −9.8%(最强自有核)
### 4.2 Ordered Direct Write(载体悖论解)
- Gv3v2: warp 段认领 + shfl 位移 rank + cap 守卫
- compact 回落 copy 级(免 counting-sort 载体)
### 4.3 bin9 Prime Table
- 16253 素数档(195KB)替 LHT2(14) 196KB 占用灾难
- 人口联锁(h_cnt[9]>0)

## 5 Signature Routing(1.5 页)★ 核心架构
### 5.1 Architecture
- 三级路由(方法级 dispatcher → 签名 bin → 行级自路由)+ 编译层轴
- 菜单/决策表分离(变体注册 vs 模型生成)
### 5.2 Validation Chain
- S0:est 量级 = 分离维度(WORSE≈15 vs BETTER≈49)
- S2:固定几何 = 零收益(比值 1.000)→ **只有逐行自适应有效**
- 相位消融:hash 插入 71.7% / 提取 23.2% / 表清 5.2%
- **三连负结果**(read-first +71% / warp 聚合 / B 行预取 +109%)
  = batched 核已近硬件极限 = "固定改核没空间,路由才有"
### 5.3 Decision Table Generation(设计,未实施)
- 物理 cost 模型(闭式/标定面)
- bandit 在线层(决策漂移自适应)

## 6 Evaluation(1.5 页)
### 6.1 Setup
- H100, 428 阵并集, compute-only 口径
### 6.2 Main Results
- **vs ocean: GM 0.8054(净赢 19.5%), 主场 0.9728(翻面), 赢面 70.5%**
- vs HSMU(2024): GM 0.2138(4.7× 快, 100% 赢)
- vs TileSpGEMM(2020): GM 0.3550(2.8× 快, 95% 赢)
- vs FlexSpGEMM(2026): TBD
### 6.3 Ablation(confirm14→17 阶梯)
- 每刀单独可归因, nnz 全程逐位
- 旗舰阵: Hook 0.990(翻面)/ Bump 1.005 / cage15 1.013
### 6.4 Signature Analysis
- 族胜率(结构 100%/图网络 12%/CFD 21%/delaunay 0%)
- 输面分桶(bin9 桶/微行巨阵桶/known-cost)
### 6.5 Methodology Lessons
- 方差定标(微阵 CV=18.9% → 交错 ≥10 次)
- 编译混沌 vs 系统噪声(奥卡姆剃刀)

## 7 Discussion(0.5 页)
- 编译层红利是 CUDA 版本/arch 敏感的(ptxas 行为)
- 签名路由的推广(SpMM/图核/任何 kernel 族选择)
- 局限:S3 自路由未实施;FlexSpGEMM 的 ML 路由 vs 我们物理交叉

## 8 Conclusion(0.25 页)

---

## 图表清单
| # | 内容 | 数据来源 |
|---|---|---|
| Fig.1 | confirm14→17 GM 阶梯(每级标注刀名) | confirm CSVs |
| Fig.2 | PTX 梯扫描(61 孤峰) | study/57 §五十二 |
| Fig.3 | per-file 三层架构图 | 手绘 |
| Fig.4 | 相位消融柱状图(71.7/23.2/5.2) | phase_ablation |
| Fig.5 | 签名维度分离(WORSE vs BETTER est 分布) | S0 数据 |
| Fig.6 | 族胜率热力图 | confirm16 |
| Tab.1 | 基线对比总表 | baseline report |
| Tab.2 | 消融表(每刀的独立贡献) | confirm chain |
| Tab.3 | 编译层现象汇总(六例) | study/57-60 |

## 缺失项(发布门槛)
1. FlexSpGEMM 基线(已解压,构建中)
2. 第二 GPU 复验(A100/4090)
3. 72 阵 held-out 泛化验证
4. S4-S6 实施(决策表+四臂对照)→ 可选(架构已验证,实施是加分)
