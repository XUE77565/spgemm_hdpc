# Harmonia 调研(SJTU 稀疏加速,ISCA 2026)— 用户讲座线索(09-18)

> 触发:用户听讲座获知"上海交大 Harmonia 稀疏加速工作",要求检索并评估可用性。
> 铁律执行:只吸收机理不 copy 形态;先查公开前作;形态入默认前必须转正+守门。
> **结论先行:纯硬件加速器论文(非 GPU 软件),无形态可 copy——天然合规;
> 机理层 3 条可用(dispatcher 叙事互证/离线模型/运行时检测),论文 C 线必引。**

## 1. 身份(已核实)

- **"Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication"**,
  ISCA 2026, pp. 2127-2141, DOI `10.1109/ISCA66397.2026.00151`
- 作者:Jingkui Yang(=共同一作,国防科大 NUDT)/ **Fangxin Liu(=共同一作,\*通讯,SJTU CS 助理教授)** /
  Xin Ju / Ning Yang / Chenyang Guan / Junjie Wang / Zongwu Wang / Mei Wen(NUDT)/ Jian Liu /
  Li Jiang / Haibing Guan(SJTU)
- 组:SJTU 软硬件协同设计组(mxhx7199.github.io)+ NUDT;组内前作多 ReRAM/PIM 稀疏加速(SoBS-X/ERA-BS/SME)
- **平台 = 专用稀疏加速器(硬件 ASIC 级),不是 GPU 软件**

## 2. 检索状况(09-18)

- IEEE Xplore 反爬空返回(3 次);ResearchGate 条目 411277002 → 403;作者 publications 页
  Paper 链接为占位符 `#`(尚未挂 PDF);无 arXiv 预印本
- 摘要经 **Semantic Scholar API(DOI 查询)** 获取(概述级转述)
- **全文未拿到**。重试路径:校园网 IEEE / 作者页日后挂 PDF / RG 直链
  (拿到后补:层级具体层数/检测特征集/切换开销数字)

## 3. 机理(摘要级)

1. **动机**:不规则稀疏模式使常规平台低效 → 硬件加速器;**现有加速器依赖固定 dataflow
   (inner-product / outer-product / row-based),各只适合特定 sparsity pattern**——
   模式一变效率崩
2. **层级调度三件套**:
   - **轻量离线模型** → 初始 tiling 策略 + dataflow 映射
   - **运行时检测 active sparsity pattern** → 动态选最优 dataflow 模式
   - 硬件快重配置 + 负载均衡
3. 数字:1.75× 性能 / 2.47× 能效 vs SOTA 加速器;运行时稀疏漂移下吞吐稳健

## 4. 前作链(铁律:先查公开前作)

- **Flexagon(ASPLOS'23,Muñoz-Martínez 等,Univ. Murcia,arXiv 2301.10852)**:
  首个 SpMSpM **可重构多 dataflow** 加速器,Merger-Reduction Network + DSwitcher;
  引用 114+——**Harmonia 的直接地基**(注意:Murcia 组作品,非 Harmonia 组自产)
- STONNE 模拟器生态(Murcia)建模 Flexagon
- GPU 软件侧对应物:nsparse/spECK 等单法 per-matrix 选择;**我们 = per-row 分箱 + dispatcher**
  ——在 GPU 软件栈上做 Harmonia 在硬件栈上做的事,无撞车,互补论证

## 5. 可用性判定(铁律口径)

**可用(机理级,3 条)**:

1. **论文叙事(C 线 dispatcher 的 related work 弹药)**:"固定 dataflow 只适合特定 pattern"
   是 ISCA'26 级同构论证——正是我们 dispatcher/多 kernel 体系的存在意义。
   定位:Harmonia 硬件侧同题解,我们 GPU 软件侧同题解(无需专用硬件)。
   **与 inno/mmspgemm_comparison.md(merge3 查重)互补,dispatcher 查重以本文档为准**
2. **离线轻量模型机理**:与 DISPATCH_MODEL(disp_train/disp_synth 训练)同构先例——
   离线学"结构统计→策略"映射,而非运行时全量搜索;设计依据可引
3. **运行时 pattern 检测→动态切换**:同构于我们的 est/binning(DITER_EST)+ rowmode(avgB 阈值)
   + occgate(③阶段);其"层级调度"提示路由成本模型可分 **row → row-block → phase** 三层
   (记为③阶段候选机理,不动现刀)

**不可用(2 条)**:硬件快重配置/PE 阵列/能效数字(平台不同);tiling 策略细节绑定其 PE 结构

**合规性**:纯硬件论文,无代码/形态可 copy——天然"只吸收机理";无形态入默认,
不触发转正+守门流程

## 6. 行动项

1. ✅ 本 survey 入档(即本文件)
2. 论文 related work dispatcher 段:Harmonia(ISCA'26)+ Flexagon(ASPLOS'23)入表
3. ③阶段 occgate/LADDER 成本模型:分层调度思想记入候选
4. 全文到手后补机理细节(层级数/检测特征/切换开销)

## 7. 信息源

- Semantic Scholar Graph API `paper/DOI:10.1109/ISCA66397.2026.00151`(标题/作者/摘要)
- mxhx7199.github.io 与 /publications/(作者列表、PDF 占位、组内发表清单)
- ResearchGate publication/411277002(403,存在性)
- arXiv 2301.10852(Flexagon 全文,前作)
