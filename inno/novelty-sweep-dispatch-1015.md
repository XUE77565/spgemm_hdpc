# 查新:形态调度 / 三层裁定 / ML 分发的先作扫描(10-15)

> 触发:用户问"以前没有人用这种方法进行调度吧"→"要不做一下查新"。
> 方法:WebSearch 多轮;**Wang 2025 已取一手摘要**(BIT 机构库 pure.bit.edu.cn);
> 其余为二手摘要 + 既有文献知识,按结论三档纪律标置信。
> **本文件 = 论文 related work 划界与主张收窄的依据**;配档 inno/mmspgemm_comparison.md(MMSpGEMM 撞轴)。

---

## 一、结论先行(定谳)

**"用 ML/特征做 SpGEMM 逐阵选择"本身不是空白。** 三件前作:

| 前作 | 干了什么 | 与我们主张的关系 |
|---|---|---|
| **IA-SpGEMM**(Xie et al., ICS 2019, ~90 引) | SuiteSparse 语料训练,**逐输入选 format+算法**(7 种 SpGEMM 变体),SMAT(SpMV)血统 | **最接近的"input-aware 分发"祖师**,必引;但它选的是**同一两阶段结构内的变体**,非管线形态 |
| **Wang 2025**(ACM TACO 28(4)? DOI 10.1145/3774654, BIT, Dec 2025) | 一手摘要定谳:"**trains a lightweight ML model to predict the optimal method for estimating the size of result matrix**" + 行分组 kernel 发射 + hash 负载因子启发式 | ML **只用于估计器选择**(symbolic 阶段结果规模估计方法);**单形态 hash 管线内**;正好是我们 L0-③ 估计器维度的最近前作,必引 |
| **ocean**(arXiv 2604.19004, 2026) | 估计先行+运行时自纠;decideMatrixType = **静态阈值**在两形态间切 | 对手即前作:形态间切换存在,但是**单变量阈值**;我们=多变量+三层 |

**我们仍可主张的 novelty(论文语言按此收窄)**:

1. **形态级分发**:整个管线形态三选一(merge3 3相 / 轻hash 4-5相 / 全形态 17→8→5,
   相位数+kernel 家族+估计/证书机制整体切换)。前作的"选择"粒度是 format/累积器变体
   (IA-SpGEMM)或估计方法(Wang 2025),不是管线架构本身;
2. **三层裁定原则**:解析门(α·Σflop vs β·Σnnz+γ·A_rows,binning 后零预跑)→
   状态机(margin 证书直方图先验)→ 学习型(只管物理不可约的)。**前作全是单层**
   (一个阈值或一个模型);分层本身 = 架构贡献;
3. **证书兜底**:溢出证书判废→形态级回退(merge3 永远活着),让 aggressive 决策敢判死刑;
   hash 溢出回退我们 8 月已有(创新点 2),证书=其系统化(判废证据链,非仅 retry);
4. **系统级否定结果**:逐 kernel 时间 ≠ 系统成本(集体效应,L0 发现,337 数据背书)。

## 二、检索过程中的方法论记录(防再踩)

- 二手摘要会**放大语义**:第一轮搜索把 Wang 2025 说成"ML 预测最优 SpGEMM algorithm",
  ACM 官方摘要说"行分组+线程块配置",一手摘要实际="ML 预测**估计方法**选择"——
  三种说法两个错。**一手摘要定谳制度**保留(本档已执行)。
- dl.acm.org / pure.bit.edu.cn 的 WebFetch 均被域名校验挡;**webReader(MCP)可穿透**,记入工具链。

## 三、必引清单(related work 骨架)

| 类别 | 引文 | 用途 |
|---|---|---|
| input-aware SpGEMM 选择 | Xie et al., ICS 2019 (IA-SpGEMM) + SMAT (SpMV 血统) | 承认"逐阵选择"先作;划界粒度 |
| ML in SpGEMM | Wang et al., TACO 2025 (DOI 10.1145/3774654) | 估计器选择先作;我们 L0-③ 划界 |
| 对手 | ocean, arXiv 2604.19004 | 静态阈值 vs 三层;估计+自纠哲学对照 |
| 自适应 kernel | Winter et al. 2019 (adaptive); Parger et al. 2020 (spECK); Nagasaka (nsparse) | 行级自适应传统(不是形态级) |
| merge 轴 | MMSpGEMM, PACT 2025(详见 inno/mmspgemm_comparison.md); Dalton merge-based | merge 路线定位 |
| 算法选择框架 | Rice 1976; SATzilla; OSKI (Vuduc 2005) | 三层裁定的理论挂靠 |
| 2026 新作扫描 | Trident partitioning SpGEMM (arXiv 2026) 等 | 证明领域活跃,无形态分发撞车 |

## 四、语言修正(能说 / 不能说)

- ❌ "以前没有人用 ML/特征调度 SpGEMM"(IA-SpGEMM 2019 就是);
- ❌ "首个 input-aware SpGEMM";
- ✅ "首个**管线形态级**分发(相位结构整体切换)+ **按可判定性分层**的裁定架构
  (解析可解不学习/证书反馈用状态机/物理不可约才学习)+ **判废证书链**";
- ✅ 否定结果表述不变(逐 kernel 时间≠系统成本是我们的独立发现)。

## 五、遗留(投稿前 DDL 项)

- [ ] IA-SpGEMM 机制描述目前来自检索摘要;投稿前下 PDF 复核
      (ssslab.cn/assets/papers/2019-xie-spgemm.pdf):它是**预测式**(特征→模型→选择)
      还是**逐输入试跑自调优**(measure-then-cache)——这决定我们"在线学习 vs 离线自调优"
      划界语言的强度;
- [ ] ocean arXiv 2604.19004 全文精读(其 related work 可能已引 IA-SpGEMM——若引了,
      我们的三层划界反而更稳);
- [ ] Wang 2025 全文(是否有除估计器外的第二个学习维度)。
