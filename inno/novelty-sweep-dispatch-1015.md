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
| **IA-SpGEMM**(Xie/Tan/Liu/Sun, ICS 2019, ~90 引) | **一手摘要已核(S2 API)**:"automatically determines the best format and algorithm"——深度模型 **MatNet**,**2700+ SuiteSparse 阵训练,预测式**(sparse features + density representations → 直接预测,非试跑自调优);CPU+GPU 评测 | **最接近的"学习型逐阵分发"祖师**,必引;但它选的是**format+算法变体(同一两阶段结构内)**,非管线形态;单层(一个模型),无解析门/证书/回退 |
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

- [x] ~~IA-SpGEMM 是预测式还是试跑式~~ → **已定谳(10-15,一手摘要)**:预测式。
      MatNet 深度模型、2700+ 阵训练、sparse features + **density representations**
      (矩阵密度图像类特征,比我们 O(1) 头部特征重得多——这本身是我们可说的差异点)。
      ssslab.cn PDF 链接已 404,S2 API 摘要为准;全文复核降级为可选项。
- [ ] ocean arXiv 2604.19004 全文精读(其 related work 可能已引 IA-SpGEMM——若引了,
      我们的三层划界反而更稳);
- [ ] Wang 2025 全文(是否有除估计器外的第二个学习维度)。

### 追记(10-15,IA-SpGEMM 一手摘要后的划界细化)

- 划界语言可以从"在线学习 vs 离线自调优"改为更硬的三点:①**粒度**(它=format/算法
  变体,我们=管线形态:相位结构+kernel 家族+估计/证书机制整体切换)②**架构**
  (它=单层深模型黑盒;我们=按可判定性三层:解析门不学习/证书反馈走状态机/不可约才
  学习——且学习器只吃 binning 副产品特征,O(1))③**安全网**(它无;我们=判废证书链
  +merge3 形态兜底)。
- 泄漏纪律呼应:MatNet 2700 阵训练的 train/test 划分方式论文里值得查一眼——我们的
  LOO-by-matrix/跨种群/时序外推验证协议若比它严,可作为方法学卖点写一句。
