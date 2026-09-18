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

## 2. 检索状况(09-18 傍晚更新:全文已到手)

- 上午:IEEE 反爬/RG 403/作者页占位,只有 Semantic Scholar API 摘要
- **用户提供全文 PDF**(仓库根目录 `Harmonia_*.pdf`,不入库——IEEE 版权 Restrictions apply)
- 下文 §3 为全文版机理,§6 为全文后再判决

## 3. 机理(全文版,09-18 傍晚补全)

**三层层级调度框架**(Fig.4):

1. **静态解析层(离线)**:只用粗粒度描述符(矩阵形状/全局密度/数据类型)+ 硬件参数
   (PE 阵列/SRAM 容量),以**最大化操作强度 OI = Ops/Bytes_loaded** 为目标,在 SRAM
   可行性约束(期望足迹 ≤ β·S_SRAM,β=0.8 安全系数)下枚举候选 tile 形状选最优
   → 产出基线 tile 形状 (T_M,T_K,T_N)/遍历序/SRAM 三分 partition。期望 nnz 用独立性近似。
2. **动态画像层(块驻 SRAM 后,每块一次)**:tile 元数据给精确 nnz → ρ_tile;对驻留侧 A
   轻采样 → 行级方差/nnz 聚类/PE 行负载不均衡。微调规则:低密度**扩** tile(提复用)/
   高密度**缩**(防 psum 溢出)/聚类**对齐** tile 边界。dataflow 选择:
   InP=中度稀疏+行均匀 / Row=行方差大或聚类 / OutP=A 均匀+B 低密度
3. **动态调优层(硬件衬底内,终层)**:反馈计数器(SRAM 压力/psum 溢出标志/MRN 合并深度
   与停顿/PE stall 周期)→ 异常检测 → 纠正动作(**切 dataflow**;**微重分 tile** 缩/放)。
   **成本门:Gain > α·Cost**(Cost = 重配置+流水线冲刷+缓冲复位,实测 20-50 周期);
   **滞回防振荡**:异常计数须连续 T=2-4 周期超阈才触发;**最坏情况有界**:切换失败
   回退静态基线,罚金恰为一次重配置延迟(50 周期)

**三条 Insight(动机实验支撑)**:①层间耦合——tile 形状/占用会**反转** dataflow 排名,
不能分层独立优化;②静态解析模型在 tile 级稀疏变异下预测误差 **3.1-3.7×**(Vesper 类),
必须动态反馈;③硬件灵活性(Trapezoid 类)需要系统级调度层才能兑现 → "逻辑异构"

**硬件**:Trapezoid 式 PE 行(DN+MRN)上仅加 SW-HW 调度接口(反馈计数器 <0.5% 阵列面积/
重配置引擎/tiling 控制器),**数据通路零改动**;三 dataflow(InP 脉动流 B/Row 选择性路由 B/
OutP rank-1 广播)全靠 DN/MRN/缓冲重编程实现。

**评估**:周期级模拟器(Trapezoid 32 行架构),基线 = **同一架构上**的 Vesper/Misam-like/
HYTE-like 调度器 + 静态最优 oracle;1.75× 性能/2.47× 能效;工作负载 = SuiteSparse SpMSpM +
剪枝 DNN(LLaMA/ResNet/VGG/OPT)。**无 GPU 对比——纯加速器对加速器(模拟)**

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

## 6. 机理映射到我方栈(全文后再判决)

| Harmonia 层 | 我方对应物 | 可吸收机理(不 copy 形态) |
|---|---|---|
| 静态解析层(OI 最大化+β 可行性约束) | est/分箱(DITER_EST/PB2_W,现为启发式) | **OI=Ops/Byte 作窗口/桶尺寸的原则性目标函数**;β 安全系数思路同我 retry 余量 |
| 动态画像层(行方差/聚类/不均衡采样) | rowmode(avgB 单特征→cursor/dense) | **行方差/聚类特征进 mode 选择**(现在只有均值一个特征) |
| 动态调优层(反馈+成本门+滞回+最坏有界) | retry(表满重试)= 其粗糙版;occgate(③阶段) | **Gain>α·Cost 成本门 + 滞回 + 失败回退静态基线** = retry 刀与 occgate 的鲁棒设计模板(与鲁棒架构铁律兼容:有出处的模型,非补丁门) |
| 层间耦合分析(Insight 1) | DISPATCH_MODEL 联合训练 | 佐证"分箱×mode×kernel 不能各旋钮独立调" |
| 静态误差 3.1-3.7×(Insight 2) | est vs 实测偏差 | 佐证 retry 纠偏机制存在的必要性 |

**查重警示(C 线)**:**Misam(决策树从矩阵统计选 dataflow)是 DISPATCH_MODEL 最近前作,
论文必引**;定位链:Flexagon(离线选模板)→Trapezoid(结构可重构缺系统调度)→
Harmonia(ISCA'26,层级调度补系统层)= 硬件侧;我方 = GPU 软件侧同构(分箱↔tile、
kernel 家族↔dataflow、retry/occgate↔动态调优),互补不撞车。

**定谳(全文后)**:模拟器评估 + 加速器对加速器、无 GPU 实现——**无形态可 copy 最终确认**,
全部吸收走机理;Harmonia 的"评估在同架构上只换调度器"的对照方法论值得我们对照实验借鉴。

## 7. 行动项(全文后更新)

1. ✅ 全文已读,机理/映射入档(本文件 §3/§6)
2. 论文 related work dispatcher 段四件套:Harmonia(ISCA'26)+ Flexagon(ASPLOS'23)+
   Trapezoid(HPCA'24?待核)+ **Misam(决策树选 dataflow,DISPATCH_MODEL 最近前作)**
3. retry 刀/occgate 设计采用 **Gain>α·Cost + 滞回 + 最坏有界回退** 模板
4. OI 目标函数列为 est/窗口尺寸候选机理(不动现刀;转正需归因对齐+守门)

## 8. 信息源

- Semantic Scholar Graph API `paper/DOI:10.1109/ISCA66397.2026.00151`(标题/作者/摘要)
- mxhx7199.github.io 与 /publications/(作者列表、PDF 占位、组内发表清单)
- ResearchGate publication/411277002(403,存在性)
- arXiv 2301.10852(Flexagon 全文,前作)
