# dense/cursor 攻坚(用户钦定主线,四步路线;2026-09-21 立卷)

> 路线:①口径对齐表(本节)→②kernel 级 us-vs-ocean profiling→③瓶颈读码+单刀→④创新超越。
> 前情:c-64=cursor 主场(19,444 diter 行 99.5% cursor,96.4% 输出);spec82 占用刀后
> dense_direct 13.8ms;kernel 级 hash 已三阵全胜(study/21+22),dense/cursor 载体质量从未
> 直接对拍过=本卷要补的洞。

## §1 口径对齐表(①阶段交付,09-21;素材=docs/76 §三/study/21+22/77+今晨实测)

### 1.1 计时结构对照

| 维度 | 我方(相位和) | ocean("Average SpGEMM time") | 不对称方向 |
|---|---|---|---|
| 计时器 | HashProf 相位 cudaEvent 对(GPU 流 elapsed) | cuTimer cudaEvent 单括号包全程 | — |
| TOTAL | **Σ相位**(相位间 host 间隙不计) | 单括号(内部 host 空洞全计) | **偏我方** |
| 排除项 | h2d/d2h/mtx 读 | 同(彼 D2H 1499ms 病理在计时窗外) | 同 |
| 统计 | 末轮+双 expand min | 10 warmup+10 轮 mean | **偏我方** |
| warmup | 有 | 10 轮 | 近似 |

**三处不对称全部偏向我方 → 所报比值均为保守口径**(真实差距只会更大);论文注记义务:
对比表须注明"Σ相位 vs 单括号"结构差(docs/76 §三原判,09-04 源码级验证)。

### 1.2 我方相位 tag 边界定义(供②阶段对号)

| tag | 内容 | 边界注意 |
|---|---|---|
| h2d/d2h | 传输(排除) | — |
| count_flop | 乘积计数 | ocean 对应 product_calc(analysis) |
| mh_construct/mh_merge/est_scan/binning | 估计器链+分箱 | ocean=analysis+est(~HLL/ER/采样) |
| accumulate | hash 行累积(hash_spa) | **含 kernel 内 extract**;ocean 排序在独立 kernel |
| retry | 溢出行重跑(全相位含 host 编排) | ocean=out_overflow 补丁(飞行中切表) |
| cnnz_scan | 行 nnz 扫描 | — |
| compact+sort | 提取区拷贝+排序 | ocean=epilogue(sortOutputFused,copy 段) |
| dense_count/dense_direct | dense/cursor 路径计数+累积 | **本卷主对象**;dense_direct 含 cursor 模式 |

### 1.3 kernel 级口径(②阶段的尺子,study/21+22 已定型)

- **口径 A+**(窗对齐):我方 HASH_RT_NOEXTRACT=1+HASH_UO_XTRACT=1(窗=探测循环,不含
  extract/排序)vs ocean ROWCYCLES(仅 AccumulatorHash 发射,**dense/ultra 行无逐行数据**=
  密侧对拍的坑,见 §3)
- **口径 C**(块宽对齐):我方 bin 梯对齐彼档位宽;hash 侧已用此口径三阵全胜
  (F2 0.356/c-64 0.906/crankseg 0.581,09-18 定谳)
- 逐行 DD_ROWTIME(3 列:行/cycles/mode,cycles 在 $2)vs ocean denseNumericIter 需彼侧仪器化
  二进制(ROWCYCLES 只覆盖 hash → **dense 侧无现成逐行数据,②阶段须用相位级/ncu 级替代**)

### 1.4 时效纪律(09-21 实录课)

- **ocean 参考值随日漂移**:F2 = 7.66(09-18)→ **7.410(09-21 复测,−3.3%)**;c-64 = 20.80(09-18)→
  20.753(09-21)。**跨线官宣必须同日双侧净窗复测**,历史值只作趋势
- 我方 off 带同理漂移(±2-5%),电池一律配对同窗

### 1.5 dense 系 gap map(77 章 dense 部未平项,09-21 盘点)

| # | 差异点 | 77 判定 | 现状(09-21) | 剩余空间 |
|---|---|---|---|---|
| 1 | 符号形态 1-bit 位图 vs 8-bit 标记 | ocean 碾压 | DSTRUCT Phase 1a 否决(c-64 +36% atomicOr 争用);**争用形态未解** | 位图+无争用写法(如 per-warp 位图聚合)=②候选 |
| 2 | numeric 9B/列 vs 我 17B/列(含 4B 游标) | ocean 胜(账面) | 游标是我方机制胜点(书签续扫);dense_struct_numeric 8.25B/列已存在(结构行用) | 双形态融合:游标列可选化(宽行才买) |
| 3 | 发射:精确座位直写 vs 扫标记半串行 | ocean 胜 | 方案5 直写已落地(dense 行);**cursor 模式仍是扫标记** | cursor 行的直写化=③候选 |
| 4 | 窗宽档位七档 vs 单档 | ocean 胜 | 档位梯已落地(v33,in-2004 −30%) | 已平 |
| 5 | 超大行 LARGEST | 平 | 已抄完(c-big −63%) | 已平 |
| 6 | 瘦行小档 | 平 | 两边都有 | 已平 |
| 7 | 路由判据 | 我胜 | 字节模型+占用门+dup 门 vs 彼档位号比较 | 保持 |
| 8 | (新)占用物理 | — | **我方独有台账**:寄存器悬崖/占用恒等式(spec82/83 刀) | 已入 DISPATCH_MODEL 候选 |

**未平的真差距收敛为三件:①位图符号形态的争用写法 ②cursor 模式发射直写化 ③游标列按行买卖**——
全部是③阶段的刀口候选,②阶段画像定优先级。

## §2 ②阶段执行清单(kernel 级 us-vs-ocean profiling;B 级可共租跑)

1. **mode 分层人口普查**(四阵:c-64/crankseg/F2/in-2004):DUMP_ROUTING 落盘
   est/span/flop/bkid+mode → cursor vs dense 行群画像(每 mode 的行数/flop 占比/输出占比)
   ——B 级,共租可跑
2. **dense_direct 13.8ms ncu 画像**(c-64,刀后状态):smsp__inst_executed/warps 驻留/
   SMEM 停顿/长计分牌,对比 ocean denseNumericIter 同指标(彼侧无逐行,用相位级 ncu 聚合)
   ——B 级,共租可跑(时长膨胀只看比值)
3. **窗税定量**(study/22 遗留:30µs/窗 clear+prefix):DD_ROWTIME 按窗数分层,宽行窗数分布
4. **溢出税复查**:mode 分层账(spec82 法)在 13.8ms 状态复测(+47%/行→现值)
5. 产出:②判决表(载体质量 cursor vs denseNumericIter 逐项)+③刀口优先级

## 状态

- 09-21 09:03 单发#10 立 §1+§2;②阶段执行待净窗/共租窗均可(B 级)
- 并行线:RETRY2 刀候默认全绿(commit b5809a4,等用户过目);mempool 决策备忘(8赢2输)
