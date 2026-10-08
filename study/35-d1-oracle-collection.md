# D1 oracle 采集报告(study/35,10-08)

> study/34 D 线第一步:v1.1 口径(标定集含 first100 抽样)。目标=逐行"最优路"真值表
> ≥10 万行,供 D2 四成本律拟合(C_hash/C_dense/C_k7/C_merge)。

## 一、前置门(全部通过)

- **Xid 哨兵**:批前/批中/批后 journalctl -k 均 0(journalctl 口径,dmesg_restrict 红线后)
- **GPU 健康**:双 H100 空闲起跑;电池全程 util 正常,峰值温 74°C
- **仪器确认**(码内+二进制 ASCII 探针双验):HASH_ROWTIME(hash.cu:4663,9 列 dump:
  r a_len flop est span ovf cycles d bkid)/ DD_ROWTIME(:4678,档外老内核 cursor|search)/
  DSN_RT(:6343,档位梯 numeric)/ MRG3_ROWTIME(merge.cu:1037)/ DUMP_ROUTING(:4817,
  est/span/flop/bkid/a_len 五数组二进制)
- **k7 臂车辆 = tools/k7_bench**(独立台,与主管线 HASH_ROWTIME 同批行 host join,设计即此)
  - **sanitizer 先行(10-08)**:cuda-12.8 memcheck pre2(中 dup fixup 尾部人口,wedge#8
    教训必盖项)全绿:0 errors + 值门 0/0/0 PASS + count 444,531/444,531 精确,无卡死
  - 布局 v5(修复后)确认在源;wedge#8 退役令解除条件满足(根因已修+域内验证绿)

## 二、采集设计

标定集 13 阵:big6(bloweybq/pre2/c-43/Si10H16/in-2004/nd24k,覆盖 k7 纯靶/R1 双适用/
边际行死区×2/dense 主场/鲸鱼 dup 重型)+ first100 规模分层抽样 6(bcsstk32/bcsstk23/
bcsstm23/bcspwr08/bcspwr05/bcsstm01,C_merge 粮仓)。

臂(spgemm_test 单进程 5 轮 warmup 后热态单发;×3 rep 取中位;裸 env 避大阵 mempool 红线;
每发 timeout --signal=TERM 自杀):

| 臂 | 配方 | 仪器 | 覆盖 |
|---|---|---|---|
| H | DITER_MIN_FLOP=999999999(allhash) | HASH_ROWTIME | hash 梯全域行特征+成本+真 d |
| D | DITER_MIN_FLOP=1 DITER_EST_MIN=1 SPANF=999999 OCCGATE=0(开门) | DD_ROWTIME+DSN_RT | dense 合法域(dup_ok 结构性排除高 dup) |
| K | k7_bench <mtx> 3(独立台) | 内置 rowtime | k7 谓词域(a_len≤32∧span≤32768∧dup≤1.67∧碰撞≤512) |
| M | METHOD=merge3(first100 侧) | MRG3_ROWTIME | merge3 全行 |
| R | 默认路由 | DUMP_ROUTING | 现役人口画像(oracle-gap 对照) |

## 三、域界发现(诚实账,非失败)

1. **bloweybq = L0 纯域界**:D 臂 100% 行进**矩阵级 dense 累积器**(est 占比 100.3%,
   C 输出全稠密 100M nnz)——逐行 dense_direct 不适用,DD/DSN 空=域界非 bug。
   该阵 oracle = k7 vs 矩阵级累积器(L0 层对决,k7 A/B 系统 2.2× 已在案);truth 表
   贡献 C_k7 样本。
2. **c-43/Si10H16/in-2004 D 臂 dsn 空**=档内(span≤7277)无人,dd(档外老内核)有数
   ——部分成功,C_dense 照常拟合(dd 行)。
3. **nd24k**:dup 62.8 → dup_ok 结构性排除 dense(逐乘积原子经济学),D 臂覆盖以
   数据为准(见 §五)。

## 四、D6 前置查重结论(10-08,动笔前义务)

- 逐行混合派发前作:**Winter 2019**(Adaptive SpGEMM,逐行策略动态选择)、**spECK 2020**
  (hash/dense/direct 局部切换)——全为**启发式阈值**,无学习型、无证书兜底
- 学习型选择前作:**Zhao 2018**(DL 格式选择)、**Dufrechou 2021**(ML 分类器 SpMV)、
  **Dinkelbach 2022**(学习型成本模型,云场景)——全为**矩阵级 + SpMV**
- **Ocean 2026**:HLL 逐行估计但单物理轴阈值
- **占位判定:学习型逐行 SpGEMM 派发 + 核内证书兜底 + 物理先验混合 = 无人占位,D6 干净**;
  必引邻作四家(Winter/spECK/Ocean/学习型格式选择谱系),定位=逐行 dispatch+证书+物理先验

## 五、采集结果(10-08 电池:74/90 发成功,550s;16 FAIL 全为 §三域界,零数据损失)

| 阵 | rows | hash 测 | dense 测 | k7 测 | mrg3 测 | best 行 |
|---|---|---|---|---|---|---|
| bloweybq | 10001 | 10001(全 HEAVY 域 cyc=0) | 0(L0 域界) | 10000 | — | 10000 |
| pre2 | 659033 | 659033 | 418695 | 444531 | — | 632839 |
| c-43 | 11125 | 11125 | 6093 | 5522 | — | 11066 |
| Si10H16 | 17077 | 17077 | 15838 | 6 | — | 17075 |
| in-2004 | 1382908 | 1382908 | 319109 | 186502 | — | 834709 |
| nd24k | 72000 | 72000 | 71047 | 0(k7 域空:a_len 798 鲸) | — | 71912 |
| first100×6 | 52892 | 全 | 0(小阵无 dense 采样) | 4563 | 52892 | 52892 |
| **合计** | | | | | | **1,630,593**(判据 10 万的 16 倍)✅ |

**现役人口画像(R 臂 DUMP_ROUTING)**:bloweybq 100% diter(矩阵累积器);pre2 hash 梯
60 万行滞留(bin6 261K/bin5 99K/bin3 91K)=R1 靶群复认;c-43 三路混合(3021 diter+4300 ESC);
Si10H16 bin5-6 1.3 万行=死区人口。与既定地图全吻合。

**c-43 三路分裂实证**:best = hash 5325 / k7 5164 / dense 577——静态门错价边际行人口
(c-43 −10%/Si10H16 −5% 回归的行级根源)确有三路真实选择空间,PVD 定价靶群成立。

## 五b、D2 首轮拟合(v0/v1/v2 三刀,附诊断)

- **四律立住**:C_hash R²=0.952(n=130 万,含 κ·a_len 链项后)/ C_dense[cursor] 0.938 /
  C_k7 0.833 / C_mrg3 0.790(first100 一致率 91-100%)
- **C_dense[search] 形态失败但根因隔离**:按阵分拟合 c-43 R²=0.986(corr(flop)=0.99,
  形态对)而 pre2 R²=0.126 / in-2004 R²=0.011(corr(span)=0.11,flop=0.07,中位 2.2M cyc
  与特征零相关)——非形态错,是**巨行人口的测量语境现象**。两假设待 D2 专轮:
  ①强制臂 co-residency 副作用(全阵 dense 化后的争用语境 ≠ 生产路由语境);
  ②带宽饱和平台(巨行扫窗字节饱和,成本= max(flop·δ, 平台) 而非线性)
- **argmin 一致率**:v1 全局单律 59.8% / v2 三子律 46.6%(search 律崩连带 in-2004 32.7%/
  pre2 38.6%)——**D2 判据(>85%)未过**,根因已隔离到 search 巨行子律;oracle-gap 口径
  同因失真(v1 口径 1.121)
- 系数存档:truth/costlaws_v1.json, costlaws_v2.json(D2 迭代起点)

## 六、交付物

- `scripts/collect_d1_oracle.py`(采集,可复跑)/ `scripts/build_d1_truth.py`(join 构建)/
  `scripts/fit_d1_costlaws.py`(D2 拟合,本报告只出原始 argmin,系数拟合见 D2 报告)
- `compare/d1_oracle/`(原始 tsv/bin/log)/ `compare/d1_oracle/truth/*.truth.tsv`(真值表)

## 七、[10-08 晚] 补采 + search 幽灵律终审 + D2 v3b(本轮desk专轮收口)

### 补采电池(5 阵+DN 臂,26/47 OK,域界全分类)

- 新增 **DN 臂**(默认路由+DD/DSN 仪器)=自然语境 dense 逐行周期,与 D 臂(强制)同行配对
- crankseg_2(中 dup 15-60 人口)/F2(甜点带)/c-64(cursor 主场)H/D/DN/R 齐;1138_bus+ash219
  (非 bcs 族,C_merge 泛化)H/M 齐;F2=全微行 dense 不可达域界+K 域空(dup 15-60);
  ash219.K=非方阵工具域界(主二进制 H/M 正常)
- **真值表 v3b 全集:18 阵,1,798,177 行**

### search 幽灵律终审(两假设一并判决,数据定谳)

1. **自然路由 search 人口≈0 全域**:in-2004 DN 臂 153K dd 行中 search 仅 **7 行**;
   c-64 DN 19,347 cursor vs **97 search** → C_dense[search] 出律(PVD v1 不需要它)
2. 强制臂 26.6 万"平台行"真相:flop p50=132 微行(avgB<64→search 态)被硬塞 1024 线程
   窗内核,每行付 ~2.2M cyc 固定开张成本=**坏假设决策的成本,非 dense 真单价**
3. 7 行真怪物 search:DN/D=0.983(语境无关,成本本来就真)
4. **cursor 语境温和**:crankseg DN/D=1.015,in-2004 DN/D=1.112(±11% 带)
5. 单价语境漂移普适实证:search 态 per-flop 17.5(c-43 小阵)→301(in-2004 百万行发射);
   **hash 同病**:实测 cyc/flop pre2=35.6 vs in-2004=92.2(2.6×)

### D2 v3b(修 oracle 污染+dense 合法域 avgB≥64 后)

- **11/15 判阵 ≥84%**:first100 六阵 91-100%/c-64 91.8%/crankseg 84.3%/c-43 92.9%/
  nd24k 94.2%/1138_bus 99.7%
- 律:C_hash .926 / C_dense[cursor] .920 / C_k7 .831 / C_mrg3 .800 / dsn 梯 n=1545 薄样本
- **残症(未定谳)**:in-2004 19.4%/pre2 38.3%/Si10H16 67.6% = 单价语境漂移压在边际
  (pre2 best 结构 dense 274K/k7 153K 全是边际行;池化单律天花板)
- **D2 下一轮设计分叉(等用户定向)**:
  A. 成本律加语境特征(发射规模/bin 人口等 dispatch 时可算量)——合规(非运行时自调)
  B. 两级标定:矩阵级语境系数 × 行级律(L0 思想下放)
  C. hash/dense 律改 per-matrix 标定表(51 阵屏离线扫)——最重但最准
