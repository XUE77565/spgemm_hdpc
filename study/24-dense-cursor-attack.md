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

## §2.1 mode 分层人口普查(09-21 09:4x,单发#11,B 级;r2t,DUMP_ROUTING+DD_ROWTIME 各 1 发)

| 阵 | diter 行(%) | cursor | dense(位图) | flop 占比 | est 占比 | 行均 est/span | cursor 均cycles |
|---|---|---|---|---|---|---|---|
| c-64 | 19,444(38.1%) | 19,347 | 97 | **98.9%** | 96.6% | 13,835/50,643 | ~266K |
| crankseg_2 | 12,526(19.6%) | **12,526** | **0** | 25.9% | 36.1% | 5,527/**57,981**(全幅) | **~443K** |
| F2 | **0** | 0 | 0 | 0 | 0 | — | —(纯 hash 阵) |
| in-2004 | 10,380(0.8%) | 10,373 | 7 | 2.1% | 10.8% | 3,369/4,096 | ~35K |

**四个定谳**:
1. **c-64 旧数复立**(19,444/99.5% cursor/96.6% est)——RETRY2 时代无迁移
2. **"dense"位图模式实践性灭绝**(四阵合计 104 行)——hash_dense_direct 的 dense 实为 **cursor 单一
   形态**;gap map #2(cursor 发射直写化)升为头号刀口,#3(游标列)全员持有
3. **crankseg 双载体确证**:12,526 全幅宽行 cursor(443K cycles/行,重型人口)+ hash 主体——
   其系统差距 1.83× 是双载体混合账(c-64 式 cursor 账 + 路由账)
4. **F2 纯 hash 阵**(0 diter 行)——retry 刀选 F2 做主战场的人口学依据坐实;in-2004 的档位梯
   结构行走独立 kernel(dense_struct),diter 只是 0.8% 尾巴
- 注:gap #1(位图符号形态)靶点在 **dense_count 相位**(c-64 2.64ms,服务全部 diter 行计数遍)
  ——非灭绝项,但 DSTRUCT 争用教训(c-64 +36%)在宽行(span 50K=1.6K 位图字)上机理存疑,③重审
- 溢出税复查(件 4)未做(轮预算),下轮与 ncu 画像并

## §2.2 刀后 ncu 画像第一读数+决赛设计(09-21 11:4x,单发#13,B 级)

### ncu 刀后状态(c-64,r2t,--set basic,launch#10;对照 spec82 刀前)

| 指标 | 刀前(spec82,09-18) | **刀后(09-21)** | ocean denseNumericIter(旧,09-14 彼未变仍有效) |
|---|---|---|---|
| warps 驻留 | 49.97% | **99.65%** ✅ 占用刀效果 ncu 级确认仍持有 | 99.67% |
| Compute(SM) 吞吐 | — | 49.52% | (待查旧档对表) |
| Duration(ncu 膨胀) | 26.9ms | 21.18ms | 20.6ms |

- **判读:占用已平(99.65 vs 99.67),内核不再是占用瓶颈——剩余差距在指令/延迟侧**
  (Compute 仅 49.5% = 约一半 SM 周期空闲,停顿分解下轮补:ncu section 名版本坑,--set basic 可靠)
- ⚠ ncu 滤镜坑入档:`-k 'regex:hash_dense_direct'` 报"no kernels profiled"但
  `regex:hash_dense_(count|direct)_kernel` 命中——正则写法敏感,用全名枚举式

### 决赛 B 台设计(强制对拍,三杠杆已侦察齐)

| 杠杆 | env | 现状 |
|---|---|---|
| 模式门(cursor↔dense/search) | **PB2_AVGB**(运行时 avgB 阈,传入 kernel;默认 0=docs/66 门) | 强制全 cursor=调低;全 dense=调高;方向待实证 |
| diter 人口门 | **DITER_EST_MIN**(默认 1024)+**DITER_MIN_FLOP**(默认 2048) | 调 0=全员 diter |
| ocean 侧 | bench.json workflow/dense bin 可钉性 | **待查(下轮 desk)**;钉不了则用其自然 dense 阵提指标 |

- 对拍阵:pre2 型(中行族,全 dense 化可行)+c-64 型(宽行,彼痛点)各一;判分=同人口逐行
  (DD_ROWTIME)或相位级,同窗配对

### §2.2b 停顿分解定谳+ocean 可钉性(09-21 12:4x,单发#14)

**hash_dense_direct 停顿表(c-64,cycles/issue)**:

| 停顿源 | direct | count | 判读 |
|---|---|---|---|
| **long_scoreboard** | **11.78** | 5.66 | **全局访存延迟=真瓶颈(~55% 停顿)**——B 链续扫/A 行/窗口流的全局读 |
| short_scoreboard | 3.64 | 1.08 | SMEM 链次因(窗内探测) |
| wait | 3.30 | 2.80 | 固定延迟依赖 |
| not_selected | 2.05 | 2.30 | warp 充足 → **非吞吐/指令数瓶颈** |

- **③刀口定性(决赛 A 的靶)**:cursor 内核 = **访存延迟瓶颈**(IPC 2.0/Issue 49.9%/占用 99.65%),
  刀口=压 B 链全局读的延迟暴露:预取/双缓冲下一窗 B 行/更宽装载/B16 已开下的流重组;
  次刀口=SMEM 探测链(3.64)。对表 spec82:溢出税 +32% 指令是地址重算型——那些指令喂得饱
  (SM Busy 49.9%),墙在等数据
- **ocean 可钉性(workflow 级 ✓,行级未验)**:analysis.json 有 use_estimation/ultrasparse/
  **use_fat_workflow** 三开关(baseline.json 全 false 先例)——关 hash 侧工作流可逼 dense 主路;
  bench.json 仅计时参数;行级 dense_bin 钉法未验,决赛用"自然 dense 阵+workflow 开关"组合
- ncu 停顿表采集法入档:--metrics 显式列 smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio
  (--set detailed 的 WarpState 段在此版本不打印)

### §2.3 决赛台首烟(09-21 13:2x,单发#15):模式杠杆定谳+人口墙发现

- **模式杠杆修正**:PB2_AVGB 无效(模式判据=编译期 AVG_FLOP_THR=warp×2);真开关=**PB2_CURSOR
  0=强制 search(全体 dense 化)/2=恒 cursor**(源码注释"实验用"现成);c-64 三发模式计数验证语义来源
- **人口墙(重要发现)**:pre2(659,033 行)在 **DITER_EST_MIN=0+DITER_MIN_FLOP=0+SPANF=100000
  三 env 全开下仍 diter bin=0**——中行族被更深的结构性门(非 env 暴露层)拦死,我方路由当前
  **不允许**中行族进 dense/cursor 载体(与 study/22 "c-64 中带无主行型"同族);两强制模式相位和
  31.05ms 全同(0 diter 行,模式开关空转)✓ 自洽
- **结论:决赛台缺人口杠杆**——需 **FORCE_DITER_ALL 强制刀**(binning 终判处 env 覆盖,行进
  diter 合法只是不经济——正是决赛要测的经济学);pre2 强制后人口=65 万行,对拍可行
- ocean 侧 workflow 开关首烟未跑(轮预算);下轮=强制刀实施+决赛台重试

### §2.3b 人口墙破+决赛首战果(09-21 14:2x,单发#16)

- **门考古破案**:v4 分支准入要 `diter_thr>0`——DITER_MIN_FLOP=0 是拆门不是开门;正确强制栈=
  **DITER_MIN_FLOP=1 DITER_EST_MIN=1 SPANF=100000 OCCGATE_COEFF=1000000 LADDER=0**
  (超档行 est/flop 地板是结构化 EST_DENSE_MIN/2 非 env 层,但 SPANF/OCCGATE 松开即可过)
- **pre2 强制人口=418,695 行(63.5%)进 diter**;默认路由 31ms → 强制全 cursor 399ms(count 83+direct 316)
  = 强制经济学代价 ~13×(这正是决赛要测的量)
- **决赛首战果(同人口 418K 行,单发 B 级)**:
  | 模式 | dense_count | dense_direct | 判 |
  |---|---|---|---|
  | **cursor(PB2_CURSOR 默认)** | 83.2 | **316.2** | ✅ |
  | **search(PB2_CURSOR=0)** | 84.9 | **960.6** | **cursor 同人口完胜 3.04×** |
- **书签机制价值首次受控定量化**:316 vs 960——换窗重搜 vs 续扫的全部差价;论文 C2 的实证弹药
- ocean 侧漏 convert(csr 不存在静默无输出),config 已还原;下轮补转换+三方对表

### §2.3c 三方对表+spec86 立(09-21 14:4x,单发#17)

| 配置 | pre2 计时 | 口径 |
|---|---|---|
| 我方默认路由 | ~31ms(相位和) | 65 万行 hash/档位梯分担 |
| **ocean 默认路由** | **15.20ms**(10 iters,自然 dense 分流) | **系统差距 2.04×=中行族载体经济学的账** |
| 我方强制全 cursor(418K 行) | dense 相位 399ms | 强制人口,测载体绝对成本 |
| 我方强制全 search | 960ms | 同上,书签差价 3.04× |

- 注:强制人口(全行)与彼自然 dense 人口(子集)不同基——**同人口对拍须彼侧行集识别**(其
  ROWCYCLES 不覆盖 dense 行,下阶段用 ncu 相位级或 population 反推);当前有效结论=系统差距
  2.04× 定量 + 我方载体两模式的受控差价
- **spec86 立**(inno/hash_opt_2026/86):读码定谳 cursor 内环双流散载(B_col/B_val 两独立全局读
  stride 8)且 **B16 交织从未移植进此内核**(signature 无参,buffer 已存在未传)——变体 A=B16
  移植(先落)/B=书签预取(书签即预取地址)/C=__ldg;验收=c-64 13.6→<12+long_scoreboard 11.78→<8

### §2.4 决赛主阵(in-2004)首轮对表(09-21 17:3x,单发#23)

| 口径 | 我方 | ocean | 比 |
|---|---|---|---|
| **系统默认路由**(相位和 vs 10 iters) | 50.3ms | **32.3ms** | **1.56× 彼胜——决赛阵差距定量** |
| 我方强制全员 cursor(1.38M 行) | count 122+direct 1405ms | — | 强制全员的绝对成本(窗口随 span 爆炸=人口滥用基线,非决赛数) |
| 我方强制全员 search | count 124+direct 1739ms | — | 书签差价 1.24×(窄 span 阵上差价缩水) |
| **彼 dense 阶梯**(ncu,枚举 12 发) | — | **七档实例化**(<7>7413块×128thr 300µs/<8>6723×256 377µs/<9>4085×512 351µs+SubWarpDispatcher+denseNumericIter) | 彼 denseNumeric 单迭代 ~1-1.3ms(ncu 膨胀,比值口径)服务其路由人口 |

- **彼侧解剖新知:ocean 的 dense 是七档模板实例化阶梯+SubWarp 派发+Iter 内核**——比我方目录
  (cursor+档位梯两载体)更细的载体内分工;这是 77 章"七档联动"的 kernel 级实锤
- **决赛协议缺最后一块:人口同一化**——强制全员(我方 1405ms)vs 彼自然子集(~1.3ms)不同基;
  正解=从彼 analysis/stats 提取其 dense 行集→我方强制到**同一行集**→逐相位对拍(下轮)
- 系统级 1.56× 即当前决赛记分(保守口径我方相位和 vs 彼单括号,三不对称全偏我方仍输=真差距)

### §2.5 人口同一化判据+对局重定(09-21 18:1x,单发#24)

- **彼判据提取(Analysis.cuh 240-268)**:`range`(链列跨度)≤ DENSE_NUMERIC_BIN_SIZES_GPU[b]
  找最小档 dense_bin_id;est 落 HASH_NUMERIC_BIN_SIZES 梯得 bin_id;**dense_bin_id ≤ bin_id 即送
  dense**(与我方 DTIER_CMP 双键同构——docs/76 §22b 判断证实)
- **人口账目对表(in-2004)**:彼 dense 阶梯块数和 ≈7413+6723+4085≈**18.2K 行**(每块 1 行);
  我方 diter 仅 10.4K 但**档位梯人口 ≈2 万级(v33 时代赢家)** ⇒ **决赛同人口对局重定:
  我方档位梯 dense_struct 家族 vs 彼 denseNumeric 七档阶梯**(不是 cursor vs 彼——cursor 人口
  是宽行子集,彼宽行走 hash)
- 下一步配方:我方 DUMP_ROUTING(in-2004)离线复算彼判据(需彼 DENSE_NUMERIC_BIN_SIZES 值,
  Common.h 提取)→ 行集交集/差集 → 我方按行集强制(临时 env)→ §2.5b 终版逐相位对表

### §2.5b 决赛终版记分(09-21 18:5x,单发#25)

- **档宽血缘实锤**:彼 DENSE_NUMERIC_BIN_SIZES={236,451,906,1816,3636,7277}(SMEM÷并发梯÷9B−4)
  =我方 v33 梯同公式同值——两梯同源
- **行集复算撞墙(诚实)**:朴素判据复算=617K 行(44.6%,含 est 中位 145 的微行),彼实际阶梯
  18.2K——**上游 workflow 优先级(ultra/est 先截流微行)未复算**,精确同人口对拍需补彼
  workflow precedence(下轮或接受代理)
- **决赛记分(同阵 in-2004,代理口径)**:
  | 侧 | 人口 | dense 族相位成本 | 每行成本 |
  |---|---|---|---|
  | 我方(count 9.13+direct 7.72) | ~21K diter 行(est 中位 3369) | **16.8ms(相位和)** | ~800ns/行 |
  | 彼阶梯(ncu 12 发枚举) | 18.2K 行(窄 est/span 档) | **~1-1.3ms(ncu 膨胀)** | ~55-70ns/行(ncu) |
  **=每人口成本差一个数量级**(ncu 膨胀打折后仍 ≥5-8×);且两侧人口 profile 不同
  (彼收最窄行,我收最肥行)——系统 1.56× 的主体在此
- **③刀口判据落定**:差距不在单点 kernel 而在**载体内分工粒度**(彼七档实例化+SubWarp 派发,
  每档窗/块/驻卡咬合)vs 我方两载体粗分工;结构件(融合/流水化)预期收益 < 此分工差距——
  **真正的 ③主刀=载体内多档化/派发化**(形态移植须走铁律:机理吸收)

### §2.6 同行强制协议 v2:三雷与半程(09-21 22:0x,单发#27 补跑)

- **协议(用户终版钦定)**:强制两边 kernel 跑**字面同行集**(挑对家甜点行),最终=逐类载体全面超过;
  手段=嵌入方阵 **P=A[R,:]∪单位对角**(P×P 的 R 行乘积与原阵**完全相同**,恒等行两边同负)
- **本轮三雷(全入档)**:①ocean 要方阵(S=A[R,:] 2000×1.38M 被拒"Incompatible dimensions")
  ②A[R,R] 方阵虽合法但列限制杀死行特征(16.9M nnz→3,164,退化基板;ocean 0.179ms/我 0.364ms
  无意义)③P 构造的 identity 追加与读竞态(grep -vxFf 边读边追,条目丢失)
- **正确配方(下轮执行)**:单遍 awk 按**行号序**内联产出(R 行条目+非 R 行对角同遍发射),
  免排序免追加;头行=首个非注释行(原阵第 2 行是横幅);我方 reader 吃 pattern 非 real
- 选行已备:/tmp/duel_rows.txt(2000 行,彼判据甜点区,est 中位 216/span 中位 257)
- **教训:单发时刻过点即死(20:13 排程被会话忙吞,21:47 用户抓到)**——单发链红线+1:
  排程避开可预期的忙窗;用户消息触达时先 CronList 清死条目

### §2.6b 首战(09-21 23:5x,单发#28 补完;P 建成=同行基板成立)

- **P**:1,382,908 方阵=R 行条目 102,959+恒等对角 1,381,942(双跑同 P×S 公平性构造保证);
  两文件分离+归并的**安全配方**成型(雷区四号入档:awk 生成器对角内循环遇越界行号洪水,
  **260GB 灌爆 /tmp 事故**——生成器必须 clamp r∈[1,n0] 快速失败;TaskStop+rm 止血,磁盘 613G 余)
- **首战记分(卡净 14MiB)**:ocean **0.968ms**(10 iters 全程) vs 我方 **9.245ms**(相位和全程)
  = 9.5×;**但我方人口残缺**:2000 甜点行仅 **539 行**进 diter(EST_MIN=64 下其余 1461 行滞留
  hash,accumulate 3.98ms 是它们的账)——**载体级对拍尚未成立**,9.5× 是系统级混合分
- 下轮:查门(1461 行被哪道拦:est 在 P 上重算后偏低?dup 门?)→补齐人口→真载体对拍;
  另 ocean 0.97ms 含 1.38M 恒等行(其 ultra 路径消化)两边同负✓

### §2.6c 强制刀落地未生效+协议保真缺陷(09-22 00:2x,单发#29)

- **真强制刀已写**(FORCE_DITER_EST env→binning 终判处 bid=BIN_DITER 覆盖全门)但**未生效**:
  est≥8 行 1670(恰=R 行,恒等行 est=1 天然排除✓)而 bkid13 仅 146——**插入点疑落错 kernel**
  (anchor 注释段编译过但非 4139 发射路径;下轮 grep 验证归属)
- **协议 v2 保真缺陷(根本性发现)**:R 行 est 在 P 上**塌缩**(中位 216→31,q25=11)——恒等 B 行
  把乘积稀释(行的工作量特征变了,甜点性消失)——**同行≠同工作量**;
  **协议 v3 修正**:三成分 P=R 行 + **supp(R)∪非R 的 k 行保留原 A[k,:]**(R 行引用的 B 行全真)
  + 其余行对角 → R 行乘积与原阵完全一致且工作量保真
- 数据:P 上 R 行 est 分布 min3/中位31/q75 77/max5748;恒等行 est=1(天然隔离层)

## 状态

- 09-21 09:41 单发#11:§2.1 人口普查入档(四定谳;cursor 单一形态确立)
- 09-21 09:03 单发#10 立 §1+§2;②阶段执行待净窗/共租窗均可(B 级)
- 并行线:RETRY2 刀候默认全绿(commit b5809a4,等用户过目);mempool 决策备忘(8赢2输)
