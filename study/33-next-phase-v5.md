# 计划 v5(10-05 深夜定稿,T10 广域终报 + 口径连环案 + v32 金标准之后)

## 用户钦定的战略方向:武器库宽度 × 调度器智能 = 架构级竞争力

> 多创建 kernel,各自在甜点行上碾压;最终创新 dispatcher 把每行送进最优武器。

## 三步走(可并行部分标注)

### 第一步:修 v33d 路由门(一小时,零新码,最高优先)
- **病灶**:v33d(657e9d7)六处默认翻转(DITER_FLOP_TIER=2048 等)把小阵行从 hash 赶进 dense
- **证据**:去掉 dense 路由 23 阵后 v32 GM 1.490 → HEAD GM 1.381(**我们好 7%**);
  修门预期广域 1.65→~1.35
- **执行**:给 v33d 门加小阵守卫(diter 行数 < 阈值 或 矩阵 nnz < 阈值时保持 hash)
- **守门**:nnz 逐位 + 值门 + 八阵屏零回归

### 第二步:建 kernel₇(双 pass 顺序直写,dup≈1 专用)
- **靶人口**:长 a_len×长 B(23 阵 GM 1.41,占广域损失 44%)
- **机理**:Pass1 全产品只 mark bitmap(1-bit 原子Or,免 F64 原子)→ prefix →
  Pass2 每产品普通写到已知终位(**F64 原子完全消除**,dup>1 的少数行用 atomicAdd 补)
- **创新性**:ocean 也没有此形态(他们也 100% 付原子税)——这是利用"dup≈1 时原子
  完全多余"的人群特征的独有武器
- **预期**:② 人口 23 阵 1.41→<1.0
- **守门**:memcheck + racecheck + nnz + 值门 + 有序 + 同行集 ×5

### 第三步:创新 dispatcher(④ 阶段,与第二步并行收集数据)
- **结构化成本模型**:est/span/dup/a_len 四维 → 各 kernel 预测成本 → argmin
- **训练数据**:DUMP_ROUTING + 82K 行逐行计时(battery① SPEC99 数据)已备
- **路由准确率目标**:>90%(oracle 上限 98.7%)
- **论文**:dispatcher + 多武器生态位 = 架构贡献

### 并行探索(不阻塞主线)
- **TMA 异步载入**:H100 硬件 DMA 引擎,B 行段异步双缓冲(与旧"软件流水"判负机理不同)
- **线程块集群 + 分布式 SMEM**:多 block 共享表/位图(破 228KB SMEM 墙,所有旧尝试都在单 block 内)

## 里程碑
| 点 | 判据 | 预期 |
|---|---|---|
| M1 | v33d 门修复+八阵屏零回归 | 广域 1.65→1.35 |
| M2 | kernel₇ 值门全绿+同行集×5 | ② 人口 1.41→<1.0 |
| M3 | dispatcher 路由准确率>90% | 广域综合<1.0 |
| M4 | 广域电池净赢 | 论文头条 |

## 认证基线(不变)
- 行级公平窗:0.892/57.6%/0.952(hash 薄赢)
- 系统认证:c-64 0.974✅/F2 1.013/msc 1.171/in-2004 1.148/nd24k 1.116/crankseg 1.209/pre2 1.741(09-30 T2)
- T10 广域 52 阵:1.653(10-05,口径已修正)
- 金标准 vs v32:去掉 dense 路由阵后 HEAD/v32=0.927✅(kernel 优化真兑现)

## 核心资产
- 六个已出厂 kernel 的甜点全胜(0.41-0.83)
- 负结果地图五卷(自旋17刀+D1/D1v2/D2a/T3/K1/K3+D3 三型)
- 口径方法论(fair-window/bkid 分类/off-vs-off/金标准 worktree)
- 打桩仪器全套(ROWCYCLES/DENSECYCLES/ESCCYCLES/DSN_RT/HASH_ROWTIME)
- B 流载入:L1 扇区 +0.6%(平)/DRAM −38%(优)
- R1' 发现:pre2 型行 oracle 混合 0.997/a_len≤10→cursor 判据 1.037(备而未发)

---

## [10-06 凌晨] 前提修正(口径第四层之后,study/32 追加节)

1. **第一步"修 v33d 门"的预期重算**:原判断"修门 1.65→1.35"的分子(1.65)与机理(小阵 dense
   固定开销不摊销)双双被冷轮假象污染。修正后基线 = **暖轮净窗 51 阵 GM 1.4167**(锚交叉验证过);
   c-43 三臂暖轮全 ≈6.8ms(默认/关守卫/全hash)→ 路由门在暖轮口径下**不再是主要矛盾**。
   小阵守卫(DITER_SMALL_NNZ=4M)已实施,是否翻默认以 23 阵交替序 A/B 电池为准(进行中);
   wash 则撤,有赢则守门三件套后入默认。
2. **第二步 kernel₇ 靶群名单更新**(暖净窗):bloweybq 3.39/333SP 2.09/pf2177 2.05/af_2 2.04/
   HTC 1.94/para-4 1.90/af_shell4 1.88/hugetrace 1.85/pre2 1.84 —— **全部 hash 主体阵(0 diter 行)**,
   与"dense 主导 23 阵 GM 2.07"的旧叙事相反;dense 路由 23 阵修正后 GM 1.378 与 hash 主体 1.449
   两翼同病。kernel₇(双 pass 顺序直写,dup≈1)的主场判定不变,但靶阵以本名单为准。
3. **M1 里程碑改判**:原 M1(v33d 门修复→1.35)改为"口径第四层修正→1.4167 新基线+守卫 A/B 定谳"。
4. **工程红线追加**:①USE_MEMPOOL=1 大阵 host arena 溢出(c-64 需 1.86GB)中断——大阵电池裸 env;
   ②后台电池运行期间禁止换 ./spgemm_test(WRITE_MTX dump 的轮间磁盘 churn 拉低时钟,rep3 报废教训);
   ③A/B 配对必须交替顺序(G 先跑有时钟爬坡系统性偏置)。
5. **kernel₇ 靶群解剖(10-06 desk,自 T10 原始日志)**:
   | 阵 | 比值 | a_len | avg_B | dup | 判 |
   |---|---|---|---|---|---|
   | bloweybq | 3.39 | 14 | 10037 | **1.0** | **kernel₇ 纯靶**(dup=1 完美) |
   | pre2 | 1.84 | 18 | 634 | 1.3 | kernel₇/R1' 双适用 |
   | 333SP | 2.09 | 12 | 36 | 1.9 | 短×短(另案:est 小批处理带) |
   | pf2177 | 2.05 | 149 | 6632 | 3.2 | 中 dup,bitmap mark 仍省 |
   | HTC | 1.94 | 8 | 1196 | 3.2 | 同上 |
   | af_2/af_shell4 | 2.04/1.88 | 70 | 1216 | **12.9** | 高 dup 非 kernel₇(原子必要) |
   | nd24k | 1.74 | 798 | 165K | **62.8** | 鲸鱼行,另案 |

## [10-06 01:0x] 第一步终局:守卫入默认(case-control 定谳)

- **A/B 电池**(23 阵 ×3 交替序,compare/guard_smallnnz_ab_battery.log):
  同路由对照组 G/N GM **1.0177**(=先手时钟偏置+噪声基线)/异路由处理组 **0.9858** →
  **校正后守卫效应 0.9686(−3.1%)**;总体 raw 1.0105 ÷ 基线 ≈ 0.993
- 逐阵:pli −8.4%/bbmat −9%/opt1 −3~5%(一致);唯一 consistent 回归 c-43 +1.6%
  (=v33d 电池当年 c-64 +1.4% 噪声带同级先例,接受)
- **物理注脚**:est~1100-1300 的边际行回 hash 赢(opt1/pli),est 2-4K 的(Si10H16/c-43)
  在 hash 死区(24-48KB 表撞占用崖)回 hash 反输——静态地板两头不讨好,**终解=dispatcher
  成本模型**(与既有 LADDER/§6 双梯合流),守卫是当下可证的保守修正
- **守门三件套全绿**:nnz G/N 8/8 bitwise 同/memcheck F2+Si10H16 0/值门双 PASS
  (rel ≤2.2e-15)/七锚零回归(全同路由同 nnz,偏差噪声带内)/Xid 0
- **勘误**:前文"net100 上 11118 行"系 count-pass(含 BIN_LARGEST)与 diter-bin 口径混用;
  net100 diter-bin 两臂同为 18802,守卫对其无作用(入对照组)

## [10-06 02:1x] 追加:vs-v32 配对分解 + accumulate 退步真凶定谳(用户三问触发)

### 同阵配对(51 阵,v32 CSV 时代 vs 今日 T10,我方/ocean 分侧)
- 我方今/v32 = **0.995**(CSV 是 v32+mempool,同配置估算 HEAD≈v32×0.97,c-43 同日直比 6.79/6.99 佐证)
- **ocean 今/彼 = 0.963(CSV 时代 ocean 数疑似含冷,小阵尤甚:c-43 彼 2.00→今 1.71,大阵 ±0.5%)**
- 净比值变化 1.032 = "配对退步"主体是 ocean 侧口径差,**但真退步家族是实的**

### 真凶(相位级差分,HEAD vs v32 worktree 同日)
| 家族 | 净窗 H/V | 病灶 | 机制 |
|---|---|---|---|
| af_2/af_shell4 | 1.366/1.348 | accumulate 7.98→12.42(+56%) | est_expand 1.15→**1.80**(+57% 严丝合缝) |
| HTC | 1.332 | accumulate 8.19→18.84(+130%) | 同上,margin 白付 |
| FEM | 1.210 | accumulate +62% | 同上 |
| c-64/bloweybl | 1.042/1.049 | dense_direct+count +1.5ms | 另案(第二优先) |

- 根子:hash.cu:4362 `g_est_cont → est_expand=1.8`(study/13,"连续 est 余量由 expand 扛;F2 ovf 1319→0")
  —— v32 是 EXPAND 扫描裁决 1.15-1.40("紧 est 完胜"电池在注释里)。**又一例只在锚族验证就翻默认**
- A/B 定谳(HASH_EXPAND env):HTC 1.15 → accumulate **−46%**/TOTAL −15%(ovf 6↔6,margin 免疫结构性溢出=死重);
  af_2 1.15 → 更差(ovf 0→6:连续 est 低估尾部需要 margin;病根=估计质量,v32 离散估计 1.15 即够);
  FEM 微差同 af_2 型;REXP=2 不覆盖(E≤4096 恒等)
- **修法方向(下一场主刀候选):margin 按实测溢出行为自适应**(round1 1.8 → ovf_cnt 结构性即收紧),
  af_2 型需估计器工作(低估尾部补偿/离散档对齐)
### Q4(CPU)定谳:相位↔净窗 gap 中位 9.7%(hugetrace 39%/333SP 36%)= 计时窗内
### launch 间隙+host 逻辑+未打桩传输——我们自己的 17 相位管线串行,ocean 3 相位结构免疫;
### 共租 CPU 安静。另:mempool v32 时代能跑 c-64 今夜 host arena 溢出(配置回归待查)

### [10-06 02:3x 对账(用户指正)] est 系统的既定决策未被推翻

- **连续化 est + expand 扛余量(study/13)是定谳决策,继续有效**:F2 ovf 1319→0、锚族大赢
  (F2 −28%/pre2 −39%/3Dspectral −42%)的机制就是它。本节上面的发现是**边界补充**:
  1.8 余量在【结构性溢出阵】(HTC 型,margin 免疫)上是死重——这是它从未被测过的
  人群,不是它的错。
- **af_2/af_shell4 的 +37% 未定谳**(收紧 expand 反差 → 非 HTC 同机制;嫌疑=连续 est
  在该族的估计质量/行档分配,待专案)。
- **流程修正(用户 02:2x 批评成立)**:①宣布"X 是问题"前必须先对账 X 的既定决策出处
  和原验证人群;②结论台账给每条定谳加【验证人群/未验证人群】栏——新数据落在未验证
  人群 = 补边界,不 = 推翻;③跨版本比较一律同日 worktree 配对(CSV 时代 ocean 含冷
  偏差 0.963 已实测)。

## [10-06 03:0x] 守门纪律升级(用户钦定):八阵屏 → 常设广域屏

> 今晚连环案的公共根因 = 验证电池太窄(每刀只在锚族验证就翻默认)。用户裁定:换屏。

1. **每刀入默认前**:必过 **51 阵 T10 广域屏**(暖轮净窗解析,`scripts/t10_warm_reparse.py`
   口径),判据 = 零未解释回归 >2% + 守门三件套(nnz/值门/memcheck)照旧
2. **版本级发布**(v 字头翻默认包):**337 阵全刷**(data/ocean/square 全集,~1.5h/臂)
3. **跨版本结论**:一律同日 worktree 配对;跨月 CSV/ocean 数据只做筛查不做定谳
   (CSV 时代 ocean 含冷偏差 0.963 已实测)
4. **结论台账**:每条定谳标注【验证人群/未验证人群】(结论稳定纪律,见 memory)
5. **expand 类逐阵参数决策**:归 dispatcher 成本模型里程碑(M3)维度,禁止单人手门
   (ADAPT_EXPAND 08-30 判负前科;"缝缝补补"否决,用户 10-06)
6. A/B 方法:交替配对序 + 同路由对照组 case-control(compare/guard_smallnnz_ab_battery.log 范式)

## [10-06 05:0x] 夜班战报:基线落定 + kernel₇ 出鞘(独立台全绿)

### ① 修正口径认证基线(HEAD+守卫 414161a)
**51 阵 GM = 1.3895**(暖净窗 ×3;ocean 复用 10-05 日志)——守卫在记分板上再收 −1.9%
(1.4167→1.3895)。germany_osm 仍无效(C_nnz=-1 溢出回退 merge,待 merge 路径测量)。
新最差名单头部:bloweybq 3.40/333SP 2.08/pf2177 2.05/af_2 2.03/c-71 1.98/HTC 1.94。

### ② kernel₇(双遍位图 + rank 直写,dup≈1 人口)独立台首战
- **靶修正**:bloweybq = 全 dense 路由阵(10001/10001 diter 行;校准表旧解析错误已更正),
  病根 = dense_direct 主导的 12.9ms 净窗;行形态 a_len≈14 × B≈717,dup 1.0001(每行 ~1 碰撞)
- **形态**:Pass1 atomicOr 位图(1-bit,免 F64)→ popcount = 精确 nnz → cum 秩表 →
  Pass2 认领(atomicOr 32-bit)→ 胜者**纯普通写**到 rank 位,败者入 SMEM fixup 表 →
  __syncthreads → 统一 atomicAdd(**屏障定序**:块内全局写可见 ⇒ 败者加必在胜者写后)
- **关键设计三则**:①2D(k,p) 切分——a_len=14 时 kk-stride 只用 14/128 线程 = v1 878ms
  的元凶,切分后 B 流合并读,296× 提速;②atomicExch 不解决次序(两原子次序仍任意,
  值门 FAIL 实证)→ fixup 协议;③**count 证书**:flop−distinct ≤ FIX_MAX(512)的行才进
  快道——溢出不可能,容量有数学背书
- **战果(bloweybq)**:count 0.49ms + numeric 2.73ms = **3.2ms vs 管线 12.91ms ≈ 4×**;
  顺带消灭估计器链(mh_merge 1.8ms)与 compact+sort(2.0ms)——k7 的 count 就是精确计数
- **守门全绿**:memcheck 0 / racecheck 0 hazards / nnz 10000/10000 行精确 /
  值门 列错0-值错0-无序0 PASS / Xid 全程 0
- **下一步(集成,按新纪律)**:BIN_K7 路由谓词(a_len≤32 ∧ span≤32768 ∧ dup 估计≤1.67,
  est/flop 从估计器来)+ 复用 dense 管线的 count→scan→write 槽位 + 51 阵广域屏零回归
  + 守门三件套 → 才谈翻默认

### ⚠ [10-06 04:1x] 夜班尾事件:GPU 驱动拥堵(未升级 wedge)
- pre2 k7 探针二跑被 30min 截杀后,nvidia-smi 连续 >30-60s 超时;6 进程 D 态
  (nvidia-smi×4+k7_bench×2,kill 不掉)+ 1 僵尸;**Xid/NVRM 零记录(journalctl 实读)**
  = 驱动级拥堵非硬件故障,可能自愈。已停一切 GPU 交互等晨检(一次 nvidia-smi 即知;
  仍堵=协调重启老剧本)
- 教训:①dmesg 空=dmesg_restrict,早期 Xid 哨兵可能假阴性→哨兵改 journalctl -k;
  ②后台 GPU 任务必须自带 timeout 自杀,勿依赖 harness 截杀(SIGKILL 驻核进程=拥堵源)
- **夜班交付不受影响**(全部事发前 commit):基线 1.3895 + kernel₇ bloweybq 4× 全绿 +
  纪律升级 4 commit;pre2 k7 定位未完(count 3.8ns/行正常,numeric 未打完)

## [10-06 下午] kernel₇ 集成设计(Phase 1,desk 定稿待实施)

### 路由(compute_bucket_kernel 新增 BIN_K7,优先级在 diter 检查之前)
谓词(核内全有现成量):
- `K7` env(默认 0,51 阵屏全绿后再议翻默认)
- a_len ∈ [1,32](2D 切分假设)
- span_len[i] ≤ 32768(位图 ≤4KB,SMEM 经济学)
- dup 预算:flop[i]−est[i] ≤ 512 ∧ flop[i]/est[i] ≤ 1.67(估计器侧,近似——精确证书在核内)
- est ≥ 256(不偷 batch_s/ultra 人口的行)

### 管线槽位(复用 dense 方案5 机器,k7 = 换 count/numeric 两核)
k7_count(标记+popcount)→ 现有 cnnz_scan/scan 机器出 offsets → k7_numeric 直写终态 C
(有序+精确 nnz,该 bin 的行**整条免 compact+sort、免估计器余量**)。行集走 BIN_K7 的
scatter/计数,与 BIN_DITER 同构。

### 溢出安全(核内证书 + 旗标重跑,废 racy 兜底)
- numeric 里 fcnt 达 FIX_MAX → 该 block 置全局 per-row 状态旗标,本行不再写
- 主核后补一小发清理 launch:只重跑旗标行,走 exch 慢道(有界,罕见)
- 路由估计器侧 dup 是近似的——溢出可能但安全,快道纯净性保持

### 守门(按新纪律)
1. 三件套 ×N:bloweybq(dup≈1 纯靶)+ pre2(中 dup,fixup 尾部人口——wedge#8 教训:
   **必须盖 fixup 尾**)+ 合成高碰撞小阵(slot 全区间)
2. 51 阵广域屏零未解释回归 >2%(盯 c-64/dense 人群不变)
3. 记分板目标:bloweybq 3.40 → ~0.9;基线 1.3895 → ~1.35

### wedge#8 根因与修复(同日定谳)
SMEM 布局重叠(fvl 1024..2047 ∩ mark≥1538;fcnt∈fvl[256])——pre2 中 dup 人口 slot≥257
踩位图 → rank 越界 → illegal instruction → context 损坏 → SIGKILL → wedge。
bloweybq 碰撞≈1 只用 slot0 故全绿(sanitizer 对界内语义重叠盲)。修复=重排
(fvl→0..1023/frk→1024..1535/fcnt→1536);教训三条入 [[gpu-safety-discipline]]。
