# study/57:B 线战役档(kernel 收官;loop 驱动,10-24 开档)

> 任务书:study/55 §B线 + study/56 交接;loop prompt 首行=让位于用户指示。
> 数据:compare/batch437/reanchor_bins.csv(scripts/reanchor_bins_beats.py 重生成)。
> 判死名单(不碰):溢出簿记移除(熔断器)/FLAT 扩 hash 行/bloweybq dense 窗 kernel/
> hybrid 值外置/带状 CAS 墙/ESC64 复活。
> 纪律:取证先行不改码/nnz 门先行/同会话 A/B 唯一刀级/批跑禁重建/口径只用 prof/stats.json。

## 一、重锚定谳(tick1,纯磁盘):微输带的真结构

旧账"12→5 拍"是 T4a 前的 StocF bin3 算术。重锚(428 阵 prof×ocn_stats×confirm14,
七刀折算后 gap_now)后**微输带不是弥散的,是两块结构人口**:

| 带 | 微输带阵数 | gap_now Σ | 拍数比(我/ocean,ns/乘积/SM) | 代表阵 |
|---|---|---|---|---|
| **b23**(bin2-3) | 31 | 158ms | **1.7-2.3×**(T4a 后仍在) | CurlCurl_4 2.26 / rgg 残 1.88 / af_shell10 1.74 / nlpkkt80 1.73 / af_1_k101 |
| **b45**(bin4-5) | 39 | 145ms | **1.1-1.35×** | nlpkkt120 1.52 / Bump_2911 1.30 / Geo_1438 1.33 / Serena 1.27 / Hook 1.29 / cage15 1.24 |
| b6p / mixed | ~22 | ~0 | — | 拍数已平(TSOPF/crankseg/Zd_Jac 亚 ms) |

绝对拍数(b45 典型):我 1.9-2.0ns(≈3.4 拍@1.76GHz)vs ocean 1.3-1.5ns——**b45 的
单位效率离 ocean 只差 25-35%,不是 StocF 型 2.4×**;b23 的 1.9-2.3× 才是拍数差主带。

**病理名单(不入 B 拍数账,单独立案)**:
| 阵 | 症状 | 定性 |
|---|---|---|
| c-big | gap 83.8ms,拍数比 64×,hf=0.46 | 非拍数:疑似 dense/diter 路由差(ocean 捕 7.7% dense 行,我方 0.03%;study/53 已注)→ 归 T4c 追查/A 线 |
| Ga3As3H12 | 21.1 vs 2.9ms(7.3×),b6p | ocean 走完全不同通路(wf=1);算法选择差非拍数 |
| webbase-1M | 5.7×,hf=0.134 | 大部分行非梯;另查 |
| bloweybq | ocn_core=0.04ms | dense 窗 kernel=判死名单;k7 路由归 A 线 |

## 二、刀序(预注册)

- **T-B1(tick2,磁盘)**:SASS 静态对账——bin2/3/4/5 热实例 vs ocean hashNumeric 的
  指令/分支质量;死分支清单(WM/D5H/span/extract 模板质量在热实例的展开);
  产出=极简实例的剥什么清单(语义零变约束下)。
- **T-B2(tick3,GPU 小窗)**:HASH_ROWTIME×ROWCYCLES 逐行对拍 6 代表阵:
  CurlCurl_4(纯 b23,拍数比 2.26)/ rgg(T4a 后 b23 残 1.88)/ af_shell10(b23/45 混)/
  nlpkkt120(纯 b4,1.52)/ Bump_2911 或 Serena(b45,1.3)/ cage15(b45,1.24)+ 对照
  ldoor(近平)。产出=逐 bin 拍数差定谳(探测环/乘积枚举/extract 残余三选一归因)。
- **T-B3**:极简实例刀(bins 2-6 热实例专用化)。
  **成败线:bin2-6 人群 accumulate ≥20% ∧ 值门三件套(nnz 先行)∧ 控制阵无害
  (bin4-6 已赢阵/dense 阵)∧ 同会话 A/B → confirm 批盖分**。
- T-B4(副攻):T3 二轮融合单趟(delaunay/ESC 带,设计在 study/53;c-big 若定性为
  diter 路由差则并入 A 线成本模型)。

## 三、方法论注

- prof(batch437,10-15 采集)= 七刀前 accumulate;结构量(bins/flop)刀不变;
  b23/b45 拍数比对现役 hash_spa 核有效(T4a 动 extract、T4c 动 diter 路由、
  T4e 动 LLB 分派,均不动梯核 numeric 内环)。
- 微输带计数差异:重锚得 100 阵(study/56 记 111)=重锚缺 prof 侧车阵+窗口边界;无碍结构结论。
- flop 口径不一致 11 阵(Bump/Flan/Cube_Coup/audikw/bone010 等,upper_tri 口径癖)——
  拍数比仅供方向,gap(ms)不受影响。

## 四、tick2 定谳[beat 列已于 tick4 作废,见 §八](10-24,SASS 静态对账 + 发射点考古)

**核心发现:生产数据里已经跑着一个受控 A/B——b45 走 LHT 常量折叠实例(HASH_LHT
默认 1,09-24 翻默认,服务 bins 4-9;夜班#4 扩到 bin4-6),b23 永远走运行时 ht 形态
(发射点 spgemm_kernel_hash.cu:6405 `bi >= 4` 门)**。b23 拍数比 1.9-2.3× vs b45
1.1-1.35× = 同一 kernel 家族、同批数据上"折叠 vs 不折叠"的现成对照。

**SASS 账单**(scripts/sass_account.sh,/tmp/{ours,ocn}.sass,修饰符剥离后直方图):

| 实例 | total | BRA | BSSY+BSYNC | BREAK | ATOM | LDS | SHFL |
|---|---|---|---|---|---|---|---|
| 我 运行时形态(bins0-3) | 3640 | 295 | 166 | 25 | 67 | 252 | 91 |
| 我 LHT bin4/5 | 4096 | 402 | 253 | **192** | 118 | 252 | 60 |
| 我 LHT bin6/7 | 3352 | 352 | 234 | 192 | 108 | — | 70 |
| ocean hashNumeric<6,331> | **768** | **52** | 28 | **1** | 9 | 34 | **0** |
| ocean <10,16253,hybrid> | 960 | 59 | 36 | 2 | 10 | — | 6 |

全类别均匀 4-5×(IMAD 651 vs 153 / ISETP 458 vs 120 / LDS 252 vs 34 / SHFL 91 vs 0)
= 全家桶模板税,非单点毒。

**三条推论**:
1. **静态≠动态再证**:LHT 实例静态比运行时形态更大(4096 vs 3640)却动态更快
   (b45 拍数比更低)——折叠的收益在热环常量(ht/mask/G 梯免每乘积重算),不在码量。
   刀 B(剥死分支)的预期收益必须来自动态仪器,静态账只能给嫌疑清单。
2. **刀 A 定义(候选,最便宜)**:SPA_LAUNCH_LHT 门 `bi >= 4` 扩到 `bi >= 2`
   (bin2→LOG_HT 7 / bin3→LOG_HT 8),新 env HASH_LHT23 默认关(=现行为逐位);
   几何不动(g_blk_lad:bins0-3=64T/块、4-7=256T);门条件同源(probe_rf ∧
   !phist ∧ !wm)。期望逐位一致(纯常量折叠),09-24 翻默认先例=八阵屏+值门逐位。
3. ocean 的 BREAK=1 vs 我 192(LHT)——探测环 break 机器是刀 B 的主嫌疑,
   但 T4b 已证溢出簿记不可剥;须先逐行对拍定动态占比再动。

**刀序修订**:tick3 = 刀 A 探针(bins2-3 LHT 扩展,同会话 A/B:CurlCurl_4/
af_shell10/rgg/StocF/af_k101 族 + 控制阵 ldoor/cage15/c-64/F2;nnz 门先行)。
原 T-B2 逐行对拍(HASH_ROWTIME×ROWCYCLES)后移到刀 A 裁定后——若刀 A 已收
b23 大半,残余才值得逐行仪器。

**方法论入账**:mawk 1.3.4 错编译陷阱——动作块 ≥2 语句 + `next` 静默丢计数
(二分定位:v4/v7 单语句通,v5/v6/v8 多语句全 0);SASS 解析一律两段式单语句规则。

## 六、tick3 定谳(10-24,刀 A 探针:HASH_LHT23 bins2-3 常量折叠扩展)

**实现**:发射点 `bi >= 4` 门扩 `bi >= (g_lht23 ? 2 : 4)`,bin2→LHT(7)/bin3→LHT(8),
env HASH_LHT23 默认 0(=现行为);几何不动(g_blk_lad 64T 不变),期望逐位。
构建 ✓ LOG_HT=7/8 实例入二进制。

**同会话交错 A/B(×2 轮,靶阵,bare env)**:

| 阵 | A 中位(两轮) | B 中位(两轮) | Δ | nnz |
|---|---|---|---|---|
| CurlCurl_4 | 8.460/8.463 | 8.298/8.282 | **−2.0%** | ✓ |
| rgg_n_2_22_s0 | 20.657/20.692 | 19.784/19.752 | **−4.4%** | ✓ |
| af_shell10 | 21.875/21.913 | 21.596/21.629 | **−1.3%** | ✓ |
| StocF-1465 | 9.459/9.468 | 9.101/9.031 | **−4.2%** | ✓ |
| af_1_k101 | 7.963/7.968 | 7.651/7.628 | **−4.1%** | ✓ |

**定谳:一致小额正(−1.3~−4.4%),五阵全胜、双轮复现、nnz 逐位;但远低于 ≥20%
预注册线(差 5-15×)。折叠不是 b23 拍数差的主要质量(回收 ≤1/5)。**

**处置**:
1. 代码+env 保留(HASH_LHT23 默认关);单刀不 worth confirm(union 影响估 ~0.1%),
   **攒刀**:与下一把 ≥20% 级刀捆绑进同一次 confirm(降 churn 纪律)。
2. **火力转向乘积枚举形态**(study/54 §十四原始指向):b23 剩余拍数差的主嫌疑。
   下 tick = 枚举形态取证设计(读码:G-group 步进/边界 + LLB 的 G 选择;
   仪器:HASH_ROWTIME 逐行周期 × 行 flop 分布 → 失衡税定量)。
3. 方法论:直跑缺电池 env(MP_HOST_MB/USE_MEMPOOL/METHOD)会挂 cuSPARSE 基线段
   (error 11)——控制阵必须走 ab_env.py 同款 env(已修 knifeA 脚本)。

**控制阵收尾(电池 env,A/B×2)**:ldoor 平(−0.1%)/ cage15 **−1.2%**(87 万 bin2-3
行受益,无害)/ c-64 平 / F2 平;nnz 全同 ✓。**StocF 电池 env 交叉**:A=7.108(与
study/54 §十的 7.22 复现一致——bare env 的 9.46 是环境抬高的绝对值,相对差仍真)vs
B=6.851(−3.6%)。**刀 A 终账:9 阵全绿 −1.2~−4.4%,零伤害,攒刀待捆绑。**

## 八、tick4 定谳(10-24 晚):失衡税证伪 + tick1 陈账反转(重要修正)

**① 静态失衡税证伪**(scripts/enum_imbalance_tax.py,组功=Σ⌈len_k/G⌉,税=max/mean):

| 阵 | 带 | 税 mean | p90 | beat(旧账) |
|---|---|---|---|---|
| CurlCurl_4 | b23 | 1.44 | 1.46 | 2.26 |
| rgg | b23/45 | 1.42/1.65 | 1.74/2.07 | 1.88 |
| af_1_k101 / af_shell10 | b45 | 1.37 | 1.37 | 1.7/1.74 |
| cage15 | b23/45 | 1.36/1.51 | 1.60/1.88 | 1.24 |
| nlpkkt120 | b45 | **1.17** | 1.14 | **1.52** |

**反转对:nlpkkt120 税最低(1.17)却 beat 高(1.52);cage15 税高(1.51)却 beat 低
(1.24)——静态组失衡税不是拍数差主驱动**,枚举失衡假设(静态形)证伪。
(SM 级驻留块重叠也会吸收行内失衡,静态税本就是上界;但方向性反转已足够否决主因。)

**② tick1 陈账反转(重要)**:reanchor 的 beat_ns/beat_ratio 列用的是 batch437
**七刀前** accumulate——对 T4a 覆盖人群严重虚高。今日实测:
- StocF:7.108ms(电池 env,与 study/54 §十 7.22 复现)= **1.16×**(非 2.88×)
- rgg:20.6ms → ~1.17×(非 1.88×)
- CurlCurl_4:8.46(bare)×0.751(StocF 校准的 env 因子)≈6.4ms vs ocean 7.91 =
  **~0.8×,accumulate 可能在赢面**(待电池 env 直测确认)

**结论:b23"拍数带 1.9-2.3×"大部分是 T4a 前的陈账;b23 accumulate 已接近/达到平。**
111 带的真实残余在哪(相位?未覆盖阵?)必须用**今日数据**重定位——tick1 表的
beat 列作废,gap_now 列(刀折算)仍有效。

**③ tick5(进行中)**:现役配置(A 臂,电池 env)12 代表阵全相位分解 × ocean
ocn_stats 逐相位对账 → 今日钱图。覆盖 b23(CurlCurl/rgg/StocF)+ b45(nlpkkt120/
Bump/Geo/Serena/Hook)+ 未覆盖阵(af_1_k101/af_shell10)+ cage15/nlpkkt80。

- [x] tick1(10-24):重锚完成,微输带结构定谳(b23 拍数带 + b45 人口带 + 病理名单);
  脚本 scripts/reanchor_bins_beats.py;数据 compare/batch437/reanchor_bins.csv。
- [x] tick2(10-24):SASS 静态对账 + 发射点考古;受控 A/B 发现(生产内生的
  LHT vs runtime 对照);刀 A 定义(bins2-3 LHT 扩展);scripts/sass_account.sh。
- [ ] tick3:刀 A 探针(GPU,同会话 A/B)
- [ ] T-B3:极简实例刀(动态仪器定嫌疑后)

## 十、tick5 定谳(10-24 晚):今日钱图——B 线主攻方向重定位

**方法**:现役配置(A 臂,电池 env)12 代表阵全相位 × ocean ocn_stats 对账
(/tmp/phase_now/,scripts/phase_gap_today.py)。

**结论:accumulate 已不是主战场**(1.07-1.29×,nlpkkt80 1.55 例外)。钱在三池:

| 池 | 人口 | 规模 | 机制 | 修法方向 |
|---|---|---|---|---|
| **compact+sort** | wf0 工作流 hash 阵(cage15 20.6/nlpkkt120 10.0/Bump 11.4/Cube_Coup 11.6/wb-edu 32) | 12 阵 Σ51ms | 我方独立全幅 compact 趟 vs ocean 排序融合在 numeric 内(epilogue≈0.03ms) | sortOutputFused 对位/方案5 直写扩 hash 行(T4d 族谨慎,一次证伪在档) |
| **est 链(mh_merge)** | 177 阵>0.3ms,Σ273ms;**微输带 72 阵 Σ79ms** | S_Berkeley 19.7/cage15 9.1/c-73×2 6.5/rajat 族 6×4.7 | **带宽税 = A_nnz×K×4B 重读**(cage15 50M×512B=25.6GB÷2.8TB/s=9.1ms ✓算术验证);est 链七刀未碰 | per-row K 降档/est=flop 直填扩界(avg_gate 机制已有)/MHSAMP 采样已有 |
| **acc 残余(b45)** | Bump/Geo/Serena/Hook/af 族 1.2-1.3× | 12 阵 Σ78ms 但 ≤20% 可收 | 原 B 线主攻,现诚实有界 | 极简实例(原 T-B3) |

**逐阵翻转算术**(GM 视角):nlpkkt120(1.23)=est 5.5+cmp 10.0+accΔ 8.8 各占
13/16/21%——**任两池修掉即翻平**;cage15(1.34)需 compact(23%)+est(12%);
Bump(1.20)需 compact 13%+acc 17%。

**刀序修订(tick6 起)**:B 线"指令流瘦身"主攻降级为三池之一;推荐下一刀
=**mh_merge 瘦身**(人口最广 72 带阵/机制已破案/旋钮已有/处女地),预注册成败线
(见 tick6);compact 融合刀随后(单阵收益最大但 T4d 前科);acc 极简实例最后。

**方法论**:ocean wf0 的 epilogue≈0 是口径陷阱——其排序融合在 numeric 内,
单比 acc 或单比 compact 都会错;必须 (acc+compact+ddirect) vs (numcore+epi) 整池比。

## 战报(滚动)
- [x] tick1:重锚(b23/b45/病理;beat 列后被 tick4 作废)
- [x] tick2:SASS 对账+受控 A/B 发现;刀 A 定义
- [x] tick3:刀 A 探针(9 阵 −1.2~−4.4%,攒刀)
- [x] tick4:失衡税证伪 + tick1 陈账反转
- [x] tick5:今日钱图三池定谳(本节)
- [ ] tick6:mh_merge 瘦身刀(预注册待写)

## 十一、tick6(10-24 晚):刀序钦定 + T-C1 compact 拷贝提速刀

**用户钦定刀序(10-24)**:① compact 融合/提速 → ② b45 acc 极简实例 → ③ mh_merge(最后)。
(质疑对话入档:mh_merge 被 challenge 的三条——est=净赢相位(T0.1 −109ms)/单独翻不动阵
(3-10%)/语义爆炸半径全成立;compact 翻阵最强(13-23% 端到端)且输出终态不变风险最低。)

**T-C1 取证链(ncu 不可用,无 ncu 定谳协议 = nsys 分解+算术+受控实验)**:
1. nsys 单面 kernel 表(cage15,自用合法):copy_warp 17.1ms/迭代 = compact 相位 83%
   (csort 仅 3.3;bin4 acc 31.2/bin5 acc 44.4/mh_merge 9.0 旁证);
2. 算术:bins≤5 条目 ~732M × 28B = 20.5GB ÷ 17.1ms = **1.20TB/s = 峰值 43%**——
   非带宽墙,是每 lane 在飞装载不足(每迭代仅 2 独立载);
3. 内核审计:合并访存良好,warp-per-row 形态正确(T4d 已证 flat 对长行倒贴);
4. ncu 2026.1+570 驱动不兼容复现(计数器全 nan+LaunchFailed)——不连发,协议切换。

**刀 C1**:hash_compact_copy_warp_v2_kernel——4 路独立装载展开(每 lane 在飞 8 载),
尾段<128 回标量;5 个发射点统一 launch_copy_warp() helper;env COMPACT_COPY_V2
默认 0=原核逐位。语义零变(同序同值构造)。
**预注册成败线**:cage15/nlpkkt120 compact+sort ≥40% 改善 ∧ C_nnz 逐位 ∧ 行内有序
检查 ∧ 控制阵(StocF/rgg/c-64/F2)无害 ∧ TOTAL 兑现。
**若 <40%**:裁定期权 = 承认 ~9-11ms 结构地板(28B/条目 × C_nnz),compact 池封顶,
按钦定序转 ② b45 acc。

**电池**:bzygud823(靶 cage15/nlpkkt120 交错×2 + 控制四阵)。

## 十二、刀 C1 终审(10-24 晚):证伪归档,compact 池关账

**结果**(bzygud823,交错×2,nnz 全同):
- cage15:20.62→19.36(−6.1%)——远低于 ≥40% 预注册线
- nlpkkt120:10.19→10.63(**+4.3% 反向**;寄存器压力吃占用症状)
- 控制(StocF/rgg/c-64/F2)全平

**定谳:ILP 展开非解——copy_warp 1.20TB/s(43% 峰值)是管线结构属性**:
每 lane 8 载只换 −6%,排除延迟遮蔽假设。与 T4b"59% 谓词=保护"同型教训:
**微观引擎读数(合并访存良好)≠ 可回收空间;瓶颈归因必须受控实验盖章。**

**compact 池关账(两个负结果 + 一个算术死刑)**:
1. C1 ILP 展开:证伪(本节);
2. count-pass 融合(方案5 扩 hash 行):算术死刑——count-only 趟 ≈ 0.5-0.6× accumulate
   (cage15 +50ms vs 省 20.7;nlpkkt120 +30 vs 省 10),无精确预计数则 compact 趟不可消除
   (ocean 靠 estik 精确 symbolic 免票,我方 est 是估计 = 管线原罪,归 A 线成本模型领地);
3. 12B/entry tmp 布局(col/val 分离数组):−14% 流量但动 extract/retry/heavy/k7 全链 = 
   高爆炸半径小收益,记"考虑过并否决"。
**compact 结构地板 = 28B/entry × C_nnz**(cage15 ~9-11ms)。C1 代码保留 env 门默认关。
刀序 ① 关账 → 转 ② b45 acc 极简实例(原 T-B3)。

## 十三、tick7 定谳(10-24 夜):b45 病灶反转——不是指令流,是并发几何

**逐行对拍(Bump_2911,HASH_ROWTIME×ROWCYCLES,study/54 §七协议)**:

| bin | 行数 | Σrt(我) | Σcyc(ocean) | 逐行比 | 我extract占 |
|---|---|---|---|---|---|
| 4 | 856k | 35.1G | 74.0G | **0.47** | 20% |
| 5 | 1994k | 88.2G | 156.7G | **0.56** | 23% |

**我方逐行周期是 ocean 的一半**(0.47-0.56×),但系统级 bin4+5 acc 75.6ms vs
ocean <7,673> 50.0ms = 1.51×。分解恒等式:**1.51× ≈ 0.53×(逐行,我赢)×
2.85×(并发逆差)**。L0"逐行≠系统"第三次应验,且首次逐行反转为赢面。

**→ ②"极简实例剥死分支"作为 b45 主刀被本数据证伪**(指令流已是赢面);
b45 的正解 = 行并发几何。机制三层:
1. 块宽:我 256T×1行/块 = 8 行/SM(线程帽);ocean <7,673> = 128T = 16+/SM;
2. **smem 帽(新发现,bin5 真枷锁)**:我 bin4 表 18KB(512槽+CS2 hist 12KB)
   /bin5 24KB(1024槽)——128T 下 bin4 可 12 块/SM,bin5 只 9 块(228KB opt-in 帽),
   块宽刀对 bin5 几乎无效;
3. **ocean 的 673 素数贴身表 = 8KB/行**——表按行贴合(est×~1.3)vs 我按 bin 档
   (est 512-1024 全给 1024 槽)——表 smem 差 3×,这是他们 16 行/SM 的物理来源。

**刀 C2(探针跑中)**:HASH_BLK_LADDER=4:128,5:128(env 机制已有,零代码);
预注册:Bump bin4-5 acc ≥20% ∧ nnz 逐位。预期:bin4 +50% 在飞(8→12)应兑现,
bin5 smem 墙制约。若 bin4 兑现 → 刀 C3 = 素数/贴身表 tiering 下扩(HT_P 机制
已在 bins7-9,形态复刻到 4-5)——比 C2 大一级的刀。

**方法论**:per-row clock64 计的是墙钟含 stall——逐行赢+系统输 = 并发差的
教科书信号;块宽梯 env 覆写机制(HASH_BLK_LADDER)是免重编探针通道,先探后刀。

## 十四、刀 C2 终审(10-24 夜):证伪——块宽不是杠杆,单行块已局部最优

**结果**(Bump 三臂交错×2,nnz 全同):A 256T=78.6ms / B 128T=92.4(+18%) /
C 64T=140.2(+78%)——单调恶化。

**机制**:块宽减半 = 驻留块×2 但每行组数减半(G=16 不变,num_groups 256/16→128/16)
→ 行延迟×2 > 驻留×2 的收益;64T 更叠加 extract 块协作迭代变长(512槽/64T=8趟 vs 2)。
**→ 我方 hash_spa 的单行单块形态在"组并行 vs 驻留"权衡上已局部最优;
ocean 的并发优势形态 = 多行封包/块(4行×16T 子组,<6,331>=64T×4行 = 128行/SM)。**

## 十五、刀 C3 设计地图(待预注册;基于 C2 教训)

**形态**:hash_spa_batched_kernel 扩展服务 bins 4-5——块内 warp 私有表,多行/块:
- bin4(ht 512,表 6KB):8 warp × 6KB = 48KB/块 → smem 帽 4 块/SM × 8 行 = **32 行/SM(现 8,×4)**
- bin5(ht 1024,表 12KB):96KB/块 → 2 块 × 8 = **16 行/SM(现 9,+78%)**
**预期上界**:并发逆差 2.85× 的可回收部分;bin4 人口(Bump 856k/Hook/Geo/Serena 族)优先。
**已有资产**:hash_spa_batched_kernel<512,4>(BATCH2,est≤256∧kc≤32)——表 512 槽的
warp-per-row 批处理形态已在;扩展 = 放开 kc 约束 + bin4 路由 + extract 兼容(CS2/legacy)。
**风险登记**:①BATCH_S 教训(G 并行度折损,BATCH2_WPB=4 即为此调)——warp 内 G≤32
对 bin4 大行(a_len 大)是否够;②与 LHT 常量折叠的实例化组合爆炸;③溢出/retry 机具
的批处理兼容(ovf 行回退单行核=天然逃生);④CS2 hist 的 smem 预算重排。
**预注册(下 tick 定稿)**:bin4 人口 acc ≥20% ∧ 值门三件套 ∧ bin5/控制无害。

## 十六、B 线首日总账(10-24;10 commits)

**定谳层(今日新知识)**:
1. 微输带三池结构(compact/est/acc)+ tick1 beat 列陈账修正
2. b23 accumulate 已近平(T4a 早已收掉;旧 1.9-2.3× 是七刀前算术)
3. b45 病灶大反转:逐行 0.53× 赢面/系统 1.51× 输 = 并发逆差 2.85×(L0 三应验)
4. ocean 并发形态 = 多行封包(4行×16T)+ 素数贴身表(8KB vs 我 bin5 24KB 含 hist)
5. 我方单行单块已局部最优(C2 单调恶化证伪块宽杠杆)

**刀账**:刀 A(LHT23 −1.2~−4.4% 九阵全绿,攒刀待捆绑)/ 刀 C1(ILP)证伪 /
刀 C2(块宽)证伪——两把负结果入论文设计空间节。
**下一步**:C3 = hash_spa_batched 扩展 bins4-5 多行封包(设计地图 §十五,
资产 <512,4> 已在,预注册待定稿);刀 A 捆绑 confirm 待 C3 落地后一起。
