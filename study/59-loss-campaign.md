# study/59 — A+ 线战役书:输面攻坚 + b45 收尾(10-28 定稿)

> 起点:confirm15 终态 **0.8311 / ocean337 主场 0.9801 / 赢面 69.3%**(§六十七)。
> 杠杆排序:输面 131 阵(>1.10 者 103)≫ 旗舰尾(Bump 1.056/cage15 1.072)> 微账。
> 纪律:全套继承(GPU 保护/同会话 A/B/d2h 交错重采/三件套/预注册/三检构建门/清场按 PID)。

## P0|输面取证分桶(先行,半天)
131 输阵逐个采 **路由([adapt] 裁定行)× bins 人口 × 相位账** → 分桶表:
- 已知桶:T5 预门误 route(bloweybq 3.07)/T6 diter 窗口族(case9 1.94/atmosmodj 1.83)/
  旧 known-cost(Stanford_Berkeley 1.86);
- 新面孔:hangGlider_3 2.35/lowThrust_4 2.33——人群未审;
- bcsstk01/02/STM07 1.3-1.5× 微阵回退;
- 产出:`results/loss_audit_1028.csv` + 每桶→靶刀映射,定 P1/P2 主次。

## P1|dense/diter 旧队列线(预期 GM 最大收益,按 P0 分桶开刀)
T5b k7 人口门(bloweybq 专靶,四夜稳定误 route)→ T3 ESC 覆盖带(usparse 对位
33 阵@1.95×,flop 33-128)→ T4b 内环瘦身(StocF 残余)。每刀预注册 + 同会话 A/B。

## P2|b45 旗舰收尾(平行小刀)
- bins6/7 素数档扩容(2719/5449,HH_LAUNCH 加档,cage15 专靶,env HASH_H67 试点);
- **Bump non-acc 相位分解**:compute-only 84.67 中 acc 66+compact 11.5=77.5,
  ~7ms 住 count/est/binning/retry/cnnz——逐相位列账 vs 差额定位(kernel 外赢面)。

## P3|账目清零
bcsstk 微阵定位 + **路由时效性检查**(dispatcher/L0 门 = 旧相位成本训练;
hash_heavy 转正后 bins4/5 成本已变,边界重估可能有免费赢面)。

## P4|confirm16(全捆绑入账)→ 总账/MEMORY 更新。

## P5(平行)|论文线
方法论五件 + 主场翻面 + per-file 架构;**FlexSpGEMM(ACA-Lab 2026-07,ML 选
tile/硬件路径 + TC 混执,基线=cuSPARSE/spECK/TileSpGEMM/HSMU)进 related work
对表**——选择空间正交(我们=形态族+编译层),需占位;AA/AAT 双轨=同赛道信号,
保持监控。

## 时间线
- 半日 1:P0 取证(脚本后台)+ Bump 相位分解(平行)→ 分桶定主次;
- 其后按桶开刀(P1 或 P2 主攻),每刀绿即 commit,攒 confirm16;
- 收官:confirm16 + 总账盖章。

## P0/P2 中途发现(10-28 晚)
**Bump non-acc 相位分解(P2 第二项,已完成)**:compute-only 84.65 全相位对齐——
count_flop 1.34 + **mh_construct 1.22 + mh_merge 4.23** + est_scan 0.07 + binning 0.25
+ acc 65.94 + cnnz 0.10 + compact 11.49 = 84.65 ✓。
**定谳:Bump 对 ocean 的 4.5ms 缺口 ≈ mh 定容机器 5.45ms 开销**(MHSAMP=0 不关机器,
它喂 binning/tmp 定容,裸门违鲁棒铁律)。**P2 新刀=定容管线降成本**:pregate 强判
hash 的阵跳过 MinHash sketch,est 退 cheaper 上界(需守 tmp 定容/retry 语义);
预期 Bump 84.65→~79 < ocean 80.19 翻面。ocean total_time 含其自家 sizing——
对比合法,我方 sizing 更贵是真实差额。

## P0 定谳:输面全 hash 路由,两真桶(10-28 晚)
131 输阵 [adapt] 全判 hash(无 dense/diter 桶——预设推翻);bins 分桶:
- **bin9 大表桶(最狠,1.77-3.07×)**:bloweybq(9:10000)/hangGlider_3(9:9118)/
  lowThrust_4(9:5271)/TSOPF_FS+case9(9:7214)/pf2177(bin7-9)——bin9 走 LHT2(14)
  196KB smem 表=占用灾难(1块/SM 级);**靶刀=hash_heavy 扩 bin9(HT_P=16253 素数档
  195KB≤227KB,opt-in 已在 §六十六 全家块)**;预期整桶 1.9-3.1× 收割;
- **微行巨阵 non-acc 桶(~1.8×)**:atmosmodj/d(0:1.27M)/delaunay_n23·24(0:6-12M)/
  bloweya——acc 仅 2.6-32ms,差额住 mh 定容+binning 大 n 开销;**靶刀=同 Bump 的
  定容降本刀**(P2,一刀双吃);
- 旧账:Stanford_Berkeley 1.855(known-cost 台账)。
**档案纠错:bloweybq 旧诊断"T5 预门误 route"作废——真因 bin9 形态**(T5b 队列项
被 bin9 扩档刀取代)。
**双刀计划**:①hash_heavy 扩 bin6(2719)+bin9(16253);②定容降本(mh 退位)。

## 刀①首试:空刀定谳+重瞄(10-28 晚)
**实现**:HH 链加 bi6(2719)/bi9(16253) 素数档(env HASH_H69 试点,默认关),
构建+双审计绿,bloweybq memcheck 0 错。**五阵 A/B 全平(逐位同)= 空刀**。
**nsys 破案(P0 分桶纠错)**:bloweybq 主核 = **hash_dense_kernel 8.03ms(65.3%)**
+mh_merge 1.17+count_flop_tiny——**bin9 胖行在 hash 方法内部的 dense 累积器子路由
被消费,根本不进 generic-else 的 HH 链**。P0 按"[adapt]→hash"分桶太粗:hash 内部
还有 dense/diter/batched 子路由。
**重瞄刀①'**:靶点移到 dense_mode 拦截上游——H69 开时 bin9 行绕过 dense 累积器
→正常 binning→HH 链(需读 dense_mode 门控:hash_product 的 dense_mode 分支与
DITER 族 env;A-line T6 旧线的真正接口)。mh_merge 1.17ms 在此阵也在账上
(刀②双吃范围扩大)。H69 现有空刀代码保留(env 默认关,无害)。

## 刀①' 定谳:bin9 桶收割(10-28 晚)
**两步实现**:①h69_route_on() 压制 dense_mode/dense_win_mode(矩阵级 dense 累积器
拦截上游改道);②bin9 compact sorted 直拷门(HH 有序直写免 SEGRTS——无此门时
A/B 实证 compact 42ms 天价吃光 acc −58% 的战果)。
**五阵 A/B(compute-only,nnz 全同,memcheck 0 错)**:
| 阵 | off | H69' | 净 | 对 ocean |
|---|---|---|---|---|
| bloweybq | 12.55 | 8.46 | −32.6% | 3.07→2.07 |
| hangGlider_3 | 10.02 | 7.14 | −28.7% | 2.35→1.67 |
| lowThrust_4 | 6.55 | 3.80 | −42.1% | 2.34→1.35 |
| TSOPF_FS_b9_c6 | 8.49 | 6.12 | −27.9% | 1.94→1.40 |
| pf2177 | 4.16 | 2.20 | −47.1% | **1.77→0.94 翻面** |
**机制定谳**:T6 时代 dense 累积器对 bin9 胖行的"最优"已被 hash_heavy(16253 档,
acc −41~−58%)+ 有序直写 + 直拷组合推翻——"决策人群随内核演化"又一实例。
**下一步**:①默认翻前**人口精细化**(压制 dense 仅当 bin9 有人口——H69 全局压制
会把无 bin9 的 dense 最优阵(exdata_1 类)误赶去 hash 路=回退雷区;改 h_cnt[9]>0
联锁);②15 阵回归(off 零影响抽验)→ 默认翻 → confirm16。bloweybq/hangGlider
残余差距住 non-acc(mh 1.17+count_flop 1.1)→ 刀②双吃名单+2。

## H69 转正:人口联锁+默认翻(10-28 晚)
**实现**:dense 压制门联锁 `h_cnt[9] > 0`(无 bin9 的 dense 最优阵零影响);
bin9→HH(16253) 默认 1(HASH_H69=0 逃生门);bin6→HH(2719) 需 H69=2(未验证人口
不默认翻);compact bin9 直拷门同步 env。双审计绿。
**验证**:bloweybq 8.46/TSOPF_FS 6.13 = H69' 数复现;case9(与 TSOPF_FS 同构)
6.11 同步受益;**exdata_1 = 2.84 vs confirm15 的 2.850 = 零影响**(人口联锁实证)。
电池 v5(新默认)回归跑中。

## 电池v5全绿+confirm16启动(10-28 晚)
v5 vs v4:零回归,意外扩大——TSOPF Σ−3.79/pre2 −3.65/c-64 −0.70/**Bump −0.99**
(bin9 人口比五阵试点更广,Bump/pre2 也有);bloweybq 10.05;nnz 15/15 同。
**confirm16_428 启动**(ocean 同卡配对;预期 bin9 桶+pf2177 翻面+TSOPF/pre2/c-64/Bump 溢出收益)。

## confirm16 终账:0.8311 → 0.8054(10-28 深夜)
| 口径 | confirm15 | **confirm16** | Δ |
|---|---|---|---|
| 总 GM(n=427) | 0.8311 | **0.8054** | **−3.1%** |
| ocean337 主场 | 0.9801 | **0.9728** | −0.7% |
| first100 | 0.4521 | **0.4010** | −11.3% |
| 赢面 | 69.3% | **70.5%**(301/427) | +1.2pp |
旗舰:Bump **1.005**(差0.5%翻面)/cage15 **1.013**(差1.3%)/Hook 1.003(噪声带)/
pf2177 **0.932 翻面入账**;bin9桶:bloweybq 2.08/hangGlider_3 1.68/lowThrust_4 1.30/TSOPF_FS 1.41。
**超预测复盘**:预测带 0.820-0.826 基于 6 已证阵外推;实际**自比改善>10% 的阵 79 个**
(bcspwr/bcsstm/c-45/c-57/662_bus/Reuters911/bundle1/TSC_OPF_300 等大量已赢小中阵
"赢了再赢"→first100 GM −11.3%)——**bin9/beneficiary 人口远大于输面取样**;
方法论一课:外推人口不能只从输面取样(赢面阵的受益人口被系统性低估)。
**输面残余 126(>1.10 者 95)**:头部 bloweybq 2.08/hangGlider_5 1.90/atmosmodd·j
1.85/delaunay_n22-24 1.75-1.81/Stanford_Berkeley 1.75——**atmosmod+delaunay 桶+
bloweybq/hangGlider 残余 = 刀②(mh 退位)靶人群**;Bump 0.5%/cage15 1.3% 尾巴也在刀②。
**两日累计**:0.8873 → 0.8054(**−9.2%**),主场 1.099→0.9728,赢面 45%→70.5%。

## 刀②设计定稿(读码完毕,10-28 夜)
**机制图**:①mh 已有门——avg_product ≤ AVG_FLOP_THR 时 est=精确 flop 免 MinHash;
dup 重阵(Bump 类)走全量 MinHash 的目的=把 flop 上界压回近 distinct(免 tmp 爆 ~10×);
②MHSAMP 采样骨架已存(mh_construct/merge 的 sample_stride/n_work 参数+est_projected
覆写管道)但仅用于 dense_win 候选的密度投影;③mh_merge 4.23ms=merge 形态已调优
(MH_MERGE_R=4/B 线 MH_K 证伪史),kernel 内优化空间枯竭——**换算法不磨 kernel**。
**刀②=采样标定 κ**:hash 路由大阵上,S=64k 行 stride 采样跑 mh(管道现成,~0.12ms)
→ κ=Σdistinct_sample/Σflop_sample → est[i]=flop[i]×κ×margin(单轻核或并入 est_scan)
→ Σest×κ 投影定容(est_projected 管道现成)。retry 语义不动(κ 欠估行→cap 守卫→
flop 表 retry ✓)。风险:κ 全局均匀假设 vs 行级 dup 方差(Bump 97.8% 重行=分布集中,
κ 应稳;hangGlider 类待验)。
**判据**:Bump computeOnly <84.2(ocean 84.60 同夜)/atmosmod·delaunay 桶改善/
nnz 逐位+retry 行数审计+15 阵电池回归。实施次序:采样门扩展→κ 计算→est 核→A/B。

## 刀②首试:全局κ证伪+靶错位(10-28 深夜;预注册判据履职)
**实现**:kappa_sample(oc_sample 同体精确计数,反向比)+kappa_est(est=flop×κ)+
host 门(MHKAPPA=1 试点,S=2048,q05×1.15,κ下限0.125;弃样回全量mh安全网)。
**A/B 三靶**:
- **Bump:κ 风暴实锤**——q05=0.092(重行dup~11×)但**分布双峰**,低dup尾行(比≈1)
  被 est=flop×0.125 重害低估 → **59925行retry(co 84.65→593,7×病理)**;nnz 仍逐位同
  (retry 兜底语义工作正常)。**定谳:全局κ结构性无法替代逐行dup判别**——mh 的
  5.45ms 买的正是双峰矩阵的行级区分;Bump 方向证伪(除非行级κ=即 mh 本身)。
- **atmosmodj/delaunay_n24:靶错位**——A_rows≥100k 但 avg_product 低→est=flop
  免MinHash路径,mh 根本不跑,co 全平(4.03/46.95)。**其1.8×差额在别处**
  (count_flop/binning/est_scan/compact 的大n开销)→ 需独立 nsys 取证。
MHKAPPA 留 env 关(默认0,无害);弃样安全网未触发路径保留。
**Bump 剩余0.5%的替代弹药**:count_flop 1.34ms(kernel优化靶,边际)/接受现状等
atmosmod 取证结果重排。

## atmosmod/delaunay 桶取证定谳(10-28 深夜)
**nsys kernel 表**:atmosmodj(co 4.03/ocean~2.19)=hash_spa_batched<128,8> 2.57ms(64%)
+FLAT 1.10;delaunay_n24(co 46.95/ocean~26)=batched **24.8ms(50.8%)**+FLAT 11.6+
esc<32> 6.9。**真靶=hash_spa_batched<128,8> 的 bin0 微行巨阵人口**(est≤64 ×
1.27M-12.5M 行),我方比 ocean 同人口慢 ~2×;FLAT copy(带宽已核)与 esc 次之。
**定下一战役**:batched 核优化(读核/lineinfo/对拍 ocean 同人口形态)——非一刀,
是 A 线级 kernel 战役;预期该桶(~8-15 阵@1.8×)若压到 ~1.2×,GM 再 −1% 级。
Bump 残余(1.005)与 cage15(1.013)属边际尾,mh 方向已证伪,batched 战役优先级更高。

## batched 核战役·读码笔记(10-28 深夜)
**我方形态**(hash_spa_batched<HT=128,WPB=8>):warp-per-row,每 warp 私有 smem 表
128 槽(est≤64×2余量,5×HT=2.5KB/warp→20KB/块),k 串行×j 32 路并行(lane 跨 B 行
stride),CAS 插入+原子值加,融合有序 extract(pack_col/slot)。geometric 血统:
docs/13(333SP 55% 块地板税的对症)+docs/67 §8 模板化。
**人口画像**:atmosmodj bin0=1.27M 行(est≤64,~25-50 积/行)→ 32 lane 中 ~1-2 积/lane
=**lane 利用率嫌疑**;delaunay_n24 12.5M 行同构 4×。
**下步**:①lineinfo 源热点(-lineinfo 已在 flags;nsys 采样归源);②读 ocean 移植核
同人口机理(其 hashNumeric 对 tiny row 的线程分配/表尺寸;对拍其对 atmosmodj 的
~2.2ms 全管线分解);③候选刀:每 warp 2-4 行(行数×/warp 换 lane 并行度)/WPB 梯/
值域 gmem(HYBRID 已有管道)。判据:atmosmodj/delaunay_n24/bloweya 三阵 computeOnly
+nnz 逐位+15 阵回归(bin0 生产主力,全人口影响)。

## batched 战役·机理定谳与刀设计(10-28 深夜)
**对拍(ocean 移植核,仪器合规)**:其 hashNumeric=block-per-row + **oc_localLoadBalance
动态分组**(按 a_len/num_products/max_b_len 把块内线程分 2^k 组:组间并行 k,组内
并行 j)——对 B 行长自适应。**我方 2× 差距机制**:batched 的 warp-per-row = j 32 路
固定并行,atmosmodj 型微行(B 行 ~7 长)→ **lane 利用率 ~22%**(7/32),几何错配;
delaunay 同构 4×。
**刀设计(multi-row-per-warp)**:每 warp 承 RPW=2/4 行(lane 组=32/RPW:8-lane 组
配 B 行 ~7 恰好);每 lane 组私有表 HT=64(est≤64,LF 50% 线性探测);smem/warp=
RPW×5×64(≈1.2-2.5KB,现 2.5KB 内);extract 融合按组;人口门:bin0 且 B 行短人口
(或统一替换 bin0 主路,电池定)。备选刀=吸收 loadBalance 机理(k/j 动态分组)。
**实施注意**:bin0 生产主力(1.27M-12.5M 行人口),三件套+15 阵电池必过;
nnz 逐位+行内有序(extract 融合保序)。

## batched 刀定谳:multirow 兑现+转正(10-28 深夜)
**实现**:hash_spa_batched_multirow<HT,RPW>(组私有表,组内 k 串行×j 并行,统一界循环
防分歧 sync,extract/rank/溢出同款语义);发射门 bi==0(env BATCHMR)。
**A/B 三阵(computeOnly,nnz 全同,memcheck 0 错)**:
| 阵 | off | mr1<128,2> | **mr2<64,4>** |
|---|---|---|---|
| atmosmodj | 4.06 | 3.48(−14%) | **3.29(−19%)** |
| delaunay_n24 | 46.92 | 45.50 | **40.98(−12.7%)** |
| bloweya | 9.07 | 8.79 | 9.54(交错重采=纯噪声带,off 8.84-9.54/mr2 8.77-9.28 全重叠) |
**lane 利用率机制证实**(ocean 对 ocean 比值:atmosmodj 1.85→~1.50/delaunay 1.81→~1.58,
未全关——FLAT copy 与残余相位在账)。**BATCHMR 默认翻 2**(逃生门 0);电池 v6 跑中;
绿→confirm17。

## 电池v6全绿+confirm17启动(10-29 凌晨)
v6 vs v5:nnz 15/15 同;两面旗(c-64/bcsstk17 +5%)交错重采=噪声清零(c-64 0.330/0.331
vs v5 0.33;bcsstk17 0.244-0.255 vs 0.26);bin0 广谱阵零回归。**confirm17_428 启动**
(记分形态=confirm16+multirow<64,4> 默认;预期 atmosmod/delaunay 桶+bin0 广谱,GM 0.8054→0.79x?)。

## confirm17 中段翻车与人口门复修(10-29 凌晨;第二次全局翻默认教训)
**预览报警**(250/428 同子集):GM 0.6936→0.7309(+5.4%回退),**128 阵恶化 vs 7 改善**
——multirow<64,4> 的 8-lane 组在 B 行长族(bcsstk/bcsstm/bcspwr,j 并行度砍 4×)大回退;
sparsine 6.35×。**根因=BATCHMR=2 裸翻默认,第二次人口联锁失误**(§六十二 H69 教训
未吃透;15 阵电池覆盖不了 428 小阵多样性)。
**复修**:杀 confirm17;默认回滚 0;**标量人口门 avg_j=total_flop/A_nnz≤12** 才开
multirow(B 行均长,B 行短人口靶;env 三态 1/2/0 强制诊断)。双向验证:atmosmodj
3.25/delaunay 41.02(multirow 保持);bcsstk18/bcspwr10 回原核。confirm17 重启(门控形态)。
**纪律升级:一切默认翻必须标量人口门或 bins 联锁,电池绿≠428 绿——人口门设计
先于翻默认**(写进 loop 硬纪律)。
