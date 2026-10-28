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
