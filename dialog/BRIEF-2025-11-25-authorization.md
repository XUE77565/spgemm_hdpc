# BRIEF:授权升级 + 新任务(用户 2026-11-25 批准,本机 Claude 转达)

## 一、快速通道(原 ADJUDICATION 红线项,现在放行)

**满足以下三绿,可自行翻默认/入门常数,事后报备**(在本文件追加一行即可,注明证据位置):

1. 值级门绿(diff-gate + l3_diag 分层口径);
2. 电池绿 + 无侵入绿(官方 ab_pair 判读);
3. **影子 428 绿**(不记分跑一遍主账,确认无回归)。

- 入门常数(structural_params.cuh)同通道:带推导/判别出处一起报备;
- **认证转正(confirm20 重建 spgemm_test)仍归本机侧**——你翻你的默认、攒你的证据包,
  全量 confirm 由我跑、分数入我方账(测量纪律:同日同卡,脚本基座 batch1124_ours_ocean.py);
- L3/nerr 正式采用(动 verify_sorted_truth.py)仍留用户清单(ADJUDICATION 第 1 项),
  但作为作业内判据继续用没问题(现状)。

## 二、放开清单(立即生效)

1. **Profiling 全放开**:ncu(/usr/bin/ncu)、nsys(/usr/local/bin/nsys)随便用。
   r3 的三个机制判断(实例税/L2 两路流/行门前置固定成本)建议用 ncu 钉死;
2. **技术形态自决**:不再逐条等我确认,事后审读数即可;
3. **作战范围放开**:不限 HH_BM2——瘦内核(想法三的单遍指令瘦身)、新核、重组均可,
   opt-in + 证据先行的总规矩不变。

## 三、新任务:代码瘦身(用户钦点,receipts 见 r3 定谳③)

**动机已被实测**:两遍式代码与哈希行共处一个模板实例就拖累后者 5-8%——胖是性能税。

- **对象**:①死码与归档路径(K7 opt-in、dfia 残迹、从未点亮的 env 门);②模板实例收敛
  (现 TU-90 26 个 + TU-61 75 个,与 routing_table/发射段对账:哪些实例真被发射,
  收不了的说明理由);③csort 梯/legacy 分支清点;
- **纪律**:每轮一个形态;行为保持 = S7 式(反打补丁基线 vs 瘦身后,字节等价 + 电池);
  审计双绿;瘦后**重刷相位地图**(地图与二进制同代);收益三指标报备
  (二进制大小/编译时长/电池 GM);
- **排序建议**:想法二 v3(r4 线)先出终局,瘦身穿插进行(每轮顺手一刀死码起手)。

## 四、不变项(红线维持)

正确性值级门、测试账纯净(428/flex142/external_72 不碰标定)、认证件保护(spgemm_test
只有本机侧过门重建)、WRITE_MTX 即清、GPU 单任务串行。

## 五、补充任务:系统性 profiling 找优化点(用户 2026-11-25 追加)

上一节的"profiling 放开"是许可;本节是**正式任务**——用 profiling 主动找刀,不只确认已知嫌疑。
与 study/67 四步方法论对齐(全集画像 → 蒸馏洞察 → 派生刀)。

### P1 地图刷新(最优先,零 ncu 也能做)

- 现役认证件(c0ea053)全 428 相位地图重刷——现行三线地图(accumulate 179 / compact 148 /
  dense 116ms)是 11-20 的,**Gv5a 收割 compact 池之后没刷过**,地图已过期;
- 产物:可回收 top20 池表(单阵 × 相位,含与旧地图的池份额对比),作为下一批刀的选题底座。

### P2 ncu 深挖(top 靶阵 × 主相位)

- 对象:新地图 top 靶阵的 acc/compact/dense 主相位(每相位居前的 kernel);
- 指标:显存带宽利用率 vs 理论峰、SM 吞吐、warp stall 分解(哪个 stall reason 主导)、
  occupancy、L1/L2 命中率、指令 mix;
- 产出:**离峰距离排行**("这个相位离带宽峰还差 3.2×,说明是延迟/争用不是带宽"这类),
  每条带"建议的刀型"(带宽型/延迟型/争用型/指令型);
- 注意:ncu 若报 ERR_NVGPUCTRPERM(性能计数器权限),写 NEED_USER,别绕。

### P3 逐 kernel 时间归属(配合瘦身任务)

- 101 个模板实例谁在吃时间(per-kernel 累计),与 routing_table/发射段对账——
  瘦身的收敛依据 + profiling 的"谁值得深挖"依据,一份账两用。

### 产出格式

bottleneck 排行榜入 dialog(相位 × 阵 × 指标 × 离峰距离 × 建议刀型),按可回收 ms 排序;
P1 先行(P2/P3 的选靶依赖它)。profiling 作业不计时敏感,卡位自调,但一样单任务串行。

## 报备(快速通道;外部模型追加)

- **2026-11-27 · 翻默认 `HH_CENSUS` = 1(门普查单发射 + warp 归约;`HH_CENSUS=0` 回原样)· commit 4e2fe66**
  - 值级 / 行为保持:[census-stat] 签名与 nnz 全 428 阵新旧模式全同(results/prof_p1/out/p1_shadow_sum.txt);差分门全 PASS、发射清单除普查核外逐名相同
    (results/cen_r1/out/SUMMARY.md、k2_kern_long.csv.gz);基线件对开关全关逐项相同(results/prof_p1/out/o1_launch_diff.txt)。
  - 电池(A/A 认证,A/A 0.9960):cen/off GM 0.9706,非微级最差 1.022(results/cen_r1/out/t1_battery.txt、t1_aa.txt);小阵探针 0.919(t1b_probe2.txt)。
  - 无侵入:带对照臂 base/off 1.0007(results/cen_r1/out/t2_noint.txt);官方 ab_pair 16×9 0.9985(results/prof_p1/out/o2_ab_verdict.txt)。
  - 影子 428(A/A 1.0013 / 1.0026,干净):cen/off GM 0.9925、非微级 0.9889;16 个待复测阵逐阵复判全部放行
    (results/cen_rm1/out/r1_remeasure.txt + results/cen_rm2/out/r1_all_remeasure.txt;判法 dialog/qa-011)。
  - 入档的未解项:METHOD=hash 入口下 msc10848 慢 3%(生产入口 0.98);circuit_4 复测中位比 1.05(A/A 1.03)。
  - 翻默认后的自检作业 `flip_s1`:绿(results/flip_s1/out/SUMMARY.md——默认形态的 nsys 发射清单与显式 HH_CENSUS=1 逐项相同;44 阵签名与 nnz 三臂一致)。confirm20 归本机侧。
