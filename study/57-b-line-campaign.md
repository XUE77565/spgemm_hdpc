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

## 四、战报

- [x] tick1(10-24):重锚完成,微输带结构定谳(b23 拍数带 + b45 人口带 + 病理名单);
  脚本 scripts/reanchor_bins_beats.py;数据 compare/batch437/reanchor_bins.csv。
- [ ] tick2:SASS 静态对账
- [ ] tick3:GPU 逐行对拍
- [ ] T-B3:极简实例刀
