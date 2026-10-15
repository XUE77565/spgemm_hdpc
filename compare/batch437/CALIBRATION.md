# compare/batch437/CALIBRATION.md:权威批口径对齐表(10-15,发车前用户拦车促成)

> 双实测定谳(ocean stats.json 实测 + SpGEMM.cuh 源码一手 + 我方 prof 全相位 dump)。
> 结论:**基本对称**;唯一不对称项偏 ocean 亏(诚实标注);binning 双方都算
> ("ocean 没算 binning"的旧记忆不成立——timing 分项与墙钟圈双证)。

| 项 | 我方 compute-only | ocean total_time | 对称性 |
|---|---|---|---|
| 定义 | TOTAL(GPU) − h2d − d2h(WARMUP_N=3 后末轮单发) | iteration 墙钟均值(10 warmup + 10 bench,SpGEMM.cuh:224-279) | — |
| 输入上载 H2D | **扣**(独立相位 h2d) | **圈外**(bench 循环前) | ✓ |
| 输出回传 D2H | **扣**(独立相位 d2h,如 c-64 64.9ms) | **圈外**(timer 停在 numeric 后) | ✓ |
| 决策性 D2H(估计→门) | **无**(diter 门设备上判定,零预跑;TOTAL 相位清单实证无 D2H 项) | **计入**(analysis_mem_cpy 等,圈内:SpGEMM.cuh:484-519) | ✗ **ocean 亏**(对我方有利方向,记分时如实声明) |
| binning | 计入(binning 相位在 TOTAL 内) | 计入(symbolic/numeric/estimation.binning 分项均在圈内) | ✓ |
| convert/mtx 解析 | 圈外 | 圈外(convert 独立二进制) | ✓ |
| 统计口径 | 末轮单发 | 10 轮均值 | 轻微差异(净窗下方差 ±3%,记录) |

## 保险机制(口径争议可后验)

- 我方:逐 (mat,arm) 全 stderr(含全部 [hash-prof] 相位行)→ `prof/<mat>__<arm>.log`
- ocean:逐阵完整 stats.json → `ocn_stats/<mat>.json`
- 任何一方日后改口径定义,无需重跑 30 小时,直接从 sidecar 重算。

## 关联

- 历史层:docs/76 分相对拍(road_usa,彼 stats.json vs 我方 profiler,同族口径);
- 权威基线(10-11):compare/ocean_337/(同 bench_detail.json + total_time 字段,与本次同口径);
- manifest:compare/README.md(批完成后登记)。
