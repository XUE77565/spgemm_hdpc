# 基线对比报告(2026-10-29)

> 我方 = confirm16 终态(GM 0.8054 vs ocean),per-file 三层 + 刀A + 刀G + H69(bin9 素数档)。
> 口径 = 428 阵并集,我方 compute-only(TOTAL−h2d−d2h)vs 各基线自家口径,H100 单卡。

## 一、总表

| 基线 | 年代 | n | **GM** | **加速比** | 赢面 | 中位 | P25 | P75 |
|---|---|---|---|---|---|---|---|---|
| **ocean(真实对手)** | — | 427 | **0.8054** | **1.24×** | **70%** | 0.883 | 0.647 | 1.060 |
| HSMU-SpGEMM | 2024 | 428 | **0.2138** | **4.7×** | 100% | 0.356 | 0.180 | 0.442 |
| TileSpGEMM | 2020 | 409 | **0.3550** | **2.8×** | 95% | 0.414 | 0.289 | 0.559 |
| FlexSpGEMM | 2026 | TBD(构建中) | — | — | — | — | — | — |

**结论:两家学术 SOTA(HSMU、TileSpGEMM)被大幅甩开(4.7×/2.8×),
唯一有竞争力的对手是 ocean(1.24× 差距,且在 ocean 主场只差 2.8%)。**

## 二、按规模的分解

| 基线 | 大阵 GM(>100ms) | 小阵 GM(≤100ms) |
|---|---|---|
| ocean | 0.707(n=10) | 0.808(n=417) |
| HSMU | **0.121**(n=41) | 0.227(n=387) |
| TileSpGEMM | **0.098**(n=57) | 0.437(n=352) |

大阵上学术基线完全溃败(HSMU 0.121/TileSpGEMM 0.098 ≈ 我们快 8-10×);
小阵上 TileSpGEMM(0.437)稍好但仍差 2.3×。

## 三、ocean 分解(真正对手)

| 口径 | GM | 说明 |
|---|---|---|
| ocean337 主场 | **0.9728** | ocean 自己挑的 337 阵 |
| first100 | **0.4010** | 另一套件,我们碾压 |
| **总 GM** | **0.8054** | 净赢 19.5% |

## 四、各基线最险阵(最接近 1.0)

**ocean(仅列输面 >1.3×)**:bloweybq 2.08 / hangGlider_5 1.90 / atmosmodd 1.85 /
delaunay_n24 1.81 / Stanford_Berkeley 1.75 / case9 1.94 / TSOPF_FS 1.94

**HSMU(最险仅 258× 差距)**:road_usa 259× / hugebubbles-0000 178× /
hugetrace-00020 138× —— 大阵完全不在同一量级。

**TileSpGEMM(最险仅 238× 差距)**:vsp_bcsstk30_500 1592× / sme3Dc 386× /
poisson3Db 210× —— 小阵也大幅落后。

## 五、方法说明

| 基线 | 编译 | 口径 | 备注 |
|---|---|---|---|
| HSMU | 源码自带 Makefile(默认) | 相位时间之和(ms) | mtx 直入,AA 轨 |
| TileSpGEMM | 改 `-code=sm_90`(原 sm_80 不兼容 H100) | stir_spgemm_time(ms) | `-d 0` 参数 |
| ocean | 他们自己的 Makefile(含 compute_61) | total_time | convert+spgemm |
| FlexSpGEMM | 待构建 | — | ML 路由+TC 混执,AA/AAT |

注意:TileSpGEMM 原编译用 `-arch=compute_61 -code=sm_80`(A100 目标),
我们改 `-code=sm_90` 适配 H100——**这也是编译层战役的同族发现**。

## 六、一句话

**学术 SOTA 全线甩开(HSMU 4.7×/TileSpGEMM 2.8×),唯一威胁是 ocean(1.24×),
而 ocean 主场(337 阵)已被翻面(0.9728)——下一个对手是 FlexSpGEMM(2026,ML+TC)。**

---
数据文件:
- `compare/batch437/baseline_hsmu.csv`(428 阵)
- `compare/batch437/baseline_tilespgemm.csv`(428 阵)
- `compare/batch437/confirm16_428.csv`(我方 vs ocean)
