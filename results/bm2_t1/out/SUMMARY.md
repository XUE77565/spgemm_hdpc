# bm2_t1 汇总(想法二 v3.1:按需划分的值级回归、宽跨度/bin9 值级、ncu、A/A 认证口径的电池与无侵入)

## S0 环境
- 主机: H100-187  开始: 2026-11-25T18:52:56Z  快照: /home/xueyizhou/exp-runs/bm2_t1/code  nproc=32
- nvcc(PATH): /usr/bin/nvcc / Build cuda_12.0.r12.0/compiler.32267302_0
- python3: Python 3.12.3 / numpy 2.3.4 scipy 1.18.1
- 磁盘: /dev/sdb1        15T  8.5T  5.3T  62% /home
- **用卡: GPU 0**
- 主机(开跑时): load=11.07 12.24 11.72 外来>50%CPU进程数=2 卡=0, 14 MiB, 0 %;1, 14 MiB, 0 %;

## S1 微复现器
- src/hh_bm2.cuh 的 sha256 与 bm2_r4 / bm2_r5 的 S1 验过的相同(两种子 ALL-PASS,168480 行 bad=0)→ 沿用
- S1: green

## S2 构建
- make TARGET=spgemm_t1: rc=0,用时 66s;审计: TU-90 模板实例: 26 个;TU-61: 51 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;
- WRITE_MTX 件 spgemm_t1w: rc=0(只重编 main.cu)
- S2: green

## S3 冒烟(各臂 1 发:nnz 与失败检查;单发的时间比不作读数)
- G1 默认(按需划分)off/m1/m2/m2p/m2L,6 阵 → green;走两遍式的行数合计 = 316999
- G2 同批划分(HH_BM2_LAZY=0,bm2_r5 的形态)off/m2e/m2pe,4 阵 → green;走两遍式的行数合计 = 97690
- G3 强制宽跨度(m2X:legacy 形态也收 + SMEM 预算 57856 字;**生产默认到不了的形态**,只为值级封条)off/m2X,4 阵 → green;走两遍式的行数合计 = 208144
- 主线臂 = 按需划分(默认 env)

## S4 值级(v3.1 回归 + 宽跨度封条 + bin9 探查)

### S4a v3.1 回归:off / m1 / m2(默认 env=按需划分)差分门 + 对 scipy 真值的归一误差

判法:差分门 PASS ∧ nnz 同 off ∧(l3_diag 有输出时)该臂 nerr 最大值 ≤ 1e-10。行例程本身的官方三判已在 bm2_r5 的 S4a 做过(20 组全绿)。

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) | 官方口径错行(仅参考) | nerr 最大 | 本作业判 |
|---|---|---|---|---|---|---|---|
| bcsstk17 | off | 1406936 | 0 / 0 | - | 2969 | 7.081592e-16 | 对照 |
| bcsstk17 | m1 | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=475822 worst_rel_vs_global=1.824e-16 | 2968 | 7.690489e-16 | 绿 |
| bcsstk17 | m2 | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=503735 worst_rel_vs_global=1.824e-16 | 2961 | 7.582301e-16 | 绿 |
| cfd2 | off | 13418644 | 0 / 0 | - | 0 | 7.736264e-16 | 对照 |
| cfd2 | m1 | 13418644 | 90070 / 0 | PASS col_mismatch=0 unordered=0 text_diff=1790262 worst_rel_vs_global=4.353e-16 | 0 | 8.599947e-16 | 绿 |
| cfd2 | m2 | 13418644 | 90070 / 0 | PASS col_mismatch=0 unordered=0 text_diff=1799345 worst_rel_vs_global=4.353e-16 | 0 | 7.736264e-16 | 绿 |
| msc10848 | off | 6185646 | 0 / 0 | - | 10 | 1.300396e-15 | 对照 |
| msc10848 | m1 | 6185646 | 3423 / 0 | PASS col_mismatch=0 unordered=0 text_diff=2649049 worst_rel_vs_global=3.799e-16 | 9 | 1.300396e-15 | 绿 |
| msc10848 | m2 | 6185646 | 3423 / 0 | PASS col_mismatch=0 unordered=0 text_diff=2704460 worst_rel_vs_global=1.900e-16 | 8 | 1.408762e-15 | 绿 |

### S4b legacy 形态的划分掩码路径:pre2(m2L)

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |
|---|---|---|---|---|
| pre2 | m2L | 323445872 | 121999 / 23205 | PASS col_mismatch=0 unordered=0 text_diff=279977 worst_rel_vs_global=1.092e-27 |

- S4a+S4b(主线): green;走两遍式的行数合计 327387

### S4c 宽跨度封条:web-Google(m2X)

**封条注**:m2X = HH_BM2=2 + HH_BM2_L1=1 + HH_BM2_SMX=57856,是生产默认到不了的形态(legacy 形态默认不开;SMEM 预算默认不放大)。
这里只验它的值对不对,防以后门变化把没验过的形态放出来;不计时,不代表建议启用。

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |
|---|---|---|---|---|
| web-Google | m2X | 29710164 | 36737 / 9321 | PASS col_mismatch=0 unordered=0 text_diff=0 worst_rel_vs_global=0.000e+00 |

### S4d bin9 探查(dialog/qa-004:电池与训练池里 bin9 没有 HH 行;在 PARSEC 族等候选阵里找)

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |
|---|---|---|---|---|
| GaAsH6 | m2 | 37611445 | 25720 / 0 | PASS col_mismatch=0 unordered=0 text_diff=3904276 worst_rel_vs_global=7.850e-16 |
| Ga3As3H12 | m2 | 62181563 | 30211 / 0 | PASS col_mismatch=0 unordered=0 text_diff=11354395 worst_rel_vs_global=1.176e-15 |

- bin9 探查(阵:走了两遍式的臂=bin9 行数;m2 = 生产可达形态,m2X = 强制形态): H2O:无 GaAsH6:m2=967 Si34H36:无 Ga3As3H12:m2=2691;做了值级的 2 个
- S4 主线: green | 宽跨度封条(m2X): green | bin9: 生产可达形态 green / 强制形态 green

## K3 ncu 硬件计数(hash_heavy 对 hash_bm2:缓存命中、显存吞吐、指令数、占用;尽力而为)
- ncu: /usr/bin/ncu / Version 2022.4.1.0 (build 32308335) (public-release)
- ncu 试点 4 种组合都没产出报告(各组的 rc 与报错行在 k3_pilot.log),K3 跳过

## T0 负载门
