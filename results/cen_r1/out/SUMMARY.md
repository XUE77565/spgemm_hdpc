# cen_r1 汇总(门普查单发射版与普查+划分合一:行为保持 → 设备侧时长 → A/A 认证的电池、小阵探针、带对照臂的无侵入)

## S0 环境
- 主机: H100-187  开始: 2026-11-26T00:36:14Z  快照: /home/xueyizhou/exp-runs/cen_r1/code  nproc=32
- nvcc(PATH): /usr/bin/nvcc / Build cuda_12.0.r12.0/compiler.32267302_0
- python3: Python 3.12.3 / numpy 2.3.4 scipy 1.18.1
- 磁盘: /dev/sdb1        15T  8.6T  5.2T  63% /home
- **用卡: GPU 0**
- 主机(开跑时): load=12.81 9.52 9.75 外来>50%CPU进程数=4 卡=0, 14 MiB, 0 %;1, 14 MiB, 0 %;

## S1 微复现器
- src/hh_bm2.cuh 的 sha256 与 bm2_r4 / bm2_r5 的 S1 验过的相同(两种子 ALL-PASS,168480 行 bad=0)→ 沿用
- S1: green

## S2 构建
- make TARGET=spgemm_c1: rc=0,用时 54s;审计: TU-90 模板实例: 26 个;TU-61: 51 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;
- WRITE_MTX 件 spgemm_c1w: rc=0(只重编 main.cu)
- S2: green
- 基线源码件 spgemm_base:sha256 核对通过,make rc=0;审计: TU-90 模板实例: 26 个;TU-61: 51 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;

## S3 冒烟(各臂 1 发:nnz 与失败检查;单发的时间比不作读数)
- G1 两遍式(宏重构后的回归)off/m2/m2p/m2L/m2e,6 阵 → green;走两遍式的行数合计 = 317379
- G2 单发射普查 off/cen/cenm2,7 阵 → green;走两遍式的行数合计 = 48845
- G3 普查 + 划分合一 off/cm2/cm2L,7 阵 → green;走两遍式的行数合计 = 219689

## S4a 门普查行为保持([census-stat] 行 = 每个 bin 的 rows / gated / total,连同结果 nnz;三种模式逐行比对)
- 比对了 24 阵(另有 0 阵基线模式没跑出结果、0 阵基线模式自己两遍就不同,都不计入);其中非空的 (阵,bin) 组合 75 个
- **不一致的阵: 0 个**
- 合一形态已一并比对

## S4b 值级:各臂对 off 的差分门

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |
|---|---|---|---|---|
| bcsstk17 | m2 | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=507408 worst_rel_vs_global=1.824e-16 |
| bcsstk17 | m2e | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=504754 worst_rel_vs_global=1.824e-16 |
| bcsstk17 | cen | 1406936 | 0 / 0 | PASS col_mismatch=0 unordered=0 text_diff=372101 worst_rel_vs_global=1.824e-16 |
| bcsstk17 | cm2 | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=505667 worst_rel_vs_global=1.824e-16 |
| cfd2 | m2 | 13418644 | 90070 / 0 | PASS col_mismatch=0 unordered=0 text_diff=1799057 worst_rel_vs_global=4.353e-16 |
| cfd2 | cen | 13418644 | 0 / 0 | PASS col_mismatch=0 unordered=0 text_diff=1368125 worst_rel_vs_global=3.265e-16 |
| cfd2 | cm2 | 13418644 | 90070 / 0 | PASS col_mismatch=0 unordered=0 text_diff=1795979 worst_rel_vs_global=4.353e-16 |
| msc10848 | cm2 | 6185646 | 3423 / 0 | PASS col_mismatch=0 unordered=0 text_diff=2702503 worst_rel_vs_global=1.900e-16 |
| GaAsH6 | cm2 | 37611445 | 25720 / 0 | PASS col_mismatch=0 unordered=0 text_diff=3916538 worst_rel_vs_global=9.158e-16 |
| web-Google | cen | 29710164 | 0 / 0 | PASS col_mismatch=0 unordered=0 text_diff=0 worst_rel_vs_global=0.000e+00 |
