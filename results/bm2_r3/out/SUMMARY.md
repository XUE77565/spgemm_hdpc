# bm2_r3 汇总(想法二 HH_BM2 第三轮:行门前置 v2 + 行门标定 + 只评测电池)

## S0 环境
- 主机: H100-187  开始: 2026-11-24T22:11:43Z  快照: /home/xueyizhou/exp-runs/bm2_r3/code
- nvcc(PATH): /usr/bin/nvcc / Build cuda_12.0.r12.0/compiler.32267302_0
- python3: Python 3.12.3 / numpy 2.3.4 scipy 1.18.1
- 磁盘: /dev/sdb1        15T  8.4T  5.4T  61% /home
- **用卡: GPU 0**
- 开跑时两卡: 0, 14 MiB, 0 %;1, 14 MiB, 0 %;

## S1 微复现器
- src/hh_bm2.cuh 的 sha256 与 bm2_r1 验过的相同 → 沿用 r1 的 ALL-PASS(两个种子,112320 行,bad=0);本轮改的是行门与发射,行例程未动
- S1: green

## S2 构建
- make TARGET=spgemm_bm2: rc=0,用时 55s;审计: TU-90 模板实例: 26 个;TU-61: 75 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;
- WRITE_MTX 件 spgemm_bm2w: rc=0(只重编 main.cu)
- S2: green

## S3 冒烟
- 小阵 off/m1/m2/m2g(带门,只为走一遍门的代码路径)各 1 发:nnz 与失败检查 → green;走两遍式的行数合计 = 376470

## S4a 值级·对照组:off / m1 / m2 各自对 scipy 真值

判法(dialog/qa-001 双方对齐的作业内口径):差分门 PASS ∧ 官方 L1=L2=0 ∧ 该臂 nerr 最大值 ≤ 1e-10(nerr=|got−want|/Σ|乘积|);官方 L3 错行数照列,并与 l3_diag 用同一判式重算的行数对账(不等=两套口径脱钩=红)。

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) | 官方 L1 / L2 / L3 错行 | l3_diag 重算错行 | nerr 最大 | 本作业判 |
|---|---|---|---|---|---|---|---|---|
| bcsstk17 | off | 1406936 | 0 / 0 | - | 0 / 0 / 2950 | 2950(=) | 7.380603e-16 | 对照 |
| bcsstk17 | m1 | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=483739 worst_rel_vs_global=2.737e-16 | 0 / 0 / 2976 | 2976(=) | 6.561526e-16 | 绿 |
| bcsstk17 | m2 | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=504432 worst_rel_vs_global=1.824e-16 | 0 / 0 / 2980 | 2980(=) | 8.851990e-16 | 绿 |
| af23560 | off | 2432504 | 0 / 0 | - | 0 / 0 / 0 | 0(=) | 4.414567e-16 | 对照 |
| af23560 | m1 | 2432504 | 19026 / 0 | PASS col_mismatch=0 unordered=0 text_diff=219555 worst_rel_vs_global=3.513e-16 | 0 / 0 / 0 | 0(=) | 6.233885e-16 | 绿 |
| af23560 | m2 | 2432504 | 19026 / 0 | PASS col_mismatch=0 unordered=0 text_diff=215706 worst_rel_vs_global=2.342e-16 | 0 / 0 / 0 | 0(=) | 4.432738e-16 | 绿 |
| cfd2 | off | 13418644 | 0 / 0 | - | 0 / 0 / 0 | 0(=) | 8.104660e-16 | 对照 |
