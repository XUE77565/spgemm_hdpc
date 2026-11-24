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
| cfd2 | m1 | 13418644 | 90070 / 0 | PASS col_mismatch=0 unordered=0 text_diff=1764922 worst_rel_vs_global=4.353e-16 | 0 / 0 / 0 | 0(=) | 7.571288e-16 | 绿 |
| cfd2 | m2 | 13418644 | 90070 / 0 | PASS col_mismatch=0 unordered=0 text_diff=1715117 worst_rel_vs_global=3.265e-16 | 0 / 0 / 0 | 0(=) | 8.104660e-16 | 绿 |
| msc10848 | off | 6185646 | 0 / 0 | - | 0 / 0 / 11 | 11(=) | 1.291460e-15 | 对照 |
| msc10848 | m1 | 6185646 | 3423 / 0 | PASS col_mismatch=0 unordered=0 text_diff=2649406 worst_rel_vs_global=3.799e-16 | 0 / 0 / 10 | 10(=) | 1.408762e-15 | 绿 |
| msc10848 | m2 | 6185646 | 3423 / 0 | PASS col_mismatch=0 unordered=0 text_diff=2707639 worst_rel_vs_global=1.900e-16 | 0 / 0 / 9 | 9(=) | 1.296029e-15 | 绿 |
| s3dkq4m2 | off | 13282553 | 0 / 0 | - | 0 / 0 / 88463 | 88463(=) | 6.528014e-16 | 对照 |
| s3dkq4m2 | m1 | 13282553 | 88410 / 0 | PASS col_mismatch=0 unordered=0 text_diff=4479490 worst_rel_vs_global=4.853e-16 | 0 / 0 / 88609 | 88609(=) | 6.798740e-16 | 绿 |
| s3dkq4m2 | m2 | 13282553 | 88410 / 0 | PASS col_mismatch=0 unordered=0 text_diff=5266752 worst_rel_vs_global=4.853e-16 | 0 / 0 / 88672 | 88672(=) | 7.665008e-16 | 绿 |

- 交叉标定(官方 L3 错行数 对 l3_diag 重算):15 组里不等的 0 组
- 并排清单(前 60 条):s4_l3diag_<阵>.txt / .csv;按行聚合全量:s4_l3rows_<阵>.csv.gz;官方三判原文尾:s4_truth_<阵>_<臂>.txt
- 条目级全量清单(阵 条数 字节 sha256 C上路径;≤1.5MB 的同时在 OUT_DIR 的 s4_l3full_<阵>.csv.gz):
    bcsstk17 30142 1362273 f2f272fba136685157193dddcb87710a0956665f0190e1ecd48cbbb3295c960d /home/xueyizhou/exp-runs/bm2_r3/keep/l3full_bcsstk17.csv.gz
    msc10848 18 995 61312fc64b671bc8e5844c68f52c7b5d370f1d3bae889eefdd9692e07c548089 /home/xueyizhou/exp-runs/bm2_r3/keep/l3full_msc10848.csv.gz
    s3dkq4m2 1227469 61806607 9743d09c3387c5a131ab044271a0755f046692170338d981ecafef52152f7adb /home/xueyizhou/exp-runs/bm2_r3/keep/l3full_s3dkq4m2.csv.gz
    s3dkq4m2 按行聚合清单超 1.9MB,改留 /home/xueyizhou/exp-runs/bm2_r3/keep/s4_l3rows_s3dkq4m2.csv.gz

## S4b 值级:HH_BM2=1/2 对 =0 的差分门 + 三臂对 scipy 真值的归一误差(不跑官方逐行三判:这几阵 4-6 千万 nnz,python 逐行太慢)

判法:差分门 PASS ∧ nnz 同 off ∧(l3_diag 有输出时)该臂 nerr 最大值 ≤ 1e-10。

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) | 官方口径错行(仅参考) | nerr 最大 | 本作业判 |
|---|---|---|---|---|---|---|---|
| TSOPF_RS_b678_c2 | off | 40139105 | 0 / 0 | - | 0 | 4.399041e-16 | 对照 |
