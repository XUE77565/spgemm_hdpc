# bm2_r5 汇总(想法二第四轮重投:两遍式独立核 v3 + 行门标定 + 只评测电池;bm2_r4 在 S2 构建失败,见 run.sh 头部)

## S0 环境
- 主机: H100-187  开始: 2026-11-25T03:15:07Z  快照: /home/xueyizhou/exp-runs/bm2_r5/code
- nvcc(PATH): /usr/bin/nvcc / Build cuda_12.0.r12.0/compiler.32267302_0
- python3: Python 3.12.3 / numpy 2.3.4 scipy 1.18.1
- 磁盘: /dev/sdb1        15T  8.4T  5.4T  61% /home
- **用卡: GPU 0**
- 开跑时两卡: 0, 14 MiB, 0 %;1, 471 MiB, 0 %;

## S1 微复现器
- src/hh_bm2.cuh 已变(2fc25346de6deabc2eda6d49739ba0627a976969896b861e3798517bab59339b),重跑 hh_repro5
- ALL-PASS: trials/cfg=40 seed=42 bad=0;ALL-PASS: trials/cfg=25 seed=2026 bad=0;
- 行数合计: rows=168480 ok=116709 ovf=7413 skip=44358 bad=0(9 档几何 × B16/值区模式/Pass1 来源 6 种组合)
- S1: green

## S2 构建
- make TARGET=spgemm_bm2: rc=0,用时 146s;审计: TU-90 模板实例: 26 个;TU-61: 51 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;
- WRITE_MTX 件 spgemm_bm2w: rc=0(只重编 main.cu)
- S2: green

## S3 冒烟
- 小阵 off/m1/m2/m2p/m2L/m2g(m2g 带门,只为走一遍门的代码路径)各 1 发:nnz 与失败检查 → green;走两遍式的行数合计 = 356504

## S4a 值级·对照组:off / m1 / m2 / m2p 各自对 scipy 真值

判法(dialog/qa-001 双方对齐的作业内口径):差分门 PASS ∧ 官方 L1=L2=0 ∧ 该臂 nerr 最大值 ≤ 1e-10(nerr=|got−want|/Σ|乘积|);官方 L3 错行数照列,并与 l3_diag 用同一判式重算的行数对账(不等=两套口径脱钩=红)。

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) | 官方 L1 / L2 / L3 错行 | l3_diag 重算错行 | nerr 最大 | 本作业判 |
|---|---|---|---|---|---|---|---|---|
| bcsstk17 | off | 1406936 | 0 / 0 | - | 0 / 0 / 2951 | 2951(=) | 6.608492e-16 | 对照 |
| bcsstk17 | m1 | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=472308 worst_rel_vs_global=1.824e-16 | 0 / 0 / 2964 | 2964(=) | 7.156079e-16 | 绿 |
| bcsstk17 | m2 | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=503605 worst_rel_vs_global=1.824e-16 | 0 / 0 / 2976 | 2976(=) | 6.530673e-16 | 绿 |
| bcsstk17 | m2p | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=503988 worst_rel_vs_global=1.824e-16 | 0 / 0 / 2975 | 2975(=) | 7.960852e-16 | 绿 |
| af23560 | off | 2432504 | 0 / 0 | - | 0 / 0 / 0 | 0(=) | 4.402007e-16 | 对照 |
| af23560 | m1 | 2432504 | 19026 / 0 | PASS col_mismatch=0 unordered=0 text_diff=226745 worst_rel_vs_global=2.342e-16 | 0 / 0 / 0 | 0(=) | 4.963373e-16 | 绿 |
| af23560 | m2 | 2432504 | 19026 / 0 | PASS col_mismatch=0 unordered=0 text_diff=223476 worst_rel_vs_global=4.684e-16 | 0 / 0 / 0 | 0(=) | 4.432738e-16 | 绿 |
| af23560 | m2p | 2432504 | 19026 / 0 | PASS col_mismatch=0 unordered=0 text_diff=224472 worst_rel_vs_global=4.684e-16 | 0 / 0 / 0 | 0(=) | 5.319859e-16 | 绿 |
| cfd2 | off | 13418644 | 0 / 0 | - | 0 / 0 / 0 | 0(=) | 8.299201e-16 | 对照 |
| cfd2 | m1 | 13418644 | 90070 / 0 | PASS col_mismatch=0 unordered=0 text_diff=1788949 worst_rel_vs_global=4.353e-16 | 0 / 0 / 0 | 0(=) | 7.772588e-16 | 绿 |
| cfd2 | m2 | 13418644 | 90070 / 0 | PASS col_mismatch=0 unordered=0 text_diff=1793499 worst_rel_vs_global=4.353e-16 | 0 / 0 / 0 | 0(=) | 8.104660e-16 | 绿 |
| cfd2 | m2p | 13418644 | 90070 / 0 | PASS col_mismatch=0 unordered=0 text_diff=1790347 worst_rel_vs_global=4.353e-16 | 0 / 0 / 0 | 0(=) | 8.104660e-16 | 绿 |
| msc10848 | off | 6185646 | 0 / 0 | - | 0 / 0 / 8 | 8(=) | 1.142301e-15 | 对照 |
| msc10848 | m1 | 6185646 | 3423 / 0 | PASS col_mismatch=0 unordered=0 text_diff=2665903 worst_rel_vs_global=1.900e-16 | 0 / 0 / 8 | 8(=) | 1.291460e-15 | 绿 |
| msc10848 | m2 | 6185646 | 3423 / 0 | PASS col_mismatch=0 unordered=0 text_diff=2699660 worst_rel_vs_global=3.799e-16 | 0 / 0 / 8 | 8(=) | 1.260667e-15 | 绿 |
| msc10848 | m2p | 6185646 | 3423 / 0 | PASS col_mismatch=0 unordered=0 text_diff=2687613 worst_rel_vs_global=1.900e-16 | 0 / 0 / 9 | 9(=) | 1.252459e-15 | 绿 |
| s3dkq4m2 | off | 13282553 | 0 / 0 | - | 0 / 0 / 88427 | 88427(=) | 6.528503e-16 | 对照 |
| s3dkq4m2 | m1 | 13282553 | 88410 / 0 | PASS col_mismatch=0 unordered=0 text_diff=4211252 worst_rel_vs_global=4.853e-16 | 0 / 0 / 88308 | 88308(=) | 6.640253e-16 | 绿 |
| s3dkq4m2 | m2 | 13282553 | 88410 / 0 | PASS col_mismatch=0 unordered=0 text_diff=5204041 worst_rel_vs_global=4.853e-16 | 0 / 0 / 88570 | 88570(=) | 8.164730e-16 | 绿 |
| s3dkq4m2 | m2p | 13282553 | 88410 / 0 | PASS col_mismatch=0 unordered=0 text_diff=5306041 worst_rel_vs_global=4.853e-16 | 0 / 0 / 88682 | 88682(=) | 7.145776e-16 | 绿 |

- 交叉标定(官方 L3 错行数 对 l3_diag 重算):20 组里不等的 0 组
- 并排清单(前 60 条):s4_l3diag_<阵>.txt / .csv;按行聚合全量:s4_l3rows_<阵>.csv.gz;官方三判原文尾:s4_truth_<阵>_<臂>.txt
- 条目级全量清单(阵 条数 字节 sha256 C上路径;≤1.5MB 的同时在 OUT_DIR 的 s4_l3full_<阵>.csv.gz):
    bcsstk17 31404 1649093 c0f691232f0e526ee3f988d2486cf9cd4cb485d3a63df9ae6a8c179c8ee092fb /home/xueyizhou/exp-runs/bm2_r5/keep/l3full_bcsstk17.csv.gz
    msc10848 12 762 b327e2d4c2de2cec9d43fc5a455029d936dd29b02adcc23e891fc09039daebb4 /home/xueyizhou/exp-runs/bm2_r5/keep/l3full_msc10848.csv.gz
    s3dkq4m2 1278118 76145127 16de26babcbd5b748a676ae1fad26b9eff7b8fa7c6df6a682b2d2147b118c3af /home/xueyizhou/exp-runs/bm2_r5/keep/l3full_s3dkq4m2.csv.gz
    s3dkq4m2 按行聚合清单超 1.9MB,改留 /home/xueyizhou/exp-runs/bm2_r5/keep/s4_l3rows_s3dkq4m2.csv.gz

## S4b 值级:HH_BM2=1/2 对 =0 的差分门 + 三臂对 scipy 真值的归一误差(不跑官方逐行三判:这几阵 4-6 千万 nnz,python 逐行太慢)

判法:差分门 PASS ∧ nnz 同 off ∧(l3_diag 有输出时)该臂 nerr 最大值 ≤ 1e-10。

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) | 官方口径错行(仅参考) | nerr 最大 | 本作业判 |
|---|---|---|---|---|---|---|---|
| TSOPF_RS_b678_c2 | off | 40139105 | 0 / 0 | - | 0 | 4.399041e-16 | 对照 |
| TSOPF_RS_b678_c2 | m1 | 40139105 | 17002 / 0 | PASS col_mismatch=0 unordered=0 text_diff=785087 worst_rel_vs_global=9.934e-19 | 0 | 4.399041e-16 | 绿 |
| TSOPF_RS_b678_c2 | m2 | 40139105 | 17002 / 0 | PASS col_mismatch=0 unordered=0 text_diff=778663 worst_rel_vs_global=9.934e-19 | 0 | 4.399041e-16 | 绿 |

### ldoor(翻 bin 家族:legacy 形态也交两遍式独立核的 m1L/m2L 臂;1.45 亿 nnz,只跑差分门)

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |
|---|---|---|---|---|
