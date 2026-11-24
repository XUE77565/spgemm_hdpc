# bm2_r2 汇总(想法二 HH_BM2 第二轮)

## S0 环境
- 主机: H100-187  开始: 2026-11-24T18:19:22Z  快照: /home/xueyizhou/exp-runs/bm2_r2/code
- nvcc(PATH): /usr/bin/nvcc / Build cuda_12.0.r12.0/compiler.32267302_0
- python3: Python 3.12.3 / numpy 2.3.4 scipy 1.18.1
- 清 bm2_r1 遗留转储: 23G → 1.9G
- 磁盘: /dev/sdb1        15T  8.4T  5.4T  61% /home
- **用卡: GPU 0**
- 开跑时两卡: 0, 14 MiB, 0 %;1, 14 MiB, 0 %;

## S1 微复现器
- src/hh_bm2.cuh 的 sha256 与 bm2_r1 验过的相同(25efd6afa6dbd60879d99d5f8dfb2f38d68f3798f7c04c60fc138e11787d4184)→ 沿用 r1 的 ALL-PASS(两个种子,112320 行,bad=0)
- S1: green

## S2 构建
- make TARGET=spgemm_bm2: rc=0,用时 58s;审计: TU-90 模板实例: 26 个;TU-61: 75 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;
- WRITE_MTX 件 spgemm_bm2w: rc=0(只重编 main.cu)
- S2: green

## S3 冒烟
- 小阵 off/m1/m2 各 1 发:nnz 与失败检查 → green;走两遍式的行数合计(m1+m2)= 97310

## S4a 值级·对照组:off / m1 / m2 各自对 scipy 真值

判法:差分门 PASS ∧ 官方 L1=L2=0 ∧(官方 L3 过,或 off 臂同阵官方 L3 也不过且该臂 nerr 最大值 ≤ 1e-10)。nerr=|got−want|/Σ|乘积|。

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) | 官方 L1 / L2 / L3 错行 | 官方最大 rel | nerr 最大 | 本作业判 |
|---|---|---|---|---|---|---|---|---|
| bcsstk17 | off | 1406936 | 0 / 0 | - | 0 / 0 / 2970 | 9.524e-02 | 6.421360e-16 | 对照 |
| bcsstk17 | m1 | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=481906 worst_rel_vs_global=1.824e-16 | 0 / 0 / 2979 | 9.524e-02 | 7.543677e-16 | 绿(官方 L3 双臂同红,归一口径过) |
| bcsstk17 | m2 | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=504258 worst_rel_vs_global=1.824e-16 | 0 / 0 / 2977 | 4.113e-02 | 8.008414e-16 | 绿(官方 L3 双臂同红,归一口径过) |
| af23560 | off | 2432504 | 0 / 0 | - | 0 / 0 / 0 | 3.458e-10 | 4.633749e-16 | 对照 |
| af23560 | m1 | 2432504 | 19026 / 0 | PASS col_mismatch=0 unordered=0 text_diff=222885 worst_rel_vs_global=2.342e-16 | 0 / 0 / 0 | 9.611e-11 | 4.520124e-16 | 绿 |
| af23560 | m2 | 2432504 | 19026 / 0 | PASS col_mismatch=0 unordered=0 text_diff=220448 worst_rel_vs_global=2.342e-16 | 0 / 0 / 0 | 3.458e-10 | 6.233885e-16 | 绿 |
| msc10848 | off | 6185646 | 0 / 0 | - | 0 / 0 / 7 | 1.157e-01 | 1.236279e-15 | 对照 |
| msc10848 | m1 | 6185646 | 3423 / 0 | PASS col_mismatch=0 unordered=0 text_diff=2655373 worst_rel_vs_global=3.799e-16 | 0 / 0 / 8 | 5.785e-02 | 1.199213e-15 | 绿(官方 L3 双臂同红,归一口径过) |
| msc10848 | m2 | 6185646 | 3423 / 0 | PASS col_mismatch=0 unordered=0 text_diff=2703535 worst_rel_vs_global=3.799e-16 | 0 / 0 / 7 | 5.785e-02 | 1.383707e-15 | 绿(官方 L3 双臂同红,归一口径过) |
| s3dkq4m2 | off | 13282553 | 0 / 0 | - | 0 / 0 / 88468 | 1.311e+05 | 6.798740e-16 | 对照 |
| s3dkq4m2 | m1 | 13282553 | 88410 / 0 | PASS col_mismatch=0 unordered=0 text_diff=4479943 worst_rel_vs_global=4.853e-16 | 0 / 0 / 88608 | 1.311e+05 | 6.836041e-16 | 绿(官方 L3 双臂同红,归一口径过) |
| s3dkq4m2 | m2 | 13282553 | 88410 / 0 | PASS col_mismatch=0 unordered=0 text_diff=5265832 worst_rel_vs_global=4.853e-16 | 0 / 0 / 88633 | 1.311e+05 | 7.547422e-16 | 绿(官方 L3 双臂同红,归一口径过) |

- 官方判错条目的并排清单(行/列/want/Σ|乘积|/各臂 got、|差|、rel、nerr):s4_l3diag_<阵>.txt 与 .csv;官方三判原文尾:s4_truth_<阵>_<臂>.txt

## S4b 值级:HH_BM2=1/2 对 =0 的差分门 + 三臂对 scipy 真值的归一误差(不跑官方逐行三判:这几阵 4-6 千万 nnz,python 逐行太慢)

判法:差分门 PASS ∧ nnz 同 off ∧(l3_diag 有输出时)该臂 nerr 最大值 ≤ 1e-10。

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) | 官方口径错行(仅参考) | nerr 最大 | 本作业判 |
|---|---|---|---|---|---|---|---|
| TSOPF_RS_b678_c2 | off | 40139105 | 0 / 0 | - | 0 | 4.399041e-16 | 对照 |
| TSOPF_RS_b678_c2 | m1 | 40139105 | 17002 / 0 | PASS col_mismatch=0 unordered=0 text_diff=779350 worst_rel_vs_global=1.987e-18 | 0 | 4.399041e-16 | 绿 |
| TSOPF_RS_b678_c2 | m2 | 40139105 | 17002 / 0 | PASS col_mismatch=0 unordered=0 text_diff=759905 worst_rel_vs_global=1.987e-18 | 0 | 4.399041e-16 | 绿 |
| web-Google | off | 29710164 | 0 / 0 | - | 0 | 0.000000e+00 | 对照 |
| web-Google | m1 | 29710164 | 0 / 46058 | PASS col_mismatch=0 unordered=0 text_diff=0 worst_rel_vs_global=0.000e+00 | 0 | 0.000000e+00 | 绿 |
| web-Google | m2 | 29710164 | 0 / 46058 | PASS col_mismatch=0 unordered=0 text_diff=0 worst_rel_vs_global=0.000e+00 | 0 | 0.000000e+00 | 绿 |
| nd3k | off | 18491738 | 0 / 0 | - | 2 | 1.099738e-15 | 对照 |
| nd3k | m1 | 18491738 | 0 / 0 | PASS col_mismatch=0 unordered=0 text_diff=8948192 worst_rel_vs_global=5.115e-16 | 0 | 1.104094e-15 | 绿 |
| nd3k | m2 | 18491738 | 0 / 0 | PASS col_mismatch=0 unordered=0 text_diff=8934839 worst_rel_vs_global=6.394e-16 | 1 | 1.099738e-15 | 绿 |
| nd6k | off | 42198384 | 0 / 0 | - | 1 | 1.623558e-15 | 对照 |
| nd6k | m1 | 42198384 | 0 / 0 | PASS col_mismatch=0 unordered=0 text_diff=22609773 worst_rel_vs_global=6.047e-16 | 2 | 1.849564e-15 | 绿 |
| nd6k | m2 | 42198384 | 0 / 0 | PASS col_mismatch=0 unordered=0 text_diff=22612852 worst_rel_vs_global=7.256e-16 | 1 | 1.743201e-15 | 绿 |

- 本轮值级里走两遍式的行数合计(m1+m2): 274124
- S4(a+b): green

## S5 三臂电池(同一二进制 spgemm_bm2;off=HH_BM2 关,m1=值区 SMEM,m2=值直写全局;5 轮交错)
- 卡状态(S5 前): 0, 14 MiB, 0 %;1, 14 MiB, 0 %; 计算进程数: 0
- ab_arms rc=0;逐阵表(含两遍式 flop 占比/行占比、accumulate 相位比)见 s5_battery.txt,逐发 s5_battery.csv,逐 bin 普查 s5_battery_stat.csv
    ===== 判读:m1 / off(逐阵中位比;测量分级)=====
    GM(m1/off) = 1.0110 | 非微级最差 = 1.1872 | nnz 异常/实验臂缺数据阵 = 0 | 微级 4 阵 | 有效 26 阵
    裁定: 红
    ===== 判读:m2 / off(逐阵中位比;测量分级)=====
    GM(m2/off) = 1.0130 | 非微级最差 = 1.2541 | nnz 异常/实验臂缺数据阵 = 0 | 微级 4 阵 | 有效 26 阵
    裁定: 红
- 卡状态(S5 后): 0, 14 MiB, 1 %;1, 14 MiB, 0 %; 计算进程数: 0

## S6 行门判别(dup 轴 HH_BM2_DUP=flop×100/est 下限;密度轴 HH_BM2_SPANX=span/est 上限;只用 data/train_disp)
- data/train_disp: 共 213 个 .mtx;8MB-800MB 且不与测试账同名的取到 7 个
- 用阵: hcircuit g7jac120sc bcircuit af23560 g7jac100 graham1 mark3jac140sc 
- ab_arms rc=0;见 s6_gates.txt / .csv / _stat.csv
    ===== 判读:m1 / off(逐阵中位比;测量分级)=====
    GM(m1/off) = 0.9853 | 非微级最差 = 1.0332 | nnz 异常/实验臂缺数据阵 = 0 | 微级 6 阵 | 有效 7 阵
    ===== 判读:m1d250 / off(逐阵中位比;测量分级)=====
    GM(m1d250/off) = 1.0171 | 非微级最差 = 0.9909 | nnz 异常/实验臂缺数据阵 = 0 | 微级 6 阵 | 有效 7 阵
    ===== 判读:m1sx32 / off(逐阵中位比;测量分级)=====
    GM(m1sx32/off) = 0.9661 | 非微级最差 = 0.9798 | nnz 异常/实验臂缺数据阵 = 0 | 微级 6 阵 | 有效 7 阵
    ===== 判读:m2 / off(逐阵中位比;测量分级)=====
    GM(m2/off) = 0.9653 | 非微级最差 = 0.9878 | nnz 异常/实验臂缺数据阵 = 0 | 微级 6 阵 | 有效 7 阵
    ===== 判读:m2d250 / off(逐阵中位比;测量分级)=====
    GM(m2d250/off) = 1.0363 | 非微级最差 = 0.9854 | nnz 异常/实验臂缺数据阵 = 0 | 微级 6 阵 | 有效 7 阵
    ===== 判读:m2sx32 / off(逐阵中位比;测量分级)=====
    GM(m2sx32/off) = 0.9746 | 非微级最差 = 1.0174 | nnz 异常/实验臂缺数据阵 = 0 | 微级 6 阵 | 有效 7 阵

## S7 无侵入(A=spgemm_base:本提交反打补丁后的基线源码;B=spgemm_bm2 默认 env=开关关;官方 ab_pair.py 16×3)
- 基线源码 sha256 核对通过
- make TARGET=spgemm_base: rc=0,用时 53s;审计: TU-90 模板实例: 26 个;TU-61: 51 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;
    GM(B/A) = 0.9918 | 最差 = 1.0342 | nnz 异常 = 0
    微级(不作单阵裁决): arc130, 1138_bus, bcsstk17, F2 — 单阵结论无效,如需判决改 10× 交错
    裁定: 绿-进下批(≤2% 带内)
- 卡状态(S7 后): 0, 14 MiB, 2 %;1, 14 MiB, 0 %; 计算进程数: 0

## S8 值级·大阵差分门(翻 bin 家族;放在计时之后只为省等待时间,红了同样作数)

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |
|---|---|---|---|---|
| pre2 | m2 | 323445872 | 118675 / 26529 | PASS col_mismatch=0 unordered=0 text_diff=272061 worst_rel_vs_global=1.092e-27 |
