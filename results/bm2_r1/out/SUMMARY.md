# bm2_r1 汇总(想法二 HH_BM2 第一轮)

## S0 环境
- 主机: H100-187  开始: 2026-11-24T16:52:41Z  快照: /home/xueyizhou/exp-runs/bm2_r1/code
- nvcc(PATH,构建用这个,与裸 make 相同): /usr/bin/nvcc / Build cuda_12.0.r12.0/compiler.32267302_0
- python3 scipy: 有
- **用卡: GPU 0**
- 开跑时两卡: 0, 14 MiB, 0 %;1, 14 MiB, 0 %;

## S1 微复现器(tools/hh_repro5.cu,编进去的是生产头 src/hh_bm2.cuh)
- 种子 42×40 / 种子 2026×25: ALL-PASS: trials/cfg=40 seed=42 bad=0;ALL-PASS: trials/cfg=25 seed=2026 bad=0;
- 行数合计: rows=112320 ok=77806 ovf=4942 skip=29572 bad=0
- S1: green

## S2 构建
- make TARGET=spgemm_bm2: rc=0,用时 59s
- 审计: ✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;
- WRITE_MTX 件 spgemm_bm2w: rc=0,用时 3s(只重编 main.cu)
- S2: green

## S3 冒烟
- 小阵 off/m1/m2 各 1 发:nnz 与失败检查 → green;普查走两遍式的行数合计(m1+m2)= 97310

## S4 值级(HH_BM2=1/2 对 HH_BM2=0 的 mtx_gate_fast;scipy 在则加 verify_sorted_truth 三判)

| 阵 | 模式 | nnz | 走两遍式行/未走行(HH bin 合计) | gate(vs off) | 三判(vs scipy 真值) |
|---|---|---|---|---|---|
| bcsstk17 | m1 | 1406936 | 9201 / 0 | PASS(col_mismatch=0 unordered=0 text_diff=481051 worst_rel_vs_global=2.737e-16) | 判定: FAIL(L1 集合错行: 0   (零抵消软判 benign: 2) L2 乱序行:   0 L3 数值错行: 2978   全局最大相对误差: 3.579e-02 ) |
| bcsstk17 | m2 | 1406936 | 9201 / 0 | PASS(col_mismatch=0 unordered=0 text_diff=506093 worst_rel_vs_global=2.737e-16) | 判定: FAIL(L1 集合错行: 0   (零抵消软判 benign: 2) L2 乱序行:   0 L3 数值错行: 2977   全局最大相对误差: 3.947e-01 ) |
| af23560 | m1 | 2432504 | 19026 / 0 | PASS(col_mismatch=0 unordered=0 text_diff=222744 worst_rel_vs_global=2.342e-16) | 判定: PASS(L1 集合错行: 0   (零抵消软判 benign: 6258) L2 乱序行:   0 L3 数值错行: 0   全局最大相对误差: 3.458e-10 ) |
| af23560 | m2 | 2432504 | 19026 / 0 | PASS(col_mismatch=0 unordered=0 text_diff=218950 worst_rel_vs_global=2.342e-16) | 判定: PASS(L1 集合错行: 0   (零抵消软判 benign: 6258) L2 乱序行:   0 L3 数值错行: 0   全局最大相对误差: 3.458e-10 ) |
| cfd2 | m1 | 13418644 | 90070 / 0 | PASS(col_mismatch=0 unordered=0 text_diff=1749232 worst_rel_vs_global=4.353e-16) | 判定: PASS(L1 集合错行: 0   (零抵消软判 benign: 2595) L2 乱序行:   0 L3 数值错行: 0   全局最大相对误差: 1.261e-11 ) |
| cfd2 | m2 | 13418644 | 90070 / 0 | PASS(col_mismatch=0 unordered=0 text_diff=1735415 worst_rel_vs_global=3.265e-16) | 判定: PASS(L1 集合错行: 0   (零抵消软判 benign: 2595) L2 乱序行:   0 L3 数值错行: 0   全局最大相对误差: 1.261e-11 ) |
| s3dkq4m2 | m1 | 13282553 | 88410 / 0 | PASS(col_mismatch=0 unordered=0 text_diff=4484437 worst_rel_vs_global=4.853e-16) | 判定: FAIL(L1 集合错行: 0   (零抵消软判 benign: 267) L2 乱序行:   0 L3 数值错行: 88620   全局最大相对误差: 6.554e+04 ) |
| s3dkq4m2 | m2 | 13282553 | 88410 / 0 | PASS(col_mismatch=0 unordered=0 text_diff=5247095 worst_rel_vs_global=6.471e-16) | 判定: FAIL(L1 集合错行: 0   (零抵消软判 benign: 267) L2 乱序行:   0 L3 数值错行: 88663   全局最大相对误差: 1.311e+05 ) |
| msc10848 | m1 | 6185646 | 3423 / 0 | PASS(col_mismatch=0 unordered=0 text_diff=2658441 worst_rel_vs_global=1.900e-16) | 判定: FAIL(L1 集合错行: 0   (零抵消软判 benign: 0) L2 乱序行:   0 L3 数值错行: 9   全局最大相对误差: 5.785e-02 ) |
| msc10848 | m2 | 6185646 | 3423 / 0 | PASS(col_mismatch=0 unordered=0 text_diff=2702206 worst_rel_vs_global=1.900e-16) | 判定: FAIL(L1 集合错行: 0   (零抵消软判 benign: 0) L2 乱序行:   0 L3 数值错行: 9   全局最大相对误差: 5.785e-02 ) |
| bcsstk30 | m1 | 8946070 | 21375 / 0 | PASS(col_mismatch=0 unordered=0 text_diff=0 worst_rel_vs_global=0.000e+00) | 判定: PASS(L1 集合错行: 0   (零抵消软判 benign: 0) L2 乱序行:   0 L3 数值错行: 0   全局最大相对误差: 0.000e+00 ) |
| bcsstk30 | m2 | 8946070 | 21375 / 0 | PASS(col_mismatch=0 unordered=0 text_diff=0 worst_rel_vs_global=0.000e+00) | 判定: PASS(L1 集合错行: 0   (零抵消软判 benign: 0) L2 乱序行:   0 L3 数值错行: 0   全局最大相对误差: 0.000e+00 ) |
| c-64 | m1 | 155006041 | 3236 / 380 | PASS(col_mismatch=0 unordered=0 text_diff=8420396 worst_rel_vs_global=1.383e-16) | 判定: PASS(L1 集合错行: 0   (零抵消软判 benign: 10254) L2 乱序行:   0 L3 数值错行: 0   全局最大相对误差: 2.148e-11 ) |
