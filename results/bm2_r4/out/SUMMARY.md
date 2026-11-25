# bm2_r4 汇总(想法二第四轮:两遍式独立核 v3 + 行门标定 + 只评测电池)

## S0 环境
- 主机: H100-187  开始: 2026-11-25T02:12:22Z  快照: /home/xueyizhou/exp-runs/bm2_r4/code
- nvcc(PATH): /usr/bin/nvcc / Build cuda_12.0.r12.0/compiler.32267302_0
- python3: Python 3.12.3 / numpy 2.3.4 scipy 1.18.1
- 磁盘: /dev/sdb1        15T  8.4T  5.4T  61% /home
- **用卡: GPU 0**
- 开跑时两卡: 0, 14 MiB, 0 %;1, 14 MiB, 0 %;

## S1 微复现器
- src/hh_bm2.cuh 已变(2fc25346de6deabc2eda6d49739ba0627a976969896b861e3798517bab59339b),重跑 hh_repro5
- ALL-PASS: trials/cfg=40 seed=42 bad=0;ALL-PASS: trials/cfg=25 seed=2026 bad=0;
- 行数合计: rows=168480 ok=116709 ovf=7413 skip=44358 bad=0(9 档几何 × B16/值区模式/Pass1 来源 6 种组合)
- S1: green

## S2 构建
- make TARGET=spgemm_bm2: rc=2,用时 21s;审计: 
- **构建或审计失败**,日志尾:
src/main.cu(197): warning #177-D: variable "C_col_idx" was declared but never referenced

src/main.cu(198): warning #177-D: variable "C_val" was declared but never referenced

src/main.cu(241): warning #177-D: variable "C_row_ptr" was declared but never referenced

src/main.cu(242): warning #177-D: variable "C_col_idx" was declared but never referenced

src/main.cu(243): warning #177-D: variable "C_val" was declared but never referenced

src/main.cu(284): warning #177-D: variable "C_row_ptr" was declared but never referenced

src/main.cu(285): warning #177-D: variable "C_col_idx" was declared but never referenced

src/main.cu(286): warning #177-D: variable "C_val" was declared but never referenced

src/main.cu(326): warning #177-D: variable "C_row_ptr" was declared but never referenced

src/main.cu(327): warning #177-D: variable "C_col_idx" was declared but never referenced

src/main.cu(328): warning #177-D: variable "C_val" was declared but never referenced

nvcc -O3 -arch=sm_90 -std=c++14 -lineinfo   -Iinclude -I/usr/local/cuda/include -c src/spgemm_kernel_formulations.cu -o src/spgemm_kernel_formulations.o
nvcc -O3 -arch=sm_90 -std=c++14 -lineinfo   -Iinclude -I/usr/local/cuda/include -DHASH_TB=90 -c src/spgemm_kernel_hash.cu -o src/spgemm_kernel_hash.o
nvcc -O3 -arch=sm_90 -std=c++14 -lineinfo   -Iinclude -I/usr/local/cuda/include -c src/spgemm_adaptive.cu -o src/spgemm_adaptive.o
src/spgemm_adaptive.cu(26): warning #177-D: function "env_int" was declared but never referenced

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

src/spgemm_adaptive.cu(31): warning #177-D: function "env_double" was declared but never referenced

nvcc -O3 -arch=sm_90 -std=c++14 -lineinfo   -Iinclude -I/usr/local/cuda/include -c src/spgemm_merge.cu -o src/spgemm_merge.o
src/spgemm_kernel_hash.cu(4937): error: declaration of a member with the same name as its class

At end of source: error: expected a "}"
src/spgemm_kernel_hash.cu(4936): note #3196-D: to match this "{"

src/spgemm_kernel_hash.cu(6956): error: too many initializer values

src/spgemm_kernel_hash.cu(6960): error: class "HashAccumCtx" has no member "h_bm2c"

src/spgemm_kernel_hash.cu(6961): error: class "HashAccumCtx" has no member "rows2"

src/spgemm_kernel_hash.cu(6962): error: a nonstatic member reference must be relative to a specific object

src/spgemm_kernel_hash.cu(6963): error: class "HashAccumCtx" has no member "hh_used"

src/spgemm_kernel_hash.cu(6964): error: class "HashAccumCtx" has no member "hh_flip"

At end of source: error: expected a ";"

9 errors detected in the compilation of "src/spgemm_kernel_hash.cu".
make: *** [Makefile:24: src/spgemm_kernel_hash.o] Error 2
make: *** Waiting for unfinished jobs....
src/spgemm_kernel_formulations.cu(480): warning #177-D: function "dpi" was declared but never referenced

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

src/spgemm_kernel_formulations.cu(481): warning #177-D: function "dcpi" was declared but never referenced

- S2: red

## 结束
- 结论: 红:S2 构建/审计失败;后续阶段未跑
- 磁盘: /dev/sdb1        15T  8.4T  5.4T  61% /home
- 结束时间: 2026-11-25T02:13:17Z,总用时 0 分钟
