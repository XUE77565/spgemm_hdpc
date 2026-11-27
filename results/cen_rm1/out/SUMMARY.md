# cen_rm1 汇总(影子待复测阵加轮复测;两遍式在生产入口下的值级;合一形态输家 / 赢家画像)

## S0 环境
- 主机: H100-187  开始: 2026-11-27T06:58:48Z  快照: /home/xueyizhou/exp-runs/cen_rm1/code  nproc=32
- nvcc(PATH): /usr/bin/nvcc / Build cuda_12.0.r12.0/compiler.32267302_0
- python3: Python 3.12.3 / numpy 2.3.4 scipy 1.18.1
- 磁盘: /dev/sdb1        15T  8.7T  5.1T  64% /home
- **用卡: GPU 0**
- 主机(开跑时): load=6.40 6.05 6.70 外来>50%CPU进程数=2 卡=0, 14 MiB, 0 %;1, 14 MiB, 0 %;

## S1 微复现器
- src/hh_bm2.cuh 的 sha256 与 bm2_r4 / bm2_r5 的 S1 验过的相同(两种子 ALL-PASS,168480 行 bad=0)→ 沿用
- S1: green

## S2 构建
- make TARGET=spgemm_m1: rc=0,用时 59s;审计: TU-90 模板实例: 26 个;TU-61: 51 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;
- WRITE_MTX 件 spgemm_m1w: rc=0(只重编 main.cu)
- S2: green

## S3 冒烟(各臂 1 发:nnz 与失败检查)
- off/cen/cm2,6 阵 → green

## R1 待复测阵加轮复测(off / aa / cen;每阵 21 发、逐阵逐轮交错;env 与影子、confirm20 相同:METHOD=adaptive WARMUP_N=3 USE_MEMPOOL=1 MP_HOST_MB=8192 D6EXP=0)
- 名单:影子里 A/A 超 ±5% 的 6 个不裁决阵、cen 臂 >1.10 的可裁决阵 rajat16、rajat 族其余成员,以及 cen 臂 >1.02 的非微级阵
- 主机(R1 前): load=8.08 7.24 7.05 外来>50%CPU进程数=2 卡=0, 14 MiB, 0 %;1, 14 MiB, 0 %;
