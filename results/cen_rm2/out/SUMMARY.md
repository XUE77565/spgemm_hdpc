# cen_rm2 汇总(影子待复测阵:上一轮没测完的部分补完;每一发记墙钟秒数)

## S0 环境
- 主机: H100-187  开始: 2026-11-27T13:06:28Z  快照: /home/xueyizhou/exp-runs/cen_rm2/code  nproc=32
- nvcc(PATH): /usr/bin/nvcc / Build cuda_12.0.r12.0/compiler.32267302_0
- python3: Python 3.12.3 / numpy 2.3.4 scipy 1.18.1
- 磁盘: /dev/sdb1        15T  8.7T  5.1T  64% /home
- **用卡: GPU 0**
- 主机(开跑时): load=9.29 7.80 7.54 外来>50%CPU进程数=2 卡=0, 14 MiB, 0 %;1, 14 MiB, 0 %;

## S1 微复现器
- src/hh_bm2.cuh 的 sha256 与 bm2_r4 / bm2_r5 的 S1 验过的相同(两种子 ALL-PASS,168480 行 bad=0)→ 沿用
- S1: green

## S2 构建
- make TARGET=spgemm_m2: rc=0,用时 59s;审计: TU-90 模板实例: 26 个;TU-61: 51 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;
- WRITE_MTX 件 spgemm_m2w: rc=0(只重编 main.cu)
- S2: green

## S3 冒烟(各臂 1 发:nnz 与失败检查)
- off/cen,4 阵 → green

## R1a 补测 A 组(rajat25 / Zd_Jac6_db / rajat20 / rajat31;off / aa / cen;21 轮;单发上限 600 秒,整段预算 270 分钟)
- 主机(R1a 前): load=8.57 7.99 7.64 外来>50%CPU进程数=2 卡=0, 14 MiB, 2 %;1, 14 MiB, 0 %;;心跳文件 /home/xueyizhou/exp-runs/cen_rm2/rm/r1a.hb
