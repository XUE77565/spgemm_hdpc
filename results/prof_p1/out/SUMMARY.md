# prof_p1 汇总(全 428 影子:off / aa / cen / cm2;默认形态相位地图;上一轮的三项补账;ncu 机制计数)

## S0 环境
- 主机: H100-187  开始: 2026-11-26T12:04:34Z  快照: /home/xueyizhou/exp-runs/prof_p1/code  nproc=32
- nvcc(PATH): /usr/bin/nvcc / Build cuda_12.0.r12.0/compiler.32267302_0
- python3: Python 3.12.3 / numpy 2.3.4 scipy 1.18.1
- 磁盘: /dev/sdb1        15T  8.7T  5.1T  63% /home
- **用卡: GPU 0**
- 主机(开跑时): load=9.50 10.43 10.67 外来>50%CPU进程数=3 卡=0, 14 MiB, 0 %;1, 66323 MiB, 0 %;

## S1 微复现器
- src/hh_bm2.cuh 的 sha256 与 bm2_r4 / bm2_r5 的 S1 验过的相同(两种子 ALL-PASS,168480 行 bad=0)→ 沿用
- S1: green

## S2 构建
- make TARGET=spgemm_p1: rc=0,用时 61s;审计: TU-90 模板实例: 26 个;TU-61: 51 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;
- WRITE_MTX 件 spgemm_p1w: rc=0(只重编 main.cu)
- S2: green
- 基线源码件 spgemm_base:sha256 核对通过,make rc=0

## S3 冒烟(各臂 1 发:nnz 与失败检查)
- off/cen/cm2/m2,7 阵 → green;走两遍式的行数合计 = 97690

## P0 影子采数脚本试跑(清单里最小的 8 个阵 × 1 轮;只验脚本通不通,读数不用)
- shadow428 rc=0,shadow_sum rc=0;有效发 32 / 32;# shadow428: bin=./spgemm_p1 gpu=0 method=adaptive 清单 428 阵,找到 8,缺 0;轮数(小/中/大)=1,1,1 分界=10,200 MB
- # shadow_sum: 清单内 8 阵,基线臂有数据的 8 阵;失败发 0;臂: off aa cen cm2 # 分级:微 8 / 中 0 / 稳 0;nnz 各臂各轮不全同的阵 0 个;签名不全同的阵 0 个(有签名的 8 阵) 

## O1 发射清单比对(nsys;核名含模板实参 + 发射次数;基线源码件 base 对 本件开关全关 off)
