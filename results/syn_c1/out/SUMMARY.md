# syn_c1 汇总(两遍式行门 HH_BM2_FX 的标定:合成训练阵 + 真实训练池,生产入口逐行配对周期 → R=1 交点)

## S0 环境
- 主机: H100-187  开始: 2026-11-27T16:01:36Z  快照: /home/xueyizhou/exp-runs/syn_c1/code  nproc=32
- nvcc(PATH): /usr/bin/nvcc / Build cuda_12.0.r12.0/compiler.32267302_0
- python3: Python 3.12.3 / numpy 2.3.4 scipy 1.18.1
- 磁盘: /dev/sdb1        15T  8.7T  5.1T  64% /home
- **用卡: GPU 1(钉死,CUDA_VISIBLE_DEVICES=1;GPU 0 归 B 端的 confirm20,本作业的任何进程都看不到它)**
- 心跳文件: /home/xueyizhou/exp-runs/syn_c1/HEARTBEAT(每一发、每一步覆写;要判断是不是挂了看它的开始时刻,单发上限 900 秒、转储 1800 秒)
- 主机(开跑时): load=12.57 11.95 10.04 外来>50%CPU进程数=3 卡=0, 2802 MiB, 0 %;1, 14 MiB, 0 %;

## S1 微复现器
- src/hh_bm2.cuh 的 sha256 与 bm2_r4 / bm2_r5 的 S1 验过的相同(两种子 ALL-PASS,168480 行 bad=0)→ 沿用
- S1: green

## S2 构建
- make TARGET=spgemm_y1: rc=0,用时 71s;审计: TU-90 模板实例: 26 个;TU-61: 51 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;
- WRITE_MTX 件 spgemm_y1w: rc=0(只重编 main.cu)
- S2: green

## S3 冒烟(各臂 1 发:nnz 与失败检查)
- off/m2/cm2,4 阵 → green

## G0 合成训练阵(tools/train_synth/gen_synth.py;固定种子;只在作业目录里)
- 生成器 rc=0,用时 147s;得到 14 个阵,共 2.5G;清单 g0_manifest.tsv

## G1 合成阵逐行配对标定(METHOD=adaptive;off / m1=值在 SMEM / m2=值直写全局;同批划分)
