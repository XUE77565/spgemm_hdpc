# prof_k1 汇总(设备侧 profiling:nsys 逐 kernel + ncu;带负载门的 v3 电池与无侵入重测)

## S0 环境
- 主机: H100-187  开始: 2026-11-25T07:42:23Z  快照: /home/xueyizhou/exp-runs/prof_k1/code  nproc=32
- nvcc(PATH): /usr/bin/nvcc / Build cuda_12.0.r12.0/compiler.32267302_0
- python3: Python 3.12.3
- 磁盘: /dev/sdb1        15T  8.4T  5.4T  61% /home
- **用卡: GPU 0**
- 主机(开跑时): load=8.41 6.87 7.24 外来>50%CPU进程数=3 卡=0, 14 MiB, 0 %;1, 5097 MiB, 0 %;
- 负载门: load1 < 6 连续两次(间隔 60s)且无外来 >50% CPU 进程,且本卡无别的计算进程、另一张卡利用率 <10%;等待上限 600 分钟

## K1 构建
- make TARGET=spgemm_k1: rc=0,用时 56s;审计: TU-90 模板实例: 26 个;TU-61: 51 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;
- K1: green

## 负载门
- 开跑时主机不满足安静判据(load=9.01 7.33 7.38 外来>50%CPU进程数=3 卡=0, 14 MiB, 0 %;1, 5097 MiB, 0 %;)→ 先跑 K 段(不吃主机计时),之后等安静窗口

## K2 逐 kernel 的 GPU 侧累计时长(nsys;设备侧时间戳,不吃主机 CPU 争用)
- 主机(K2 前): load=9.01 7.33 7.38 外来>50%CPU进程数=3 卡=0, 14 MiB, 0 %;1, 5097 MiB, 0 %;
- nsys: /usr/local/bin/nsys / NVIDIA Nsight Systems version 2024.6.2.225-246235244400v0
- 试点通过:profile 选项 [-t cuda -s none --cpuctxsw=none --force-overwrite=true],报告 cuda_gpu_kern_sum(输出方式 stdout),试点表 35 行(表头样例在 k2_pilot_head.csv)
