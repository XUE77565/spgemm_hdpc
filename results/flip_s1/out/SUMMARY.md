# flip_s1 汇总(HH_CENSUS 翻默认后的自检:默认形态确实走单发射普查,产物与两种显式设置一致)

## S0 环境
- 主机: H100-187  开始: 2026-11-27T15:24:39Z  快照: /home/xueyizhou/exp-runs/flip_s1/code  nproc=32
- nvcc(PATH): /usr/bin/nvcc / Build cuda_12.0.r12.0/compiler.32267302_0
- python3: Python 3.12.3 / numpy 2.3.4 scipy 1.18.1
- 磁盘: /dev/sdb1        15T  8.7T  5.1T  64% /home
- **用卡: GPU 0**
- 主机(开跑时): load=6.21 7.08 7.40 外来>50%CPU进程数=2 卡=0, 14 MiB, 0 %;1, 14 MiB, 0 %;

## S1 微复现器
- src/hh_bm2.cuh 的 sha256 与 bm2_r4 / bm2_r5 的 S1 验过的相同(两种子 ALL-PASS,168480 行 bad=0)→ 沿用
- S1: green

## S2 构建
- make TARGET=spgemm_f1: rc=0,用时 55s;审计: TU-90 模板实例: 26 个;TU-61: 51 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;
- WRITE_MTX 件 spgemm_f1w: rc=0(只重编 main.cu)
- S2: green

## C1 冒烟(def=不设 / c0=HH_CENSUS=0 / c1=HH_CENSUS=1;各 1 发)
- 6 阵 × 3 臂:nnz 与失败检查 → green

## C2 门普查签名与 nnz(三臂逐阵相同?)
- 比对 44 阵(清单最小的 40 个 + bcsstk17 / cfd2 / web-Google / ldoor):签名或 nnz 三臂不全同的 0 阵;没凑齐三臂的 0 阵;失败发 0
- C2: green

## C3 发射清单(nsys;核名 + 发射次数)
- def 对 c1(显式 HH_CENSUS=1):比对 4 阵,same(same = 默认形态的发射清单与显式开完全一致)
- def 对 c0(显式 HH_CENSUS=0)的差异项(应只有门普查核:原样的 hh_gate_census_off_kernel 多、单发射的 hh_gate_census1_kernel 少):
    == bcsstk17
            2 → 0      hh_gate_census1_kernel(const int *, const int *, const int *, const int *, long long *)
            0 → 12     hh_gate_census_off_kernel(const int *, const int *, int, int, const int *, const int *, unsigned int, long long *)
    == cfd2
            2 → 0      hh_gate_census1_kernel(const int *, const int *, const int *, const int *, long long *)
            0 → 12     hh_gate_census_off_kernel(const int *, const int *, int, int, const int *, const int *, unsigned int, long long *)
    == ldoor
            2 → 0      hh_gate_census1_kernel(const int *, const int *, const int *, const int *, long long *)
            0 → 12     hh_gate_census_off_kernel(const int *, const int *, int, int, const int *, const int *, unsigned int, long long *)
    == web-Google
            2 → 0      hh_gate_census1_kernel(const int *, const int *, const int *, const int *, long long *)
            0 → 12     hh_gate_census_off_kernel(const int *, const int *, int, int, const int *, const int *, unsigned int, long long *)
- def 对 c0:门普查核之外的差异项 0 条;hh_gate_census1_kernel 出现在差异里的阵次 4
- C3: green

## 结束
- 结论: 绿:默认形态走单发射普查(发射清单与显式 HH_CENSUS=1 相同),签名与 nnz 三臂一致
- 主机(结束时): load=9.85 8.71 8.23 外来>50%CPU进程数=3 卡=0, 14 MiB, 0 %;1, 10970 MiB, 0 %;
- 磁盘: /dev/sdb1        15T  8.7T  5.1T  64% /home
- 结束时间: 2026-11-27T15:46:23Z,总用时 21 分钟
