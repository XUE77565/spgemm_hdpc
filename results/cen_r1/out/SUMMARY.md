# cen_r1 汇总(门普查单发射版与普查+划分合一:行为保持 → 设备侧时长 → A/A 认证的电池、小阵探针、带对照臂的无侵入)

## S0 环境
- 主机: H100-187  开始: 2026-11-26T00:36:14Z  快照: /home/xueyizhou/exp-runs/cen_r1/code  nproc=32
- nvcc(PATH): /usr/bin/nvcc / Build cuda_12.0.r12.0/compiler.32267302_0
- python3: Python 3.12.3 / numpy 2.3.4 scipy 1.18.1
- 磁盘: /dev/sdb1        15T  8.6T  5.2T  63% /home
- **用卡: GPU 0**
- 主机(开跑时): load=12.81 9.52 9.75 外来>50%CPU进程数=4 卡=0, 14 MiB, 0 %;1, 14 MiB, 0 %;

## S1 微复现器
- src/hh_bm2.cuh 的 sha256 与 bm2_r4 / bm2_r5 的 S1 验过的相同(两种子 ALL-PASS,168480 行 bad=0)→ 沿用
- S1: green

## S2 构建
- make TARGET=spgemm_c1: rc=0,用时 54s;审计: TU-90 模板实例: 26 个;TU-61: 51 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;
- WRITE_MTX 件 spgemm_c1w: rc=0(只重编 main.cu)
- S2: green
- 基线源码件 spgemm_base:sha256 核对通过,make rc=0;审计: TU-90 模板实例: 26 个;TU-61: 51 个;✓ 实例集不相交(探针P2 风险解除);✓ opt-in 全覆盖;

## S3 冒烟(各臂 1 发:nnz 与失败检查;单发的时间比不作读数)
- G1 两遍式(宏重构后的回归)off/m2/m2p/m2L/m2e,6 阵 → green;走两遍式的行数合计 = 317379
- G2 单发射普查 off/cen/cenm2,7 阵 → green;走两遍式的行数合计 = 48845
- G3 普查 + 划分合一 off/cm2/cm2L,7 阵 → green;走两遍式的行数合计 = 219689

## S4a 门普查行为保持([census-stat] 行 = 每个 bin 的 rows / gated / total,连同结果 nnz;三种模式逐行比对)
- 比对了 24 阵(另有 0 阵基线模式没跑出结果、0 阵基线模式自己两遍就不同,都不计入);其中非空的 (阵,bin) 组合 75 个
- **不一致的阵: 0 个**
- 合一形态已一并比对

## S4b 值级:各臂对 off 的差分门

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |
|---|---|---|---|---|
| bcsstk17 | m2 | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=507408 worst_rel_vs_global=1.824e-16 |
| bcsstk17 | m2e | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=504754 worst_rel_vs_global=1.824e-16 |
| bcsstk17 | cen | 1406936 | 0 / 0 | PASS col_mismatch=0 unordered=0 text_diff=372101 worst_rel_vs_global=1.824e-16 |
| bcsstk17 | cm2 | 1406936 | 9201 / 0 | PASS col_mismatch=0 unordered=0 text_diff=505667 worst_rel_vs_global=1.824e-16 |
| cfd2 | m2 | 13418644 | 90070 / 0 | PASS col_mismatch=0 unordered=0 text_diff=1799057 worst_rel_vs_global=4.353e-16 |
| cfd2 | cen | 13418644 | 0 / 0 | PASS col_mismatch=0 unordered=0 text_diff=1368125 worst_rel_vs_global=3.265e-16 |
| cfd2 | cm2 | 13418644 | 90070 / 0 | PASS col_mismatch=0 unordered=0 text_diff=1795979 worst_rel_vs_global=4.353e-16 |
| msc10848 | cm2 | 6185646 | 3423 / 0 | PASS col_mismatch=0 unordered=0 text_diff=2702503 worst_rel_vs_global=1.900e-16 |
| GaAsH6 | cm2 | 37611445 | 25720 / 0 | PASS col_mismatch=0 unordered=0 text_diff=3916538 worst_rel_vs_global=9.158e-16 |
| web-Google | cen | 29710164 | 0 / 0 | PASS col_mismatch=0 unordered=0 text_diff=0 worst_rel_vs_global=0.000e+00 |
| pre2 | cen | 323445872 | 0 / 0 | PASS col_mismatch=0 unordered=0 text_diff=246781 worst_rel_vs_global=1.092e-27 |
| pre2 | cm2L | 323445872 | 121999 / 23205 | PASS col_mismatch=0 unordered=0 text_diff=278385 worst_rel_vs_global=1.501e-27 |

- S4b: green;走两遍式的行数合计 358885(GaAsH6 是 bin9 有行的阵;cm2L 是合一 + legacy 形态,只验值)

## K2 逐 kernel 的 GPU 侧累计时长(nsys;设备侧时间戳,不吃主机 CPU 争用)
- nsys: NVIDIA Nsight Systems version 2024.6.2.225-246235244400v0;臂: off cen m2 cm2 ;22 阵 × 2 轮,WARMUP_N=1
- 采数:176 发,失败 0 发
- kern_sum rc=0;全文 k2_kern.txt,长表 k2_kern_long.csv.gz

```
# kern_sum: 读到 176 个文件,有表的 176 个,读不出表的 0 个
# nsys 表头: Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name

===== ① 逐阵:Σkernel GPU 时长(ms,各轮中位;含预热轮)与对 off 之比;括号里是两遍式相关核族(hash_heavy_kernel + hash_bm2_kernel + hh_bm2_part_kernel)的比 =====
mat                    off_ms    轮间%            cen/off(核族比)            cm2/off(核族比)             m2/off(核族比)
Bump_2911             202.298    0.6        0.9593 (0.9997)         0.9579 (0.9958)         0.9998 (0.9997) 
FEM_3D_thermal2         9.475    0.0        0.9846 (1.0000)         0.9545 (0.8312)*        0.9692 (0.8396)*
Hook_1498             101.490    0.5        0.9636 (1.0005)         0.9630 (0.9971)         0.9972 (0.9964) 
StocF-1465             26.719    0.1        0.9956 (0.9864)         0.9959 (0.9938)         1.0017 (0.9965) 
TSOPF_RS_b678_c2       12.669    0.3        0.9948 (1.0019)         0.9100 (0.8687)*        0.9153 (0.8684)*
af23560                 0.918    0.0        0.9492 (1.0035)         0.7790 (0.6330)*        0.8282 (0.6542)*
bcsstk17                1.103    0.3        0.9791 (1.0031)         0.9020 (0.8267)*        0.9314 (0.8445)*
bcsstk30                4.266    0.4        0.9840 (0.9982)         0.8813 (0.7989)*        0.8914 (0.7975)*
bloweybq               24.608    0.1           0.9992 (nan)            0.9992 (nan)            0.9965 (nan) 
c-64                   35.239    0.1        0.9935 (0.9741)         0.9994 (0.7457)*        1.0077 (0.7509)*
c-big                 287.953    0.9        1.0225 (0.9942)         0.9978 (0.9937)         1.0119 (0.9928) 
cage15                239.810    0.0        0.9433 (1.0000)         0.9441 (0.9999)         0.9999 (1.0000) 
case9                  16.638    0.1           0.9992 (nan)            1.0001 (nan)            0.9996 (nan) 
cfd2                    4.417    0.2        0.9633 (0.9987)         0.7985 (0.6868)*        0.8335 (0.6968)*
ldoor                  55.775    0.1        0.9496 (1.0000)         0.9502 (0.9995)         0.9998 (0.9997) 
msc10848                3.807    0.8        0.9861 (0.9932)         0.9698 (0.9437)*        0.9551 (0.9680)*
nlpkkt80               48.902    0.0        0.9334 (0.9994)         0.9354 (0.9996)         0.9991 (0.9997) 
offshore                6.095    0.0        0.9648 (1.0003)         0.9665 (0.9948)*        1.0006 (0.9983)*
pre2                   48.178    0.1        0.9829 (1.0014)         0.9832 (1.0004)         1.0005 (1.0018) 
rgg_n_2_21_s0          31.212    0.1        0.9979 (1.0098)         0.9978 (0.7977)*        1.0004 (1.0160)*
s3dkq4m2                5.332    0.1        0.9715 (1.0003)         0.9872 (1.0197)*        1.0156 (1.0245)*
web-Google             13.220    0.1        0.9857 (0.9996)         0.9873 (0.9939)         0.9996 (0.9928) 
(* = 该臂在这阵上确有 hash_bm2_kernel 发射)
GM_KERN(cen/off) = 0.9772 | 阵数 22 | 最差 1.0225 | 有两遍式发射的阵 0 个,其 GM = -
GM_KERN(cm2/off) = 0.9461 | 阵数 22 | 最差 1.0001 | 有两遍式发射的阵 11 个,其 GM = 0.9193
GM_KERN(m2/off) = 0.9690 | 阵数 22 | 最差 1.0156 | 有两遍式发射的阵 11 个,其 GM = 0.9385

```
- 比值是设备侧 Σkernel 时长之比(含预热轮,各臂相同),不含主机侧的发射与同步开销。门普查核族在各臂的累计时长见 k2_kern.txt 的 ②
- CUDA API 调用次数(6 个小阵,每臂一发 = 预热 1 次 + 计时 1 次;k2_api_launch.txt,表头样例 k2_api_sample.csv):
    bcsstk17 off 发射=92 流同步=26 设备同步=1 同步拷贝=0
    bcsstk17 cen 发射=0 流同步=0 设备同步=0 同步拷贝=0
    bcsstk17 m2 发射=94 流同步=26 设备同步=1 同步拷贝=0
    bcsstk17 cm2 发射=0 流同步=0 设备同步=0 同步拷贝=0
    af23560 off 发射=0 流同步=0 设备同步=0 同步拷贝=0
    af23560 cen 发射=56 流同步=26 设备同步=1 同步拷贝=0
    af23560 m2 发射=0 流同步=0 设备同步=0 同步拷贝=0
    af23560 cm2 发射=56 流同步=26 设备同步=1 同步拷贝=0
    msc10848 off 发射=0 流同步=0 设备同步=0 同步拷贝=0
    msc10848 cen 发射=66 流同步=26 设备同步=1 同步拷贝=0
    msc10848 m2 发射=0 流同步=0 设备同步=0 同步拷贝=0
    msc10848 cm2 发射=0 流同步=0 设备同步=0 同步拷贝=0
    cfd2 off 发射=0 流同步=0 设备同步=0 同步拷贝=0
    cfd2 cen 发射=0 流同步=0 设备同步=0 同步拷贝=0
    cfd2 m2 发射=0 流同步=0 设备同步=0 同步拷贝=0
    cfd2 cm2 发射=0 流同步=0 设备同步=0 同步拷贝=0
    web-Google off 发射=0 流同步=0 设备同步=0 同步拷贝=0
    web-Google cen 发射=0 流同步=0 设备同步=0 同步拷贝=0
    web-Google m2 发射=0 流同步=0 设备同步=0 同步拷贝=0
    web-Google cm2 发射=0 流同步=0 设备同步=0 同步拷贝=0
    case9 off 发射=0 流同步=0 设备同步=0 同步拷贝=0
    case9 cen 发射=0 流同步=0 设备同步=0 同步拷贝=0
    case9 m2 发射=60 流同步=20 设备同步=1 同步拷贝=0
    case9 cm2 发射=0 流同步=0 设备同步=0 同步拷贝=0

## K3 ncu 探查(bm2_t1 里 /usr/bin/ncu 2022.4.1 四种组合都报 Failed to prepare kernel for profiling)
- 驱动: 570.195.03;nsys: NVIDIA Nsight Systems version 2024.6.2.225-246235244400v0
- /opt/nvidia/nsight-compute/2025.1.1/ncu → /opt/nvidia/nsight-compute/2025.1.1/ncu(Version 2025.1.1.0 (build 35528883) (public-release)):试点 rc=0,**出了报告**
- 可用的 ncu: /opt/nvidia/nsight-compute/2025.1.1/ncu(application 重放)。机制计数留到下一个作业做(本作业不再加长)

## T0 负载门
- 等了 15 分钟,主机仍不满足安静判据(load=7.59 9.41 10.81 外来>50%CPU进程数=2 卡=0, 14 MiB, 0 %;1, 14 MiB, 0 %;)。照跑:计时有效性由 A/A 对照臂认证,判据不过整段作废

## T1 只评测电池(同一二进制;臂: off aa cen m2 cm2 ;24 阵 × 7 轮交错)
- 主机(T1 前): load=7.59 9.41 10.81 外来>50%CPU进程数=2 卡=0, 14 MiB, 0 %;1, 14 MiB, 0 %;
- ab_arms rc=0;逐阵表 t1_battery.txt,逐发 t1_battery.csv,逐 bin 普查 t1_battery_stat.csv
- 段内主机负载: load1 中位 10.05 / 最大 22.77(200 个采样,每分钟一次;其中 200 次见到外来 >50% CPU 进程)
- A/A 对照臂: AA_N 24;AA_GM 0.9960;AA_NONMICRO_MIN 0.9767;AA_NONMICRO_MAX 1.0212;AA_VERDICT clean;(逐阵见 t1_aa.txt)
- **T1 计时判定: A/A 认证通过**(下面的 GM 与裁定有效)
    ===== 判读:aa / off(逐阵中位比;测量分级)=====
    GM(aa/off) = 0.9960 | 非微级最差 = 1.0212 | nnz 异常/实验臂缺数据阵 = 0 | 微级 4 阵 | 有效 24 阵
    裁定: 绿(GM≤1.02 且无非微级阵>1.10 且 nnz 全同)
    ===== 判读:cen / off(逐阵中位比;测量分级)=====
    GM(cen/off) = 0.9706 | 非微级最差 = 1.0221 | nnz 异常/实验臂缺数据阵 = 0 | 微级 4 阵 | 有效 24 阵
    裁定: 绿(GM≤1.02 且无非微级阵>1.10 且 nnz 全同)
    ===== 判读:m2 / off(逐阵中位比;测量分级)=====
    GM(m2/off) = 0.9725 | 非微级最差 = 1.0229 | nnz 异常/实验臂缺数据阵 = 0 | 微级 4 阵 | 有效 24 阵
    裁定: 绿(GM≤1.02 且无非微级阵>1.10 且 nnz 全同)
    ===== 判读:cm2 / off(逐阵中位比;测量分级)=====
    GM(cm2/off) = 0.9605 | 非微级最差 = 1.0378 | nnz 异常/实验臂缺数据阵 = 0 | 微级 4 阵 | 有效 24 阵
    裁定: 绿(GM≤1.02 且无非微级阵>1.10 且 nnz 全同)

## T1b 小阵探针(臂: off aa cen cm2 ;25 轮交错;看用不上两遍式的小阵上少 5 次发射值多少、合一形态有没有新增固定成本)
- 第 1 次:ab_arms rc=0(11 阵);逐阵表 t1b_probe1.txt,逐发 t1b_probe1.csv
- 段内主机负载: load1 中位 9.47 / 最大 18.54(116 个采样,每分钟一次;其中 116 次见到外来 >50% CPU 进程)
- A/A 对照臂: AA_N 11;AA_GM 1.0294;AA_NONMICRO_MIN nan;AA_NONMICRO_MAX nan;AA_VERDICT polluted;
    ===== 判读:aa / off(逐阵中位比;测量分级)=====
    GM(aa/off) = 1.0294 | 非微级最差 = 0.0000 | nnz 异常/实验臂缺数据阵 = 0 | 微级 11 阵 | 有效 11 阵
    ===== 判读:cen / off(逐阵中位比;测量分级)=====
    GM(cen/off) = 0.9666 | 非微级最差 = 0.0000 | nnz 异常/实验臂缺数据阵 = 0 | 微级 11 阵 | 有效 11 阵
    ===== 判读:cm2 / off(逐阵中位比;测量分级)=====
    GM(cm2/off) = 0.9600 | 非微级最差 = 0.0000 | nnz 异常/实验臂缺数据阵 = 0 | 微级 11 阵 | 有效 11 阵
- **T1b 第 1 次: A/A 判据未过 → 这一次整段作废**,整段重跑一次
- 第 2 次:ab_arms rc=0(11 阵);逐阵表 t1b_probe2.txt,逐发 t1b_probe2.csv
- 段内主机负载: load1 中位 9.52 / 最大 16.99(124 个采样,每分钟一次;其中 124 次见到外来 >50% CPU 进程)
- A/A 对照臂: AA_N 11;AA_GM 1.0055;AA_NONMICRO_MIN nan;AA_NONMICRO_MAX nan;AA_VERDICT clean;
    ===== 判读:aa / off(逐阵中位比;测量分级)=====
    GM(aa/off) = 1.0055 | 非微级最差 = 0.0000 | nnz 异常/实验臂缺数据阵 = 0 | 微级 11 阵 | 有效 11 阵
    ===== 判读:cen / off(逐阵中位比;测量分级)=====
    GM(cen/off) = 0.9193 | 非微级最差 = 0.0000 | nnz 异常/实验臂缺数据阵 = 0 | 微级 11 阵 | 有效 11 阵
    ===== 判读:cm2 / off(逐阵中位比;测量分级)=====
    GM(cm2/off) = 0.9374 | 非微级最差 = 0.0000 | nnz 异常/实验臂缺数据阵 = 0 | 微级 11 阵 | 有效 11 阵
- **T1b 第 2 次: A/A 认证通过**

## T2 无侵入:off=本件默认 env / aa=同 off 的对照 / base=基线源码件(反打补丁),同批交错,16 阵 × 7 轮
- 说明:bm2_t1 的官方 ab_pair(16×3,无对照臂)被两个微级阵的噪声判红。这里把基线件作为一条臂放进同一个交错电池,与 A/A 臂同批;
  判读 = base/off 的 GM 与逐阵比(off 是分母,所以 >1 表示本件更快),有效性看 A/A。
- 主机(T2 前): load=14.50 10.72 9.53 外来>50%CPU进程数=4 卡=0, 14 MiB, 0 %;1, 5426 MiB, 0 %;
