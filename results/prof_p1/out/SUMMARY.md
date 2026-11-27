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
- 采数 44 发(22 阵 × 2 件),失败 0 发;LAUNCH_DIFF_N 22;LAUNCH_DIFF_MATS 0;LAUNCH_DIFF_VERDICT same;(明细 o1_launch_diff.txt)
    # 臂 base 对臂 off:比对 22 阵;臂 base 的发射总次数 2062(含预热轮)
    设备侧 Σkernel:GM_KERN(off/base) = 0.9996 | 阵数 22 | 最差 1.0094 | 有两遍式发射的阵 0 个,其 GM = -

## O2 官方 ab_pair 16×9(参考轨:两个二进制、无对照臂;按 dialog/qa-008 只入档,不作裁定;裁定以上一轮 T2 的带对照臂版为准)
- 主机(O2 前): load=20.84 14.11 12.29 外来>50%CPU进程数=5 卡=0, 14 MiB, 0 %;1, 80825 MiB, 100 %;
    GM(B/A) = 0.9985 | 最差 = 1.0429 | nnz 异常 = 0
    微级(不作单阵裁决): arc130, 1138_bus, bcsstk17, F2 — 单阵结论无效,如需判决改 10× 交错
    裁定: 绿-进下批(≤2% 带内)
- 段内主机负载: load1 中位 10.53 / 最大 20.89(72 个采样,每分钟一次;其中 72 次见到外来 >50% CPU 进程)

## O3 msc10848 加轮复核(off / aa / cen / cm2 / m2,21 轮交错)
- ab_arms rc=0
          msc10848      2.051      2.045   0.9971      中      1.438     1.441   1.0021         -         -  
          msc10848      2.051      2.112   1.0297      中      1.438     1.528   1.0626         -         -  
          msc10848      2.051      2.087   1.0176      中      1.438     1.490   1.0362         -         -  
          msc10848      2.051      2.044   0.9966      中      1.438     1.401   0.9743         -         -  

## K3 ncu 硬件计数(hash_heavy 对 hash_bm2;application 重放;60 分钟预算)
- ncu: /opt/nvidia/nsight-compute/2025.1.1/ncu / Version 2025.1.1.0 (build 35528883) (public-release)
- 采数 11 组(阵 × 臂),成功 11 组;原始宽表 k3_raw_<阵>__<臂>.csv.gz,ncu 报告留在 /home/xueyizhou/exp-runs/prof_p1/keep/
- ncu_sum rc=0;挑选指标的并排表 k3_ncu.txt(# ncu_sum: 文件 11 个,读出指标的 11 个,读不出的: 无)

## P1 全 428 影子(off / aa / cen / cm2;METHOD=adaptive,生产默认 env;轮数按文件大小分档:<10MB 5 轮 / 10–200MB 3 轮 / ≥200MB 2 轮)
- 等了 15 分钟,主机仍不满足安静判据(load=9.77 8.17 8.36 外来>50%CPU进程数=2 卡=0, 14 MiB, 0 %;1, 14 MiB, 0 %;)。照跑:有效性由 A/A 对照臂认证,判据不过整段作废
- shadow428 rc=0;# 完成:428 阵,5972 发,失败 0 发,用时 868 分钟
- 段内主机负载: load1 中位 8.20 / 最大 36.76(868 个采样,每分钟一次;其中 868 次见到外来 >50% CPU 进程)
- shadow_sum rc=0;全文 p1_shadow_sum.txt,逐阵 p1_shadow_permat.csv,相位地图 p1_shadow_map.csv.gz,逐发 p1_shadow.csv.gz,相位长表 p1_shadow_phases.csv.gz

```
# shadow_sum: 清单内 428 阵,基线臂有数据的 428 阵;失败发 0;臂: off aa cen cm2
# 分级:微 92 / 中 202 / 稳 134;nnz 各臂各轮不全同的阵 0 个;签名不全同的阵 0 个(有签名的 428 阵)

===== A/A 对照(aa / off)=====
AA_N 428(非微级 336)
AA_GM_ALL 1.0013
AA_GM_NONMICRO 1.0026
AA_NONMICRO_OUT5 6(占非微级 1.8%;上限 2%)
AA_VERDICT clean
不裁决阵(A/A 偏离超 ±5% 的非微级阵): rajat17(1.376) rajat18(1.280) rajat28(1.106) Zd_Jac6_db(1.083) rajat25(1.072) bcsstk33(0.947)

===== 判读:cen / off =====
GM(cen/off) = 0.9925(428 阵)| 非微级 GM = 0.9889(336 阵)| 微级 GM = 1.0061 | 各档(小/中/大文件)GM = 1.0028 / 0.9906 / 0.9797
非微级阵 >1.10:4 个(其中可裁决的 1 个):rajat25(1.331,不裁决) rajat17(1.281,不裁决) rajat16(1.267) rajat28(1.142,不裁决)
非微级阵 <0.95:12 个;>1.02:13 个
非微级最快 12: bcsstk33(0.908) nlpkkt120(0.919) Dubcova3(0.923) nlpkkt80(0.933) nemeth26(0.939) cage13(0.940) cage14(0.942) cage15(0.943) conf5_4-8x8-15(0.944) af_shell3(0.945) hood(0.948) olafu(0.948)
非微级最慢 12: rajat25(1.331) rajat17(1.281) rajat16(1.267) rajat28(1.142) rajat18(1.076) circuit_4(1.044) TSOPF_RS_b2052_c1(1.032) c-45(1.027) std1_Jac3(1.023) CurlCurl_4(1.022) av41092(1.021) Stanford_Berkeley(1.021)
裁定: 红

===== 判读:cm2 / off =====
GM(cm2/off) = 0.9490(428 阵)| 非微级 GM = 0.9365(336 阵)| 微级 GM = 0.9958 | 各档(小/中/大文件)GM = 0.9488 / 0.9421 / 0.9647
非微级阵 >1.10:12 个(其中可裁决的 11 个):fem_filter(1.450) torso1(1.423) nd12k(1.241) pattern1(1.233) nd24k(1.222) nemeth23(1.164) Si41Ge41H72(1.156) nemeth22(1.139) rajat25(1.135,不裁决) nemeth25(1.129) nemeth24(1.117) Ga41As41H72(1.102)
非微级阵 <0.95:148 个;>1.02:29 个
非微级最快 12: pkustk12(0.612) net50(0.650) filter3D(0.670) hangGlider_4(0.675) hangGlider_5(0.680) net75(0.689) hangGlider_3(0.696) net100(0.696) conf5_4-8x8-15(0.710) case9(0.712) bloweybq(0.716) TSOPF_FS_b9_c6(0.721)
非微级最慢 12: fem_filter(1.450) torso1(1.423) nd12k(1.241) pattern1(1.233) nd24k(1.222) nemeth23(1.164) Si41Ge41H72(1.156) nemeth22(1.139) rajat25(1.135) nemeth25(1.129) nemeth24(1.117) Ga41As41H72(1.102)
裁定: 红

```
- 相位地图与各池前 20 阵(默认形态 = 臂 off;cen / cm2 的池总量差也在其中),全文在 p1_shadow_sum.txt:
```
===== 相位地图(各阵各相位取各轮中位,再跨阵求和;单位 ms)=====
相位                          off           aa          cen          cm2
TOTAL(GPU)             26609.59     26687.20     26586.11     26454.08
accumulate              2991.74      2993.82      2995.99      2884.82
binning                  130.38       130.52        53.78        59.81
cnnz_scan                 15.75        15.80        15.83        15.96
compact+sort             760.90       761.14       759.83       759.58
count_flop               123.46       123.57       123.22       123.44
d2h                    16938.62     16972.05     16934.61     16961.96
dense_count              282.83       290.44       291.51       285.07
dense_direct            1699.67      1729.15      1742.48      1698.18
est_scan                  13.69        14.07        13.77        13.75
h2d                     3237.38      3230.90      3239.88      3231.54
mh_construct              47.07        46.94        46.88        46.98
mh_merge                 291.46       291.28       291.47       290.90
retry                     72.65        72.49        72.48        66.75

----- 池(臂 off):池总量与前 20 阵 -----
池 accumulate             总量    2991.74 ms;前 20 阵合计    1202.85 ms
    Stanford_Berkeley(183.9) c-big(81.8) Flan_1565(75.3) Bump_2911(72.4) dielFilterV3real(72.3) cage15(71.2) Cube_Coup_dt0(69.7) audikw_1(62.3) nd24k(59.0) pkustk12(57.7) bone010(52.3) Long_Coup_dt6(48.7) nlpkkt120(43.5) boneS10(40.0) wb-edu(38.3) Serena(37.1) Geo_1438(35.7) Ga41As41H72(34.7) Hook_1498(34.3) Ga3As3H12(32.9)
    臂 cen    该池总量    2995.99 ms(对 off +4.24 ms)
    臂 cm2    该池总量    2884.82 ms(对 off -106.92 ms)
池 compact+sort           总量     760.90 ms;前 20 阵合计     319.62 ms
    wb-edu(34.8) cage15(27.1) audikw_1(26.3) dielFilterV3real(25.5) Cube_Coup_dt0(21.6) Flan_1565(16.8) Bump_2911(16.6) Long_Coup_dt6(15.4) 3Dspectralwave(14.2) c-big(14.2) dielFilterV2real(12.4) eu-2005(11.9) delaunay_n24(11.9) Serena(11.4) pkustk12(11.3) in-2004(11.0) nlpkkt120(10.3) Hook_1498(10.2) inline_1(8.5) pre2(8.5)
    臂 cen    该池总量     759.83 ms(对 off -1.07 ms)
    臂 cm2    该池总量     759.58 ms(对 off -1.33 ms)
池 dense_direct           总量    1699.67 ms;前 20 阵合计    1524.91 ms
    mouse_gene(483.0) TSOPF_FS_b300_c2(269.5) gupta3(135.9) SiO2(75.0) c-73b(57.2) c-73(57.2) TSOPF_FS_b300(56.8) gupta2(51.5) pattern1(40.9) rajat20(37.5) rajat28(34.7) case39(33.1) rajat16(32.1) rajat17(30.9) rajat25(29.2) rajat18(28.1) gupta1(22.9) nd6k(17.1) net150(16.4) vsp_model1_crew1_cr42_south31(16.0)
    臂 cen    该池总量    1742.48 ms(对 off +42.81 ms)
    臂 cm2    该池总量    1698.18 ms(对 off -1.49 ms)
池 mh_merge+dense_count   总量     574.28 ms;前 20 阵合计     327.65 ms
    mouse_gene(68.8) TSOPF_FS_b300_c2(38.7) gupta3(21.6) Stanford_Berkeley(20.2) c-73(18.3) c-73b(16.9) SiO2(16.3) rajat20(13.5) rajat28(12.8) rajat16(11.7) rajat18(11.6) rajat25(11.5) gupta2(11.4) rajat17(11.0) TSOPF_FS_b300(10.4) cage15(9.0) pattern1(7.7) gupta1(5.5) mult_dcop_02(5.4) mult_dcop_01(5.4)
    臂 cen    该池总量     582.98 ms(对 off +8.69 ms)
    臂 cm2    该池总量     575.97 ms(对 off +1.69 ms)
各池前 20 阵合计的总和(可回收 top20 口径): 3375.03 ms
```

## P2 地图对齐发(off 臂单发,HASH_LHT23=0:对齐 scripts/phase_decomp_current.sh 的旧口径,只为与本机侧 atlas 对照;不是生产默认)
- shadow428 rc=0;# 完成:428 阵,428 发,失败 0 发,用时 64 分钟
- 旧口径地图 p2_lht0_map.txt / p2_lht0_map.csv.gz(长表 mat,arm,phase,ms)

## 结束
- 结论: S3 冒烟 green;P0 试跑通;O1/O2/O3、K3 见各段;P1 影子的裁定看 p1_shadow_sum.txt 里的 AA_VERDICT 与各臂的裁定行;P2 旧口径地图已出
- 主机(结束时): load=6.82 7.08 7.90 外来>50%CPU进程数=2 卡=0, 14 MiB, 0 %;1, 14 MiB, 0 %;
- 磁盘: /dev/sdb1        15T  8.7T  5.1T  64% /home
- 结束时间: 2026-11-27T06:44:34Z,总用时 1120 分钟
