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
- shadow428 rc=137;[   7/16] rajat28                          19.0MB 档1 轮数21 累计 441 发(失败 0) 195 分钟
- 段内主机负载: load1 中位 7.44 / 最大 10.35(225 个采样,每分钟一次;其中 225 次见到外来 >50% CPU 进程)
- remeasure_sum rc=0;逐阵全文 r1_remeasure.txt,逐发 r1.csv.gz,相位长表 r1_phases.csv.gz

```
# remeasure_sum: 8 阵;臂 off aa cen;实验臂 cen
== c-45
   该阵 A/A 中位比 1.0000 → 稳定阵(基线臂四分位距/中位 = 0.9%)
   cen   中位比 0.9989 | 最小比 0.9989 下四分比 0.9867 上四分比 0.9986 → 放行
== bcsstk33
   该阵 A/A 中位比 1.0063 → 稳定阵(基线臂四分位距/中位 = 3.9%)
   cen   中位比 0.9768 | 最小比 0.9731 下四分比 0.9840 上四分比 0.9723 → 放行
== circuit_4
   该阵 A/A 中位比 1.0286 → 稳定阵(基线臂四分位距/中位 = 2.2%)
   cen   中位比 1.0525 | 最小比 1.0339 下四分比 1.0240 上四分比 1.0824 → 放行
== rajat18
   该阵 A/A 中位比 1.1041 → 不稳阵(基线臂四分位距/中位 = 9.7%)
   cen   中位比 1.0635 | 最小比 0.9863 下四分比 1.0034 上四分比 1.0832 → 放行
== rajat16
   该阵 A/A 中位比 1.0756 → 不稳阵(基线臂四分位距/中位 = 23.4%)
   cen   中位比 1.0614 | 最小比 0.9840 下四分比 1.0007 上四分比 0.9911 → 放行
== rajat17
   该阵 A/A 中位比 1.0822 → 不稳阵(基线臂四分位距/中位 = 18.2%)
   cen   中位比 1.0868 | 最小比 1.0076 下四分比 1.0244 上四分比 1.0756 → 放行
== rajat28
   该阵 A/A 中位比 1.0062 → 稳定阵(基线臂四分位距/中位 = 20.6%)
   cen   中位比 1.0360 | 最小比 0.9977 下四分比 1.0198 上四分比 1.0590 → 放行
== rajat20
   该阵 A/A 中位比 1.0236 → 稳定阵(基线臂四分位距/中位 = 10.9%)
   cen   中位比 0.9535 | 最小比 1.0012 下四分比 0.9532 上四分比 0.9716 → 放行
REMEASURE_RED cen 0
```
- 判法(跑之前写定,scripts/remeasure_sum.py 文件头;是否采用由 dialog/qa-011 定):稳定阵看中位比 ≤1.10;不稳阵(该阵自己的 A/A 超 ±5%)
  看最小值 / 下四分位 / 上四分位三档比值依次 ≤1.05 / 1.05 / 1.10。

## R2 值级·生产入口(METHOD=adaptive;各臂对 off 的差分门)

此前两遍式的值级都在 METHOD=hash 下做。生产入口把密行也送进哈希梯,HH 行人口不同,这里补验:影子里合一形态最快与最慢的阵各取几个。

| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |
|---|---|---|---|---|
| case9 | cen | 52372050 | 0 / 0 | PASS col_mismatch=0 unordered=0 text_diff=4355 worst_rel_vs_global=1.195e-16 |
| case9 | cm2 | 52372050 | 7217 / 0 | PASS col_mismatch=0 unordered=0 text_diff=4873 worst_rel_vs_global=7.294e-21 |
| bloweybq | cm2 | 100020001 | 10000 / 0 | PASS col_mismatch=0 unordered=0 text_diff=0 worst_rel_vs_global=0.000e+00 |
| net50 | cm2 | 40622452 | 15355 / 0 | PASS col_mismatch=0 unordered=0 text_diff=0 worst_rel_vs_global=0.000e+00 |
| hangGlider_4 | cm2 | 191342279 | 13830 / 0 | PASS col_mismatch=0 unordered=0 text_diff=6234 worst_rel_vs_global=1.486e-16 |
| filter3D | cm2 | 20161619 | 98949 / 0 | PASS col_mismatch=0 unordered=0 text_diff=2168888 worst_rel_vs_global=1.647e-16 |
| pkustk12 | cm2 | 474804911 | 89517 / 1538 | PASS col_mismatch=0 unordered=0 text_diff=0 worst_rel_vs_global=0.000e+00 |
| fem_filter | cm2 | 8424292 | 14006 / 0 | PASS col_mismatch=0 unordered=0 text_diff=937523 worst_rel_vs_global=2.532e-16 |
| nd12k | cm2 | 90906894 | 35955 / 0 | PASS col_mismatch=0 unordered=0 text_diff=50391947 worst_rel_vs_global=6.731e-16 |
| Si41Ge41H72 | cm2 | 173776869 | 32640 / 69058 | PASS col_mismatch=0 unordered=0 text_diff=28409447 worst_rel_vs_global=1.448e-15 |

- R2: green;走两遍式的行数合计 317469

## R4 行门标定·生产入口(只用 data/train_disp;METHOD=adaptive;逐行配对周期;位图形态,off / m1 / m2)
- 出处:bm2_r5 的标定是在 METHOD=hash 下做的(位图形态 34 阵全部 R<1,不需要门)。生产入口把密行也送进哈希梯,影子里输的那类行当时没进 HH bin。
  这里换成生产入口重做,看训练池里有没有会输的行、(FX, CX) 门能不能把它们分出来——门值只能从这里或物理式来,不看 428。
- 训练池候选 49 阵(1MB-800MB,非测试账同名,不含 af23560)
- 采数:147 发;没出行计时的 0 发(dispatcher 路由到 merge3 的阵没有哈希行,属正常);逐行周期文件 49 组
- bm2_gate_cal rc=0;见 r4_cal.txt / r4_cal_cells.csv / r4_cal.env
    # bm2_gate_cal: 训练阵 43 个,HH 行合计 118276(位图形态 111672 / legacy 形态 6604)
    ===== 形态 leg=0(位图):计入目标的阵 34 个(HH 行 ≥ 200)=====
      全收(模式 1,不设门): 平均 R = 0.7975,最差阵 R = 0.9755
      全收(模式 2,不设门): 平均 R = 0.7783,最差阵 R = 0.9694
      → 选:模式 2,FX=∞,CX=∞(平均 R=0.7783)
    ===== 形态 leg=1(legacy):计入目标的阵 1 个(HH 行 ≥ 200)=====
      全收(模式 1,不设门): 平均 R = 1.0000,最差阵 R = 1.0000
      全收(模式 2,不设门): 平均 R = 1.0000,最差阵 R = 1.0000
      → 最优门的平均收益 < 0.5%:该形态不开
      实例税读数(模式 1,未走两遍式的 6604 行;v3 起两边同为原实例,应≈1):Σ周期(开) / Σ周期(关) = 1.0060
      实例税读数(模式 2,未走两遍式的 6604 行;v3 起两边同为原实例,应≈1):Σ周期(开) / Σ周期(关) = 1.0005
    # auto 臂(位图形态): HH_BM2=2,HH_BM2_FX=0,HH_BM2_CX=0  平均 R=0.7783

## R3 合一形态(cm2)在生产入口下的输家 / 赢家画像(只作画像;428 是测试账,不据此定门值)
- 逐 bin 普查([bm2-stat]:每个 bin 走 / 未走两遍式的行数与 Σflop、SMEM 字数):r3_bm2stat.txt(71 行)
- nsys 设备侧(off 对 cm2,METHOD=adaptive,各 1 轮):48 发,失败 0 发;全文 r3_kern.txt,长表 r3_kern_long.csv.gz
```
===== ① 逐阵:Σkernel GPU 时长(ms,各轮中位;含预热轮)与对 off 之比;括号里是两遍式相关核族(hash_heavy_kernel + hash_bm2_kernel + hh_bm2_part_kernel)的比 =====
mat                    off_ms    轮间%            cm2/off(核族比)
Ga41As41H72            85.169    nan        1.1008 (1.1488)*
Si41Ge41H72            73.503    nan        1.1518 (1.2031)*
TSOPF_FS_b9_c6         14.677    nan        0.7142 (0.4929)*
bloweybq               24.528    nan        0.7147 (0.4937)*
case9                  15.111    nan        0.6951 (0.4735)*
conf5_4-8x8-15          4.537    nan        0.7062 (0.6551)*
fem_filter              6.737    nan        1.4979 (1.7448)*
filter3D                6.885    nan        0.6446 (0.5224)*
hangGlider_3           19.940    nan        0.6901 (0.4815)*
hangGlider_4           47.920    nan        0.6624 (0.5073)*
hangGlider_5           50.485    nan        0.6585 (0.5045)*
nd12k                  64.735    nan        1.2444 (1.2786)*
nd24k                 132.104    nan        1.2305 (1.2607)*
nemeth22                2.789    nan        1.1555 (1.2272)*
nemeth23                3.088    nan        1.1319 (1.2644)*
nemeth24                3.090    nan        1.1250 (1.2689)*
nemeth25                3.083    nan        1.1239 (1.2623)*
net100                 22.053    nan        0.6986 (0.5182)*
net50                   9.031    nan        0.6234 (0.4916)*
net75                  14.750    nan        0.6771 (0.5726)*
pattern1              149.601    nan        1.2283 (1.6894)*
pkustk12              139.981    nan        0.6078 (0.4976)*
rajat25                98.856    nan        1.2167 (0.9607)*
torso1                 63.209    nan        1.4523 (1.4833)*
(* = 该臂在这阵上确有 hash_bm2_kernel 发射)
GM_KERN(cm2/off) = 0.9049 | 阵数 24 | 最差 1.4979 | 有两遍式发射的阵 24 个,其 GM = 0.9049

```
- ncu(off 对 cm2,6 阵,application 重放):7 组,成功 7 组;逐发射的指令数 / 显存读字节 / 缓存命中见 r3_ncu_long.csv 与 r3_ncu.txt

## 结束
- 结论: S3 冒烟 green;R1 的逐阵判定看 r1_remeasure.txt 的 REMEASURE_RED 行;R2 值级 green;R3 画像见各文件
- 主机(结束时): load=7.62 7.44 7.42 外来>50%CPU进程数=2 卡=0, 14 MiB, 0 %;1, 14 MiB, 0 %;
- 磁盘: /dev/sdb1        15T  8.7T  5.1T  64% /home
- 结束时间: 2026-11-27T13:05:13Z,总用时 366 分钟
