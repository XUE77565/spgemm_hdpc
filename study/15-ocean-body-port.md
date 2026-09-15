# Ocean 体移植:read-first 探测入默认(2026-09-15 凌晨)

> 一句话:**把彼 hashNumericKernel 的探测体(read-first + wrap + j*11)移植进我们的发射环境做逐变量对拍——同 est/bin/block/温度计下 accumulate −4~−13%(crankseg 24.4→21.2ms),×5 电池 geomean −2.4%,值级 bit-exact,已翻默认。crankseg 逐行输比 1.638→1.465。**

## 方法(用户钦定的"逐变量对照"终极版)

`hash_spa_ocean_kernel`(src/spgemm_kernel_hash.cu):彼探测体 × 我们的 init/k-j 循环/extract/溢出守卫/env。env `HASH_OCEAN`(默认 1;=0 回我 CAS 体)。同环境逐行对拍把 1.64× 拆成"体差"与"环境差":
- **体差(probe 机理)= 13.5%**(crankseg accumulate;逐行 1.638→1.465)
- 环境差(表 2.7× vs 1.75× 等)≈ 余下 1.47× 的主体,但表紧化已三轮证伪(2 幂梯谷底)

## 电池(×5,compare/port_battery.log + 复检)

| 阵 | Δ% | | 阵 | Δ% |
|---|---|---|---|---|
| pre2 | **−12.1** | | web-Google | −3.9 |
| crankseg_2 | −5.3 | | Zd/msc/nd24k/Ga3 | −0.7~−1.5 |
| in-2004 | −3.3 | | c-64/c-43 | +0.4/+0.3(复检=噪声) |
| F2 | −2.5 | | **geomean** | **−2.4%** |

- accumulate 口径:crankseg −13% / c-64 −12% / F2 −4%(hash 相位赢面,系统被 dense/sort/est 稀释)
- msc10848 值级 bit-exact 过 cuSPARSE ✓;nnz 11/11 ✓

## 沿途尸检(本篇的关键排除,勿重蹈)

1. **Lemire 归约 −9%**(Knuth×Lemire vs j*11×mask);FIT 非整表第三轮死 → 素数档梦碎,pow2+mask 定稿
2. **j*11 直接配 Lemire = 灾难 78×**(值域 70 万全折 slot 0,单链走穿)——Lemire 前置必须全值域 hash
3. 移植体需要自己的 SMEM opt-in(否则 >48KB 表发射 invalid argument → 粘滞错变 thrust "invalid device ordinal")
4. GFORCE 全程:G=32 是 U 形谷底(4→512 单调劣化)——G 维度无余量,ncu 发散信号源于 probe 体而非 G 拓扑
5. PRIV(W 私有表):W8 封顶仍 +13%,W>8 被 SMEM 物理封死——结构性出局
6. upper_tri 行号相关 r=+0.010 平坦——过滤税排除

## 逐行记分(study/14 口径,matched 块;v39 补测完毕 2026-09-15)

| 阵 | 可比行 | CAS 体(v38) | **彼体移植(v39)** |
|---|---|---|---|
| F2 | 71,502 | 65.0% / 0.847 | **75.6% / 0.795** |
| c-64 | 7,214 | 49.8% / 1.002 | **52.3% / 0.962(翻赢)** |
| crankseg_2 | 51,225 | 27.2% / 1.638 | **28.5% / 1.467** |

**三场两胜一负,全体 13 万行加权胜率 ≈ 56%**;余差距 = crankseg 1.47×(环境:pow2 表 2.7× vs 彼 1.75×,紧化三轮证伪 + init/extract ∝ 表)。

## 资产

- 代码:hash_spa_ocean_kernel + HASH_OCEAN(默认 1)+ HASH_FIT_PCT(移植路径,实验遗留)+ GFORCE
- /tmp/bin_pre_port(翻默认前二进制);compare/port_battery.log

---

## 勘误(2026-09-16 转正刀,两则 + 一条路线修正)

**勘误一:v39 的 −2.4% geomean 掺了"免排序"水分,且移植周输出 C 小表行无序(存量正确性 bug #2)。**
移植体 extract 注释称"与主 kernel 同(ht≤CSORT_HT 走 count-sort 有序写)",但代码只有无序写一条路;下游小 bin(`bi ≤ ilog2c(CSORT_HT/32)`)只做 compact_copy(假设上游已序)→ 无序行直达 C。nnz 红旗与值级 set 对比均序盲,一周未察。实测(WRITE_MTX dump + 行内严格升序检查):

| 阵 | 移植体输出乱序违规 | 转正后 |
|---|---|---|
| crankseg_2 | 6,908 处(首例 row 1783) | **0** |
| F2 | 2,277,120 处(首例 row 7) | **0** |

∴ 本篇 −2.4% geomean 与 commit 1c4d254 的系统级声明须加注:其中含跳过排序作业的成分;**accumulate 相位的机理归因(read-first 体差 13.5%,clock64 逐行口径)不受影响,仍然成立**。

**勘误二:转正刀已执行——机理以自有形态并入 `hash_spa_kernel`,移植体整块删除。**
- read-first 探测(命中普通读零原子/空槽 CAS 占位)进 `hash_spa_kernel` 内层循环,`template<int MODE, int PROBE_RF=1>` 编译期分支,主 bin/retry(MODE=0)/count(MODE=1)/direct(MODE=2)全发射点统一;逃生门 `HASH_PROBE_RF=0`(=CAS-first,v38 语义)
- `hash_spa_ocean_kernel`/`HASH_OCEAN`/`HASH_FIT_PCT` 全库零残留;散列常数 Knuth 2654435761u 本就自有;注释引 nsparse `kernel_spgemm_hash_s.cu:298-315`/spECK `HashMap.cuh` 公共先例
- diff −132/+84;值级三绿(msc10848 default/storm/CAS 门,DITER off);小表行有序合同恢复(上表)

**路线修正:extract 路径成为新刀候选。** 三点定位(crankseg accumulate):v38(CAS+kernel 内 count-sort,有序)24.4ms → 移植体(RF+无序)21.1ms → 转正版(RF+count-sort,有序)22.9ms。read-first 机理净赚 −6.2%(对 v38);移植体对转正版多出的 8.5% = 它欠的排序作业。若把小 bin 的排序从 kernel 内 O(count²) count-sort 挪到下游 csort_fused 基数排序(v37 细梯已证 bin6/7 便宜),有望同时拿"无序 extract 的便宜 + 有序合同"——待独立成刀验证。
- 创新点池更新:**"对拍驱动的内核体移植+同环境逐变量归因"方法学**;bin 级 dup 条件化(体已统一,此点并入 G/配置)
