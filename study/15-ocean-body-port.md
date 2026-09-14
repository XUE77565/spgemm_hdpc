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

## 逐行记分(study/14 口径,matched 块)

| 阵 | CAS 体(v38) | **彼体移植(v39)** |
|---|---|---|
| F2 | 65.0% / 0.847 | (系统 −2.5%,逐行待补) |
| c-64 | 49.8% / 1.002 | (同上) |
| crankseg | 27.2% / 1.638 | **28.6% / 1.465**(dup>20:1.689→1.502) |

## 资产

- 代码:hash_spa_ocean_kernel + HASH_OCEAN(默认 1)+ HASH_FIT_PCT(移植路径,实验遗留)+ GFORCE
- /tmp/bin_pre_port(翻默认前二进制);compare/port_battery.log
- 创新点池更新:**"对拍驱动的内核体移植+同环境逐变量归因"方法学**;bin 级 dup 条件化(体已统一,此点并入 G/配置)
