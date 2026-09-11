# v37:csort 细梯 + block 梯修 occupancy(2026-09-11,study/06 §7 两把配置刀)

> 一句话:**csort bin6 `<512,8>`→`<128,16>` 一刀 −43%(2643→1498µs)**;hash_spa 盲抄 Ocean slots/thread 不变量被证伪(轻档倒退),修正后只保留 bin8+ 的 occupancy 修复;初版 bin8-9 `<512,32>` 撞块寄存器限额被验尸(§3.5)。值级 bit-exact 过 cuSPARSE 对拍。

## 1. 改动(env 门控,默认关,电池过了才翻默认)

**刀 1:CSORT_LADDER=v2**(spgemm_kernel_hash.cu compact 段):
- bin6(ht2048):`<512,8>` CAP4096 → **`<128,16>` CAP2048 紧贴表容**(F2 主档 42k 行,主赢点)
- bin7(ht4096):`<512,8>` → `<256,16>`(同 CAP,IPT×2 线程减半)
- bin8-9:**不动**(见 §3.5 验尸)
- 出处:Ocean sortOutputFused 哲学(IPT 恒 16、TPB 紧贴表容);旧配置对 bin6 CAP 冗余 2×,BlockRadixSort TempStorage 挤 occupancy

## 3.5 验尸:v2 初版 bin8-9 `<512,32>` launch 不可(硬件不变量教训)

- 症状:首夜电池 8/11 阵 on 侧全灭(`CUDA error at :2870 too many resources`);首采的 nsys "10.51ms" 实为崩溃前部分和(bin8 csort 从未出现在 trace 里)
- 定罪(ptxas -v):`<512,32>` = 157 regs/线程 → 512×157 = **80384 > 65536 块寄存器限额 → launch 不可**;对照 `<256,64>` 229×256 = 58624 贴线可跑、`<128,16>`/`<256,16>` 80 regs 从容
- **不变式入档:csort 配置可行性 = TPB × regs(TPB,IPT) ≤ 65536**;加 IPT 换线程数前必须过这道公式(IPT 每 ×2 寄存器近乎 ×2)
- 处置:bin8-9 回旧 `<256,64>`(F2 仅 12 行,无证据值得冒险);修复后 F2/pre2/web-Google/Zd_Jac2 全程 `Result C` 收尾 ✓

**刀 2:HASH_BLK_LADDER**(per-bin block 梯,支持 v2/legacy/逐档覆写):
- v2 全套(96/192/384/512/1024)被 F2 证伪:bin4 +50%、bin5 +4.5%(我们 G-group 结构 ≠ Ocean,轻档本该胖块)
- **修正梯 v3 = 只改 bi≥8 → 1024T**:修真 occupancy 缺陷(ht≥8192 表 ≥96KB ×512T 只驻 2 CTA = 1024 线程;1024T 恢复 2048)
- 用法:`HASH_BLK_LADDER=8:1024,9:1024,10:1024`(在 legacy 基础上覆写)

## 2. F2 per-launch 归因(nsys,修复版合法 trace:compare/kernel_F2/nsys_v37fix_F2.nsys-rep)

| launch | 基线 | v37 | Δ |
|---|---|---|---|
| hash bin4-7(94/1754/3270/820µs) | 同配置 | 94/1772/3274/805 | 平价(噪声带) |
| hash bin8 @1024T | 98.5 | 74.5 | −24%(12 行,方向性) |
| **csort bin6 `<128,16>`** | **2643.2** | **1502.9** | **−43.2%** |
| csort bin7 `<256,16>` | 326.1 | 314.4 | −3.6% |
| csort bin8(回旧 `<256,64>`) | 36.4 | 36.8 | 平 |
| **TOTAL GPU** | **11.74** | **10.59** | **−9.8%**(电池 −10.1% 互证) |

溢出重试路径(hash_global 1426µs)未动——那是第 2 刀(安全系数/载体)的靶。

## 3. 正确性论证 + 实证

- 不变式:非重试行 `row_nnz ≤ ht ≤ CAP`;重试行 `row_nnz = flop 定表精确值` 可超 CAP → csort 只排前 CAP 项写废,**末段 retry 善后(`launch_csort<..>(d_retry_n, d_retry_rows, d_retry_off, ..)`)整行覆写**(跑在 per-bin csort 之后)——与旧语义一致(csort_fused 本就不带 skip,靠覆写遮蔽)
- 实证:msc10848(360 重试行 + bin6 2922 行 + bin5 3537 行)WRITE_MTX 三跑:
  - legacy == cuSPARSE:max abs err = 0(bit-exact)
  - **v3 == cuSPARSE:max abs err = 0(bit-exact)**
- nnz 红旗:F2 = 39648121 两版一致

## 4. ×5 配对电池(compare/v37_ladder_battery.log,2026-09-11 晚净窗 load~5-7)

`AB_RUNS=5 ab_env "HASH_BLK_LADDER=8:1024,9:1024,10:1024 CSORT_LADDER=v2"` × 11 阵:

| matrix | Δ% | | matrix | Δ% |
|---|---|---|---|---|
| pre2 | **−15.1** | | msc10848 | −0.6 |
| in-2004 | **−10.7** | | c-43 | −2.6 |
| F2 | **−10.1** | | Zd_Jac2 | +5.1(复测 −0.5,噪声) |
| Ga3As3H12 | **−10.0** | | web-Google | −0.0 |
| nd24k | **−7.3** | | crankseg_2 | +2.1(复测 −0.6,噪声) |
| c-64 | −0.3 | | **geomean** | **−6.4%** |

- 六大赢全在 bin6/7 行多的阵(pre2 bin6/7 380K 行 = 排序税鲸鱼,正中靶心);平价阵 = 无 bin6/7 行或被 DSTRUCT 路由走
- Zd/crankseg 首轮 +5.1/+2.1 → 同窗 ×5 复测 −0.5/−0.6 = 噪声带内摆动,零真回归
- nnz 红旗 11/11 全过

## 5. 默认翻转 + 确认(compare/v37_default_confirm.log)

- `CSORT_LADDER` 默认 v2 开(=legacy 回旧);`HASH_BLK_LADDER` 默认 = 旧三档 + bi≥8→1024T(=legacy 纯旧;=b:t 逐档覆写)
- 反向确认 ×5(off=新默认 vs on=legacy):legacy 慢 F2 +10.0%/pre2 +17.1%/in-2004 +12.4%,Zd/crankseg 平 → 翻转坐实

## 6. 遗留(下一刀候选,study/06 §7 顺位)

1. **溢出安全系数/载体**(F2 上 1.96ms 纯税;Ocean 用 ×1.5~2.0 est 系数 + SMEM dense 位图载体)
2. ht1024 档输 17% / ht4096 档输 34% 的 kernel 内因(block 梯已排除同责)
3. mh_merge 0.40ms → HLL 0.17ms 的 est 管线差
