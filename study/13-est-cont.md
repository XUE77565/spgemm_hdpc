# est 连续化 + expand 扛余量:geomean −13.6% 入默认(2026-09-14)

> 一句话:**去掉 est 的双重 pow2 吸附(mh_merge 尾 `est=next_pow2` × binning `target=2×est` = 2.3-4.6× 浪费),余量改由 expand 扛(HASH_EXPAND 1.15→1.8,恰收敛到 Ocean HLL 安全系数区间 1.5-2.0);×5 电池 geomean −13.6%,溢出 1319→3,值级 bit-exact 过 cuSPARSE,已翻默认。**

## 改动(已入默认)

- `EST_CONT`(默认开,=0 回旧):mh_merge 尾 `est = temp` 连续输出,不再 snap next_pow2
- `HASH_EXPAND` 默认 1.8(连续 est 时;显式 env 优先):余量从"双重吸附"移到"单系数",与 Ocean hll_expansion_coes 同位
- `HASH_TGT_ALPHA` 默认联动 150(连续 est 时;legacy 时 200):binning 单次吸附
- 逃生门:`EST_CONT=0 HASH_TGT_ALPHA=200`(实测 legacy 慢 F2 +22%/web-Google +85%/msc +37%)

## 机理链(study/12 的根因 → 本篇干预)

- 旧:est = pow2(E×1.15) → 表 = pow2(2×est) → 表/真值 2.3-4.6×(F2 71,505 行仅 5 个 est 值 = 分类器)
- 新:est = E×1.8 连续 → 表 = pow2(1.5×est) → 表/真值 2.7-5.4×?? 不——**行级分辨率出来了**:每行进对 bin,档内均匀 → 探测链短 + retry 归零
- 实测 F2:over-alloc 1.63→1.79×(内存几乎不变),retry 1319→3,acc+retry+compact 10.15→8.15ms(−20%)

## ×5 电池(11 阵,compare/estcont_battery.log)

| 阵 | Δ% | | 阵 | Δ% |
|---|---|---|---|---|
| web-Google | **−45.7** | | crankseg_2 | −8.5 |
| msc10848 | **−27.6** | | in-2004 | −7.6 |
| c-43 | **−20.7** | | c-64 | −3.4 |
| F2 | **−17.9** | | pre2 | +1.6 |
| Zd_Jac2 | −11.3 | | nd24k | +3.8 |
| Ga3As3H12 | +2.5 | | **geomean** | **−13.6%** |

- 7 赢 4 平微输;nnz 红旗 11/11 过;msc10848 值级 bit-exact 过 cuSPARSE

## 已知耦合(下轮靶)

- **nd24k 相位账 −75% 但全阵 +3.8%**:est×1.8 动了 dense/heavy 路由门(est≥2048 diter 触发等)→ 重路由税吃回相位赢。修法:路由门用"未膨胀 est"或重校门限
- pre2/in-2004 的 accumulate 付表变大税(各自最优 expand 1.4-1.6),矩阵级自适应 expand(现 ADAPT_EXPAND 状态机只认 1.15/1.4)值得重校
- expand 扫描(F2):1.4→−17%/1.6→−18%/1.8→−20%/2.0→−26%(ovf=0);全局折中 1.8(nd24k/pre2 在 1.4-1.6 更优,故留 env)

## 资产

- compare/estcont_battery.log + estcont_default_confirm.log;/tmp/bin_pre_estcont(翻默认前二进制)
- HASH_TGT_ALPHA/EST_CONT/HASH_EXPAND 三旋钮联动语义见代码注释
