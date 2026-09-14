# 中间档梯手术验尸:干预失败,真根源 = est 量化(2026-09-14)

> 一句话:**"合身档"干预全线失败(FIT=75/65:F2 +9~20%/crankseg +21%/nd24k +18%,retry 不变 = 探测链税非溢出);Lemire 归约本身 −4-5%;根因锁定 = F2 的 71,505 行 est 只有 5 个唯一值(256/512/1024/2048/4096)——我们的"估计器"实为分类器,bin 内真值 span [250,1000+] 无分辨力,任何单表必错。**

## 干预记录(补丁存 compare/rowlevel_F2/lemire_fitted.patch,已回退)

1. 探测 Lemire 化:`(hash×ht)>>32` 替 mask(任意表容免取模,nnz 全对 39648121 ✓ 正确性成立)
2. 发射侧 `HASH_FIT_PCT`(默认 100=legacy):bi≥4 表 = pct%×2 幂档

| FIT% | F2 accumulate | crankseg | nd24k |
|---|---|---|---|
| 100(仅 Lemire) | 6.052(基线 5.71,**−5% 税**) | — | — |
| 80 | 6.596 | — | — |
| 75 | 7.278 | 26.18(21.58) | 68.6(57.9) |
| 65 | 6.843 | — | — |

- retry 计数全程 1319 不变 → 慢因 = 装载因子↑ → 线性探测链长,非溢出
- FIT=100 的 −5% = Lemire 高位映射 + wrap 前进在 2 幂表上的净税 → 不能留,已回退(基线 5.714 复验)

## 三重证伪后的根因链(study/11 + 本篇)

bin5/7 行集输带(1.30×/1.45×)的假说全部干预失败:α 因子(量化和承重墙)、LLB(wash)、read-first(≤4%)、合身表(探测链)。**共同上游 = est 无 bin 内分辨力**:

- Ocean:HLL est 连续 → 每行贴身入档(673 表 LF~0.5)→ 表即合身
- 我们:est ∈ {2^k}(5 个值)→ est=512 的 bin 装着真值 250~1000+ 的行 → 1024 表白付 init、665 表打爆尾部 → **梯粒度不是病,est 分辨率才是**
- est 量化源头:mh_merge 尾部 sketch→est 换算(下一场定位;est_fill 旁路 = min(flop,n) 非 pow2 佐证量化在 sketch 侧)

## 下一场刀谱(两件套,顺序不可倒)

1. **est 连续化**:mh_merge 输出改连续(bottom-k 原始估计 k×2^32/kth_min 类),或 sketch 换 HLL/Ocean-C 精度档;电池验证 est/true 分布收敛
2. **复用本篇基础设施**:lemire_fitted.patch 的 Lemire 探测 + HASH_FIT_PCT 直接可用(nnz 已验),配连续 est 后 per-bin 合身表才有意义;届时 Lemire −5% 税需与合身收益合并评估

## 资产

- 补丁:compare/rowlevel_F2/lemire_fitted.patch(Lemire 探测 ×2 处 + HASH_FIT_PCT 发射侧)
- 代码保留:HASH_TGT_ALPHA(study/11,默认 200 零差)
- 基线复验:回退后 F2 accumulate 5.714ms ✓
