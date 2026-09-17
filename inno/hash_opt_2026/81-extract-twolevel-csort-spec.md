# ①阶段 ②号刀:extract 两级计数排序(替 O(d²) count-sort)规格

## 为什么要做(①阶段唯一"减总功"方向)
线程预算恒等式证明:块宽/表长等"重分配"型旋钮全部对消(15eadf6),唯一能破恒等式的是**减少每行总功**。
extract 的 count-sort 是唯一挂账的算法级浪费:排序税 5.9%(口径 B,A→B 差)。

## 现状解剖(hash_spa_kernel extract,行 1862-1875)
- 中小行(ht ≤ CSORT_HT=1024,且非 UO):先 compact(非空槽前移,得 d=distinct),
  再 **count-sort:每元素 k 扫全 d 数 rank = O(d²)/blockDim**;d=1024 时 1M 次比较 ÷256 线程 = 4K op/线程
- 大行(ht>1024):无序 extract + 下游 csort(已在 compact 相位)
- 税的分布推断:O(d²) → 税集中在 d 128-1024 的中行(F2 bin5-6/msc/c-64 轻带);d≤64 行税可忽略

## 新算法:两级 SMEM 计数排序(two-level counting sort)
1. compact 后 sh_col[0..d) 已就位(SMEM)
2. **Pass1 高位直方图**:hist[c >> SH]++(SH 取 ceil(log2(span/1024)),bin 数 ≤1024;
   4KB SMEM 额外,当前 CSORT_HT 档表 12KB,装得下)
3. **1024 元素前缀和**(256 线程,标准两遍,~40 op)
4. **Pass2 落位**:每元素 rank = bin_base[bin] + (# 同 bin 更小者)。
   同 bin 平均元素数 = d/1024 ≤ 1 → 同 bin 冲突极少,桶内用微型 count-sort(b² ≤ 4)或原子序号
   (原子序号不稳序?同 bin 内元素列值不同——bin 内按低位再数一次即可,桶内 O(b²) 可忽略)
5. 输出契约不变:tmp_key/base+rank 有序 → 下游 compact 逐位不动

复杂度:O(d + 1024) vs O(d²);d=1024 → 2K vs 1M op(÷500);d=256 → 1.3K vs 65K(÷50)。

## 逃生门与判据
- env HASH_EXTRACT=2 启用新路径,默认 0(legacy count-sort);UO 口径仪器不动
- 判据**必须口径 B**(含排序):A 会把这刀收益藏掉
- 守门:17 位值级 ×4 阵 + 有序扫描(这刀直接动序,有序检查是命门)+ ×5 电池 ×10 阵净窗
- 主判阵:F2 / msc / c-64 轻带(d 128-1024 密集);crankseg 大多走大行路径(预期中性)

## 风险表
| 风险 | 缓解 |
|---|---|
| bin 内冲突序不稳(同 bin 两元素) | 桶内显式 O(b²) 比较(b≤2 实测),不用原子序号 |
| SH 按行 span 取,span 未传 kernel | 用 col 全域位宽(18 位)>>…或从 span_lo/span_len 传(已有数组) |
| SMEM 超 48KB 档(CSORT_HT 档表 12KB+hist 4KB+d 紧凑区) | 预算核算:12+4+≤4 < 20KB,安全 |
| 直方图清零成本 O(1024)/行 | 与表 clear 同量级,计入净收益测算 |

## 实施顺序
1. kernel 内新路径(仅 CSORT_HT 分支内改,大行路径零改动)
2. 口径 B 电池(F2/msc/c-64 轻带)×5 净窗;排序税 5.9% → 目标 <2%
3. 值级+有序守门;绿 → 入默认,口径 A/B 双测刷新对拍基线
