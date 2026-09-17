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

## 深夜解剖补充(09-18,compact+sort 2.46ms 内部地图)
- 结构:小行(≤CSORT_HT)kernel 内 count-sort O(d²) + 纯拷贝;bin6/7(ht 2048-4096)= csort_fused/csort = **per-row cub BlockRadixSort(32 位 4 pass,已高度优化非裸比较排序)**;bin8/9 = SEGRTS(cub 分段基数,默认关)
- F2 的 2.46ms 载体:主力 bin6(47.9K 行 d~1000-2000)的 per-row radix + bin4/5 的 in-kernel count-sort + 拷贝
- 修订方向(按性价比):①小行 count-sort 两级化(本规格主体,O(d²)→O(d+1024),bin4/5 载体)②d≤2048 时 counting(直方图+落位,2 pass)vs radix(4 pass)的交叉点实测——counting 有 2× 潜力但 cub 已高度调优,须实测③SEGRTS 通道(bin8/9)重估④拷贝段 warp-batched 已有(Fix#11)
- 判据口径 B 含排序;F2 目标 2.46→<1.5ms

## 实施记录(09-18 凌晨,已编码已守门)

### 代码位图
- `hash_spa_kernel` 两个 count-sort 位点全替换:MODE=2 直写(原 1864)/MODE=0 tmp(原 1876);位图 = Pass0 min/max(SMEM+warp 归约)→ 逐行自适应 SH(桶≤1024)→ Pass1 直方图 → Pass2 排他前缀(连续所有权)→ Pass3 桶分组重排(寄存器 staging 4 槽/线程)→ Pass4 桶内 O(b²) 位次直写
- 门槛:ht∈[512,1024] 启用(bin4/5);ht<512 税轻不动,ht>CSORT_HT 走原大行路径零改动
- SMEM:hist 三数组(h_cnt2/h_start/h_fill 各 1024 int)=12KB,host 侧 `ht∈[512,1024]` 才追加;bin5 表 12+12=24KB×8 块=192KB<228KB 不压 occupancy
- env HASH_EXTRACT=2 开,默认 0;二进制 spgemm_c2(WM 版)

### 守门结果(全绿)
- MODE=0 位点:值级 4/4(F2/msc/crankseg PASS @1e-9;c-64 only_test=20,504 与既往 artifact 逐数一致豁免)+ **有序 4/4 SORTED-OK**(本刀命门)
- MODE=2 位点(D5H=1,F2 hash_direct 6.5ms 相位执行实证):值级 PASS + SORTED-OK
- A 级口径 B 电池(×5):净窗守卫挂 /tmp/ab_c2_wall.sh(卡 1 共租 69GB 等待中)

### 尸检两雷(守门体系各抓获一次,方法论活例)
1. **SMEM 指针偏移**:hist 起点写成 `smem + 2*ht_size`(落进 sh_val 区!表布局=ht×4B+ht×8B=12×ht B,hist 应在 `smem + 3*ht_size`)→ 值区被直方图清零覆写+计数变野值 → F2 illegal access。教训:**SMEM 指针算术逐字节核算每个区段,int* 基址下 double 区占 2 int 槽/元素**
2. **跨步所有权前缀**:Pass2 用线程跨步持桶(b 归线程 b%B)+段基址公式——公式假设"前线程全部桶在我之前",跨步时序不成立 → F2 85,963 有序违规/msc 15,111(值级反而 PASS:多重集对、仅序错)。修复=连续所有权([t·ib,(t+1)·ib)。教训:**块级前缀和的 ownership 布局必须与段基址公式的顺序假设一致**
3. 顺带:大阵 WM 守门必须 `METHOD=hash` 单方法门控(T1 cuSPARSE 大阵 insufficient resources 是已知限,truth17 大阵本就是 scipy 生成);比对目标 = self_product_adaptive/hash.mtx 而非 self_product.mtx(T1)
