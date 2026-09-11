# 行级多阵 hash kernel 对拍定谳(2026-09-11 深夜,6 阵 ×5 档)

> 一句话:**低重复(dup≤2)的行,我们的 hash kernel 全面赢或平(c-64 三档 0.36-0.87×,Ga3 两档 0.51/0.98×);输幅随 dup 单调增长(F2 dup7-10 → 1.18-1.58×,crankseg dup26 → 1.86×,Ga3 whale dup87 → 60×),Pearson r=0.83。机理定位:我们命中路径每次探测都 atomicCAS,Ocean 先普通读、命中零原子。**

方法:双边 nsys per-launch(1 块=1 行),按行数配对档(±35% 容差);我方 DUMP_ROUTING 出逐档 dup(=flop/est)。资产:/tmp/rowband/(sqlite×10 + routing dump×5)+ 本表。pre2 ht2048 的 7.3× 判为行集错配作废(彼侧同批行分进 331 小表+dense,分类体系不同);Ga3 ht8192 的 60× 为真匹配(两边都是顶部重行)。

## 逐档记分(我方 v37 vs Ocean,ns/row 比值;<1=我赢)

| matrix | 档 | 我/彼 | dup | 判 |
|---|---|---|---|---|
| c-64 | ht1024 | **0.36×** | 1.6 | 大赢 |
| c-64 | ht2048 | **0.60×** | 1.9 | 赢 |
| c-64 | ht4096 | **0.87×** | 1.7 | 赢 |
| Ga3As3H12 | ht2048 | **0.51×** | 1.3 | 赢 |
| Ga3As3H12 | ht1024 | 0.98× | 1.8 | 平 |
| pre2 | ht1024 | 1.11× | 1.1 | 平 |
| F2 | ht512 | **0.46×** | 9.8 | 赢(小表链短,CAS 税低) |
| F2 | ht2048 | **0.81×** | 7.2 | 赢(主档) |
| F2 | ht1024 | 1.18× | 9.0 | 输 18% |
| F2 | ht4096 | 1.58× | 7.6 | 输 58% |
| Ga3As3H12 | ht4096 | 1.27× | 2.0 | 输 27%(低 dup 例外:表大) |
| crankseg_2 | ht2048 | 1.08× | 28.0 | 平 |
| crankseg_2 | ht4096 | **1.86×** | 26.2 | 输 86% |
| nd24k | ht4096 | 1.09× | 43.5 | 平 |
| Ga3As3H12 | ht8192 | **60.7×** | 86.9 | 灾难(whale 行进 SMEM hash) |

## 机理(代码级,三税叠加)

1. **命中税(主犯,随 dup 放大)**:我们每乘积 `atomicCAS(&sh_col[slot],-1,j)` —— 命中也要付 SMEM 原子;Ocean `ids[map_id]` 普通读先行,命中零原子、只有空槽才 CAS(spgemm/spECK 系同款,Hashmap.cuh operator[])。dup=每键命中次数 → CAS 税 ∝ dup。dup≤2 时税可忽略(我们赢,说明探测函数/G 结构本身不差);dup 7-26 时税滚成 18-86%。
2. **同址 atomicAdd 串行化(两边共有,高 dup 主导)**:值累加每乘积一次 atomicAdd,H100 同址吞吐 ≈1/4(docs/72 已有此常数);Ga3 whale(dup 87)两边都被打爆,但他们路由侧把这类行送 dense,我们送 SMEM hash → 60×。这半边是路由税。
3. **表尺寸错配放大探测+初始化成本**:ht4096 在低 dup 也输 27%(Ga3):48KB 表逐行 memset + 长Probe链;他们 2719 素数表 33KB。

## 刀(按证据排序)

1. **read-first 探测**(命中零原子,空槽才 CAS;Ocean/spECK 同款)— 治税 1,预计收复 F2 两档 + crankseg 大部。小口子:hash_spa 插入循环改 6 行。
2. **高 dup 行路由走 dense/LARGEST**(税 2 的路由半边;Ga3 whale 类)— 与 study/08 的溢出/whale 刀合并。
3. ht4096 低 dup 档:素数表 or 表 init 分摊(税 3,小头)。
