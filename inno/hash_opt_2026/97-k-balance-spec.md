# spec97:k 分布负载均衡(重行截距第三成分,09-24 18:3x 立)

## 靶子(§3.1p 终测推导)

重行截距三成分:清表(已排除可修)/末栅栏等最慢组/**本刀**/调度(不可修)。
现状:`for (p = rs+my_group; p < re; p += num_groups)` 轮转分 k——各组 Σ B 行长不均,
nd24k(a_len 312×变长 B 行)32 组各 ~10 k:E[max/mean] ≈ 1.5-1.8× → **慢组拖全块,
末栅栏等待 ≈ 0.6×均工作 ≈ 数万周期 ≈ 截距同量级**。thread-0 时钟含此等待。

## 设计(env HASH_KBAL=1,大表 bin 试点)

1. 表清+栅栏后:块协作把行 k 表的 B 行长载入 SMEM scratch(a_len ≤ KBAL_MAX=2048,
   超限回退轮转),Hillis-Steele 含前缀(2048 元素/1024 线程=2 元素/线程×11 步)。
2. 每组二分搜自己边界:target_g = total×(g+1)/num_groups → 连续 k 段 [k0,k1)。
3. 探测宏 p 环改 (p_lo,p_hi,p_st)=(k0,k1,1) 或 (rs+my_group,re,num_groups)——
   **环体零改动**(界/步本就是循环不变寄存器,每乘积成本不变)。
4. 成本:2-3 次额外栅栏+O(a_len log) SMEM 扫(~几十周期/行)vs 不均衡等待(万级)。

## 验收

- nd24k/in-2004 accumulate ×3 配对(−3%+ 起判)+逐行 1-4K 带改善;
- 值门三件套(F2+in-2004 truth17);甜点带零回退(kbal 只开大表 bin);
- 红线:先门后屏/Xid/opt-in(smem +8KB→大表 bin 本就 opt-in)/env 读码。

## 判死条件

若 A/B 无赢(<2%):说明栅栏等待非主成分(截距=清表+调度主导)→ 截距三成分全排除,
重行定谳"结构税",关门。

## 实施状态(09-24 19:xx):WIP 停车,待净上下文重攻

- 实现+接线完成(内核块/宏参数化/env/smem),但 **illegal access 未定位**:
  证据矩阵=单槽写净/全局读净/多槽装载崩;kbal=0 全净,kbal>0 时连 kbal 未启用的
  bin4 等价发射也崩(F2/in-2004 同)——疑编辑间隐性互作用(待查:宏展开后 p_lo
  生存期/调试 printf 的栈影响/STACK 24→40)。
- **WIP 补丁=/tmp/wab/spec97_kbal_WIP.patch**(完整 diff);干净基线已还原复验。
- 重攻法:净上下文+CUDA_LAUNCH_BLOCKING+逐 bin 关断二分;或重写为独立模板实例
  (kbal 进模板参,不碰主宏)。
