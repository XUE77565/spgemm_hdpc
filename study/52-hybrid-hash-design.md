# study/52:混合 hash 设计文档(梯 2-5 值外置;10-20,自动化链 C 相)

> **只设计不改码**(汇报后人工过目再动)。靶=Phase A 定谳的剩余输面人群。

## 一、靶区(Phase A 实测,10-20)

| 阵 | est p50 | dup | 主 bin | accumulate 差距 |
|---|---|---|---|---|
| ldoor | 273 | **9.5** | 梯4(89%) | 1.33×(28 vs 21ms) |
| rgg_n_2_22 | 81 | 2.6 | 梯2-3(81%) | 1.83×(33 vs 18) |
| StocF-1465 | 117 | 1.9 | 梯3(92%) | 2.5×(15 vs 6) |

共性:**中 est(64-1024)× 中高 dup 的 hash_spa 梯子行**——正是 H100 无 smem F64 原子
→ CAST.SPIN 自旋墙人群(17 连败判死的硬件边界)。ocean 在同人群用 §3.3 混合设计赢我们。

## 二、机理(ocean §3.3 源码级,study/44 可借清单 #1;只借机理不抄形态)

```
现状(我方梯子):  smem 表存 [key(4B) + val(8B)];重复键的值累加 = F64 smem 原子
                 → H100 无原生 → ATOMS.CAST.SPIN 循环;dup 高 = 同槽自旋排队
混合设计:        smem 只存 key(4B/槽,32-bit CAS 原生快);值放 gmem 数组 val[slot]
                 → 值累加 = gmem 原生 FP64 RED(RED.E.ADD.F64,fire-and-forget 单指令,
                   不需要返回值 → 无往返等待)
提取:            扫 smem key → 对有效槽 gather gmem val(多一趟 gmem 读,按需)
```

**为什么赢**:①键的 CAS 是 32-bit 原生(浅);②值的 RED 是单发指令,L2 侧对同槽串行
但 warp 不自旋等待;③smem 占用减半(免 8B 值/槽)→ 梯容量或行/块翻倍。

ocean 自证:行长 3× 无显著惩罚;gmem 原子延迟仅 2-4× smem(且 fire-and-forget 免掉)。

## 三、实现方案(供评审,全部 env 门控默认关)

| 项 | 方案 |
|---|---|
| 开关 | `HYBRID_V=1`(与既有 `g_hybrid/d_hybrid_val` 基建衔接——BIN_BATCH_S 路已有同型先例 5752 行!) |
| 范围 | 先梯 2-5(est 64-1024,dup≥1.5 行才走);bin0/BATCH_S/ESC 不动 |
| 数据布局 | `d_hval[tier_base + slot]`(gmem,val 8B/槽);smem 表只留 key+flag |
| 插入 | 探针命中→`atomicAdd(d_hval+abs_slot, v)`(原生 RED);新键→CAS 占槽后 RED 初值 |
| 提取 | 现有 extract 加一路 gather(或并入 compact:FLAT 条目级拷贝天然适配——src 变二元) |
| 人口门 | 行级 dup 探针(已有的 est/flop 比)≥ 阈值才启用;低 dup 行维持全 smem(免 gmem 往返) |
| 交互 | D5H 直写行豁免(同现有);溢出 retry 语义不变;COMPACT_FLAT 的 src 读需兼容二元布局 |

## 四、风险登记

1. 提取多一趟 gmem 读:低 dup 行净亏 → 人口门必须先行(FLAT 的教训);
2. 梯子 kernel 是模板族,改造面 = 梯 2-5 五个实例;
3. D5H/FLAT 交互面(两处消费端)——值门矩阵要覆盖组合;
4. ocean 用"机理"不抄"形态":我们的槽布局/梯结构不同,引用其 §3.3 为机理出处。

## 五、验证计划(预注册)

- 值门:小阵逐位(梯 2-5 人群阵)+ ldoor/rgg/StocF Result C 锚定;
- 同会话 A/B:三靶阵 + 控制阵(c-64/nd24k 全 smem 不受扰);
- 判据:ldoor accumulate ≥15% 改善才转正;**expectation: ldoor(dup9.5)最大赢面,
  rgg/StocF 部分赢(ocean 另有 symbolic 工作流加成,不全归累积器)**;
- 批级:428 一夜 → manifest。

## 六、预期账(推测档)

ldoor 1.33→~1.1 / rgg 1.83→~1.5 / StocF 2.5→~1.9[推测];337 GM 影响 ~−0.3-0.5%。
与 ESC64(已证伪)互补:那刀换累积器形态,这刀换值的物理位置——后者正对 CAS 墙。
