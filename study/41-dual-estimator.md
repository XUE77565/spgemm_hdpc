# study/41:1a 双估计器——读码设计(10-12 晚,Phase 1a 开场)

> 前置:1b 已定谳(fc49855);本文为 1a 的设计稿,只读码不占 GPU(解毒屏在跑)。
> 外部基线铁律:吸收机理,不抄形态(ocean/ 全源码在手,对照已做)。

## 一、问题实况

估计器链(count_flop → mh_construct → mh_merge → est_scan)在重估计阵上 ~9.5ms
(docs/83 浪费工位表);c-64 实测 construct 0.14 + merge 1.72 ≈ 1.9ms / 20ms 净窗 ≈ 10%。
目标:低 dup 阵把这条链砍到 ~0。

## 二、ocean 估计器解剖(读码结论,ocean/kernels/HLL.cuh + Analysis.cuh)

1. **HLL 机制**:`hllAdd` = 对列哈希取 `__clz+1` 为 rank,`atomicMax(&scratch[idx])`
   进 2^p 个 uint8 寄存器;merge = 各被引用 B 行 sketch 的**逐字节 MAX**,读侧
   uint32×4 寄存器 `__vmaxu4` SIMD;估计 = 调和均值 + 空寄存器 linear-counting 修正
   + HLL_CONSTANT[p] 偏差表。construct 有按列分段的 sections 变体(窗选择用)。
2. **真正的省法不是 HLL 快,是 decideMatrixType 的矩阵级跳过**(Analysis.cuh:18-27):
   `total_products/A_nnz < 8`(输入膨胀低)或 compaction 低 → **return 0,整个估计器
   不跑**(est=flop 够用)。默认阈值 8(Utils.h:30)。"ocean HLL 5ns/元素"的神话
   一半是这条免跑路径的功劳。
3. 与我们 MinHash 的**机制对偶**:min-of-hash ↔ max-of-clz;算术均值(D=m²·2³²/Σmin)
   ↔ 调和均值。sketch 尺寸:我们 MH_K=32 → 128B/行(生产);HLL p=12 → 4KB/行
   (p 可降)。**我们 merge 每行引用的流量未必输**——"换 HLL 就快"不成立,
   跳过路径才是大头。

## 三、我们已有的资产(零新机制可复用)

- per-row flop 封顶:est=min(E×expand, flop)(mh_merge 出口)——低估风险已封顶;
- **设备侧 total_flop/avg_product 规约已在线**(hash.cu:4618,PIPEFUSE step-1 遗产)
  → 矩阵级闸门的输入**免费**;
- `est_fill_kernel`(est=min(flop,n) 直填)现成(MHSAMP 路径遗产);
- retry 基础设施:欠估行的兜底经济学已验证(docs/47)。

## 四、设计(10-12 晚二次修正:对账发现闸门已存在)

> ⚠ 勘误:初版本文把"矩阵级跳过闸门"当缺失主刀——**对账证明它 8 月起就是默认行为**
> (hash.cu:4622 `avg_product ≤ AVG_FLOP_THR(64)` → est=flop D2D 直填,MinHash 整段
> 跳过;c-64 相位表里没有 construct/merge 的低均度阵就是命中者)。结论稳定纪律:
> 宣布"X 缺失"前先对账既有实现。ocean 的 decideMatrixType(expansion<8)与我们是
> 同机理异形态——他们的分母是 A_nnz(=平均 B 行长),我们的是 A_rows(=平均每行
> flop,含 a_len),阈值家族不同但哲学同源。**"数学白给"的部分 8 月已经入账。**

真战场 = **不命中闸门的"真需估计"阵**(avg_product>64:web/重 dup 类)。其税的解剖
(c-64 相位实测):

```
merge 1.69ms,带宽账 = A_nnz×128B ≈ 320MB → 屋顶 ~0.1ms
→ 实测 = 屋顶的 17 倍 =【延迟受限,非流量受限】(1 warp/行,散射读 B sketch 冷数据)
construct 0.14ms(B 侧写满 MH_M=128 partition,而 merge 只读 K=32 → 写读比 4:1)
```

修正后的刀序:
- **1a-1'(主刀)merge 占用/流水线**:17× 屋顶差距的延迟病——多行/warp 或大块发射
  让在飞读增多。需 ncu 实证瓶颈位(L1 miss?冷 TLB?),GPU 空后做;
- **1a-2'(次刀)construct 写侧砍到 K partition**:4× 写流量(它本来便宜,顺手);
- HLL A/B 降级为备选(机理对偶+我们 sketch 更小,只在上面两刀不够时再议);
- 闸门本身不动(已存在;阈值 64 的重标定视 1a-1' 后的剩余税再议)。

## 五、开放取证(GPU 空时补)

- in-2004 的 2 行结构性欠估(actual=raw×1.5-2):DDUMP_ROW + 行结构 dump——
  是估计器盲区标本,若形态是"超长链/极端 skew"则同时喂 1a-1 的阈值设计。

## 六、守门预告

- 值门:跳过路径改变 est → 改变 binning/retry 分布 → 全链值级对比必须过
  (跳过阵 ×2 + 估计阵 ×2);
- 电池:低 dup 阵(333SP 型)预期 −1.5~−2ms;高 dup 阵零变化(不命中门);
- 判废安全:est=flop 零低估 → 无新证书需求(DCEST 的 flop 封顶守卫已覆盖此形态)。
