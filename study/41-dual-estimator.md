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

## 四、设计:两刀

### 1a-1 矩阵级估计器跳过闸门(主刀,吸收 ocean 的跳过机理)

```cpp
// count_flop 后(设备侧 avg_product 已有,零成本判定):
if (avg_product <= EST_SKIP_THR) {          // 默认 64(沿 AVG_FLOP_THR 哲学;电池标定)
    est_fill_kernel<<<...>>>(d_flop, A_cols, d_est, A_rows);   // est=min(flop,n)
    goto est_scan;                           // construct+merge 整段消失
}
```
- 物理出处:est=flop 是精确上界,零低估;高估代价 = 表内存+retry 经济学
  (P(retry)×cost vs 省掉的 construct+merge)——沿 docs/47 的 1.15/1.4 扫描裁决法标定阈值;
- 与 ocean 的形态差异(不抄之点):他们 threshold=8 打在输入膨胀上,我们打在
  avg_product 上且走自己的 est_fill/retry 体系;
- 顺手兑现 1c 的一部分:命中阵直接少 2 个 launch。

### 1a-2 HLL A/B(次刀,只对仍需估计的高 dup 阵)

机理上不预设胜负(我们的 K=32 sketch 更小);构造侧 uint8 vs uint32 有 4× 流量差,
merge 侧 SIMD 打平的假说要 A/B。**先跑 1a-1 的电池看剩余税量再决定要不要做**。

## 五、开放取证(GPU 空时补)

- in-2004 的 2 行结构性欠估(actual=raw×1.5-2):DDUMP_ROW + 行结构 dump——
  是估计器盲区标本,若形态是"超长链/极端 skew"则同时喂 1a-1 的阈值设计。

## 六、守门预告

- 值门:跳过路径改变 est → 改变 binning/retry 分布 → 全链值级对比必须过
  (跳过阵 ×2 + 估计阵 ×2);
- 电池:低 dup 阵(333SP 型)预期 −1.5~−2ms;高 dup 阵零变化(不命中门);
- 判废安全:est=flop 零低估 → 无新证书需求(DCEST 的 flop 封顶守卫已覆盖此形态)。
