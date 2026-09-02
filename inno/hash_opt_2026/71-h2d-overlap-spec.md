# docs/71 h2d 重叠管线(用户 09-02 提议:读入期间做 estimate/symbolic/dispatch 掩盖开销)

**机制**:h2d 走 copy engine,SM 闲 —— 早期 kernel(count_flop/mh_construct)在已到达
数据块上开跑,物理重叠。CSR 行连续 → col 块 k = 连续行区间,天然流水分割。

**口径**:compute-only = TOTAL − h2d;真重叠(stream 并行)后 kernel 藏入 h2d 窗 →
指标与墙钟同降,合法(非口径游戏,须 event 栅栏保证数据就绪)。

**三层收益**:①重叠本身(大阵藏 1.5-4%,小阵 3-8%,geomean 估 −2~4%);
②**DVFS 破局**(docs/57:SM 长闲→345MHz 冷跑小 kernel;传输期干活让后继起点更暖);
③对 Ocean 净优势(它们同是串行 h2d)。

## 设计

```
传 dA_row_ptr(全量,小)→ sync
按 chunk(8-32MB)循环:
    cudaMemcpyAsync(chunk k 的 col/val → dA_buffer 偏移, stream S1) + recordEvent(E_k)
    流水:stream S2 waitEvent(E_k) → launch count_flop/construct 于行区间 [r_k, r_{k+1})
末块后:merge/est_scan/binning(全局信息,不可藏,但本就小)
```

- count_flop_span_kernel / mh_construct_kernel 加行区间参数(或 grid-stride + range)
- mh_merge 不动(需全量 sketch);dispatch(binning)在 est 后 = 只能藏构造部分 ✓ 符合
  用户"初步 estimate"的定位
- **验收:nnz 逐位不变(纯重叠零语义变化)+ 相位表(h2d 期间 SM util>0)+ 小阵电池**

## 排队

复验/v28 终审期间禁止编译 → 本项 = v28 后第一优先(预计 2-4h 工程)。

---

## docs/13 v2 数据(2026-09-03,同窗口采集;road/OSM 族 = 最大系统性缺口 ~10 阵 @~3×)

**相位(road_usa,compute-only 41.7ms,HashProf GPU 流计时抗 CPU 争用)**:
| 相位 | ms | 占比 | 主因 |
|---|---|---|---|
| accumulate | 21.5 | 52% | 896ns/行;bin0 表 128 槽,est~23 利用率 18%,清表+扫描 6.7× 超配 |
| compact+sort | 10.6 | 25% | 440ns/行;warp-per-row 8 行/CTA = 3M CTA;5-nnz 行 27/32 lane 闲 |
| count_flop | 7.6 | 18% | warp-per-row 4.8-entry 行 = 27/32 lane 闲 |
| binning+est+cnnz | 2.1 | 5% | 正常 |

**bin 分布(BINS_DBG)**:road_usa 24M 行 / germany_osm 11.5M 行 **100% 落 bin[0]**;
web-Google 分散(bin0 740K/bin3 137K/…)——小行族专用档只影响 road 类,零外溢风险。

**三件套(375d9d0 后 commit,env 门控默认关)**:
1. BATCH_S=1:est≤32 ∧ kc≤32 → BIN_BATCH_S(64 槽表 = 2×32,α=0.5;清/扫税 −50%)
2. COMPACT_T=1:bin0/BATCH_S compact 线程纂(CTA 3M→94K,0 闲 lane)
3. COUNT_TINY=1:avg row≤8 阵 count 线程纂(MLP 64 行/warp 在飞)
正确性:三阵 nnz 逐位一致 ✓;净窗电池 = `scripts/ab_env.py "BATCH_S=1 COMPACT_T=1 COUNT_TINY=1" road_usa germany_osm hugetrace-00010 hugebubbles-00000 web-Google pre2 bcsstk30 pwtk F2 Ga3As3H12 nd24k`

**预期**:accumulate −30~50% + compact −50% + count −40% → road_usa 41.7 → ~26ms(3.49×→2.2×);
家族 ~10 阵同步受益,geomean 估 −1.5~2%。
