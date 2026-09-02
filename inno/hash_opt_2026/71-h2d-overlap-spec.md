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
