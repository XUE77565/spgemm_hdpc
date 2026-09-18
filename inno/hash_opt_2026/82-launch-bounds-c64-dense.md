# c-64 占用悬崖刀:hash_dense_direct __launch_bounds__(1024,2)(09-18 中午)

## 证据链(全实测,B级 ncu + cuobjdump)
1. c-64 相位解剖(215640c):dense_direct 18.5ms = 1.21× 差距主体;19,444 diter 行 99.5% cursor
2. ncu 头对头(同阵同法):我 cursor 4.01G inst/26.9ms/**warps 49.97%** vs ocean denseNumericIter
   4.67G inst/20.6ms/**warps 99.67%** —— 指令瘦 14% 仍慢 31% = 纯占用差
3. cuobjdump 铁证:我 hash_dense_direct **REG:50**(1024thr → 65536/(1024×50)=1 块/SM=50%)vs
   彼 denseNumericIter **REG:32**(恰 2 块=100%)
4. SMEM 本身不是限制:PB2_W 5400 列 ×21B ≈ 113KB × 2 = 226KB ≤ 228 设计就是 2 块;
   **寄存器是唯一 binding constraint**
5. 同物理先例:LADDER c-64 鲸鱼 +1071% 验尸的"占用悬崖项"(表>4K 槽)——本次是 dense 侧同病

## 刀
`__global__ void __launch_bounds__(MAX_BLOCK_THREADS, 2) hash_dense_direct_kernel(...)` 一行
→ nvcc 压 REG:50→32(STACK 24→32,溢出微增=换 2× 占用的交易)

## 守门(全绿)
- 值级:msc PASS / crankseg PASS / c-64 FAIL=artifact 形状逐数一致(only_test=20,504,rel 1.4e-16)
- 有序:3/3 SORTED-OK
- 电池:off(spgemm_c2t) vs lb2(spgemm_lb2t) ×5 相位和口径(跑中)

## 预期与判读
- 若占用兑现:dense_direct 18.5 → ~11-13ms,c-64 全阵 25.5 → ~18-20 → 对 ocean 20.57 翻盘
- 若溢出吃掉收益:回退停靠;溢出在 L1/L2,STACK 仅 +8B
- crankseg 附带受益(同 kernel 服务其 dense/cursor 行)
- 泛化候选(后续独立刀,勿混):hash_spa REG:40 在 bin8+ 1024thr 档同样 1 块/SM——
  但 hash_spa 是三阵全胜主力,动它须独立电池

## 二进制
spgemm_lb2(WM)/spgemm_lb2t(非WM) vs spgemm_c2/c2t(off 对)
