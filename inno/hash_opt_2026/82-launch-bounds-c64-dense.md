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

## 电池判决(09-18 12:30-12:36,×5 相位和口径,off 带内)
- msc 3.535→3.194(**−9.6%**)/ c-64 25.157→20.377(**−19.0%!**)/ crankseg 59.353→54.760(**−7.7**,
  off 在带缘−0.8%,配对差分可信但标注)/ F2 9.954→9.904(−0.5,不经此路径符合预期)
- **系统记分牌更新:c-64 25.16→20.38 vs ocean 20.57 = 1.223→0.991——三阵中第一阵系统级跨线翻盘**
  (领先 0.9%,边际;官宣前需净窗复核 ocean 侧 20.57)
- crankseg 2.153→2.006(同 kernel 附带收益;残余=路由差距=③阶段 occgate 阵地)
- 判决:刀入默认(代码无条件,值级+有序守门全绿);物理台账新条目:**寄存器占用悬崖**——
  SMEM 预算给足 2 块但 REG>32 时 1024thr 块仍只驻 1 块;指令瘦 14% 救不回占用腰斩

## 官宣确认(09-18 12:45,ocean 侧新跑)
- ocean c-64 新跑:20.8042ms(10 轮均,compute;D2H 1499ms 灾难再现 vs 我方 ~80ms)
- **c-64 终局记分:20.377 vs 20.804 = 0.979,系统级首胜(双边新数据,margin 2.1%)**
- 今日弧线:1.212 →(占用解剖+一行 launch_bounds)→ 0.979

## mode 分层账(09-18 午后,用户纪律问询触发:cursor/dense 独立记账)
- **cursor 与 dense 不是两个独立 kernel**:同为 hash_dense_direct_kernel 的按行模式(rowmode=
  avgB≥阈值→cursor 合并;否则 dense 位图),共享窗口 SMEM 布局(val8+flag1+pref4+cursor4/列)
- 分层差分(dd 文件 3 列:行 cycles mode,$2=cycles——又一个口径陷阱):
  cursor 19,347 行 +47.8%/dense 97 行 +45.9% 逐行 cyc —— **同幅变慢 = 纯溢出税,与模式无关**
- **赢面结构 = 占用恒等式活教材**:2× 占用 ÷ 1.47 溢出 = 1.34× 净吞吐;
  dense_direct 18.5→13.8ms 实测(18.5×1.47/2=13.6 ✓)物理全闭环
- **下一块肉(独立刀,cursor/dense 内部)**:去溢出——手工减活状态拿"自然 2 块"
  (重算代持有/冷值下沉 SMEM/拆窗循环),目标 dense_direct ~9-10ms → c-64 ~16.4 = 0.79
