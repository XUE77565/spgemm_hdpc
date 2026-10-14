# study/43:全库代码体检(10-13,代理②周边篇已复核;主文件篇待到)

> 双代理只读审查(批跑期间,零 GPU)。代理结论**逐条人工复核后**才入档;
> 本篇先入周边文件篇(10 条,3 条 P1 已亲验属实),主文件(7150 行)篇后补。

## 已复核确认(P1,修法 ≤10 行,批后动手)

1. **main.cu:437 `#endif LOG_BOTH("Time:...")`**——LOG_BOTH 挂在 #endif 同行,
   预处理器静默丢弃 → **adaptive 的 Time 行任何构建都不打印**,解析该行的计分
   对 adaptive 拿不到数。修:换行。✅亲验(cat -A 原始字节)
2. **adaptive -1 回退链泄漏**——hash 判废路径(DCEST 溢出/CNT_FUSE)仍走到
   d2h 并把 pinned C_buffer 交给 caller;adaptive(spgemm_adaptive.cu:79-84/122-125)
   覆盖 *C_buffer_out 不释放 → USE_MEMPOOL=0(编译默认,spgemm.h:20)裸跑每回退
   漏一块 cudaMallocHost(warmup×5 逐轮累加);池模式(我们的测量脚本全用它)
   arena reset 兜住故未爆。修:回退前 pinned_free(*C_buffer_out)(注意池/legacy
   双态的 free 语义配对)。✅亲验机制
3. **USE_DEV_POOL×RETRY2 持久 arena 别名**——dev_pool_reset 每轮清 bump 偏移,
   R1d"持久"静态指针下轮与新分配同址 → retry 数据被 A 上传覆写=静默算错。
   双 env 默认关;修:RETRY2 持久 buffer 改独立 cudaMalloc 或 reset 语义分层。

## 已确认(P2,排队)

4. mempool 溢出=exit(EXIT_FAILURE) 无逐次回退(鲸鱼阵+MP_HOST_MB 默认 256MB);
5. mempool_init 失败回退后 host_free(A_buffer) 走 free() 释放 pinned 指针=UB;
6. mtx 解析 r/c 无界检+sscanf 返回值不查(数据可信故未爆,加固项);
7. d3_merge_kernel lane≥k 无掩码预载(读入 val 段=分配内垃圾,结果被 mask 不污染;
   memcheck 过=未越分配;卫生修);
8. launch_bin_segrts 9 个 dev_alloc 零 dev_free(SEGRTS 默认关);
9. **spgemm.h:11 `#define DBG 1` 写死**——Makefile DBG 开关失效,无日志二进制
   不可能产出;dbg=fflush(stderr) 在 wall-clock 计时窗内(计分用 compute-only
   口径不受污染,但 Time 行恒含日志开销)。修:`#ifndef DBG` 包裹(默认保持 1
   免破坏现有 harness 的 [hash-prof] 解析)。✅亲验
10. adaptive AA/ATT dispatcher 整段复制(维护性,权重更新要改两处)。

## 复核备注(对代理结论的修正)

- #7 代理标"D3EXP=1 默认关"触发——**不准确**:D3 merge 核随 dstr_split 人群
  默认发射(pre2 即 D3 主场),但越界读落在同一分配的 val 段内,不越分配界,
  memcheck(pre2,0 err)实证无害;降级为卫生项。
- #2 的实际暴露面比代理描述窄:所有测量脚本 USE_MEMPOOL=1,仅裸二进制+判废时漏。

## 主文件篇(7150 行)

[待代理①报告到达后:同样逐条复核再入档]

## 处置序(批后)

批跑完成后按 1→2→9→(3,5)→(4,6,7,8,10) 顺序修,每条过守门;
#10 与 1c 融合同场处理。

## 主文件篇(代理①,15 条;P0-1 已行级亲验,P0-2 行号属实待深验)

**P0**:①3612-3613 FORCE_DITER_EST 强制改判在 bucket_id 写入之后(counts/scatter
错位→段溢出;✅亲验:写入序实锤,触发=该旋钮>0);②3776 dense_largest 对
d_tmp_val 无清零 atomicAdd(首触零页掩盖,warmup 多轮值翻倍;LARGEST 默认开
——**批后第一优先深验+修**)。
**P1**:RETRY2 SMEM 车 cap 读陈旧 d_roff;CNT_FUSE static d_tmp 首次定容永不扩容
(多阵同进程必中);retry 免排序路 CAP=16384 截尾大行;SEGRTS scan int 回绕
(ToLL 漏修第三处)。
**P1 优化**:in_place 判据全趟白跑(FIX2 默认 0,~30MB/轮);k7 调试残留 printf+死码。
**P2**:batched 行粒度早退后全掩码 syncwarp UB;heavy_prep 疑似条件反写;两条回退
路径漏释 5 buffer+FILE*;死代码五段(subwarp8/hkv/hybrid/check_sorted 等);
compact 循环内 pageable 同步双停摆;**17→8 融合弹药清单**(binning 5→2 发、
k7_count 6→2 发、尾声三重复抽公共核、dense_sum 门合一)。

## 总处置序(批后)

1. P0-2 深验+修(LARGEST 默认开)→ 2. P0-1 换序 → 3. 周边篇 #1(#endif)→
4. 周边篇 #2(回退泄漏)→ 5. P1 群(env 门控类)→ 6. 死代码下线与融合弹药(与 1c 同场)。
代理总评:老 core 纪律扎实,雷集中在"后加旋钮/边路 + static 跨调用假设"。

## 修雷四连 ✅(10-14 午后全部落地+守门)

| # | 修复 | 守门 |
|---|---|---|
| P0-2 | tmp_val 清零改无条件(原 INITDBG 门着,LARGEST/batched 的 atomicAdd 靠零页运气) | 双跑值门 col_mismatch=0 worst 1.4e-16;memcheck 0err |
| P0-1 | FORCE_DITER_EST 强制改判移到 bucket_id 写入前(counts/scatter 口径归一) | FORCE=100000 跑通 nnz 精确 |
| 周边#1 | #endif 同行吞行修复 | adaptive Time 行复活 |
| 周边#4+#5 | mempool 溢出 exit→回退登记表(16槽,destroy 统一释放);init 失败不再翻 flag(A_buffer pinned/free 错配) | MP_HOST_MB=16 强制溢出:回退路径存活,nnz 精确 |

残留 P1/P2 队列不变(RETRY2/CNT_FUSE/SEGRTS 等 env 门控类,与 1c 同场)。
