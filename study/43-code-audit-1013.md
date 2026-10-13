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
