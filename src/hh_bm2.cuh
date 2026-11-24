// hh_bm2.cuh — 刀三·想法二(study/69 §二十 设计 / §二十六 实施):位图标记-累积两遍式行例程
// 生产(src/spgemm_kernel_hash.cu 的 hash_heavy_kernel<...,BM2>)与微复现器(tools/hh_repro5.cu)
// 共用本文件——复现器编进去的就是生产代码本身,不是手抄的同构件。
//
// 一行一块。全部工作区从调用方给的动态 SMEM 缓冲 sm 起算(int 字),不新增静态 __shared__:
//   [0, 2cw)               bm64:每个列域词一个 64 位条目——低 32 位=位图词,高 32 位=该词之前的
//                          置位总数(独占前缀)。Pass2 一次 8B 读同时取到"位图词+前缀"。
//   [2cw, 2cw+ng)          gpre:32 词一组的组计数 → 组独占前缀(ng=⌈cw/32⌉)
//   [2cw+ng]               ntot:行内 distinct 总数
//   [sv_off, sv_off+2cap)  sval:值区(仅 VMODE=1;按位次寻址,上界 cap=行 est 槽宽)
// 流程:清位图 ‖ Pass1 标记(先读后 or)‖ 组内局部前缀+组计数 ‖ 组前缀(tid0)‖ 前缀定稿+键有序
//   直写+值区清零 ‖ Pass2 位次直寻累加 ‖(VMODE=1)值区拷出。栅栏 5 个,VMODE=1 再加 1 个。
// VMODE:1=值累加在 SMEM(软件模拟的 F64 原子,与哈希路径同款),末尾拷到 tmp_val;
//        2=值直接累加到全局 tmp_val[base+位次](原生 RED,免值区 SMEM 与末尾拷出)。
// 输出契约(与 Gv4 位图有序直写一致):tmp_key/tmp_val[base, base+n) 行内列升序,row_nnz[i]=n;
//   n>cap(est 低估)→ 不写任何输出,注册 ovf 交既有 retry 机具。
// 前提(调用方保证):每个乘积列 j 满足 0 ≤ j−cbase < 32·cw(docs/24 跨度窗:B 行列升序),
//   且 hh_bm2_fits(cw, cap, sm 的字数, VMODE) 为真。
#pragma once

// hash_heavy 一次发射的动态 SMEM 字数(int 字)——发射侧 HH_LAUNCH2 的 hhsm、核内 BM2 行门、微复现器
// 三处共用,单一出处免公式漂移:表区(col 偶对齐 + val)+ 尾区(位图 BM 词;legacy 时为 fused 384×12B 或 0)
__host__ __device__ constexpr int hh_smem_words(const int P, const int BM, const int FUS, const int LEG)
{
    return ((P + 1) & ~1) + 2 * P + (LEG ? (FUS ? 384 * 3 : 0) : BM);
}

__host__ __device__ constexpr int hh_bm2_ng(const int cw) { return (cw + 31) >> 5; }
// 值区起点(int 字):bm64 2cw + gpre ng + ntot 1,向上取偶(8B 对齐)
__host__ __device__ constexpr int hh_bm2_sv_off(const int cw) { return (2 * cw + hh_bm2_ng(cw) + 2) & ~1; }
// SMEM 字预算判定(宿主/设备共用):totw=本次发射给的动态 SMEM 字数
__host__ __device__ constexpr bool hh_bm2_fits(const int cw, const int cap, const int totw, const int vmode)
{
    return cw > 0 && cap > 0
        && (long long)hh_bm2_sv_off(cw) + ((vmode == 1) ? 2LL * (long long)cap : 0LL) <= (long long)totw;
}

// 返回:n(行内 distinct 数,已写出)或 −1(n>cap,已注册 ovf,未写输出)。块内所有线程返回值相同。
template<int LOG_NTHR, int B16ON, int VMODE>
__device__ __forceinline__ int hh_bm2_row(
    const int *__restrict__ A_rp, const int *__restrict__ A_ci, const double *__restrict__ A_val,
    const int *__restrict__ B_rp, const int *__restrict__ B_ci, const double *__restrict__ B_val,
    const uint4 *__restrict__ B16,
    const int i, const int cbase, const int cw, const int cap, const long long base,
    const long long prods, const int max_bl_raw,
    unsigned long long *tmp_key, double *tmp_val, int *row_nnz,
    int *overflow_flag, int *ovf_rows, int *ovf_cnt, int *row_ovf,
    int *sm)
{
    constexpr int NT = 1 << LOG_NTHR;
    const int tid = threadIdx.x;
    unsigned *bmw = (unsigned *)sm;                                   // bmw[2w]=位图词,bmw[2w+1]=前缀
    unsigned long long *bm64 = (unsigned long long *)sm;
    const int ng = hh_bm2_ng(cw);
    int *gpre = sm + 2 * cw;
    int *ntot = gpre + ng;
    double *sval = (double *)(sm + hh_bm2_sv_off(cw));

    for (int w = tid; w < cw; w += NT) bm64[w] = 0ull;                // 清位图(前缀半字随后全量覆写)
    __syncthreads();                                                  // 栅栏 1

    const int rs = A_rp[i], re = A_rp[i + 1];
    // 组自适应:与 hash_heavy_kernel 哈希路径同一算式(avg B 行长定组+双向 2× 容差重平衡)
    int lg;
    {
        const int a_len = re - rs;
        const int max_bl = max_bl_raw + 1;
        long long avg = (prods - max_bl) / (a_len > 1 ? a_len - 1 : 1);
        if (avg < 1) avg = 1;
        lg = 31 - __clz((unsigned)avg);
        if (lg < 2) lg = 2;
        if (lg > LOG_NTHR) lg = LOG_NTHR;
        long long max_sub = ((long long)max_bl + (1 << lg) - 1) >> lg;
        long long iters = (a_len + (1LL << (LOG_NTHR - lg)) - 1) >> (LOG_NTHR - lg);
        while (max_sub > iters * 2 && lg < LOG_NTHR) {
            lg++; max_sub = ((long long)max_bl + (1 << lg) - 1) >> lg;
            iters = (a_len + (1LL << (LOG_NTHR - lg)) - 1) >> (LOG_NTHR - lg);
        }
        while (iters > max_sub * 2 && lg > 2) {
            lg--; max_sub = ((long long)max_bl + (1 << lg) - 1) >> lg;
            iters = (a_len + (1LL << (LOG_NTHR - lg)) - 1) >> (LOG_NTHR - lg);
        }
    }
    const int group_size = 1 << lg;
    const int num_groups = NT >> lg;
    const int my_group = tid >> lg;
    const int my_id = tid & (group_size - 1);

    // ── Pass 1:标记。只读 B 的列号流(4B/乘积);先读后 or——dup 行多数乘积只付一次普通读 ──
    for (int p = rs + my_group; p < re; p += num_groups) {
        const int k = A_ci[p];
        const int ks = B_rp[k], ke = B_rp[k + 1];
        for (int q = ks + my_id; q < ke; q += group_size) {
            const unsigned b = (unsigned)(B_ci[q] - cbase);
            const unsigned bit = 1u << (b & 31u);
            unsigned *wp = bmw + 2u * (b >> 5);
            if (!(*wp & bit)) atomicOr(wp, bit);
        }
    }
    __syncthreads();                                                  // 栅栏 2

    // ── 前缀:组内局部独占前缀(写入各词高半)+ 组计数 ──
    for (int g = tid; g < ng; g += NT) {
        int acc = 0;
        const int w1 = ((g + 1) << 5) < cw ? ((g + 1) << 5) : cw;
        for (int w = g << 5; w < w1; w++) {
            const int t = __popc(bmw[2 * w]);
            bmw[2 * w + 1] = (unsigned)acc;
            acc += t;
        }
        gpre[g] = acc;
    }
    __syncthreads();                                                  // 栅栏 3
    if (tid == 0) {                                                   // 组独占前缀+总数(串行 ng 步;bin4-8 ≤~130,bin9 ≤~900)
        int run = 0;
        for (int g = 0; g < ng; g++) { const int t = gpre[g]; gpre[g] = run; run += t; }
        *ntot = run;
    }
    __syncthreads();                                                  // 栅栏 4
    const int n = *ntot;
    if (n > cap) {                                                    // est 低估:既有 retry 熔断语义
        if (tid == 0) {
            row_nnz[i] = 0;
            atomicExch(overflow_flag, 1);
            if (!atomicExch(&row_ovf[i], 1)) { const int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
        }
        return -1;
    }

    // ── 前缀定稿(加组前缀)+ 键有序直写(位图枚举,位次=前缀+词内序)+ 值区清零 ──
    {
        const unsigned long long khi = ((unsigned long long)(unsigned)i) << 32;
        for (int w = tid; w < cw; w += NT) {
            unsigned lo = bmw[2 * w];
            const unsigned hi = bmw[2 * w + 1] + (unsigned)gpre[w >> 5];
            bmw[2 * w + 1] = hi;
            long long pos = base + (long long)hi;
            const unsigned c0 = (unsigned)cbase + ((unsigned)w << 5);
            while (lo) {
                const unsigned bp = (unsigned)(__ffs((int)lo) - 1);
                tmp_key[pos] = khi | (unsigned long long)(c0 + bp);
                pos++;
                lo &= lo - 1u;
            }
        }
    }
    if (VMODE == 2) { for (int r = tid; r < n; r += NT) tmp_val[base + r] = 0.0; }
    else            { for (int r = tid; r < n; r += NT) sval[r] = 0.0; }
    __syncthreads();                                                  // 栅栏 5(前缀/清零对全块可见)

    // ── Pass 2:累加。位次 = 前缀(高半)+ popc(位图词 & 低位掩码),一次 8B SMEM 读;无探测环 ──
    for (int p = rs + my_group; p < re; p += num_groups) {
        const int k = A_ci[p];
        const double a_ik = A_val[p];
        const int ks = B_rp[k], ke = B_rp[k + 1];
        for (int q = ks + my_id; q < ke; q += group_size) {
            int j; double bv;
            if (B16ON) { const uint4 e = B16[q]; j = (int)e.x; bv = __hiloint2double(e.w, e.z); }
            else { j = B_ci[q]; bv = B_val[q]; }
            const double v = a_ik * bv;
            const unsigned b = (unsigned)(j - cbase);
            const unsigned long long e64 = bm64[b >> 5];
            const int r = (int)(unsigned)(e64 >> 32)
                        + __popc((unsigned)e64 & ((1u << (b & 31u)) - 1u));
            // 双指针+编译期分派:全局指针可证 → 弃返回值的原生 RED;SMEM 指针 → 模拟原子(同哈希路径)
            if (VMODE == 2) atomicAdd(&tmp_val[base + r], v);
            else            atomicAdd(&sval[r], v);
        }
    }
    if (VMODE != 2) {
        __syncthreads();                                              // 栅栏 6(仅 VMODE=1)
        for (int r = tid; r < n; r += NT) tmp_val[base + r] = sval[r];
    }
    if (tid == 0) row_nnz[i] = n;
    return n;
}
