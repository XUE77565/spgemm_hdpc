// ═══════════════════════════════════════════════════════════════════════════
// 刀E6(study/57 §三十三):ocean hashNumericKernel 替换消融移植(10-25,用户批准)
//
// 归属:枚举循环、探测(operator[])、localLoadBalance、init/compactAndSort 的
// 算法结构逐行来自 ocean/(挑战基线,其 repo 带 citation 章节);移植仅做管线
// 接口适配,全部以 /* PORT: */ 标注——受试对象(乘积枚举+探测环)保持原样。
// 目的:检验"23 条/乘积"的活环经济性是否可在异管线中存活(E1 测的是我方
// 自研复刻,非其字面源码——本实验补上证据链最后一环)。
// 许可与使用边界:env HASH_OCNK 默认关;默认生产行为零影响;是否转正/归属
// 声明由用户按挑战规则裁决。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

// ── localLoadBalance(ocean AccumulatorCommon.cuh:67-106 完整逐行原样;审计修复:首轮移植漏了下行/第三调整循环与钳位)──
__device__ __forceinline__ unsigned oc_localLoadBalance(unsigned n_element_a, unsigned num_products,
                                                        unsigned max_elements_b, unsigned lbound, unsigned ubound) {
    if (num_products == 0) return 0;
    unsigned avg_ops = max(1U, (num_products - max_elements_b) / max(1U, n_element_a - 1));
    unsigned log_nthr = max(1U, (31 - __clz(avg_ops)));
    if ((1U << log_nthr) * 3 < avg_ops * 2) { log_nthr += 1; }
    unsigned a_elem_per_iter = 1 << (ubound - log_nthr);
    unsigned num_iters = (n_element_a + a_elem_per_iter - 1) / a_elem_per_iter;
    unsigned max_sub_iter = (max_elements_b + (1<<log_nthr) - 1) >> log_nthr;
    while (max_sub_iter > num_iters * 2) {
        if (log_nthr >= ubound) break;
        log_nthr += 1;
        max_sub_iter = (max_elements_b + (1<<log_nthr) - 1) >> log_nthr;
        a_elem_per_iter = 1 << (ubound - log_nthr);
        num_iters = (n_element_a + a_elem_per_iter - 1) / a_elem_per_iter;
    }
    while (num_iters > max_sub_iter * 2) {
        if (log_nthr <= lbound) break;
        log_nthr -= 1;
        max_sub_iter = (max_elements_b + (1<<log_nthr) - 1) >> log_nthr;
        a_elem_per_iter = 1 << (ubound - log_nthr);
        num_iters = (n_element_a + a_elem_per_iter - 1) / a_elem_per_iter;
    }
    while (a_elem_per_iter > n_element_a && (1 << log_nthr) < max_elements_b) {
        log_nthr += 1;
        a_elem_per_iter = 1 << (ubound - log_nthr);
    }
    if (log_nthr < lbound) log_nthr = lbound;
    if (log_nthr > ubound) log_nthr = ubound;
    return log_nthr;
}

#define OC_HASH_UNUSED 0xffffffffu

// ── NumericHashmap::operator[](ocean Hashmap.cuh:30 原样;hashKernel=id*11 原样)──
// 返回值槽指针;调用方 atomicAdd。占位原子 = 原样(CAS+occupancy)。
__device__ __forceinline__ int oc_hashKernel(unsigned id) { return id * 11; }

// ── hashNumericKernel 移植(ocean AccumulatorHash.cuh:513 起)──────────────
// 差异清单(全为 PORT 接口适配,算法结构零改动):
//   ① 尾部 compactAndSort 写出 → 我方 tmp 契约(tmp_key 打包 i<<32|c / tmp_val)
//   ② CHECK_OVERFLOW 溢出分支 → 我方 row_ovf/ovf_cnt/ovf_rows/overflow_flag(retry 机具)
//   ③ ROWCYCLES/HYBRID/global-buffer 机具 → 删除(非受试对象)
//   ④ A/B 数组直喂我方指针(int 列号/偏移 long long;值域 32 位内安全)
template<int LOG_NTHR, int HASH_SIZE, bool CHECK_OVERFLOW, typename offset_t>
__global__ __launch_bounds__(1 << LOG_NTHR, 2048 >> LOG_NTHR) void ocean_hash_kernel(
    const int *A_row_ptr, const int *A_col_ind, const double *A_values,   /* PORT: index_t→int */
    int A_rows, int A_cols, int nnz_a,                                     /* PORT: 尺寸同型 */
    const int *B_row_ptr, const int *B_col_ind, const double *B_values,
    int B_rows, int B_cols, int nnz_b,
    const int *num_products,          /* PORT: = 我方 d_flop(精确乘积数,同源) */
    const int *max_elements_b,        /* PORT: = 我方 d_maxbl */
    const int *bucket_rows, int n_in_bucket,                                /* PORT: 行表 */
    const long long *output_row_offsets,   /* PORT: = 我方 d_off(est 偏移,同 wf1 语义) */
    unsigned long long *tmp_key, double *tmp_val,                          /* PORT: 我方 tmp 契约 */
    int *row_nnz,                                                         /* PORT: 我方 row_nnz */
    int *overflow_flag, int *ovf_rows, int *ovf_cnt, int *row_ovf,        /* PORT: 我方溢出机具 */
    int my_row_shift = 0)                                                 /* PORT: tmp_key 高 32 位行号(打包契约) */
{
    extern __shared__ uint8_t dynamicShared[];
    const int tid = threadIdx.x;

    unsigned *oc_ids = (unsigned *)(&dynamicShared[4 * sizeof(unsigned)]);           // 原样布局
    double   *oc_values = (double *)(&dynamicShared[4 * sizeof(unsigned) + sizeof(unsigned) * ((HASH_SIZE + 1) / 2 * 2)]);
    __shared__ int oc_occ;

    int my_row = bucket_rows[blockIdx.x];                                 /* PORT: 行表直取 */
    const int row_tag = my_row + my_row_shift;                            /* PORT: 打包行号 */

    // hmap.init(原样:清 ids/值/occupancy)
    if (tid == 0) oc_occ = 0;
    for (int i = tid; i < HASH_SIZE; i += blockDim.x) oc_ids[i] = OC_HASH_UNUSED;
    for (int i = tid; i < HASH_SIZE; i += blockDim.x) oc_values[i] = 0;
    __syncthreads();

    int start_element_a = A_row_ptr[my_row];
    int end_element_a   = A_row_ptr[my_row + 1];

    // localLoadBalance(原样调用形态)
    int log_nthr_per_a_elem = oc_localLoadBalance(
        end_element_a - start_element_a,
        num_products[my_row],
        max_elements_b[my_row],
        2, LOG_NTHR);

    int my_group_id = threadIdx.x >> log_nthr_per_a_elem;
    int num_groups = blockDim.x >> log_nthr_per_a_elem;
    int my_id_in_group = threadIdx.x & ((1 << log_nthr_per_a_elem) - 1);
    int group_size = 1 << log_nthr_per_a_elem;

    // ══ 受试活环(ocean 582-606 逐行原样)══
    for (int element_a = start_element_a + my_group_id;
         element_a < end_element_a;
         element_a += num_groups) {

        double a_value = A_values[element_a];
        int b_start = B_row_ptr[A_col_ind[element_a]];
        int b_end = B_row_ptr[A_col_ind[element_a] + 1];

        for (int b_element = b_start + my_id_in_group;
             b_element < b_end;
             b_element += group_size) {

            double b_value = B_values[b_element];
            unsigned b_col = (unsigned)B_col_ind[b_element];

            if (CHECK_OVERFLOW) {                        // 原样:每乘积 nearFull
                if (oc_occ >= HASH_SIZE * 4 / 5) break;  /* PORT: *hmap.occupancy 内联读(原为 nearFull()) */
            }
            // hmap[b_col] 原样:do-while 探测 + CAS + occupancy
            unsigned hashed_id = oc_hashKernel(b_col);
            unsigned map_id = hashed_id % (unsigned)HASH_SIZE;
            double *loc = nullptr;
            do {
                unsigned entry = oc_ids[map_id];
                if (entry == b_col) { loc = oc_values + map_id; break; }
                if (entry == OC_HASH_UNUSED) {
                    unsigned old_id = atomicCAS(&oc_ids[map_id], OC_HASH_UNUSED, b_col);
                    if (old_id == OC_HASH_UNUSED || old_id == b_col) {
                        if (old_id == OC_HASH_UNUSED) atomicAdd((unsigned *)&oc_occ, 1);  // 原样 occupancy
                        loc = oc_values + map_id;
                        break;
                    }
                }
                map_id = map_id + 1;
                if (map_id >= (unsigned)HASH_SIZE) map_id = 0;
            } while (true);
            if (loc) atomicAdd(loc, a_value * b_value);   // 原样:调用方值累加
        }
    }
    __syncthreads();
    // ══ 活环结束 ══

    if (CHECK_OVERFLOW) {   /* PORT: 溢出 → 我方 retry 机具(原为 buffer 分配) */
        if (oc_occ >= HASH_SIZE * 4 / 5) {
            if (tid == 0) {
                atomicExch(overflow_flag, 1);
                if (!atomicExch(&row_ovf[my_row], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = my_row; }
                row_nnz[my_row] = HASH_SIZE;
            }
            return;
        }
    }

    // compactAndSort(ocean 444-488 原样结构;PORT: 写出改我方 tmp 契约)
    __shared__ int oc_counter;
    if (tid == 0) oc_counter = 0;
    __syncthreads();
    int total_iter = (HASH_SIZE + blockDim.x - 1) / blockDim.x;
    #pragma unroll
    for (int it = 0; it < total_iter; it++) {
        int i = it * blockDim.x + tid;
        double my_val = 0; unsigned my_idx = OC_HASH_UNUSED;
        if (i < HASH_SIZE) { my_val = oc_values[i]; my_idx = oc_ids[i]; }
        __syncthreads();
        if (my_idx != OC_HASH_UNUSED) {
            int pos = atomicAdd(&oc_counter, 1);
            oc_ids[pos] = my_idx; oc_values[pos] = my_val;   // 原样:表内前部压缩
        }
    }
    __syncthreads();
    int count = oc_counter;
    offset_t base = output_row_offsets[my_row];
    for (int i = tid; i < count; i += blockDim.x) {
        unsigned current_idx = oc_ids[i];
        int loc = 0;
        for (int j = 0; j < count; j++) {
            if (oc_ids[j] < current_idx) loc++;
        }
        tmp_key[base + loc] = ((unsigned long long)(unsigned)row_tag << 32) | current_idx;  /* PORT: 打包写 */
        tmp_val[base + loc] = oc_values[i];
    }
    if (tid == 0) row_nnz[my_row] = count;                 /* PORT: 我方 row_nnz */
}
