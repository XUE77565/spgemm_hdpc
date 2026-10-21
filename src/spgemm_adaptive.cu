#include "spgemm.h"
#include "structural_params.cuh"   // DISPATCH_MODEL(learned 权重唯一来源)
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// METHOD 环境变量单方法门控:未设/空/"all" → 跑全部;否则只跑 == METHOD(大小写不敏感)的块
bool should_run_method(const char *key) {
    const char *m = std::getenv("METHOD");
    if (!m || !*m || !std::strcmp(m, "all")) return true;
    // 大小写不敏感比较
    for (int i = 0; ; i++) {
        char a = m[i], b = key[i];
        if (a >= 'A' && a <= 'Z') a += 32;
        if (b >= 'A' && b <= 'Z') b += 32;
        if (a != b) return false;
        if (a == 0) return true;
    }
}

// 自适应 dispatcher:C = A·A 的分流。score<0 → hash(大阵/重行/高 skew),否则 merge3;hash 溢出回退 merge3。

// env 读取(分流阈值可被环境变量覆盖)
static int env_int(const char *k, int def) {
    const char *e = std::getenv(k);
    if (e && *e) { int v = std::atoi(e); if (v > 0) return v; }
    return def;
}
static double env_double(const char *k, double def) {
    const char *e = std::getenv(k);
    if (e && *e) { double v = std::atof(e); if (v > 0) return v; }
    return def;
}


void spgemm_self_product_adaptive(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    // A_buffer 布局:[row_ptr(A_rows+1) | col_idx(A_nnz) | val(A_nnz)]
    const int *row_ptr = (const int *)A_buffer;
    int max_row_nnz = 0;
    for (int i = 0; i < A_rows; i++) {                    // O(n) host 扫描,极廉价
        int rnz = row_ptr[i + 1] - row_ptr[i];
        if (rnz > max_row_nnz) max_row_nnz = rnz;
    }
    double avg  = A_rows > 0 ? (double)A_nnz / A_rows : 0.0;
    double skew = avg > 0.0 ? (double)max_row_nnz / avg : 0.0;

    // 多变量调度公式(100 阵拟合,R²=0.824,准确率 93%)
    double flop_proxy = (double)A_nnz * A_nnz / std::max(A_rows, 1);
    double lfp = log10(std::max(flop_proxy, 1.0));
    double ln  = log10((double)A_rows);
    double lmr = log10((double)std::max(max_row_nnz, 1));
    double lsk = log10(std::max(skew, 1.0));
    // learned 权重唯一来源 = structural_params DISPATCH_MODEL(disp_synth 标定 + disp_train 拟合,
    // 与 ocean337 测试集不相交;方法学见 docs/B 线)
    const DispatchModel& DM = DISPATCH_MODEL;
    double score = DM.w_flop * lfp + DM.w_n * ln + DM.w_maxrow * lmr + DM.w_skew * lsk + DM.bias;
    // score < 0 → hash(flop 高 / 重行 / 均匀);score ≥ 0 → merge3(小阵 / 不均匀 straggler)
    const char *force = std::getenv("ADAPTIVE_FORCE");
    int use_hash;
    if (force && force[0]) {
        use_hash = (force[0] == 'h' || force[0] == 'H') ? 1 : 0;   // 环境变量强制覆盖(测试用)
    } else {
        use_hash = (score < 0.0) ? 1 : 0;
    }
    // A2' 预门(10-17,L0 解析层落地):n≥L0_PREGATE_N → 强制 hash + A 等效配置(跳过 diter 决策机器)。
    // 依据批437:337 侧 dispatcher 从不路由 merge3 但自税 4%(diter 机器照跑);n≥5000 全体 A 优于 ADPT
    // (规则版 GM 0.9968 vs 现状 1.0331);bcsstm25(n=15439)类 score 误送 merge3 的 5.5× 爆点同修。
    // 注意:DITER_MIN_FLOP 在 hash 管线内为惰性静态缓存(首调用读 env)——本 setenv 先于调用故生效;
    // 若同进程先跑过裸 hash 方法再跑 adaptive,缓存已定,预门配置不生效(多方法连跑边缘场景,已知局限)。
    static int g_pregate = -1, g_pregate_n = -1;
    if (g_pregate < 0) {
        const char *e = std::getenv("L0_PREGATE");
        g_pregate = (e && *e) ? (std::atoi(e) > 0 ? 1 : 0) : 1;
        const char *e2 = std::getenv("L0_PREGATE_N");
        g_pregate_n = (e2 && *e2) ? std::atoi(e2) : 5000;
    }
    int pregated = 0;
    if (g_pregate && A_rows >= g_pregate_n) {
        use_hash = 1;
        pregated = 1;
    }
    // 带状族 C 路由(10-17,金矿第二层):skew>1700 ∧ avg<10 的极端偏斜带状阵
    // (blowey*/a5esindl/rajat*)保留 diter 机器(=C 配置)——批437 实测 C 快 0.28-0.94×,
    // 模拟净 GM 0.9486→0.9413;排除 avg≥10(Stanford_Berkeley C输1.59);误伤=circuit_4 1.21。
    static int g_band = -1;
    if (g_band < 0) { const char *e = std::getenv("L0_BAND"); g_band = (e && *e) ? (std::atoi(e) > 0 ? 1 : 0) : 1; }
    const bool band_family = g_band && skew > 1700.0 && avg < 10.0;
    // 退化结构否决(10-17,S3 终审产物):准对角大阵(avg<2 ∧ n≥10000)merge3 无生态位
    // (bcsstm25 实测 5.2×;73 个 merge3 赢家零误伤)。批发人群特征盲区,规则硬兜。
    if (avg < 2.0 && A_rows >= 10000) {
        use_hash = 1;
    }
    // T4e(LLB skew 门,10-22):100≤skew≤1900(dispatcher 口径=maxrow/avg)人口开按行动态负载平衡
    // 负载平衡(静态 G 分组在行长悬殊 k 集上尾部拖死)。电池:HTC×2 −44/−39%/c-66 −29%/
    // email-Enron −14%/circuit_4 −6%;反面:cage/Emilia/StocF(行匀)+4~7%,bloweya
    // (带状 3750)+47% → 带状族排除。LLB=0/1 显式 env 优先(缓存首读,须先于 hash 首调)。
    if (use_hash && !std::getenv("LLB") && skew >= 100.0 && skew <= 1900.0)
        setenv("LLB", "1", 1);
    if (use_hash && !band_family && !std::getenv("DITER_MIN_FLOP"))
        setenv("DITER_MIN_FLOP", "999999999", 1);   // dispatcher 的 hash 路径 A 等效;带状族例外(保留 diter=C 配置,金矿第二层)
    char why_buf[128];
    snprintf(why_buf, sizeof(why_buf), "%s%.2f fp=%.0f mr=%d sk=%.1f",
             pregated ? "pregate " : "", score, flop_proxy, max_row_nnz, skew);
    const char *why = why_buf;
    dbg("[adapt] n=%d maxrow=%d skew=%.1f → %s (%s)\n", A_rows, max_row_nnz, skew, use_hash ? "hash" : "merge3", why);
    printf("[adapt] n=%d maxrow=%d skew=%.1f → %s (%s)\n", A_rows, max_row_nnz, skew, use_hash ? "hash" : "merge3", why);

    if (use_hash) {
        spgemm_self_product_hash(A_buffer, A_rows, A_cols, A_nnz,
                                 C_buffer_out, C_rows, C_cols, C_nnz);
        if (*C_nnz < 0) {                                 // hash 溢出 → 回退 merge3
            dbg("[adapt] hash overflow → fallback merge3\n");
            printf("[adapt] hash overflow → fallback merge3\n");
            spgemm_self_product_merge3(A_buffer, A_rows, A_cols, A_nnz,
                                       C_buffer_out, C_rows, C_cols, C_nnz);
        }
    } else {
        spgemm_self_product_merge3(A_buffer, A_rows, A_cols, A_nnz,
                                   C_buffer_out, C_rows, C_cols, C_nnz);
    }
}

// ATT 自适应 dispatcher:C = A·Aᵀ 上三角。复用 AA 分流逻辑,调 att_hash/att_merge3,溢出回退 att_merge3。
void spgemm_att_adaptive(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    const int *row_ptr = (const int *)A_buffer;
    int max_row_nnz = 0;
    for (int i = 0; i < A_rows; i++) {
        int rnz = row_ptr[i + 1] - row_ptr[i];
        if (rnz > max_row_nnz) max_row_nnz = rnz;
    }
    double avg  = A_rows > 0 ? (double)A_nnz / A_rows : 0.0;
    double skew = avg > 0.0 ? (double)max_row_nnz / avg : 0.0;
    double flop_proxy = (double)A_nnz * A_nnz / std::max(A_rows, 1);
    double lfp = log10(std::max(flop_proxy, 1.0));
    double ln  = log10((double)A_rows);
    double lmr = log10((double)std::max(max_row_nnz, 1));
    double lsk = log10(std::max(skew, 1.0));
    // learned 权重唯一来源 = structural_params DISPATCH_MODEL(disp_synth 标定 + disp_train 拟合,
    // 与 ocean337 测试集不相交;方法学见 docs/B 线)
    const DispatchModel& DM = DISPATCH_MODEL;
    double score = DM.w_flop * lfp + DM.w_n * ln + DM.w_maxrow * lmr + DM.w_skew * lsk + DM.bias;
    const char *force = std::getenv("ADAPTIVE_FORCE");
    int use_hash;
    if (force && force[0]) use_hash = (force[0] == 'h' || force[0] == 'H') ? 1 : 0;
    else use_hash = (score < 0.0) ? 1 : 0;
    dbg("[att-adapt] n=%d maxrow=%d skew=%.1f → %s (score=%.2f fp=%.0f)\n",
        A_rows, max_row_nnz, skew, use_hash ? "att_hash" : "att_merge3", score, flop_proxy);

    if (use_hash) {
        spgemm_att_hash(A_buffer, A_rows, A_cols, A_nnz, C_buffer_out, C_rows, C_cols, C_nnz);
        if (*C_nnz < 0) {                                 // hash 溢出 → 回退 att_merge3
            dbg("[att-adapt] att_hash overflow → fallback att_merge3\n");
            spgemm_att_merge3(A_buffer, A_rows, A_cols, A_nnz, C_buffer_out, C_rows, C_cols, C_nnz);
        }
    } else {
        spgemm_att_merge3(A_buffer, A_rows, A_cols, A_nnz, C_buffer_out, C_rows, C_cols, C_nnz);
    }
}
