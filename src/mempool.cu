#include "mempool.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>

// 运行时开关,默认 false(原路径)。main 启动时按 USE_MEMPOOL 覆盖。
bool g_use_mempool = false;
// device arena 独立开关(USE_DEV_POOL,默认关):省 driver 往返,但拖慢小阵 hash 扫描(compute-only 负优化)。
bool g_use_dev_pool = false;

// pinned arena 的全部状态。单线程串行使用(benchmark 一矩阵一进程),无需加锁。
static char*  g_base = nullptr;
static size_t g_cap  = 0;
static size_t g_off  = 0;

// device arena(见 mempool.h):bump-reset 模型,dev_alloc 仅步进指针(0 driver call)。
static char*  g_dev     = nullptr;
static size_t g_dev_cap = 0;
static size_t g_dev_off = 0;

// 内部:从 arena bump 切一块(16 字节对齐)。调用前需保证已 reset 到合适起点
// 溢出回退(审计#4,10-14):不再 exit 自杀——逐次直接分配并登记,free/destroy 对应处理。
// (昨天 WRITE_MTX 雷即此路径:鲸鱼阵 C 超池即杀进程。benchmark 一矩阵一进程,16 槽足够。)
static void* g_fb_host[16]; static int g_fb_host_n = 0;
static void* g_fb_dev[16];  static int g_fb_dev_n = 0;
static void fb_track(void** lst, int* n, void* p) { if (p && *n < 16) lst[(*n)++] = p; }
static bool fb_is(void** lst, int n, void* p) { for (int i = 0; i < n; i++) if (lst[i] == p) return true; return false; }
static void* arena_alloc(size_t bytes) {
    if (bytes == 0) bytes = 1;
    size_t aligned = (g_off + 15) & ~((size_t)15);
    if (aligned + bytes > g_cap) {
        fprintf(stderr,
                "[mempool] HOST arena overflow: need %zu B, cap %zu B, used %zu B "
                "→ 回退直接分配(审计#4;原为 exit)\n", bytes, g_cap, g_off);
        void* p = nullptr;
        if (cudaMallocHost(&p, bytes) == cudaSuccess && p) {
            fb_track(g_fb_host, &g_fb_host_n, p);
            return p;
        }
        fprintf(stderr, "[mempool] 回退 cudaMallocHost(%zu B) 也失败\n", bytes);
        return nullptr;
    }
    g_off = aligned + bytes;
    return g_base + aligned;
}

bool mempool_init(size_t cap_bytes) {
    if (!g_use_mempool) {
        fprintf(stderr, "[mempool] OFF — legacy cudaMallocHost (USE_MEMPOOL unset/0)\n");
        return true;  // 原路径:不分配 arena,零开销
    }
    if (cap_bytes == 0) {
        const char* e = std::getenv("MP_HOST_MB");
        cap_bytes = e ? (size_t)std::atol(e) * 1024 * 1024
                      : (size_t)256 * 1024 * 1024;  // 默认 256MB
    }
    g_cap = cap_bytes;
    cudaError_t err = cudaMallocHost((void**)&g_base, g_cap);
    if (err != cudaSuccess || !g_base) {
        fprintf(stderr, "[mempool] cudaMallocHost(%zu MB) failed: %s\n",
                g_cap >> 20, cudaGetErrorString(err));
        g_base = nullptr;
        g_cap = 0;
        return false;
    }
    fprintf(stderr, "[mempool] host pinned arena ready: %zu MB (USE_MEMPOOL=ON)\n",
            g_cap >> 20);

    // device arena:复用 ~13 个 device buffer,省 ~0.66ms driver 往返。默认 8GB(MP_DEV_MB)。仅 USE_DEV_POOL=1 分配。
    if (!g_use_dev_pool) {
        fprintf(stderr, "[mempool] device arena OFF (USE_DEV_POOL unset; compute-only 友好)\n");
        return true;
    }
    const char* ed = std::getenv("MP_DEV_MB");
    size_t dcap = ed ? (size_t)std::atol(ed) * 1024 * 1024
                     : (size_t)16384 * 1024 * 1024;
    cudaError_t errd = cudaMalloc((void**)&g_dev, dcap);
    if (errd != cudaSuccess || !g_dev) {
        fprintf(stderr, "[mempool] cudaMalloc dev arena (%zu MB) failed: %s — "
                "fallback 到 per-call cudaMalloc(性能不变,仅失去 pool 收益)\n",
                dcap >> 20, cudaGetErrorString(errd));
        g_dev = nullptr; g_dev_cap = g_dev_off = 0;
    } else {
        g_dev_cap = dcap; g_dev_off = 0;
        fprintf(stderr, "[mempool] device arena ready: %zu MB\n", g_dev_cap >> 20);
    }
    return true;
}

void mempool_destroy() {
    for (int i = 0; i < g_fb_host_n; i++) cudaFreeHost(g_fb_host[i]);   // 溢出回退块
    g_fb_host_n = 0;
    for (int i = 0; i < g_fb_dev_n; i++) cudaFree(g_fb_dev[i]);
    g_fb_dev_n = 0;
    if (g_base) {
        cudaFreeHost(g_base);  // arena 本身的锁页内存用 cudaFreeHost 释放(合法)
        g_base = nullptr;
    }
    g_cap = g_off = 0;
    if (g_dev) {
        cudaFree(g_dev);
        g_dev = nullptr;
    }
    g_dev_cap = g_dev_off = 0;
}

// device arena API(见 mempool.h)
void dev_pool_reset() {
    if (g_use_dev_pool) g_dev_off = 0;
}

static bool g_malloc_async = false;
static bool g_ma_init = false;
static void ma_init_once() {   // docs/32 修法1:Ocean 池纪律 —— releaseThreshold=MAX = 迭代间同尺寸零 driver 往返
    if (g_ma_init) return;
    g_ma_init = true;
    const char *e = std::getenv("MALLOC_ASYNC");
    g_malloc_async = (e && *e && std::atoi(e) > 0);
    if (g_malloc_async) {
        cudaMemPool_t pool;
        if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess) {
            uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
        }
        fprintf(stderr, "[mempool] MALLOC_ASYNC=ON(cudaMallocAsync + releaseThreshold=MAX)\n");
    }
}
void* dev_alloc(size_t bytes) {
    if (bytes == 0) bytes = 1;
    if (!g_use_dev_pool) {                  // legacy:原路径 cudaMalloc(A/B 同二进制)
        ma_init_once();
        if (g_malloc_async) {
            void *ap = nullptr;
            if (cudaMallocAsync(&ap, bytes, (cudaStream_t)0) == cudaSuccess && ap) return ap;
            // 罕见失败 → 落回同步路径
        }
        void* p = nullptr;
        cudaError_t err = cudaMalloc(&p, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "[mempool] cudaMalloc(%zu B) failed: %s\n",
                    bytes, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        return p;
    }
    size_t aligned = (g_dev_off + 15) & ~((size_t)15);   // 16B 对齐(double/uint4 友好)
    if (aligned + bytes > g_dev_cap) {
        fprintf(stderr,
                "[mempool] DEV arena overflow: need %zu B, cap %zu MB, used %zu MB "
                "→ 回退直接分配(审计#4;原为 exit)\n", bytes, g_dev_cap >> 20, g_dev_off >> 20);
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) == cudaSuccess && p) {
            fb_track(g_fb_dev, &g_fb_dev_n, p);
            return p;
        }
        fprintf(stderr, "[mempool] 回退 cudaMalloc(%zu B) 也失败\n", bytes);
        return nullptr;
    }
    g_dev_off = aligned + bytes;
    return g_dev + aligned;
}

void dev_free(void* p) {
    if (!g_use_dev_pool) {
        if (g_malloc_async) cudaFreeAsync(p, (cudaStream_t)0);
        else cudaFree(p);
        return;
    }
    if (fb_is(g_fb_dev, g_fb_dev_n, p)) cudaFree(p);   // 溢出回退块;池内块 no-op(reset 回收)
}

size_t dev_pool_used() { return g_dev_off; }

cudaError_t pinned_d2h_alloc(void** out, size_t bytes) {
    if (g_use_mempool) {
        if (!g_base) {   // init 失败残留态(审计#5):flag 保持 true(A_buffer 已 pinned),
                         // 逐次直接分配并登记,host_free/pinned_free 走 cudaFreeHost 语义不破
            void* p = arena_alloc(bytes);   // g_cap=0 → 必走溢出回退路径
            if (!p) return cudaErrorMemoryAllocation;
            *out = p;
            return cudaSuccess;
        }
        g_off = 0;            // reset:上一方法的 C 已被调用者消费,arena 从头复用
        *out = arena_alloc(bytes);
        return (*out) ? cudaSuccess : cudaErrorMemoryAllocation;
    }
    return cudaMallocHost(out, bytes);  // 原路径
}

void pinned_free(void* p) {
    if (!g_use_mempool) { cudaFreeHost(p); return; }
    if (fb_is(g_fb_host, g_fb_host_n, p)) cudaFreeHost(p);   // 溢出/残留态回退块;池内块 no-op
}

void host_free(void* p) {
    if (!p) return;
    if (g_use_mempool) cudaFreeHost(p);   // A_buffer 经 cudaMallocHost 锁页
    else                free(p);           // A_buffer 经 malloc(pageable)
}

size_t mempool_cap() { return g_cap; }
size_t mempool_used() { return g_off; }
