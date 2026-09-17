# ⑤ 合并 atomicAdd v2 突破规格(用户钦定本阶段主目标,09-17)

## 为什么这是突破点
高 dup 阵(crankseg 对 ocean 1.355-1.57 输)的账本:同址 atomicAdd 串行化是最大项(kernel 侧天花板)。
合并 = 把同列乘积在生产者侧聚合(dup=26 时原子+探测双坍缩 ~26×)——**nsparse/spECK 无此先例(已查重)= 自有创新点**。

## 结构约束(v1 教训 + 洞察)
1. **必须保留 G 多 k 并发**:同列 dup 乘积来自不同 k,必须同时在线才能 match 合并(warp-per-k 串行会消灭合并机会)
2. **v1 死因**:q 循环每 lane 步数不同 → `__activemask()` 碎片化 → match/shfl 边界不一致 → 结构性值损坏
3. **v2 解法 = 双层 uniform 谓词化**:
   - k 层:warp 内 32/G 个组的 k 步数统一到 warp-max(`__reduce_max_sync(0xffffffff, my_k_steps)`),耗尽组谓词空转
   - j 层:每 k 的 q 步数同法 warp-max 统一,耗尽 lane 谓词空转(sentinel j = -1)
   - 全程 full mask(0xffffffff)match/shfl——**无 activemask,无同步边界漂移**
4. B16 读在 G 结构下 = 每组连续 G×16B 块 + 相邻组 k 连续 → B 区近乎连续(ncu 复测中,若仍有差再修)

## 核心循环(v2 草案)
```cuda
int my_ksteps = (num_k_iters for my group);
int ksteps = __reduce_max_sync(0xffffffffu, my_ksteps);
for (int ki = 0; ki < ksteps; ki++) {            // warp uniform
    bool kact = ki < my_ksteps;
    int k = kact ? A_col_idx[my_p + ki*num_groups] : -1;
    int q0 = kact ? B_rp[k] + my_id : 0, qlim = kact ? B_rp[k+1] : 0;
    int jsteps = __reduce_max_sync(0xffffffffu, kact ? (qlim-q0+G-1)/G : 0);
    for (int qi = 0; qi < jsteps; qi++) {        // warp uniform
        bool act = kact && q0 + qi*G < qlim;
        int j = act ? B_j(q0+qi*G) : -1;  double v = act ? prod : 0;
        unsigned mm = __match_any_sync(0xffffffffu, j);   // sentinel 自群,无害
        int lane5=tid&31, r=__popc(mm&((1u<<lane5)-1)), gs=__popc(mm);
        double sv=v;
        #pragma unroll
        for (int off=1; off<32; off<<=1) {                // rank butterfly(v1 已写对)
            int pr=r^off, pl=(pr<gs)?(int)__fns((int)mm,-1,pr+1):lane5;
            double pv=__shfl_sync(0xffffffffu, sv, pl);
            if (pr<gs) sv+=pv;
        }
        if (act && lane5==__ffs(mm)-1) { probe(j) once; atomicAdd(slot, sv); }
    }
}
```

## 代价与守门
- 低 dup 行:match/butterfly/谓词 ~10-15 ops/乘积纯开销 → 电池看低 dup 阵(F2/pre2 类)回归幅度,若 >3% 需行级自适应(走 LLB 式成本模型,非门)
- 值级门:msc+crankseg 17 位(DITER on 默认);有序检查
- 电池:×5 10 阵 vs HASH_WM=0;高 dup 阵(crankseg/c-64)为主判定
- 判据:crankseg hash 行对 ocean 中位从 1.355 压向 ≤1.0 = 突破达成

## v2.x 实测终局(09-17)

- **v2.2/__ffs 逐位 gather = 值级正确**(msc/crankseg 17 位 PASS;雷定谳:__fns base/offset 语义陷阱,弃用)
- **性能判词:负**。逐行 WM on 中位 1.795-2.179(off = 1.355-1.571);系统 crankseg WM1 比 WM0 慢 6-15%(125→134-145ms)
- **经济学根因(重要)**:dup=26 的串行是**时间性的**(26 个 k 跨迭代先后砸同一槽),不是**并发性的**——warp 每时刻仅 32/G≈4 个 k 在线 → match 合并的理论上界 = 4×,而 uniform 谓词化的填充开销(match+gather+空转迭代)≈ 20%+ → 净负
- **能吃到时间性 dup 的路**(后续):①G×WM 联动(G=4 → 8 并发 k,合并上界翻倍,B 散读代价回升,需联合扫)②k 按 j 邻域重排(同 j 的 k 相邻化 = 时间转并发)③极端 dup 走 dense 路由(已有)
- 状态:HASH_WM 默认 0 停靠;v2.3(gnm 哨兵群修复)在库
