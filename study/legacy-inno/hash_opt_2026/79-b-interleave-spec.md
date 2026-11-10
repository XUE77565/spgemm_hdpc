# 交织 B 布局手术规格(④,2026-09-17;首刀范围 = hash_spa_kernel)

## 靶子(实测出处)
ncu 三指针(study/15 尸检):我方 L1 命中 9% vs ocean 16%;sec/req 3.51 vs 5.24;请求发起 2.5×。
根因 = B 读流两股窄流(col 4B + val 8B 各一股)+ G 拓扑散读。**本刀治布局半边**(G 拓扑半边不在本刀)。

## 设计

1. **布局**:`uint4 B16[q] = {col, pad, val_lo, val_hi}` —— 16B 对齐,一次 LDG.128 携带 col+val;
   代价 = B 副本 +33% 内存(crankseg 14.1M nnz × 16B ≈ 226MB,预算内)
2. **构建 kernel**(每管线一次):`b_interleave_kernel<<<ceil(nnzB/256),256>>>(B_rp, B_ci, B_v, B16)`
   —— 行结构不变(B_row_ptr 复用,entry 下标即 q),纯格式转换
3. **hash_spa_kernel 改造**:
   - 新参 `const uint4 *B16 = nullptr`(默认空 = 关,行为不变)
   - 内层循环:`uint4 e = __ldg(&B16[q]); int j = (int)e.x; double bv = __hiloint2double(e.w, e.z);`
     原 `B_col_idx[q]`/`B_val[q]` 两次载入 → 一次 128b
4. **门**:`HASH_BI=1` 开(默认 0);B16 仅开启时分配构建;逃生门 = env 归零
5. **守门**:编译 + 值级 17 位(msc/crankseg,DITER off)+ nnz 红旗 ×5 电池(10 阵)vs HASH_BI=0

## 不在本刀(后续梯队)
- G 拓扑散读(warp-per-k 全合并形状;与 G 梯耦合,另刀)
- 其他 14 个 B 消费 kernel(batched/ESC/dense 系;首刀验赢后按收益扩散)
- count_flop/est 相的 B 读(相位占比小)

## 预期
F2/crankseg accumulate 相位 5-15%(B 流占比最高的是高 dup 重 flop 阵 = crankseg 类);
系统级被相位稀释,hash 主导阵可见。ncu 复查三指针(sec/req 3.51→~7+,请求次数 ~减半,L1 命中回升)。
