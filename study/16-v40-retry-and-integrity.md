# v40 SMEM 重试车 + 完整性大发现 + GPU 楔死事故(2026-09-15 深夜)

> 三件事一晚:**① v40 溢出载体刀落地(重试税 −84~−94%);② 揪出存量正确性地雷(diter 路径写垃圾 + 历史值级验证全是 cu-vs-cu 空对照);③ GPU1 驱动楔死,静态排雷完成待 Xid 定谳。**

## ① v40:SMEM 重试车(已实现,真值验证过,电池被事故打断)

设计:run_retry 按行分流——`2×flop ≤ HASH_CAP ∧ flop ≥ 2^10` 的行复用移植体 `hash_spa_ocean_kernel` 走 SMEM 原子(ht=2×max ≥ 2×true 数学不可再溢);鲸鱼行(装不下)单独走老全局表路(positional rht_g/rtab_g 重建,arena 复用)。env `RETRY_SMEM`(默认 1;=0 全老路)。

| 阵 | retry 相位(前→后) |
|---|---|
| F2 | 2.33 → **0.38ms(−84%)** |
| in-2004 | 7.10 → **0.45ms(−92%)** |
| crankseg | 2.19 → 0.96ms |

- margin 复扫:重试便宜了但 crankseg/nd24k 的墙变成**探测链物理墙**(LF↑),全局 margin 维持 1.8/150;F2 单独可吃 1.5/130(−10%,留给 ADAPT_EXPAND 状态机进化)
- 11 阵电池被 GPU 事故打断(F2 通过:−0.6%),**重启后需补跑**

## ② 完整性地雷(比性能重要的发现)

1. **历史"值级 bit-exact"全是空对照**:T4f(adaptive)块根本没有 WRITE_MTX——此前"复制结果文件对比"复制的全是上一次 cuSPARSE 残留文件。**v37/v38/v39 的值级声明作废**(nnz 红旗是真的)。已补 T4f 写出(main.cu)。
2. **补上真对照后立即抓到存量 bug**:msc10848 default 配置 FAIL——**205 万重复条目/3255 行垃圾**,定位 = **diter(bin13)路径**(行内容 [连续段+列1×N] 型垃圾;nnz 计数正确所以红旗瞎)。bin13>0 的阵全部可疑(msc/in-2004/…)。`DITER_MIN_FLOP=0` 关掉后 default+storm 双双真 PASS——**hash 主路径(含 v38/v39/v40)清白**。
3. diter bug 为存量(bisect 全 env 关闭仍在),下一场第一优先修。

## ③ GPU1 楔死事故与排雷

时间线:电池(GPU1)第 12 运行(pre2 老路)None → 后续全灭 → GPU1 "requires reset" → 我 kill 电池后 GPU0 仍正常(pre2 双模式多跑通过)→ c-43 活性测试在 GPU0 自旋 12min CPU 后死 → **驱动级 D-wedge(nvidia-smi 全挂)**。

静态排雷(全部完成):
- **雷①已修**:retry_prep 不可重试行早退留垃圾 rht(裸 alloc 无 memset)→ exclusive_scan → hash_global 越界写。触发条件近乎不可能且为存量,已给合法表值兜底
- 鲸鱼子集重建/SMEM 车/GFORCE/移植体:边界守卫齐全;子集数组泄漏(MB 级,脏无害)
- diter 垃圾 = 行内写错(进程干净退出,无 fault 签名)→ 解释数据损坏、解释不了楔死
- **无静态雷能解释驱动楔死**;第三解读 = GPU1 全天被 ncu(含 2023 版失败 prepare)重编程 + 事故前 storm 重负载连发 = 压力触发驱动失效

**重启后三步定谳**:① `sudo dmesg | grep -i xid`(13/31=应用=我们 / 79/63=硬件驱动);② `compute-sanitizer --tool memcheck` 跑 storm 配置;③ 干净驱动复现当晚序列(F2×10→pre2 老路×3→c-43)。

## 顺手尸检

- int4 向量清表:+16% 回退(标量跨线程清表已最优),已回滚
- 交织 B 布局:未做(优先级让位);素数梯迷思破除(彼梯 331→673→1327 比值 2.03 ≈ 2 幂等宽,彼表小的真因 = margin 1.5-2.0 单系数 + 溢出便宜,梯粒度无关)

## 状态

- 工作树:v40 重试车 + 雷①修复 + T4f WRITE_MTX + GFORCE 旋钮(均可编译,0 error)
- GPU 状态:驱动楔死,**需重启**;GPU1 另有 "requires reset"
- 电池:11 阵 ×5 待重启后补跑(GPU0)
