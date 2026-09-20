# Retry 2.0 档位制快重试(F2 跨线刀,09-20 立项;设计基础=study/22 件3+本轮代码解剖)

## 现状解剖(09-20 读码定谳,src/spgemm_kernel_hash.cu)
1. **v40 SMEM 重试车已存在**(4816-4861,RETRY_SMEM 默认 1):溢出行按 `flop≥1024 ∧ 2×flop≤HASH_CAP`
   分流上 hash_spa 二次发射——但配表 = **批量统一 ht=pow2≥2×max(flop)**(一个肥行抬全批)
2. **准入缝隙 = study/22 解剖的病灶**:est 745/flop 13434 型行(小 est 大 flop)因 2×flop 超 HASH_CAP
   被推全局路;flop<1024 的小溢出行也全落全局
3. **全局路配表**:retry_prep(2543)h = pow2≥2×min(flop,ncols) 且**下限 2×HASH_CAP=32K 槽**
   ——小溢出行也配 32K 槽全局表(384KB arena/行)= 冷原子+大表 init,retry 1.33μs/行=主遍历 4-5×
4. **F2 固定成本 0.475ms(0 溢出行也付,study/22 待查)**:run_retry(4761)每轮先 blocking
   cudaMemcpy 读 d_ovf_cnt = accumulate 后首个 host 同步点 = 全管线 drain 落在 retry 相位账上

## 刀(三件,R0→R2 单刀序)
- **R0 固定成本**:①插桩定位(cudaEvent 细分 memcpy/scan/launch 或 nsys 一发)②若=drain:
  pinned 异步读回+event 等(只等 accumulate 不等全机)③相位和口径下验证 0 溢出行时 retry 相位归零
- **R1 档位制配表**(主刀):SMEM 车配表从"批量统一 2×max(flop)"改"**原表 pow2 上探一档 ht2=2×ht**"
  ——溢出仅意味着 distinct>ht(est 误差通常小,σ=1.04/√k),一档 2× 几乎总够;ht2>HASH_CAP 或
  再溢出 → 上探下一档(链式,≤2 SMEM 档)→ 仍败才全局 flop 路(终止性:distinct≤min(flop,ncols) 有界)
- **R2 准入放宽**:地板 flop≥1024 取消(flop<1024 的溢行走 ht2 小表 SMEM 车更便宜);
  鲸鱼(flop 大∧ht2 超帽)维持全局路不动

## 守门(三件套全量)
值级(truth17+compare_mtx_values.py;大阵 WM=METHOD=hash+self_product_hash.mtx)+
行内有序+memcheck(retry 是 study/17 双卡楔死案发地——positional 子集重建的未初始化尾段雷,
改动触碰 rows_g 补集逻辑必须 memcheck);c-64 artifact(20,504)豁免

## 电池(A 级,需净窗;本轮 09-20 双卡全占被阻塞)
- 主判 F2(retry 段 0.58ms→<0.3ms;系统 8.01→目标 ≤7.66=跨线);守卫 crankseg/c-64/msc
  +扩 bench 新阵(in-2004/nd24k/wb-edu/web-G/c-43/3tube,先建 off 带)
- off=c78t,新刀独立命名;×5 起步,跨线判定点 ×7
- env 旋钮:RETRY2=1 开关(默认开前守门全绿);RETRY_SMEM 保留兼容

## 预期与风险
- 预期:F2 retry 段 −0.3ms+;顺带解锁 est 收紧经济学(3689 注释:小阵被迫 EXPAND 1.4 就是为躲
  retry 固定成本——R0 修好后紧 est 可再试,bcsstk30 教训重评)
- 风险:①ht2 再溢出率若高(est 误差重尾)→链式档变慢,电池判;②分流逻辑改动触碰 study/17 雷
  (必须复用 rows_g 本体);③0.475 若非 drain 而是 PCIe RTT+host wake → pinned 只减半,记录实测定量

## 状态
- 09-20 11:53 loop 第 1 轮:spec 立;环境=GPU0 44.5GB+GPU1 78.8GB/98%(双卡全占)→实施可写码
  编译,C 级值级可跑(GPU0 闲置 SM),A 级电池等净窗
