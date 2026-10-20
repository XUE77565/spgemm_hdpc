# study/51:今日自动化优化链(10-20 下午起,用户汇报备赛期间)

> 输面地图刷新(10-20):靶2 判死——新 top 输家(Long_Coup 3.9×/bloweybq 3.0×/ldoor 2.7×/
> rgg 2.3×/StocF 2.2×)C/A≈1.00,无路由解药;带状族已在 10-17 收割(唯一 C/A≪1 人群)。
> 剩余输面 = accumulate 单位效率(kernel 级)。ESC64 已证伪(同人群),指向访存结构。

## 状态机(值守 loop 按序推进,勾选留痕)

- [ ] A accumulate 取证(新 top 输家四阵):Long_Coup_dt6/ldoor/rgg_n_2_22_s0/StocF-1465
      ①相位份额(我 A 臂 sidecar vs ocean ocn_stats)定位 2.2-3.9× 集中在哪个相位
      ②DUMP_ROUTING 看 bkid 分布(哪类 kernel 伺候它们的行)
      ③输出:差距解剖 + 下一刀候选排序(纯分析零风险)
- [ ] B CD 溢出修雷(study/47):定位 DCEST 链的 int32 C_nnz/margin → long long 化 +
      负值即判废 + cudaMalloc size_t 卫兵;build → 门:c-73/c-73b CD 臂出值且
      Result C nnz == C 臂(值级 WRITE_MTX 门二选一)→ 控制阵无害 → commit
- [ ] C(时间充裕)hybrid-hash 设计文档:依 A 的取证结论,写中带(索引 smem/值 gmem RED)
      的实现方案(只设计不改码,明天汇报后人工过目再动)
- [ ] 终报:入档 + commit + 通知

## 纪律
改码必 env 门控默认关/值门三件套(Result C 锚定!grep 教训)/同会话 A/B/
控制阵无害/Xid 哨兵/GPU 陌生租户即停/含糊即停不猜。汇报日不动记分二进制语义。
