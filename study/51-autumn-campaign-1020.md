# study/51:今日自动化优化链(10-20 下午起,用户汇报备赛期间)

> 输面地图刷新(10-20):靶2 判死——新 top 输家(Long_Coup 3.9×/bloweybq 3.0×/ldoor 2.7×/
> rgg 2.3×/StocF 2.2×)C/A≈1.00,无路由解药;带状族已在 10-17 收割(唯一 C/A≪1 人群)。
> 剩余输面 = accumulate 单位效率(kernel 级)。ESC64 已证伪(同人群),指向访存结构。

## 状态机(值守 loop 按序推进,勾选留痕)

- [x] A 取证(10-20 午后):**差距集中 accumulate**(rgg 33vs18=1.83×/StocF 15vs6=2.5×/ldoor 28vs21=1.33×;
  Long_Coup 已平 1.04×,confirm4 的 3.9× 为 FLAT 前旧数)。**人群钉死:中 est(64-1024)×
  中高 dup 的梯 2-5 行**——ldoor est273/dup9.5(CAS 自旋墙本尊)/StocF est117·92%梯3/
  rgg est81 dup2.6。**= ocean §3.3 混合累积器靶区**(索引smem/值gmem原生FP64 RED 绕开
  smem double CAS 墙)。ocean 对 rgg/StocF 走 symbolic 工作流(其 numeric 更快是主因)。
- [x] B CD 溢出修雷完成(10-20 午后,study/47 闭环):cnnz_scan 后加**负值即判废回退
      merge3**(int32 CSR 装不下 3.5e9 级巨输出,判废是唯一正确语义;复用 underflow 收尾);
      门:c-73/c-73b CD 臂优雅判废(打印溢出证书,不再 OOM 崩溃);控制阵 nd24k/wb-edu
      CD nnz == C 臂精确一致 ✓;全套 int64 化不做(int32 CSR 天花板,判废+回退=证书正解)
- [ ] C(时间充裕)hybrid-hash 设计文档:依 A 的取证结论,写中带(索引 smem/值 gmem RED)
      的实现方案(只设计不改码,明天汇报后人工过目再动)
- [ ] 终报:入档 + commit + 通知

## 纪律
改码必 env 门控默认关/值门三件套(Result C 锚定!grep 教训)/同会话 A/B/
控制阵无害/Xid 哨兵/GPU 陌生租户即停/含糊即停不猜。汇报日不动记分二进制语义。
