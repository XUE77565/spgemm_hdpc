# cursor 结构刀线(spec89,09-22 立;用户定调:0.98 远不够)

## 证据链
- 停顿 11.78 long_scoreboard(冷失载在窗内流式推进,spec86-B 定谳不可预取)+溢出税 1.47(spec82)
- **五连平/负收线**:B16(wash)/书签预取(负)/count 游标化(负)/批量装载 #1(wash,13.62 vs 13.64
  ——编译器已在飞多载)——**形态内极限彻底钉死,c-64 dense_direct=13.6ms 就是本形态的地板**

## 事件档案:参数错位事故(#29 遗留+#30 修复)
- #29 python 盲插把 force_est_min 插进 compute_bucket 签名**中部**→全部实参错位一格→静默腐化
  (diter bin=0,accumulate 417ms);#30 又误把参数加到 dense_direct 尾→两处连环;最终修=签名尾
  (occ_coeff 后)+发射对齐;**红线重申(库内已有):模板半编辑必查签名+发射两处——python 盲插再犯一次**
- 修复后:off 路由恢复(19444/13.635 带内✓);**FORCE_DITER_EST 接通验证**(P 上 1670=est≥8 人口精确)

## 剩余结构刀(0.98→0.88 的真路径,均多轮工程)
1. **窗界双缓冲流水**(w extract ∥ w+1 scan;SMEM×2 预算=P B2_W 21B/列×2)
2. **stage-4 直写拆核**(extract 独立发射)
3. **重算 vs 持有重平衡**(砍活状态到自然 2 块无溢出,治 1.47 税根)
4. count/direct 融合(2.64ms 相位消失)
- 预期叠加:13.6→9-10ms=c-64 全阵 0.98→~0.79(spec82 旧账)

- 09-22 05:4x 单发#41 **L2b 双 cnnz 设计定稿+依赖清单(未实施,新会话执行)**:档位梯 direct 也吃
  C_row_ptr→简单重排成环;解法=**早 cnnz(供梯基址,老行条目垃圾但不用)+老内核行 tmp/d_off+
  晚 cnnz(老 direct 出 row_nnz 后重建,供 compact)+underflow 检查随晚 cnnz 后移**;
  发射点:老 count=4448(d_rows_sorted+h_big_off 段,可独跳)/梯 count=4476 保留/cnnz=5317(10 行
  inclusive_scan)/underflow=5335;爆炸半径五处协同=新鲜上下文专轮做,勿残上下文蛮干
  **L1**=direct 双输出形态脚手架(加 tmp 输出参数对,默认 null 走旧路,编译过+off 逐位不变)
  **L2**=diter tmp 缓冲+相位重排脚手架(direct→cnnz→compact 新序,CNT_FUSE=0 时旧序不变)
  **L3**=CNT_FUSE=1 全链启用+守门;每级独立可验,事故面最小化
  失败=整 lambda 退场)②count 跳过+cnnz 换 est 前缀 ③direct 基址 exact_rp→d_off(est-gapped)
  ④**compact 现不碰 diter 行("直写终态")=#4' 引入新增挤压流量(赢面模型已含)**;
  实施需 4 处协同+签名/发射核对,下轮专轮做(勿在残上下文里动 4 接缝)
  squeeze(只拷输出 Σnnz×24B≈c-64 1.8ms/in-2004 1.25ms);**赢面人口依赖**:c-64 净省仅 ~0.8ms(−4%),
  in-2004 净省 ~7.9ms(−16%);**基建全在**:est-gapped 偏移(est_scan/d_off 既有)+hash compact 本就
  做 gapped→tight 挤压=最小改点「diter 行改走该机械+跳 count」;注意与方案5 规模门(4411)的关系
  待读;实施=下轮(需读 4400-4560 方案5 全链后落 CNT_FUSE env)
  驻留 2→1 块=占用恒等式反向(spec82 同族)——重叠收益大概率被占用腰斩吃掉;不实施,停靠
- **#4' 升为新优先:count/direct 融合经 est-gapped 免计数**——dense_count(2.64ms c-64)只为
  精确偏移(exact_rp)预配;est×1.8 本就超配 → 改 est-gapped 缓冲+compact 挤压(hash 侧同款
  既有形态),count 相位整个消失;欠配行走 retry(既有);预期 c-64 −2.6ms/全阵 −13%→~0.87
- #3(stage-4 拆核)同样面临 SMEM/global 往返税,停靠
