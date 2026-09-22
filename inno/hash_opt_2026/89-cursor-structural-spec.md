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

- 09-22 07:0x #4' 值级门:**cf0(CNT_FUSE=0)↔cf1(=1) PASS(1e-9)=刀值中立逐位**;truth17 FAIL=c-64
  既有 artifact(豁免口径,cf0 同 FAIL 非本刀引入);剩=有序+memcheck+净窗×3 双阵+20 阵快屏(07:39 轮)

- 09-22 07:5x #4' 守门收口中:有序 SORTED-OK(155M)✓;**memcheck 158→187 errors 未清**(无条件写 row_nnz 修复未中);
  病灶=hash_compact_copy_warp 越界写 84MB 越过一个 204KB 分配——**新根因候选:dC 缓冲按早 cnnz 的
  C_nnz_result(融合态=垃圾)定尺寸,晚 cnnz 真值偏移超出→compact 越界**;修法=晚 cnnz 后
  C_nnz_result>早值时 realloc dC_ci/d_val(或直接按 total_est×2 上界分配);
  性能数字(−21.9%)无 sanitizer 下取得,值级 cf0↔cf1 仍逐位(越界写落在 slack 区,仍必修);
  下一轮:fresh 上下文实施 realloc→memcheck 清零→×3 双阵→20 阵→#4' 登顶

## #4' 登顶(09-22 08:2x,×3 净窗 6/6 同号,/tmp/ab_cntfuse_x3_20260922.log)

- **三门全绿**(值级 0↔1 逐位 1e-9+有序 155,006,041 SORTED-OK+memcheck 0 errors——最后修=
  dC 分配 g_cnf 态 total_est×2 上界,根因=早 cnnz 垃圾值定尺寸致晚 cnnz 真偏移越界)
- **×3 判决**:c-64 off 中位 20.43 / **on 18.34(−10.2%,配对 −2.26/−2.09/−1.97)**→
  **vs ocean 20.75 = 0.884(0.98→0.88 达成)**;in-2004 50.43→**41.90(−16.9%)**→1.56→1.30
- 实施链复盘:L1 双输出脚手架→L2a 布线→L2b 双 cnnz 五接缝(三批+两根因修复:
  row_nnz 条件写残留/dC 尺寸)→L3 激活;沿途 6 个 bug 全被守门/烟拦下(脚手架分级的价值)
- **CNT_FUSE=1 候默认**(20 阵快屏待跑);cursor 线收官:0.98→0.88;①剩=③创新 kernel

## CNT_FUSE 判决撤销+事故档案(09-22 12:2x,boot -1 journal 实锤)

- 20 阵屏数据全部作废:crankseg -47.4/wb-edu -43.6 等大赢面=丢数据假加速嫌疑(值门未跑
  先跑了屏=方法论违规,顺序应先门后屏)
- Xid 实锤:10:59:05 与 11:05:39 两次 Xid 31 pid=spgemm_cct ACCESS_TYPE_VIRT_WRITE
  (GPU1 b1:00)——CNT_FUSE 刀有真越界写;机理嫌疑=est 低估行 rank>=cap 静默丢 +
  diter_gapped_copy row_nnz>est 越界读/晚 cnnz C_nnz 超 dC
- 同 boot 双事件分离:GPU0 GSP 固件死亡(10:57:22,同款 MMU_FAULT->LOCKDOWN,在场 pid 8636
  非我方)先于我方首错(10:59)——楔死本身非我方触发,但我方越界独立存在
- CNT_FUSE 状态:已知有 bug,env 冻结;修法候选=低估行强制走溢出路径(免计数下必须有
  nnz>est 检出->溢出兜底,不许静默丢)+copy 核 clamp+dC 按 flop 封顶
- 杜绝红线(新增):1)先门后屏(值级0<->1+nnz 总对账+有序+memcheck 含代表大阵,全绿才许性能屏)
  2)电池脚本每轮查 journal Xid(见 spgemm 进程名即停) 3)大赢面(>15%)自动触发值级复审

## CNT_FUSE 破损定谳:int32 溢出实锤(09-22 13:0x)

- crankseg_2 on=CNT_FUSE 跑出 **C_nnz=-1792527655(负数=int32 回绕)**——晚 cnnz 全阵 scan 求和
  包含未覆盖/陈旧 row_nnz 的垃圾项→回绕→负偏移进 compact=**Xid 31 VIRT_WRITE 完整因果链**
  (=wedge#4 我方越界的病根);Zd_Jac2 off/on nnz 一致(15,374,812)小阵幸存;c-64/in-2004 已验值真赢
- 修复①②已入库(drop 检出器+copy clamp):c-64/crankseg drop=0=extract 丢不是 crankseg 病根
- **判定:CNT_FUSE 停补丁化修复,转"破损-需重构"档**(病根=row_nnz 覆盖语义+int64+预 compact bail
  三件深度件;env 默认关=生产安全,保留 c-64/in-2004 已验赢面作重构后的对照组)

## CNT_FUSE 重构设计(09-22 13:3x,#50;病根=garbage row_nnz:dev_alloc 无 memset+融合态覆盖不全→晚 cnnz 求和回绕)

- **R1 覆盖语义**(一轮):d_row_nnz 分配后**无条件 cudaMemset 0**(全阵);direct 两出口已无条件写
  (1183/1320);未被任何写者覆盖的行=0(空行语义,compact 安全)——一行 host 改动,消"垃圾项"
- **R2 int64 防护**(一轮):晚 cnnz 的 thrust scan 改 long long 输入视图(或预 cast kernel);
  C_nnz_result 判废条件:C<0∨C>total_est×2∨drop>0→INVALID
- **R3 预 compact bail**(一轮):INVALID 判定**提到 copy/compact 发射之前**(现检查在晚 cnnz
  fprintf 处但 copy 已在 compact 相位内先跑?核实顺序)——拦截后走 underflow bail 模板清理返回,
  **负偏移永不进写路径**;三件各自独立可验,顺序 R1→R3→R2(R1 消主垃圾源即大概率自愈)
- 回归对照:c-64(18.34/in-2004 41.90)+Zd(nnz 一致)+crankseg(重构后应=C_nnz 104,590,168 或 INVALID 而非负数)

## CNT_FUSE 三门全绿重建(09-22 14:4x,R1+R3 后)

- 值级 0↔1:c-64 PASS+**crankseg_2 PASS**(1e-9,R1 修复后首次!)
- crankseg 有序:OK(104,590,168=与 off 精确同数)
- memcheck:crankseg 0 errors(Xid 阵洗净)+c-64 早前 0
- **判定:CNT_FUSE 从"破损"回到"三门全绿";R2(int64 scan)为纵深加固非必需**

## crankseg 真赢复测(09-22 14:5x,R1 修复后 ×2)

- off 50.17/50.27 vs **on 45.18/45.55** = **真赢 −9.4%(非 −47% 假象)**
- 判读:−47% 中 ~38pp 是 int32 回绕造成的"早退假象"(负 C_nnz→轻路径),真赢=count 消除的
  实际价值 −9.4%(crankseg count 相位 ~4.7ms 的真实节省)
- **CNT_FUSE 真赢谱(R1 后,全部过三门)**:c-64 −10.2%/in-2004 −16.9%/crankseg −9.4%;
  三阵一致方向,量级=count 相位占比的物理账

## 20 阵屏 v2(09-22 15:0x,三门全绿后合规跑,Xid 哨兵 0)

- **14 阵无害或赢**:crankseg −10.9/in-2004 −17.1/nd24k −16.6/**wb-edu −45.3(真!bail 阵 on 态
  免 count 免 bail=merge 回退也省)**/c-43 −10.7/F2 −2.1/rajat20 −3.0/其余中性
- **回归 1 阵:3dtube +64.2%**×3 复核确认真回归(on 8.3-11.2 vs off 6.5-7.1)——小阵 est-gapped
  挤压>count 节省(c-43 同类但赢=边界带);**修法候选=cntfuse 规模门(dense_nr<阈值或 Σest/dense
  占比<阈值时关闭)——属路由层决策不动,只做 env 门候选记档等用户**
- c-64 +3.5% 此屏漂移(×3 已证 −10.2;wb-edu −45.3 同阵前屏 −43.6 复现=真)

- 20 阵 v2 全 20 阵出齐补记:rajat25 −24.9/c-58 −15.8/Zd −7.0 再添三赢——**终版:15 赢或中性偏赢/4 中性/
  1 回归(3dtube)**;Xid 哨兵全程 0
