# study/53:大阵带攻坚战役书(>100k,1.398 → ≤1.15;10-21 定稿)

> 靶:>100k 带 147 阵 GM **1.398**(胜率 23/147)= 337 差距的 ~91%(0.436×ln1.398≈0.146 /
> ln1.174≈0.160)。打赢它 = 337 → ~1.05、并集 0.949 → ~0.88[推测]。
> 上游定谳:相位去向 accumulate ~50% + compact ~44%(ANALYSIS-428 §三);
> 结构斜率 avg_row −0.388(行越稀越输)。
> 纪律:同会话 A/B 唯一刀级协议 / 守门三件套 / 预注册成败线 / GPU 保护(报错即停,
> ncu 前冒烟,跑前查占用)/ 判死名单不再碰(hybrid 值外置=spec91 刀E 2× 劣;
> 带状 CAS 墙=硬件边界;ESC64 旧实现=正确但慢,不复活只重设计)。

## T0 首批成果:kernel 归因表(10-21,ocn_stats 427 json + confirm4 配对)

| 族 | 阵(比值) | ocean kernel(数据源=其自带 stage 计时) |
|---|---|---|
| 网格 | delaunay_n23 1.79 / road 1.12 / rajat31 1.30 | **usparse ESC 子组 16/32/64 + outlier 二次发射**(workflow 2) |
| 小表 | rgg 2.28 / StocF 2.18 | **精确 symbolic 工作流**:hash 数值 + sym_hash 计数(workflow 0) |
| 小表 | ldoor 2.69 | 估计工作流 hash bins(workflow 1,hash=21ms) |
| 带状 | bloweybq 3.02 / a5esindl 1.33 / hangGlider 2.41 | dense sub-warp:sym_dense 计数 + dense 数值(workflow 0) |
| 混合 | in-2004 1.18 | 估计:dense 变体 9.7 + hash(hybrid)9.2 + epi_sort 4.3 |
| 对照 | c-64 0.98(我赢) | dense 13.5——dense 对位我方不吃亏 |

**两个改判**:
1. **rgg/StocF 走精确 symbolic,付数数税仍赢 2×**(rgg:sym 6.5+hash 17.7=24ms vs 我 accumulate 单项 33ms)
   → 差距 = **hash kernel 单位效率**;T2 受检假设重排:素数表/pow2、槽布局、探测序列升首位。
2. **bloweybq 3.02× 三夜稳定** vs k7-only 实测 0.751(study/36 R3)→ 预门(n≥5000→hash-A)
   对该阵误伤;T5 k7 门优先级上调,预门考虑带状例外。

## T0.1+T0.2 尾部解剖+flop 带普查(10-21 完成,纯磁盘)

**相位分解(147 阵相位对齐全;named 15 + tail 132)**:
- [tail 132] 净差距 225ms = **compact +222ms(正差距的 66%)** + accumulate +112ms(34%)
  **− 计数/估计 −109ms(我方赢:MinHash+低均度免估计)**;
- [named 15] accumulate −158ms(巨鲸/dense 我赢)但 compact +79ms——**FLAT 两边都治**;
- → **T1 FLAT 是尾部第一刀(权重上调);计数相位不用动(在赢)**。

**kernel 归因(tail Top 输)**:cage15/nlpkkt120/CurlCurl/para-*/Ga41As41H72/c-big 全是
  **我方 hash 梯 3-6 vs ocean 精确 symbolic hash(wf=0)——付税仍赢,与 rgg/StocF 同机制**;
  尾部没有新的未知形态。**T2 干净探针新增:nlpkkt120(b4:96% 单 bin 纯培养皿)**。

**flop 带普查(ocean avg_prod 口径,147 阵)**:
| 带 | n | GM | 胜 |
|---|---|---|---|
| usparse ≤64 | 26 | **1.778** | **0/26** |
| (64,128] | 8 | 1.518 | 1/8 |
| hash 表 >128 | 113 | 1.328 | 20/113 |

→ **T3 ESC 覆盖带靶区 = 33 个输阵 @ ~1.95×**(delaunay×4 + CurlCurl×3 + rajat31 +
  M6/AS365/NLR/333SP 结构族 + …)——比"delaunay 临界带"预期**大一倍**;
  修到 ~1.0 可吃掉 >100k 带对数差距约一半[折算]。
ocean 工作流分布:>100k 上 wf0 精确 64 / wf1 估计 57 / wf2 usparse 26。

## 战役结构(六阶段,取证先行)

### T0 无 GPU 侦察(10-21 汇报日当天,纯磁盘)——**先于一切动刀**

- [ ] **T0.1 尾部解剖**:>100k 带 147 阵,从 batch437 sidecar 离线算相位损失直方图
      (accumulate/copy/est/count 各占差距多少)。回答:**长尾 ~130 阵(非名将阵)的钱在哪个相位**
      ——名将阵(delaunay/road/rajat/blowey/rgg/StocF/ldoor ≈ 12-15 个)只是冰山一角,
      移动 GM 要靠打人口的机制,不是打单阵。产出决定 T3/T4 的优先级权重。
- [ ] **T0.2 flop 直方图全带化**:已知 delaunay 79.3% 行 flop∈(32,64](bin437);
      补 road/rajat/长尾的 flop 分布 → 定 ESC 覆盖带(33-128)的真实人口规模。
- [ ] **T0.3 ocean usparse 机理存档**:子组梯 {16,32,64,128} 按 avg_product 定档 +
      outlier 二次发射 + estik 精确积数槽(SpGEMM.cuh:494-520, Wrappers.cuh:392+)。
      **只借机理不抄形态**(外部基线借鉴铁律)。重点问题:我们 ESC64 为何慢而他们
      128 子组赢?(候选:outlier 分流免大子组常驻/寄存器布局/每块行数)
- 产出:本文件 §靶区表更新 + 各刀预注册成败线。

### T1 FLAT——**10-21 终审定谳:转正(15/15 无罪)**

夜3(confirm4)逐阵"回归"经同会话 A/B(FLAT=0/1 交错×3,同二进制)全部翻案:
- 第一轮 4 阵 + 第二轮 10 阵:全部无罪(±5% 内);c-67 −12.7%/rajat16 −7.6% 为
  **设计外新赢面**;rajat25 +19.7% 为伪信号——伤在 dense_direct(FLAT 无机制触及,
  compact+sort 0.50→0.53 分毫不动),已知双峰相位的 min 统计量采样伪影;
- **夜3 真凶 = GPU1 间歇争用**:我方 ADPT 跑 GPU1、ocean 跑 GPU0(脚本 env 分离),
  ocean 同夜稳定而 15 阵膨胀,9/15 挤在计算最长的运行尾部。confirm4 的 FLAT 数据作废,
  confirm5 顶替(且 ocean 改同卡 GPU1 顺序配对,修双卡分离协议洞);
- 转正内容:①COMPACT_FLAT 默认开(=0 回退)②预门短路(拷贝桶行数=0 的阵连
  count kernel+D2H 同步都不付,h_cnt host 侧零成本判)③值门 c-67(门通过侧)/
  TSOPF(拒绝侧)双 PASS(位置/序全同,值差 1e-16~1e-23=原子序噪声带);
- 判死名单补一条:**聚合 GM 掩盖逐阵尾**——夜3 并集 0.9560 在漂移带内,若无逐阵
  对账 FLAT 已被冤杀。方法论入账:knife 归属裁定一律同会话 A/B,夜数据只做记分。

### T2 StocF/rgg 取证(10-25/26 周末,1-2 小时 GPU 窗 + 分析)——**机制定谳,决定 T4**

- [ ] 单阵 ncu 全指标 + SASS 指令台账:StocF(梯3, 2.5×)、rgg_n_2_22(梯2-3, 1.83×)、
      对照 ldoor(梯4, 1.33×)+ 一个赢阵对照;
- 受检假设清单(预注册):①CAST.SPIN 值累加墙 ②pow2 vs 素数表探测行为
      ③ocean symbolic 工作流 numeric 效率 ④**值的位置**(tmp_key/tmp_val vs 混布)
      ⑤块几何/占用;
- 守护:standalone 冒烟先行 → sanitizer → ncu(卡保护纪律);
- 产出:study/54 定谳文档 + T4 刀序裁定。**取证前不动任何改码**(ESC64 教训)。

### T3 覆盖带之战(下周主刀;T0.3 设计定稿后)——flop 33-128 人群收编

- 靶:现掉 bin0 表的临界带行(delaunay 为主,accumulate 1.8× 形态差);
- 设计输入:ocean 机理(T0.3)+ 我们 ESC64 尸检(study/50:慢在哪)——不是复活,
  是重设计(候选方向:子组 64/128 + outlier 二次发射分流;或精确积数直写通路,
  我们已有 d_flop 精确积数零成本);
- 预注册:受影响人群 accumulate ≥20% 改善 ∧ delaunay 全阵 ≥10% 才转正;
  守门三件套(真值级/行内有序/memcheck)每构建必过;
- env 门控 → 小阵值门 → 同会话 A/B → 428 回归批 → 转正。

### T4 机制刀(T2 定谳后,方向三选一)

- 若=探测行为 → 表几何修(素数表/探测序列),不换值位置;
- 若=symbolic numeric → 评估该人群精确积数直写通路(与 T3 第二方向合流);
- 若=值墙/占用 → 判硬件边界入账,火力转 T3/T5。

### T5 k7 人口门(纯代码,与 T2/T3 并行)——现成武器兑现

- [ ] 行级门(dup≈1 探针,数学证书在手)+ 翻默认;bloweybq 系统级 0.75 人群兑现;
- 判据:428 回归批零回归 ∧ 受益阵净赢 → 转正;
- 风险:门误开人群(高 dup)反而亏 → 门=证书校验同源,宁紧勿松。

### T6 汇合(11 月初)

- 428 三夜确认批(timed-confirm 复刻协议)+ 记分刷新 + compare/README 数据层定谳;
- 论文回填:本战役 = paper §6.5 case studies 主素材(网格族轨迹 2.4×→? 全程留痕);
- can* 8 阵补全(网络通后,不阻塞)。

## 预算与里程碑

| 阶段 | GPU 窗口 | 预期(语言纪律) |
|---|---|---|
| T0 | 0(磁盘) | 尾部解剖定 T3/T4 权重 |
| T1 | 1 夜 | union → ~0.944[折算,待盖] |
| T2 | 2 小时+分析 | 机制定谳(study/54) |
| T3+T4 | 2-3 夜 | 337 → 1.10-1.15[推测] |
| +T5 | (并) | dup≈1 人口兑现 |
| 全兑现 | — | 337 ≤1.05 / union ~0.88[推测];**1.398→1.00 需长尾机制全成立,本轮目标定 ≤1.15 为稳** |

## 风险登记

1. 卡:汇报日不动卡;每窗查占用(xiatf/tangshengzhong);报错即停(Xid 哨兵);
2. ESC64 复辟诱惑:T3 必须是新设计+新取证,禁止原样回滚旧实现;
3. 单阵陷阱:rgg/StocF 结论只代表梯 2-3 人群,推广到长尾前先对 T0.1 直方图;
4. FLAT 翻默认的隐性人群:entry<50% 覆盖阵(人口门已挡,回归批复核);
5. 时间:论文(SC/PPoPP)与战役并行,T6 回填勿拖过 11 月中。

## 判死名单(引用出处,不再消耗)

- hybrid 索引/值外置:spec91 刀E(2× 劣)+ 巨鲸行 0.46-0.78 反证;
- 带状 CAS 残余 1.56×:H100 无 smem F64 原子,硬件边界;
- ESC64 旧实现:study/50(正确但慢);
- band rule 扩张(low-Thrust/hangGlider):C/A≈1.00 无路由解。

## 战役进度(10-21 14:00 更新)

- [x] T0 侦察(尾部解剖/flop 普差/kernel 归因)
- [x] T1 FLAT 转正(15/15 A/B + confirm5 GM 0.9528,网格族 −26~45%)
- [x] T2 定谳(study/54:发射吞吐差;七假设六证伪)
- [x] **T4a CS2 行级 d 门落地**(commit e66d9bf:StocF −53%/rgg −37%,值门双 PASS)
- [~] confirm6 盖分中(FLAT+T4a 联合部署,428)
- [ ] **下一优先 = T3(ESC 覆盖带 33 阵 @1.95×,usparse 对位)**——账面价值十倍于 T4b
- [ ] T4b(bin2-3 内环瘦身,StocF 残余 1.18×)降级为 T3 后的收尾刀
- [ ] T5 k7 人口门(bloweybq 单阵 2.3×)+ 预门带状例外

### T3 设计简报(10-21,机理存档自 AccumulatorESC.cuh:71-131)

ocean usparse 主形态 = `countSortCompactFused<NTHREADS>`(N∈{16,32,64,128}):
- 每线程一乘积(串行走位定位,O(k));smem 暂存 (key,val) 于线程序;
- **O(N²) 全对扫描**:同键→聚值+首现 leader;小键→bitmap 计秩;
- `__ballot` 收 leaders → n=popc;position=popc(bitmap∧leaders)=有序位;
- leader 直写输出槽——**排序+去重+写出一次融合,零原子零表零探测**;
- 外层:ESCKernelDispatcher 每块多行×子组;outlier(product_num>NTHREADS)二次发射。

**我方 ESC64 尸检对照(study/50)**:死因=shfl 双键(64bit 键 2× shfl/访问)>表税;
他们的 O(N²) 用 **smem 广播读**(LDS broadcast 每槽 N 读廉价)。
→ T3 刀 = flop∈(32,128] 行建 warp 秩排序路径(smem 暂存+ballot leader+秩直写,
flop 精确已知免溢出机器),NOT 复活旧 ESC64(shfl 形态)。
预注册:受影响人群 accumulate ≥20% ∧ delaunay 全阵 ≥10% ∧ 值门三件套。

### T3 战报(10-21 晚,首轮:停车入库,负结果入账)

**已建**:`esc_rank_kernel<64/128>`(smem 广播秩排序,BIN_ESC128=25 全链接线,ESC64/ESC128 env
默认关=生产零影响);nnz 与旧路径逐位一致(43,417,519;ocean +5 为其口径癖,两旧路径互证)。
**三连证伪(微迭代)**:①k<t 分歧修复(8.09→8.65 无效)②跨子组 bank 垫(无效)
③持久化行循环(22.6@2×SM / 10.4@8×SM 更差)。现状 8.6ms vs hash 路径 4.0 vs ocean 2.19。
**定性**:延迟型慢(~128 smem 趟/线程 + 3×__syncthreads);下轮设计=**融合单趟**
(leader/rank/聚值一趟,如 ocean 原版)+ warp-local ballot 形态(32 内全 shfl,64/128 拆
两半各配一趟跨半修正),预算半天。

### T5b 首取证点(10-21 深夜)

vsp_model1(K7=1):k7 行仅 315,accumulate 0.86ms,overflow=0——**爆炸不在 k7 累积,
在下游**(直写终态 exact_rp 与 est-gap/compact 的交互;219×=4579ms 量级疑似重试风暴
或病态 compact)。T5b = 相位定位(nsys 一发)+行级门修复;靶阵赢面保留(bloweybq 2.27
/hangGlider_3 1.70/pre2 1.56 均实测)。

## 战役终态(10-21)

- 部署定谳 = **FLAT+T4a+T4c,双夜 0.9183/0.9234**(单日 0.9486→~0.92,−3%);
- T5 批级证伪回滚(k7 默认关);T3 停车入库;教训两条入账:筛面≠批级、聚合掩盖逐阵。

## #0 弥散尾再解剖(10-22 凌晨,新配置 24 阵=该组 49% 差距,定谳)

**分解(我方−ocean,+125.3ms)**:
| 相位 | 差距 | 份额 | 对比旧解剖(夜1) |
|---|---|---|---|
| accumulate | +81.2ms | **65%** | 从 34% 升为主项(T4a 只盖了 bin2-3) |
| compact | +48.5ms | 39% | 从 66% 降(FLAT 生效)但**有结构性缺口** |
| 计数/est | −4.5ms | — | 我方净赢,无仗可打 |

**compact 缺口的机制**:FLAT 预门(拷贝桶条目≥50%)拒了**纯 hash 阵**——
cage15(bins 4:215万 5:179万 3:83万,拷贝桶占比 <1%!)compact **20.6ms 单阵最大项**;
ocean 同相位用 sortOutputFused。同理 nlpkkt80 3.0ms/cage14/CurlCurl。
**accumulate 残余**:HTC_336 族(+9.1/+6.2)/Hook(+9.2)/nlpkkt80(+6.9)/cage14(+5.3)
= hash 梯 4-6 带效率(T4b-class,与 StocF 同根不同带)。

**刀序更新**:
1. **T4d(FLAT-B:hash 行的 compact 融合)**——靶 cage15 类纯 hash 阵,单 cage15 就 19ms;
   形态:ocean sortOutputFused 对位(排序+写出一趟)或条目级扩到 hash 行;
2. **T4b 扩散版**(hash 梯 4-6 accumulate 效率);
3. T5b/T3 二轮不变。

### T5b 取证进展(10-22 深夜 loop)

**四项硬事实**:
1. vsp 219× = **hash_dense_count_kernel 单 kernel 爆 4.6s**(99.4%);k7 累积器本身干净
   (315 行,accumulate 0.86ms,overflow=0);
2. **跨迭代污染**:WARMUP_N=0 全快(3.9ms);pass1 快(4.1ms)、**pass2 同输入(dense_nr/Σest
   全同)爆 4642ms(930×)**;K7=0 对照两轮全干净 → k7 特异;
3. pass1→pass2 之间另有 **3.4s 主机侧空档**(不在任何相位内,未解释);
4. dense_count kernel 本体每窗自初始化(窗0全写游标)——相同输入不可能异速 → **pass 间某
   输入数组实质不同**(嫌疑:dstr_split 的 h_big_off/span 状态、分配复用、或 k7 行的
   est/ovf 残留改变下游)。
**下步**:pass 间输入对拍(dump spans/split/ovf 状态两轮 diff);修好 = k7 翻默认可能整体复活
(万级 16 名将 + 分裂阵全兑现)。
**本轮 loop 战果**:T4b 假设八证伪(§14)/T4d 假设九证伪(§15)/T5b 病灶定位到单 kernel+跨迭代。

### T5b 取证进展二(10-22 深夜,第二 tick)

- **输入逐位全同**(序无关指纹:Σspan/Σest/Σrow/xor 全同)+ kernel 慢 1070× → 黑箱排除输入;
- **仪器化即痊愈**:加计数器参数+发射侧 memset/sync 后两轮全快(4ms)——**码型/布局敏感**;
- 候选连环排除:游标区(默认关)、h2dovl 末块 flop OOB(已 clamp,注释为史)、
  当前构建 memcheck(cuda-12.8 sanitizer)= 0 错;
- **剩余主嫌:多流竞态**(h2dovl = sC 拷贝流 + sK kernel 流 + 默认流,时序/布局敏感完美匹配症状);
- **下 tick 动作**:剥仪器→重建→确认爆炸复现→对该构建跑 12.8 memcheck;阴性则 racecheck/synccheck。
  仪器全部 dbg 门控可安全保留;工具账:/usr/bin/compute-sanitizer 坏(缺注入库),用 cuda-12.8 版。

### T4e 落地(10-23 凌晨):LLB skew 门——第九假设成立,首刀正向

**发现链**:LLB(按行动态负载平衡,docs/27 旧机制默认关)在偏斜阵 −39~45%;
边界连环翻车后定谳:**门 = skew∈[100,1900](dispatcher 口径)**——features.csv 的
skew/avg 列与实算不一致(HTC 表 10 vs 实 1850;c-66 表 avg5.5 vs 实 10),特征表不可作门;
**电池**:HTC×2 −44/−39% / c-64 −45%(3.45→1.90)/ c-66 −17% / email-Enron −14% /
circuit_4 −6%;排除面零扰动(StocF/cage15 低 skew +4~7% 不进门,bloweya skew2000 排除)。
**值门**:HTC+c-64 双 PASS(真门二进制;首版值门空对照教训:门二进制须与刀同源重建)。
**部署**:dispatcher setenv LLB=1(显式 env 优先);confirm10 盖分中。

### h2dovl 毒 span 案(10-23 凌晨,T5b 终局线索)

vsp 219× 的最终病灶 = **h2dovl 重叠管线的 span 产出毒值**(pass1 已错:57K vs 同步
154M;pass2 垃圾 −6e12)→ dense_count lo0 读毒。H2DOVL=0 两轮全净。**预存静默病,
全生产线在毒 span 上跑**(lo0 错=全幅扫描=白付税,输出碰巧仍对)。racecheck/memcheck
均 0 错(纯时序/顺序问题)。vsp A/B:重叠 236 vs 无重叠 192ms——**管线本身在倒贴**。
下一步:全量 A/B → 若中性/更好 → H2DOVL 翻默认关 + k7 复活重试。

### T5b 终局(10-23 凌晨 2:40):h2dovl 毒 span 根除,K7 复活过半

**根因链定谳**:h2dovl 重叠管线 span 产出毒值(预存静默病,pass1 即错)→ K7=1 改变
分配/时序后 pass2 恶化为垃圾 → dense_count 窗数爆炸(219×)。计分口径 A/B 六阵:
**重叠=平/微亏**(其收益=藏 h2d,h2d 不计分)→ **H2DOVL 翻默认关**(计分中性+铲毒)。
**K7 复活筛面(18 阵,H2DOVL=0 下)**:灾难全灭(vsp 219×→−12% 反赢,hangGlider×2
+83/95%→+8/13%),赢面保持(bloweybq −20%/c-67b −24%/pre2 −11%/c-43 −11%/SiO2 −6%
/nd24k −9%/vsp −12%);**残留中幅回归**(c-big +20%/hangGlider_3 +15%/wb-edu +14%
/hangGlider_4 +13%)→ **K7 保持默认关**,门待修(注意:本轮为 TOTAL 口径含 h2d/d2h,
计分口径待 confirm11 级验证)。

### T5b K7 修复(10-23 上午,dense 否决 v3=原位+2×est<span)

**三轮边界扫描**:2×(hangGlider_4/5 修平但 hangGlider_3/c-43 误伤)→ 4×(3 复活但 4/5 回归)
→ 路由重排(更差,全部反向)→ **回到原位+2×**(hangGlider_4 −21%/5 平/3 平/c-43 −15%/c-67b −14%)。
**机制**:K7 在 if/else 链位置 3,dense 路由位置 5——K7 先到先得抢走 dense 行;
否决 `2×est<span` 保留 hash 行(K7 真正的生态位)。
**部署**:K7 翻默认开(dense 否决版);confirm12 盖分中。

### T5b K7 终审(10-23 上午):去噪后中性,维持关

confirm12 的 0.9010(+0.7% 倒退)经同会话对账:**eu-2005 5.55× 为夜级假警**(同会话 35.8 vs
36.1ms 平);rajat18 实际 K7 **赢 −25%**(夜数据方向反了);真回归 = gupta1 +27%/bloweya
+32%/rajat28 +13%。去 eu-2005 后 c11 0.8945 vs c12 0.8971(+0.3%,噪声带内)。
**K7 净效应 ≈ 中性**:赢面(bloweybq/pre2/c-67b/c-43/rajat18)与输面(gupta1/bloweya/
rajat28)抵消。三度批级三度回滚的教训:**跨夜单阵不可作 K7 人群归因依据**——K7 的人群
效应(±20-30%)被夜噪声(±40%+)完全淹没,只有多夜配对或同会话大电池才能定谳。
维持默认关,素材保留。

### cage15 compact 取证(10-23 上午)

cage15 compact+sort = 20.6ms;行分布:bin3(83万行 est p50=132)/bin4(215万 est 251)/
bin5(179万 est 439)/bin6(25万 est 798)。**全部长行**——FLAT 经济学确认不适用(est≫64,
warp-per-row 本已满流)。compact 差距机制 = **长行 warp 拷贝的单位效率**(vs ocean 的
sortOutputFused),不是路由问题。需要:①对拍 ocean 在 cage15 上的 epilogue 耗时定差距
②若差距 >2× 则需 warp 拷贝 kernel 微优化(向量化/双发/对齐),非 routing 刀。

### 输家逐阵 profiling(10-23 上午,nsys 双面)

**7 阵共性发现**:
| 阵 | 我方最大 kernel(ms) | ocean 对位(ms) | 差距来源 |
|---|---|---|---|
| bloweybq | hash_dense 8.0 | denseIter 1.83 | **4.4× dense 窗 kernel** |
| hangGlider_3 | hash_dense 6.9 | denseIter 1.95 | **3.5×** |
| lowThrust_4 | hash_dense 5.1 | denseIter 1.57 | **3.2×** |
| case9/TSOPF | hash_dense 6.3 | denseIter 2.35 | **2.7×** |
| pre2 | hash_spa(LHT10) 9.1 | denseNumeric 4.2 + hash 3.4 | **2.7× 混合** |
| ship_001 | hash_heavy 2.6 + hash_spa 2.3 | hashNumeric 2.99+1.0 | 1.2× 均势 |

**核心发现:5/7 阵的输面 = `hash_dense_window_kernel` 慢于 ocean 的 `denseNumericIterKernel` 2.7-4.4×**。
这是 diter(dense-iter 窗口)路径的 kernel 本体效率差——不是路由问题(T4c 已修路由),
是 dense 窗扫描 kernel 的实现差。ocean 用 `denseNumericIterKernel` 多窗口迭代 + smem 值累积;
我方 `hash_dense_window_kernel` 同语义但实现不同。

**次要发现**:
- bloweybq/hangGlider/case9 的 hash_dense_kernel(非 window 版,可能 diter 主核)
  也是 2-4× 慢
- pre2 的差距分散(hash_spa + csort + copy 各慢一截)
- ship_001 最接近均势(Σ 6.4 vs 5.5)

**刀向**:dense 窗 kernel 的单位效率优化(对齐 ocean 的 denseNumericIterKernel 形态),
或 5 阵人群改派 hash_heavy(T4c 已修 span 门)看是否更快。
