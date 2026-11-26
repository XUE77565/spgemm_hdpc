# study/69:重计算战役书——三线战争(11-20 晚开篇,loop 自动)

> 缘起:用户令"攻坚 accumulate,先从 SB 开始,挂 loop"。首轮 profiling 即翻案:
> 旧地图(旧二进制 prof 日志)画的"accumulate 63.3%"在当前件上是 **35.9%**,
> SB 的可回收 93.7ms 有 74% 在 dense_direct——**战役从单点改三线**。
> 本档 = 战役总账:地图、排刀、每刀的定谳与验证记录。

## 一、瓶颈地图(当前二进制,可回收 top20=560ms,脚本固化)

| 战线 | 可回收 | 份额 | 主力阵(单阵可回收) |
|---|---|---|---|
| accumulate | 179ms | 35.9% | c-big 41ms、Bump 29、Hook 14、delaunay_n24 21、dielFilter 15 |
| **compact+sort** | **148ms** | **29.5%** | **nlpkkt120 62ms、cage15 55、cage14 18、ldoor 14、nlpkkt80 22** |
| dense_direct | 116ms | 23.2% | **SB 94ms**、rajat16/20/25 ~12/阵 |
| mh_merge+dense_count | 55ms | 10.9% | SB(mh 20)/rajat(count) |

单阵相位(中位):SB=dense_direct 227.3ms/74%;nlpkkt120=compact 73.0ms;
cage15=compact 68.2 + acc 58.1;c-big=acc 81.8;rajat16/20/25=dense_direct 32-35ms。

**方法论铁律(新)**:地图必须与二进制同代——每轮转正/大改后重刷
(脚本:/tmp/atlas_refresh 逻辑,入库待办);旧日志画的靶开炮前验明正身。

## 二、排刀序(每刀走:微复现器→实现→值级三判→ab_pair 分级→confirm)

### 刀一:Gv5a 载体融合(compact 线,148ms 池)★首发
- **靶**:legacy 提取人口的"读回-排序-再写"载体趟。nlpkkt120 的 compact 73ms
  ≈ 其对 ocean 差距(62ms)的全部——**这一刀可能单独翻掉 nlpkkt120**;
- **设计**(study/68 §五 已有):排序融进 hash_heavy 内核的 legacy 提取——
  行内 smem 直方图(est≤CAP 段)一趟完成"累积→有序写出",消 DRAM 往返;
  span 无关(物理量 O(est)),正中宽 span 低密度人口;
- **验证**:hh_repro2 扩展 legacy-融合分支(全绿才进生产)→ 值级(nlpkkt80/120、
  cage15、ldoor)→ ab_pair(基线=HH=0)→ confirm19;
- **预期**:compact 池吃 50-70% ≈ +1.0-1.5% GM。

### 刀二:SB/rajat dense 路径(dense 线,116ms 池)
- **先解剖再设计**:SB 的 dense_direct 227ms 拆行结构(窗口宽度分布/初始化
  vs 写回占比/趟数)——SB 行=网图(sparse 输出+hub 大 B 行),为何走 dense 路?
  路由条件复核(dense 门判据 vs SB 实际行形状,疑预门误判)；
- rajat 族同理(33ms/阵×3)。
- **假说**:SB 的 dense 路由本身是误判(网图行不该走窗口),修路由可能比修内核便宜。

### 刀三:accumulate 瘦内核(accumulate 线,179ms 池,周级)
- b45 线重启:spec99 资产、彼 symbolic 1.6cyc/乘积对标、T3/ESC 档案;
- 靶阵群:c-big/Bump/Hook/delaunay/dielFilter——等刀一刀二的战场情报再开。

### 顺手:hangGlider 相位盲区
- gap 15-18ms 但可见相位仅 ~7ms——60% 差距在相位标签外(d2h/cnnz/未标内核)，
  解剖可能白捡一个发现(疑:ESC/batch 内核未计入 accumulate 相位标签)。

## 三、纪律(全程)
GPU 单任务/报错即停/测量分级(微<1ms 不作单阵裁决)/代码走别名构建→值级三判
→ab_pair 分级判读→红即停写明原因;禁未经门 commit、碰测试账、覆盖 spgemm_test。
地图刷新脚本入库(每刀开动前先看地图:"这刀打的是 148/179/116ms 哪块")。

## 四、验证记录(逐刀追加)

(待刀一微复现器)

---
*版本:11-20 v1(开篇+三线地图+排刀)。关联:study/68(§五 Gv5a 设计/§九-3 K7 终案
/§十 测量纪律)、study/57(b45 战史/spec99)、study/53(T3/ESC)。*

## 五、刀一实现规格(Gv5a 载体融合,复现器已 ALL-GREEN,待生产落地)

**内核侧**(hash_heavy_kernel legacy 分支,仅 bins4/5 v1):
- 触发条件:force_legacy==1 ∧ HT_P≤2048(bin4/512、bin5/2048;bin7+ v2 再说,smem 不够);
- 流程:legacy 提取处不走无序写,改融合提取——走表收集(原子票机,复现器同构)→
  smem 位偶排序(键值对,哨兵 SENTINEL_KEY=0x7fffffff 沉底,N_POW2=2048)→
  有序直写 tmp[base..base+k);smem 预算=表(6/24KB)+排序区(2048×12B=24KB)=
  30/48KB,launch 侧 hhsm += 24KB(fused 时);row_nnz=k(票机计数,与 bitmap 分支同语义);
- 超帽(k>cap):沿用既有 ovf→retry 熔断,融合排序自然不做(收集后判 k 与 cap)。

**契约侧**(hh_used/hh_flip 三态):
- hh_flip=2(新:fused-sorted):bins4/5 翻 bin 且走了融合提取 → compact 侧
  视同 h_hh_used=1 走 copy/FLAT(行已有序),不再进 csort 载体;
- ctx 增 hh_fused 出参;compact 的 bins4/5 条件改为
  `!(hh_sorted_host() && (h_hh_used[bi] || h_hh_flip_fused[bi]))`。

**发射侧**:HH_LAUNCH 增 FUSED_ 参(仅 bi∈{4,5} 且 hh_leg=1 时=1);
hhsm 条件追加排序区;hh_elig/hh_leg 决策逻辑不变。

**验证链**:①值级:nlpkkt80/nlpkkt120/cage15/ldoor(融合路径)三判 L1=L2=0;
②ab_pair(A=HH=0 基线,B=别名):compact 线阵(nlpkkt120/cage15/ldoor/nlpkkt80)
应显著赢,其余平;③绿→commit→confirm19。预期:compact 池吃 50-70%≈+1.0-1.5% GM。

**复现器**:tools/hh_repro3.cu(606 用例 ALL-GREEN,11-20)——生产实现须与其逐行同构
(收集/排序/写三段),分叉处仅 smem 布局(表+排序区共存)。

## 五-1、刀一编辑点锚定(11-21,供生产实现直接落锤)

- 内核 legacy 提取分支:src/spgemm_kernel_hash.cu:4626 起("提取(legacy)"注释处)——
  融合分支插在此 if-else 的 legacy 侧,条件 force_legacy∧HT_P≤2048;
- ctx 字段:4706(hh_flip 声明处,加 hh_fused);
- 出参赋值:4950(c.hh_flip=hh_leg 处,加 fused 三态);
- 发射宏调用:HH_LAUNCH(7,512/8,2048,...)(bi==4/5 两行,加 FUSED 参);
- compact 契约:7534(bins4/5 载体条件,按 §五 三态式改);
- GPU 空闲已确认;实现顺序:内核→ctx→发射→compact→别名构建→值级(nlpkkt80/120/cage15/ldoor)
  →ab_pair→confirm19。

## 六、11-20 日终状态快照(明日续点)

**今日全弧线**(study/68 §七-十 + 本档):
- 正确性:Gv3v2 蝶式 rank 定谳→Gv4 位图修复→Gv4.1 翻bin legacy→Gv4.2 普查上提,
  三 commit(28ea718/7662d4c/01417a5)+ 认证件重建;
- 记分:confirm18 **0.9251 现役**(赢面 49.6%),0.8054 退役;四行消融账:
  带病 0.8137/HH=0 0.9278/Gv4.1 0.9289/Gv4.2 0.9251(全部同日同卡);
- 纪律:测量分级(微<1ms 不作裁决)入 ab_pair(commit 0c1671a);案底=bcsstk17
  1.186→10× 0.984、bcspwr01 n=39 假自然实验、探针 TOTAL≠compute-only;
- K7 终案:全账 0.9389 净负,三度证伪 opt-in 归档(90256eb);8 真赢/32 真输;
  **资格门≠竞争力门原则**入指派表设计;
- 地图:三线(accumulate 179/compact 148/dense 116ms),旧图 63% 单点说推翻
  (方法论:地图与二进制同代);
- 刀一(Gv5a 载体融合):复现器 606 全绿(cdcfd0a)→生产实现→值级双 PASS
  (nlpkkt80/ldoor)→v1 电池 web-Google 2.03× 红→**根因=固定 2048 宽网络
  (隐式宽度路由)→v1.1 宽度截断补丁已构建(spgemm_gv5a2,审计绿)**。

**明日续点(按序)**:
1. v1 电池结果入档(/tmp/ab_gv5a.log + results/ab_gv5a_vs_hh0.csv,v1 回归记录);
2. spgemm_gv5a2 值级双向复验:web-Google(小 k)+ nlpkkt80/ldoor(大 k)+ F2,
   三判 L1=L2=0;
3. 重跑电池(A=HH=0 基线 vs B=spgemm_gv5a2):预期 web-Google 塌回 ≤1.05、
   nlpkkt/ldoor compact 赢兑现 → 绿则 commit(src 当前含未提交 Gv5a 改动,
   **红不 commit 纪律维持**)→ confirm19(ocean 复用,~2h);
4. 刀二(SB/rajat dense 线)解剖开工;
**源码状态**:src/spgemm_kernel_hash.cu 含 Gv5a v1.1 未提交改动(通过编译+审计,
值级/电池未过,按纪律不入库)。

## 七、刀一 v1→v1.2 迭代史与 v1.3 方案(11-22)

**v1.0**(固定 2048 宽位偶):电池 GM 1.6481 红;cage15/Hook/Bump 5.5×/ldoor 4.9×
——每行 log² 屏障,百万轻行人口被杀。教训:载体的真优势=批处理零 per-row 屏障。
**v1.1**(宽度截断):ldoor 单测仍 5.9×——屏障数不减,无效,证屏障是主犯。
**v1.2**(k≤384 走 O(k²) 秩排序单屏障;>384 位偶):电池 GM 1.0656——**轻行层全绿**
(ldoor 1.008/web-Google 0.972/12 阵 ≤1.01),重行层仍红(pre2 1.64/Hook 1.25/
Bump 1.20/cage15 1.09)。
**v1.2 红因双嫌疑**:①位偶屏障在 k∈(384,2048];②融合区 24KB smem 降 occupancy
殃及同核累积相(bin4 6→30KB,驻留减半)——②更符合"累积重阵全红"图案。

**v1.3 方案(待实现)**:秩-only 融合,门槛 384,融合区缩 4.6KB(occupancy 无损);
k>384 行回 legacy 无序写。**硬骨头=混合行 compact 契约**:sorted/unsorted 同 bin
并存——carrier 全 bin 重排无害但白费,真收益需 per-row 有序标志(方案:row_nnz
符号位或独立 flag 数组,copy 跳 sorted 行、carrier 只吃 unsorted 行列表)。
nlpkkt80(0.897 手测赢)不在 16 阵电池内,终判需 confirm 或换电池清单。

## 八、11-23 晨战报:刀一 v1.3 突破 + 刀二开刀

**刀一 v1.3(est 槽宽前置路由)四阵**:pre2 1.007/Hook **0.822**/ldoor **0.673**/
web-Google 0.918——判别定谳(重行双重走表)修正后翻 bin 人口载体税整体塌。
电池(基线 vs gv5a4)后台中。v1.0→v1.3 五轮=判别-修正循环(屏障→宽度→双走→
路由),每轮单因实验,非乱补。
**刀二 SB 解剖**:232ms dense_direct 服务 5263 条 hub 重行(Σnnz=36.6M,avg 7000/行;
count 43ms+dstr_split#1)——440MB/232ms=1.9GB/s,**离带宽 1500×**,内环病理
(嫌疑:全局窗零化/原子争用/窗口溢出),下一步=读 dense_direct 内核定位。
其余人口:ESC8/16/32=47.8 万轻行+bin0 8.1 万(warp 路径,非主矛盾)。

## 九、刀二 SB 病灶定谳(11-23):search 路径无空窗跳过

hash_dense_direct_kernel(src:1304,search 路径 1375-1387):`for wi∈[0,nw_win)` 全跨度
逐窗,每窗付 PB2_W 零化(dval+dflag)+每 A 索引一次 dev_lower_bound 二分,无空窗跳过。
SB hub 行(5263 条,avg 7000 出/683k 跨度≈1% 密度)→ 170-330 窗/行 × 零化+二分
= 232ms(1.9GB/s 病理)。cursor 路径有 smap 前进,search 无。
**双重病**:①路由——1% 密度宽跨行根本不该进 dense(diter 门误判);②内核——
search 路径需空窗跳过(候选:行实际列域窗表预扫,或移植 merge3 有效列域思想)。
**修法优先序**:先修路由(diter/dense 门按密度 est/span 判,把 SB hub 行送回
hash 路径)——便宜;内核跳窗作 v2。rajat16/20/25 同病(33ms/阵)。

## 十、刀一 v1.3 值级门拦截(11-23 晨):混合行契约破裂

电池绿(GM 0.9451,nnz 0 异常)但值级 FAIL:ldoor L2 乱序 17654/web-Google 3622。
根因:v1.3 重行(cap>384)走 legacy 无序写,compact 契约 hh_flip=2 仍整 bin 走
copy——sorted/unsorted 混合行,无序行直拷出厂。§七预警的硬骨头应验。
**性能证据保留但刀不可 commit**——值级门拦截 = 三层验证顺序正确(电池放行/
值级拦截,带病 commit 被挡)。
**v1.4 方案(混合契约三件)**:①d_fus_unsort 标志数组(重行无序时置 1);
②copy_warp 的 skip 接该标志(copy 跳无序行——skip 机制现成,row_ovf 同款);
③gather 内核收集标志行成紧凑列表 → launch_csort 只吃该列表(载体只跑无序行)。
预期:轻行免载体赢保留,重行回载体形态(v1.3 性能大部分保留,因大赢来自轻行
人口的 copy 化)。电池+值级双绿才 commit。

## 十一、刀一收官:confirm19 = 0.9085(11-23 晨)

commit c0ea053(Gv5a v1.4)。confirm19:427 阵,ocean 同卡复用,
**GM 0.9251→0.9085(−1.8%),赢面 197/427**。五线总账:带病 0.8137(退役)/
HH=0 0.9278/Gv4.1 0.9289/Gv4.2 0.9251/**Gv5a 0.9085(现役)**。
compact 线(148ms 池)主力收割完毕;剩余:pre2 +5.9%(带内,重行载体回归税)、
bin7-9 fused 未扩(v2 候选)。下一刀:刀二 SB/rajat dense 线(116ms 池,
病灶已定谳=search 路径无空窗跳过+路由误判,修法=先路由后内核)。

## 十二、刀二 SB 定谳(11-23):dense_direct 的 SMEM 原子争用 + dup 门放行

**画像翻案**:SB 5263 dense 行 = F=11930/est=2666/span=2141/**est-span=1.0(窗内全满)**/
dup=5.7——span 窄窗内全密,路由方向本身对;"空窗跳过"假说否(早前误诊)。
**真凶**:dup=5.7 → 每输出位 5.7 次同地址 SMEM atomicAdd(争用吞吐 1/4),
62.7M 乘积×4-6 cycle ≈ 232ms——与 dense_direct 227ms 算术吻合。
**对照实验**:DITER_MIN_FLOP=∞ 推回 hash → SB 360→261ms(−27%),diter 5263→417。
hash 的 CAS 分散恰是 dup 的克星。**修法二选一**:①DUP_LIGHT 8→4(dup≥4 行
回 hash;SB 5.7 落网外;需全账电池防 F2/c-64 类 dup<4 dense 赢家回归);
②dense_direct 内核 K1_ACC 争用消解(flag 门控累加替代 atomicAdd)。
优先①(一行常数,带电池);rajat16/20/25 同画像待验。

## 十三、刀二进展(11-23):门考古定谳 + LADDER 判别

- dup4(DUP_LIGHT 8→4)几乎无效(diter 5263→5103):SB 行不走 dup_ok 主门;
- LADDER=1(docs/67 双梯,默认关)赶走一半(diter→2680)但性能持平——
  **被赶走的一半在 hash/dense 同速**;真亏的 ~2600 行恰是 ladder 成本律判
  "dense 便宜"的行——**ladder 的争用项在 62.7M 乘积尺度低估**(物理推导的
  dup<8 前提在此失守,SB dup 5.7 实测同址争用已主导);
- DITER_MIN_FLOP=∞(全推 hash)SB −27% 且 c-64 −10%,F2/cage15 无回归;
- **结论:刀二的正解=内核级**——dense_direct 的同址 atomicAdd 争用消解
  (K1_ACC 通道或 per-thread 私分+归约),门调整已被判别为死路(梯子成本律
  本身要重标定,那是调常数不是治机理)。排队为 v2;刀三(accumulate 墙)
  是更大池,先转刀三侦察或做 K1_ACC 由用户裁决。

## 十四、K1FW 双面刃(11-23):c-64 −14% 但值级炸(19444 行集合错)

K1_ACC 已降级为裸 atomicAdd(判负废弃注释),k1fw 分支现效果=免 dflag 写
(每乘积少一次 SMEM 写)。c-64 −14%/SB −0.5%/F2 平——**但值级 L1 集合错
19444 行**(提取的 dpref 前缀依赖 dflag,免写=漏计)——**死码不冤,勿启用**。
结论:刀二的内核级正解仍是从头设计争用消解(如 per-thread 私分+块内归约,
或 dflag 改由首写原子返回值驱动——atomicAdd 返回旧值≠0 可判占用,免独立
dflag 写)。dflag-in-atomic 方案最便宜:atomicAdd 返回旧值,若加前为 0.0
则本线程是首贡献者→由它置 flag(每位置仅一次 flag 写而非每乘积一次)。
排队实施;磁盘纪律修订:WRITE_MTX 批跑后即清 results/matrices/。

## 十五、刀二 v1:dfia(dflag-in-atomic,11-23)

实施:三处乘积内环 atomicAdd 返回旧值判首写(旧值==0→置 dflag),每位置
一次 flag 写替代每乘积一次。**c-64 0.887(−11%)且值级 PASS**(L1=L2=0,
与 K1FW 的 19444 行集合错对照=首写判据修复了 dpref 漏计)。但 SB 1.025/
F2 1.012 微红(返回值依赖链在低 dup 人口的税?)——**需要 16 阵电池定净账**。
spregemm_dfia 别名已构建;下一跳电池(A=现役 confirm19 件 vs B=dfia)。
若净绿→值级四阵(SB/c-64/F2/TSOPF)→commit→confirm20;净红→SB 的 +2.5%
解剖(疑 cursor 路径的返回值依赖链)或收缩为 bin 级门控启用。

## 十六、刀二 dfia 终账(11-23):空结果,c-64 读数不稳定

电池 16×3:GM 1.0083(带内但方向偏负),11 阵可裁决全平(1.000-1.004),
唯一大变量 c-64 1.104(单测 0.887)——**同址原子争用的时序敏感性使读数
不稳定,dfia 的收益不可稳定复现**。诚实判定:**空结果,不入账**。
cage15 B r2 一次 FAIL(前两臂正常)=瞬态(与 confirm17 同款)。
处置:dfia 内核改动回滚(commit 反转),争用消解方向改由外部模型按
study/70 §四(per-thread 私分+块内归约,更强的争用消解)继续。刀二
在本次会话的结论:病灶定谳完整(dup 5.7 同址争用/梯子成本律低估/
K1FW dpref 陷阱),内核级方案已明确但 dfia 这一版不够。

## 十七、刀三(accumulate 墙)侦察基线(11-23)

b45 战役终局(study/57 §廿四)结论回顾:ocean b45 人口走 wf=1 估计流(非精确
symbolic),差距 = 纯 numeric 本体(65.1 vs 78.4ms,Bump);我方 hash_spa 四轴
证伪 = "肥内核的局部最优"——瘦内核重写是终局方向,spec99 资产在(TU-61,
bins7-8 接管,当年 lean 方向)。
当前二进制基线(confirm19 形态):Bump acc=72.5/compact=15.8/compute=99.6;
c-big acc=83.0/compact=14.2/compute=105.5。**accumulate 占 compute 73-79%**,
是纯 kernel 内环差距,与路由无关。
**资产位置**:spec99 = hash_heavy_kernel 本体(TU-61 tagged 实例);当年未批
的 lean 方向 = 精简乘积内环(每乘积最少指令数,对齐彼 1.6cyc/乘积)。
**这是外部模型(study/70 §四)的主战场之一**——本轮会话侦察完成,移交。

## 十八、刀三·想法一(并发逆差)定谳(11-23 晚)

**实验**:HASH_TGT_ALPHA 梯度(200 默认/120/80),acc 相位直接读取。
**结果**:
| 阵 | acc@200 | acc@120 | Δacc | compute Δ |
|---|---|---|---|---|
| Bump | 76.3 | 61.4 | **−19.5%** | −0.8%(被 retry 吃掉) |
| c-big | 80.1 | 82.8 | +3.3% | +1.5% |
| delaunay_n24 | 32.0 | 34.5 | +8.1% | — |
| ldoor | — | — | — | **−12.3%**(compute 级) |
| cage15 | — | — | — | −2.4% |
| F2 | — | — | — | +7.8% |

**定谳**:①**accumulate 相位确认为延迟遮蔽受限**(occupancy-sensitive)——
Bump 缩表 40% → acc 相位 −19.5%,方向对了;②但 **retry 成本吃掉了增益**
(表小→overflow 多→重算代价)→ 净 compute ≈ 平;③**人口分化**——
ldoor/cage15 大赢,c-big/delaunay 微输,F2 输——**这是 per-population 路由
决策,不是全局翻默认**(竞争力门原则第三次验证);④c-big/delaunay 的
acc 反升=它们的行确实需要大表(冲突多,表小探测链更长)。

**下一步方向**:
A. **免表缩的 occupancy 攻击**——不缩表,缩非表 smem(dpref/smap/游标区)
   → 块驻留增但不触发 retry cascade;
B. **retry 降本**——溢出行不整行重算,只对溢出部分补(增量 retry);
C. **人口自适应 alpha**——行级/矩阵级判据决定 alpha(竞争力门),但这是
   路由不是 kernel,适合外部模型(study/70 §五)。

当前推荐:先 A(改 hash_spa 的非表 smem 布局,零 retry 风险)。

## 十九、刀三·想法一A(块宽梯度)空结果(11-23 晚)

BLK_LADDER 256/512/1024 三档:Bump acc 全平(72.5/72.6/72.6)——
块宽不是 occupancy 的限制因子,SMEM 表大小才是。想法一(缩表)的 acc
收益确认但被 retry 吃掉的结论不变。转想法二(位图两遍式)设计。

## 二十、刀三·想法二(位图标记-累积两遍式)设计稿(11-23 晚)

**机制**:
Pass1(标记):每乘积一次 atomicOr(bitmap[j>>5], 1<<(j&31))——无 CAS
无探测环无链追踪,~1cyc/乘积。
Pass2(累积):第二遍读 B 行,每乘积 atomicAdd(&val[bitmap_rank(j)], product)
——bitmap_rank 由 Pass1 后的前缀给出(每列唯一槽,无同槽争用)。

**成本模型**:
- 现状(CAS):每乘积 = 读 A 值(1) + 读 B 元素(1) + 哈希计算(1) + CAS 探测
  (dup 次尝试 × ~3 cyc) + atomicAdd(1) ≈ dup × 4-6 cyc
- 位图两遍:Pass1 = 读 A + 读 B + atomicOr ≈ 3 cyc;Pass2 = 读 A + 读 B +
  atomicAdd ≈ 3 cyc;总 = ~6 cyc + 前缀扫描 O(S/32)
- **交叉点:dup > 1.5 时位图赢**(dup=5.7 → CAS ~25 cyc vs 位图 ~6 cyc = 4× )

**SMEM 预算**:位图 S/32 词(S=span)。b45 阵 span 2-8K → 位图 64-256 词 =
0.25-1KB,微不足道。槽位数 = popcount(位图) ≤ S 但 ≤ est 上界。

**与 Gv4 位图直写的关系**:Gv4 已做位图(标记+前缀+直写),但只覆盖
hash_heavy 翻 bin 人口(宽度受 BM_WORDS 限制)。想法二 = 同一思想推广到
**hash_spa 主梯**(bins 2-8 的主体人口),改 CAS 探测为位图直寻。

**实现要点**:
- 每行一块,块内两遍走 A 行引用的 B 行(两遍读同一 B 行 = L2 命中第二遍);
- 位图在 SMEM(复用现有表的 col 槽位区域,表区在两遍式下不需要);
- 前缀 = Gv4 的组前缀方案(已验证);
- 溢出:est < popcount → fallback 到现有 hash 路径(保留 retry 机具);
- compact 契约:输出已行内有序(位图前缀即位次),走 copy 路(同 Gv4)。

**预期**:b45 池(Bump/c-big/delaunay/Hook)的 acc 相位 dup>1.5 人口,
CAS 消解 ≈ 3-4× 内环加速。Bump acc 72.5 → ~20-25ms 理论值(6 cyc vs 25 cyc)。

**验证**:微复现器先行(hh_repro4.cu)→ 值级 → 电池。

## 二十一、想法二微复现器 ALL-PASS(11-23 晚)

hh_repro4.cu(span≤8192, 500 用例含 dup 1-10×):位图 atomicOr 标记 → 前缀 →
直寻累积,升序+值级全绿。正确性锚点就位,下一步=生产实现(hash_spa 的
bins 2-8 路径加位图分支)。

## 二十二、想法二生产实施锚点(11-23 深夜,移交)

**CAS 探测环位置**:src/spgemm_kernel_hash.cu:2069-2105——hash_spa 主内环。
结构:__match_any_sync(warp 内去重)→ 无伙伴路径(2072-2087 CAS+线性探测)
或 Σv 归约路径(2088-2105 leader 单探测)。两路均有完整 CAS+探测+溢出。

**位图替换方案**(不删 CAS 路径,加分支共存):
- 模板参 `int BMODE = 0`(0=CAS 原路,1=位图两遍);
- BMODE=1 时 Pass1 = 每乘积 `atomicOr(&bmp[(j-clo)>>5], 1u<<((j-clo)&31))`
  (替换 2069-2105 整段);前缀+Pass2 追加在栅栏后(复用 Gv4 组前缀);
- 需传 col_lo(=d_span_lo[i],已有)与位图 SMEM 区(复用表区 col 槽位);
- compact 契约:BMODE=1 输出已行内有序(同 Gv4)→ copy 路;
- 发射侧:HH_LAUNCH 已有完整基础设施可复用;
- **实施约 200-300 行改动,半天到一天**——适合外部模型或下轮会话。

**成本模型速算**:CAS 路径 dup=5.7 → 每乘积 ~25 cyc(探测 5 次×3+CAS 2+
val add 3+match 2);位图两遍 ~6 cyc(Pass1 3+Pass2 3)+ 前缀 O(S/32)/行。
Bump 理论 acc 72.5→~18ms(4×)。

**注**:hash_spa 的 __match_any_sync 已做 warp 内去重(1 指令),对 dup
有部分缓解——实测 CAS 环的真实成本可能低于 25 cyc/乘积(需 ncu 确认)。
位图两遍的真实优势可能更接近 2× 而非 4×。

## 二十三、刀三战役终态(11-23 深夜收口)

四个想法全部处置完毕:
- 想法一(occupancy):定谳——acc 延迟遮蔽受限实锤(Bump 缩表→acc −19.5%),
  但 retry 吃增益(净≈平);人口分化=竞争力门;块宽空结果(表大小是限制)。
- 想法一 A(块宽):空。
- **想法二(位图两遍式):全套就绪**——设计稿(§二十)+ 微复现器 500 全绿
  (hh_repro4.cu)+ 生产实施锚点(§二十二,CAS 环 2069-2105/BMODE 方案/
  200-300 行/成本模型 2-4×)。**最大潜在收益,待实施**。
- 想法三(spec99+c61):未跑(想法二优先级更高,时间不够)。
- 想法四(warp 特化):未跑(排最后)。

**刀三移交清单**(给外部模型/下轮):
1. 想法二生产实施(锚点在 §二十二,最高优先);
2. 想法三(c61 标定,一天);
3. 想法一 C(人口自适应 alpha,路由层,竞争力门);
4. 想法四(warp 特化,最后)。

## 二十四、想法二生产实施 v1(11-24,进行中)

已落地(commit 待验证):
- 模板:hash_spa_kernel 加 BMODE=0(默认);1=位图两遍式
- Pass1:乘积环内 atomicOr(&tab_bmp[...])(替换 CAS 探测段)
- 前缀+计数:MODE==1 块前插入 BMODE 分支(popcount+归约)
- span_lo 用 __device__ 全局(g_spa_spanlo_d,同 g_fus_unsort_d 模式)
- 构建:spgemm_bm 通过(审计双绿)
- 当前状态:仅前缀计数(值未提取)——需加发射侧(装载 g_spa_spanlo_d,
  指定 BMODE=1 的 bin)+ Pass2 累积提取 + 值级验证

## 二十五、想法二实施中止+架构修正(11-24)

**中止原因**:BMODE 作为模板参与现有宏体系冲突——SPA_LAUNCH_LHT2 宏把
g_spabm(运行时 env)当模板实参传,nvcc 报"must have constant value"。
20+ 个发射点的宏改造复杂度超出本轮余量。

**架构修正(下轮方向)**:不用模板参,改用 **__device__ 全局符号**(同
g_hh_sorted_d 模式)——hash_spa_kernel 内加 `if (g_spa_bm_d)` 运行时分支,
免模板实例爆炸,20 个发射点零改动。Pass1 标记 / Pass2 提取的内核代码不变,
只是开关从模板折叠变全局符号读(每乘积环一次分支,不在热径)。

**已验证的构件**(可直接复用):
- Pass1 atomicOr 标记(内核代码已写)
- 前缀+popcount 计数(内核代码已写)
- g_spa_spanlo_d 装载(已接)
- nnz 三阵全同(无侵入确认)
- 微复现器 hh_repro4(500 全绿)

**回滚说明**:git stash 已回退全部未提交改动;下轮从干净树开始按新架构实施。

## 二十六、想法二改落 hash_heavy_kernel:HH_BM2 实施与第一轮作业(11-24,外部模型)

**锚点勘误(读源码所得,先于实施)**:
1. §二十二给的 CAS 环(src:2069-2105)在 hash_spa_kernel 的 `wm_eff==4` 分支——生产不可达
   (`g_wm` 只取 0/1;LHT 实例 `WM_T=0`)。生产主内环是 HASH_INNER_LOOP 宏。
2. b45(bin4/5)和 bin7/8/9 的生产累积核是 **hash_heavy_kernel**(默认 SPEC99=1、HASH_H45/H55/H69=1
   → hh_elig);hash_spa 在 TU-61 默认只接 bin6 与 bin2/3。想打 b45 人口,必须落 hash_heavy。
3. hash_heavy 只有一个发射宏(HH_LAUNCH2),模板实参本来就由宏参给——§二十五的"模板参与宏体系冲突"
   是 hash_spa 20+ 发射点的问题,这里不存在。故开关用**模板参 BM2 + 宿主 env 经 if/else 选实例**
   (同 g_s99och 手法):BM2=0 实例里新代码整块编译期消除,与改动前同一份机器码;行级门参数与普查
   计数才走 `__device__` 全局符号(g_bm2_*,同 g_hh_sorted_d 模式)。
4. b27d0fa 留在 hash_spa 的 BMODE 半成品已回退(提交 70d1dea,src/ 与 675b227 逐字节相同)。

**实现**(`src/hh_bm2.cuh` 的 `hh_bm2_row`,生产与 `tools/hh_repro5.cu` 共用同一份):
- 行级分流在 hash_heavy_kernel 入口、清表之前:合格行走两遍式后直接返回,不合格行原样走哈希路径。
  合格 = 位图(+值区)装得进本次发射的动态 SMEM(`hh_bm2_fits`;字数与 hhsm 同源 `hh_smem_words`)
  ∧ 行门(`HH_BM2_MINFLOP`、`HH_BM2_DUP`=flop×100/est 下限、`HH_BM2_SPANX`=span/est 上限;默认 0=不设,
  门值待判别实验定——每行固定成本是 O(span/32) 的清位图/前缀/键枚举,所以密度轴和 dup 轴都留了旋钮)。
- SMEM 布局:每个列域词一个 64 位条目(低 32=位图词,高 32=该词之前的置位数)→ Pass2 一次 8B 读
  拿到"位图词+前缀",位次 = 高半 + popc(低半 & 低位掩码)。组前缀(32 词/组)沿用 Gv4 做法。
- Pass1 只读 B 的列号流(4B/乘积),先读后 atomicOr;Pass2 读 B16,`atomicAdd(值[位次])`。
- 值区两种形态都做成实例供实测:`HH_BM2=1` 值在 SMEM(软件模拟 F64 原子,末尾拷出);
  `HH_BM2=2` 值直接累加到全局 `tmp_val[base+位次]`(原生 RED,免值区与拷出)。出处:study/57 §廿六
  四格矩阵——128T 下 L2 原生快 11%,256T 下慢 4.5%,所以不预设赢家。
- 输出契约 = Gv4 位图行:行内列升序直写 tmp、row_nnz=n;n>cap 不写输出、注册 ovf 交 retry。
  位图 bin 走 copy;fused bin(hh_flip=2)不置无序标志走 copy;legacy bin 仍过载体。compact 侧零改动。
- 覆盖面:位图 bin 里 TOTW 大的档(bin5 8192 字)能装下 span 到 ~11 万列的行,超过 Gv4 的 64K 门
  (这些行原本被踢去 retry);翻 bin 的 fused 档里窄 span 行也能走。SMEM 装不下的宽 span 行仍走哈希。
- 开关:`HH_BM2`(0 默认)、`HH_BM2_BINS`(bin 位掩码,默认 0x3f0)、`HH_BM2_STAT=1`(普查,诊断跑)。
  att(A·Aᵀ)不开;OCH=1 的 A/B 实例不配 BM2。bin6 默认不走 hash_heavy,要测用 `HASH_H69=2`。

**对成本模型的保留意见(入档,待实测裁决)**:§二十的"CAS 路径 dup×4-6 cyc"不成立——哈希路径每乘积
只探测一次(CAS 只在首次插入),同列 atomicAdd 争用两种形态都有;两遍式每乘积反而多一趟 B 读 +
一次 SMEM 读,省掉的是探测链、清表与表扫描提取。study/57 实测 b45 每乘积 ≈3.4 拍,不是 25 拍。
另:Bump/Hook/cage15/ldoor 属翻 bin 家族(宽 span 行占比过门),SMEM 位图只接得住其中的窄 span 行,
普查会给出实际覆盖面。**结论以作业读数为准,不预判。**

**第一轮作业 `bm2_r1`**(jobs/bm2_r1/run.sh,红即停):S1 微复现器(9 档几何×B16×值区模式,逐行真值)
→ S2 构建+审计 → S3 冒烟 → S4 值级(HH_BM2=1/2 对 =0 的 mtx_gate_fast;scipy 在则三判)
→ S5 三臂电池 off/m1/m2(同一二进制,env 臂,16 阵+靶阵,带 accumulate 相位与普查)
→ S6 行门判别(无门 / dup 门 250 / 密度门 32,只用 data/train_disp,测试账不参与标定)
→ S7 无侵入电池(基线源码件 vs 新件关开关,官方 ab_pair.py)。
新工具:`scripts/ab_arms.py`(多臂 env 电池,判读口径同 ab_pair)。

**顺带发现,待用户裁决**:main 的 `DUP_LIGHT=4`,认证形态(8301300,confirm19)是 8。31fc667 快照带入,
675b227 只回滚了 dfia 三处。§十三已判"门调整是死路",常数未改回。本刀未动它(两臂同常数,A/B 不受影响)。

## 二十七、想法二第二轮定谳(bm2_r2)与 v2:行门前置 + 行门标定(11-25,外部模型)

**bm2_r2 读数**(results 分支 results/bm2_r2/out/;GPU 0,同一二进制 env 臂,5 轮交错):
- 正确性:微复现器 ALL-PASS;差分门 16/16 + pre2/ldoor 大阵 3/3 PASS;官方 L1/L2 全零;官方 L3 标红为现任路径固有
  (off/m1/m2 同源,dialog/qa-001);归一误差 nerr ≤1.9e-15;官方 L3 错行数与 l3_diag 重算 12/12 相等。
- 无侵入(S7,官方 ab_pair 16×3,基线源码件 vs 新件关开关):**GM 0.9918,最差 1.034(F2 微级),绿**。
- **"凡装得下就走"的三臂电池是红的**(26 阵,4 阵微级):m1/off GM **1.0110**,m2/off GM **1.0130**;
  非微级最差 Bump 1.187(m1)/ Hook 1.254(m2)。逐阵(co 比 m1/m2;accumulate 相位比;两遍式 Σflop 占全阵比):

  | 阵 | 形态 | co 比 m1 / m2 | acc 比 m1 / m2 | flop 占比 m1 / m2 |
  |---|---|---|---|---|
  | cfd2 | 位图 | 0.941 / **0.897** | 0.908 / 0.836 | 81% |
  | F2(微级) | 位图 | 0.904 / 0.895 | 0.851 / 0.820 | 84% |
  | TSOPF | 位图 | 0.994 / 0.965 | 0.986 / 0.951 | 74% |
  | bcsstk30 | 位图 | 0.995 / 0.964 | 0.993 / 0.946 | 62% |
  | FEM_3D_thermal2 | 位图 | 0.981 / 0.970 | 0.967 / 0.907 | 59% |
  | msc10848 | 位图 | 0.998 / 0.989 | 1.001 / 0.991 | 30% |
  | s3dkq4m2 | 位图 | **1.086 / 1.063** | 1.129 / 1.100 | 99% |
  | Hook | legacy | **1.186 / 1.254** | 1.331 / 1.474 | 54% / 81% |
  | Bump | legacy | **1.187 / 1.220** | 1.278 / 1.335 | 31% / 45% |
  | pre2 | legacy | 1.034 / 1.012 | 1.039 / 1.036 | 23% |
  | ldoor | legacy | 1.027 / 1.015 | 1.037 / 1.022 | 9% / 10% |
  | offshore | legacy | 1.025 / 1.023 | 1.040 / 1.038 | 14% / 15% |
  | cage15 / nlpkkt80 / dielFilter / c-big(m1) | — | 1.018 / 1.010 / 1.012 / 1.021 | 1.030 / 1.015 / 1.013 / 1.022 | **0%** |

**定谳三条**:
1. **设计稿"dup>1.5 时位图赢"被证伪,方向相反**:两遍式每乘积多一趟 B 读,乘积越多越亏。s3dkq4m2(每行 flop≈2900,
   表 512)输 6-9%,cfd2(每行 flop≈710)赢 6-10%。训练池上 dup 下限门(flop≥2.5×est)也是更差(S6:1.017/1.036 对 0.985/0.965)。
2. **形态决定胜负**:赢家全在位图形态 bin(哈希路径要付清表 + Gv4 位图提取,两遍式正好省掉);输家全在 legacy 形态 bin
   (翻 bin 家族,宽 span:哈希路径没有位图成本,两遍式反要付 O(span/32) 的清位图/前缀/键枚举)。
3. **BM2 实例对没走两遍式的行有 1-3% 的税**(零覆盖阵 co +1-2%,acc +1.3-3%):每线程在清表前多读约 10 个全局量,
   外加实例码生成的差异。这笔税若摊到全账,会吃掉窄 span 人口的收益。
结论:两遍式不是普适替代,是**位图形态、乘积数相对表长不大**这一类行的赢家——即 qa-002 Q1 机制 (c) 的菜单项。

**v2 改动(本提交;行例程 src/hh_bm2.cuh 未动,微复现器结论沿用)**:
- **行门前置** `hh_bm2_gate_kernel`:binning 后与 Gv4 门普查同一发射批,逐行写标志(bit0=位图形态下走,bit1=legacy 形态下走)
  并出每 bin 合格行数/Σflop,挂既有 D2H 批(零新增同步)。核内每线程只读一个标志,标了位的行再复核 `hh_bm2_fits`。
- **bin 级取舍**:当前形态下合格 Σflop 占比 ≥ `HH_BM2_BINMIN`(且 >0)才发射 BM2 实例,否则就是原实例——不合格 bin 零税。
- **门形态**(全是 env,默认 0=不设):`flop×100 ≤ FX×HT`、`cw×100 ≤ CX×HT`(cw=⌈span/32⌉,HT=该 bin 表长);
  位图形态一组 `HH_BM2 / HH_BM2_FX / HH_BM2_CX`,legacy 形态一组 `HH_BM2_L1 / HH_BM2_FX1 / HH_BM2_CX1`。
  撤掉第一轮的 `HH_BM2_DUP / MINFLOP / SPANX`(方向不对或已被新形态覆盖)。
- **BM2 发射的 SMEM 恒为"表区+位图尾区"**(与位图形态同块/SM:bin4 14KB×16、bin5 32KB×7、bin7 56KB×4、bin8 112KB×2),
  SMEM 判定从此与翻 bin 无关,legacy 形态 bin4 的可装跨度由约 3.4 万列升到约 4.8 万列。
- 行计时最高位标"本行走了两遍式",`HASH_ROWTIME` 落盘后可逐行配对。
- 默认仍关;`HH_BM2` 未设时行门核不发射,BM2=0 实例与改动前同一份代码。

**行门标定方法(scripts/bm2_gate_cal.py,只用 data/train_disp)**:同一行在哈希路径与两遍式下的块时长配对
(off 跑与"不设门"的 m1/m2 跑),任何门的效果离线可算:R(门)=(Σ门内行 t两遍 + Σ其余行 t哈希)/Σ t哈希,逐阵算再平均;
在 (模式, FX, CX) 网格上分形态取最小。同一块宽、同一每 SM 块数下,块时长比就是占用 SM 时间比。
af23560(=电池里的 F2)不进标定,留作样本外。门值只作 env 交给评测臂,不入源码常数(入库归 dialog 裁决清单)。

**第三轮作业 `bm2_r3`**:值级(三臂官方三判 + l3_diag,标红阵全量清单按 qa-003 方案 A 交付;TSOPF、ldoor)→ S5 行门标定
→ S6 只评测电池(off / m2all / m2leg0 / auto,5 轮)→ S6b 实例税(零覆盖阵,强制 BM2 实例但不收行)→ S7 无侵入 → S8 pre2。
判读:auto 臂 GM<1 且无非微级阵 >1.10 = 行门成立;m2leg0 对 m2all 的差 = 形态规则的贡献;S6b 的比值 = v2 之后剩余的实例税。

## 二十八、想法二第三轮定谳(bm2_r3)与 v3:两遍式独立核(11-25,外部模型)

**bm2_r3 读数**(results/bm2_r3/out/;GPU 0,同一二进制 env 臂):
- 值级(v2 集成):五阵三臂差分门 10/10、官方 L1/L2 全零、nerr ≤1.4e-15、交叉标定 15/15 相等;TSOPF、ldoor PASS。
  标红阵全量清单按 dialog/qa-003 方案 A 交付(条目级:bcsstk17 30,142 条、msc10848 18 条进 OUT_DIR;s3dkq4m2 122.7 万条留 C)。
- **S5 行门标定(训练池 43 阵,HH 行 11.8 万,逐行配对块时长)**:位图形态 34 阵全部 R<1,"全收"平均 R=0.80(值区 SMEM)/
  0.76(值直写全局),按 flop/HT 分桶直到 [5.66,8) 都 <1,加 FX/CX 上限没有更好;legacy 形态只有 hcircuit 一阵,R=1.09。
  → 标定结果 = "只开位图形态、不设行门、值直写全局"。
- **S6 电池(26 阵,5 轮)**:
  - m2all(不设门、两种形态都开):GM **1.033**,Bump 1.51 / Hook 1.33(SMEM 加宽后 legacy 形态收进更多行,比 r2 更差);
  - m2leg0 与 auto 配置相同(=标定结果),等于同配置两次重复:GM **1.0105 / 1.0037**,非微级最差 1.095(s3dkq4m2)/ 1.103(msc10848)。
    赢:cfd2 0.93(acc 0.84)、TSOPF 0.97(0.95)、bcsstk30 0.96-0.97(0.93)、FEM_3D_thermal2 0.98(0.90);
    输:s3dkq4m2 1.095(acc 1.11)、msc10848 1.07-1.10(acc 1.08-1.09;r2 里是 0.99)。
    零覆盖阵(Hook/Bump/cage15/ldoor/pre2/nlpkkt80…):acc 比 0.999-1.003(bin 级取舍生效),co 比 1.005-1.02
    (offshore 1.035;F2 的 acc 比 0.85 而 co 比 1.06)——多出来的在 binning:行门前置每 run 两次 cudaMalloc + 6 次发射。
- **S6b 实例税**(零覆盖阵,强制发射 BM2 实例但不收行):GM 1.036;cage15 / c-big / nlpkkt80 / dielFilter 的 acc +5.2~7.7%。

**定谳**:
1. **"只开位图形态"在电池上是持平,不是赢**——五个赢家被 s3dkq4m2、msc10848 和行门前置的固定成本抵掉。不默认开。
2. **实例税是码生成级的**:v2 的核内门只读一个标志,强制发射 BM2 实例仍慢 5-8%(legacy 形态还叠加 SMEM 加宽挤占 L1)。
   两遍式内联进 hash_heavy 实例,既拖累同实例的哈希行,自身内环的码生成也随宿主上下文漂移(msc10848:r2 0.99 → r3 1.08,
   同一份行例程源码)。这与 study/57 "刀4 换个写法 +12%" 是同一类敏感性。
3. **逐行块时长不足以预测大阵的相位读数**:训练池的阵 B 整个在 L2 里;s3dkq4m2 这类 B 超出 L2 的阵,Pass1 读 B_ci、
   Pass2 读 B16 是两路流,多 25% 字节且第二遍不命中(推断,v3 的 P1 臂验证)。另:v2 值直写全局形态的行计时缺收尾栅栏,
   读数偏乐观(0.76 对 0.80 的差里有这一项)。

**v3(本提交)**:
- **两遍式 = 独立核 `hash_bm2_kernel`**(行例程仍是 src/hh_bm2.cuh 的 hh_bm2_row);`hash_heavy_kernel` 与 HH_LAUNCH 宏
  **回到与基线(70d1dea)逐字相同**,BM2 模板参撤销。对 src/spgemm_kernel_hash.cu 的改动相对基线是纯增量。
- **行表划分核 `hh_bm2_part_kernel`**:binning 后单次发射覆盖 bin4-9,把每个 HH bin 的行表划成"合格行靠前 / 其余靠后"
  (d_sort2,warp 内一次预留、按 lane 序落位,近似保序),并出每 bin 的行数、Σflop、合格行数、合格 Σflop、合格行 SMEM 需求上界,
  挂既有 D2H 批。缓冲为进程级静态(预热轮分配一次,计时轮零分配)。
- **发射**(hash_accum_c61):形态允许且合格份额够 → 合格行交 `hash_bm2_kernel`(块宽同该 bin 的 hash_heavy;
  **SMEM 按合格行实际需求给**,1KB 取整——cfd2 这类行约 1-2KB,而不是跟着哈希表预算的 14KB),其余行由原样的
  hash_heavy 实例处理(行表换成 d_sort2 的后段)。不合格 bin 与默认路径:发射代码与改动前相同。
- **Pass1 列来源可选**(行例程模板参 P1SRC;env `HH_BM2_P1=1` 读 B16 的列字段,与 Pass2 同一条流);行计时补收尾栅栏。
- env:`HH_BM2`(值区模式 1/2)、`HH_BM2_L1`(1=legacy 形态也开,默认 0)、`HH_BM2_FX/_CX`、`HH_BM2_P1`、`HH_BM2_SMX`
  (两遍式行的 SMEM 预算覆盖)、`HH_BM2_BINS/_BINMIN/_STAT`。v2 的 `_FX1/_CX1/_FORCE` 与核内标志数组撤销。
- 这就是 dialog/qa-002 Q1 的机制 (c):菜单里多一个核,主线核不动;注册进 routing_table.h 等这一形态在电池上站住再做。

**第四轮作业 `bm2_r4`**:微复现器重跑(行例程多了 P1SRC,6 种组合)→ 值级(五阵四臂 off/m1/m2/m2p;TSOPF;ldoor 的 legacy 臂)
→ S5 标定(Pass1 两种来源各一次)→ S6 电池 off / m2 / m2p / m2L / auto(5 轮)→ S7 无侵入 → S8 pre2。
判读:m2 或 m2p 的 GM<1 且无非微级阵 >1.10 = 位图形态的两遍式站住;零覆盖阵的 co 比应回到 ≈1.00(划分核的固定成本);
s3dkq4m2 在 m2p 下是否翻正 = Pass1 来源假说的判决;m2L 对 m2 = legacy 形态在独立核 + 按需 SMEM 下是否仍输。

**bm2_r4 构建失败,重投 `bm2_r5`(11-25,外部模型)**:r4 的 S1 是绿的(微复现器 6 种组合 × 9 档几何,两种子共 168480 行,bad=0),
S2 构建失败、55 秒退出(exit 12),v3 在 C 上一发没跑。原因:`src/spgemm_kernel_hash.cu` 里 `struct HashAccumCtx {` 一行被编辑脚本
写了两遍,nvcc 报 4937 行 "declaration of a member with the same name as its class",其余 8 条是它的连带;A 机当时只对抽出来的
片段做了语法检查,整份文件没过。修复(c804fd4,相对基线仍是纯增 261 行):
- 删掉重复行;
- 两遍式核的发射撤掉模板函数 `bm2_launch<LNT>`,改成 hash_accum_c61 里与 HH_LAUNCH2 同形的宏(字面量模板实参、
  opt-in 紧跟发射)——依赖型模板实参的核发射在这个仓库里没有先例,不留给 nvcc 去试;实例集不变(4 档块宽 × 6 种组合,只在 TU-61);
- 划分核里合格行 SMEM 需求上界的 64 位 atomicMax 换成仓库已在用的 int 版(写 64 位槽的低 32 位)。
A 机的静态检查改成整份文件、两个翻译单元各过一遍(桩头的原子/洗牌函数按 CUDA 12.0 文档的重载集逐个列出);同一检查对 r4 的
版本报出同样的 9 条错。r5 的阶段、臂、判读与上面 r4 的完全相同,只是 S2 的 make 加了 -k(两个翻译单元的报错一次出全)。

## 二十九、bm2_r5:v3 正确性定谳;计时读数因主机争用作废,改用设备侧读数 + 负载门(11-25,外部模型)

**背景**:r5 运行期间 C 的主机 CPU 被别的任务打满(load 27–35,nproc=32;dialog/qa-006)。B 端实测同二进制配对 GM 变差 3.9–6.5%,
退化集中在毫秒级小阵。r5 开跑后我即在 dialog 里预先声明:S6 电池、S7 无侵入的 GM 与绿/红裁定不作定谳依据,不论好坏。

**有效的部分**
- 正确性:S1–S4、S8 全绿。五阵四臂差分门全 PASS、官方 L1=L2=0、nerr ≤ 1.3e-15、交叉标定 20/20;TSOPF、ldoor(m1L/m2L)、pre2(m2L)
  差分门 PASS;值级里走两遍式 88.4 万行。**v3(独立核 hash_bm2_kernel + 行表划分)集成正确。** bin9 的生产值级仍空白(11 个候选阵都没有
  bin9 的两遍式发射)。
- S5 标定(核内 clock64 周期,GPU 侧):位图形态 34 阵平均 R = 0.797(m1)/ 0.777(m2)/ 0.763(m2p),最差阵 0.95–0.97,全部 <1,不需要行门;
  legacy 形态 1 阵 R=1.12。噪声对照 1.010 / 1.013。与 r3 一致。

**仅参考的部分(S6 报表:m2 GM 0.9794、m2p 0.9814、m2L 1.0151;S7 0.9890——不引用为裁定)**
- 大阵(≥10ms,逐轮配对比很紧):零覆盖阵 m2/off = 1.001–1.004 → 划分核固定成本 0.1–0.4%(v2 的行门前置 0.5–2%);
  legacy 形态 Hook 1.27 / Bump 1.475(acc 1.50 / 1.71)→ **legacy 形态在独立核下仍大输,定谳不开**。
- 毫秒级阵(方向性):cfd2 0.849、bcsstk30 0.905、TSOPF 0.933、FEM_3D_thermal2 0.956;s3dkq4m2 1.02、msc10848 1.02(v2 是 1.095 / 1.07–1.10)。
  m2p 没让 s3dkq4m2 翻正(1.058)→ "两路流"假说是不成立的方向。
- r3 的 off 臂轮间离散度不比 r5 小(s3dkq4m2 22.9% 对 4.8%)→ r3 很可能也在争用窗口里,其电池读数同样降级;"要拆独立核"的方向结论不变。
- 零覆盖中小阵上划分核的绝对成本约 40–70µs/run(case9 +0.5%、web-Google +0.8%、rgg +0.4%):候选优化 = 把划分并进已有的
  门普查核(hh_gate_census_off_kernel 本来就对同一批行发射 6 次),或只对位图形态的 bin 划分。

**测量口径的改动(此后沿用)**
- 主机吵时的主指标 = nsys 逐 kernel 的设备侧累计时长(scripts/kern_sum.py):臂间比不含主机侧发射/同步开销,所以只回答
  "核本身快不快";主机侧固定成本仍要靠安静窗口的墙钟电池。
- 墙钟电池带负载门(B 端判据:load1 < 6 连续两次、无外来 >50% CPU 进程)与 A/A 对照臂(scripts/aa_check.py:
  |GM−1| ≤ 1% 且非微级阵在 ±5% 内才算干净);段内每分钟采 load,事后复核。
- 作业 `prof_k1`:K2 nsys(22 阵 × 4 臂 × 2 轮,兼作 BRIEF §五 P3 的第一版逐核账)、K3 ncu(5 阵)、T1 电池(off/aa/m2/m2p)、T2 无侵入。

## 三十、prof_k1:设备侧读数定了核一级的结论;v3.1 按需划分;门普查核是第一把 profiling 找到的刀(11-25,外部模型)

**prof_k1 的结果**
- K2(nsys 2024.6.2,`cuda_gpu_kern_sum`,22 阵 × 4 臂 × 2 轮,176 发 0 失败,轮间离散多数 ≤0.6%):Σkernel 设备侧时长对 off 之比——
  m2 GM **0.9697**(最差 s3dkq4m2 1.016;有两遍式发射的 11 阵 GM 0.9385);m2p 0.9733(最差 s3dkq4m2 1.055);
  m2L 1.0039(Bump 1.464、Hook 1.269)。逐阵 m2:cfd2 0.833、af23560 0.830、bcsstk30 0.901、TSOPF 0.914、bcsstk17 0.926、
  msc10848 0.949、FEM_3D_thermal2 0.971;零覆盖阵 1.000–1.006。与 r5 电池的逐阵读数方向、量级一致。
  **结论:位图形态的两遍式在核一级是赢的;legacy 形态输;m2p 不作默认。**这组数不含主机侧开销。
- K3(ncu)试点失败:`Failed to prepare kernel for profiling`(非权限)。下一作业逐个试 application 重放 / 关内存池 / 正则去掉划分核。
- T 段没跑:等满 10 小时,B 端的安静判据(load1 < 6 连续两次、无外来 >50% CPU 进程)一次都没满足(load1 8–12,外来进程 3–5 个)。
  → 向 B 端提议(dialog/qa-007):计时有效性改由 A/A 对照臂直接认证,负载门降为"先等 30 分钟,等不到照跑"。

**P3 第一版账(22 阵、基线臂、设备侧)**:hash_heavy_kernel 37.2%、dense_largest_numeric 13.1%、hash_dense_window 8.4%、
hash_compact_copy_warp 7.7%、csort_counting 7.7%、hash_spa 6.5%、mh_merge 5.1%、hh_gate_census_off_kernel 3.0%。被发射过的实例 60 个。

**第一把刀:门普查核的原子争用**。hh_gate_census_off_kernel(Gv4.2)在大阵上占设备侧 Σkernel 的 4–7%:cage15 7.0ms/run(5.8%)、
nlpkkt80 1.65ms(6.8%)、ldoor 1.46ms(5.2%)、Bump 4.26ms(4.2%)、Hook 1.92ms(3.8%)。读源码:每行一到两次 atomicAdd 打在同一个
bin 的计数地址上,无 warp 归约;bin4–9 各发射一次、每次铺满 A_rows 线程。方案:单次发射 + warp 内归约(结果是整数和,逐位相同),
并可与两遍式的行表划分并成一个核。待做(opt-in 先行)。

**v3.1(commit e898272):按需划分**
- 问题:v3 只要开了 HH_BM2,每个阵每次 run 都多一次划分核发射 + 清零 + 回传(设备侧 3–15µs,大阵到 0.3ms;加上主机侧发射开销)。
  全 428 里小阵多,这笔固定成本会吃掉影子的 GM。
- 做法:`HH_BM2_LAZY=1`(默认)——等 binning 的既有同步点之后,宿主已知各 bin 行数与门普查结果,只在"掩码内的 HH bin 确有行且
  形态允许"时才发射划分核(网格只盖 HH 行段),随后一次同步取回计数。其余阵零新增发射、零新增同步。`HH_BM2_LAZY=0` = v3 原样。
- `HH_BM2_BINS` 默认 0x3f0 → 0x3b0(bin6 默认不走 hash_heavy)。相对基线仍是纯增(284 行)。

**bin9**:普查事实——电池 24 阵与训练池 43 阵里 HH 行只落在 bin4/5/7/8。按 B 端建议:web-Google 用强制形态 `m2X`
(HH_BM2_L1=1 + HH_BM2_SMX=57856,生产默认到不了)做差分门并带封条注;另在 PARSEC 族等候选阵里探查真正的 bin9 行。

**作业 `bm2_t1`**:S2 构建 → S3 冒烟(按需 / 同批 / 强制宽跨度三组各自判)→ S4 值级(v3.1 回归 + pre2 + web-Google 封条 + bin9 探查)
→ K3 ncu → T1 电池 off/aa/m2/m2p × 7 轮(A/A 认证)→ T1b 小阵固定成本探针(m2 对 m2e)→ T2 无侵入。

## 三十一、门普查单发射版(HH_CENSUS=1,opt-in;11-26,外部模型)

- **靶**:§三十 的第一把刀。hh_gate_census_off_kernel 每行 1–2 次 atomicAdd 打同一个 bin 的计数地址,无 warp 归约;bin4..9 各发射
  一次、每次铺满 A_rows 线程。大阵上占设备侧 4–7%;同一批行、读得更多的划分核只要它的 1/30(cage15:0.24ms 对 7.0ms),
  所以瓶颈是原子争用而不是访存。B 端读码核实,无已知坑(dialog/qa-007)。
- **做法**:`hh_gate_census1_kernel`——一次发射盖 bin4..9,warp 内归约后每 warp 至多两次原子加。结果是 64 位整数和,与原核逐位相同,
  下游的 hh_leg 判定因此逐 bin 相同。`HH_CENSUS=1` 才走;默认 0 时原循环原样执行(源码上是在原 for 前插了 if/else)。
- **预期**(待测):大阵设备侧省掉普查的大部分(cage15 约 −5%、nlpkkt80 约 −6%、ldoor 约 −5%、Bump 约 −4%、Hook 约 −3.5%);
  所有走 hash 路径的阵每次 run 少 5 次核发射,小阵上按主机侧发射开销计也有收益。
- **证据计划**:① `HH_CENSUS_STAT=1` 的 [census-stat] 行在两种模式下逐行相同(电池 + 428 里抽一批);② 差分门(开对关的转储应逐字相同);
  ③ nsys 设备侧 Σkernel 对比;④ A/A 认证口径的电池与小阵探针;⑤ 全 428 影子(它影响每一个走 hash 路径的 run)。
- **后续**:与两遍式的行表划分并成一个核(开 HH_BM2 时再少一次发射和一次同步)。

## 三十二、bm2_t1 定谳;普查 + 划分合一(11-26,外部模型)

**bm2_t1(exit 0,329 分钟)**
- 值级全绿:v3.1 回归三阵 + pre2(m2L);web-Google 强制宽跨度 m2X(36,737 行,封条);**bin9 生产可达样本** GaAsH6(967 行)、
  Ga3As3H12(2,691 行)差分门 PASS。
- T1 电池(A/A 认证口径,7 轮,段内 load1 中位 10.5):A/A GM 1.0001、非微级 0.988–1.031 → 干净。**m2/off GM 0.9862,非微级最差 1.023 → 绿**;
  m2p 0.9830(最差 s3dkq4m2 1.068)。逐阵 m2:cfd2 0.896、TSOPF 0.924、bcsstk30 0.920、FEM 0.972;s3dkq4m2 1.023(真的小输);零覆盖大阵 ≈1.000。
- T1b 小阵探针:A/A GM 0.9816 → 整段作废(微级阵 11 轮不够)。
- T2 无侵入(官方 ab_pair 16×3,无对照臂):GM 1.0296 判红,红在 1138_bus 1.32、F2 1.15 两个微级阵;非微级 12 阵 GM 1.002。没有判决力,需带对照臂重做。
- ncu 2022.4.1 四种组合都不通(Failed to prepare kernel for profiling,非权限)。

**按需划分的代价**:多出的那次同步在有负载的主机上不便宜——cfd2 从 r5 同批划分的 0.85 退到 0.90;offshore("掩码内有行但一行不收")+1.5%。
同批划分又让用不上两遍式的小阵白付一次发射。

**普查 + 划分合一(commit 2ea31a5)**:`HH_CENSUS=1` + `HH_BM2=2` + `HH_BM2_LAZY=2` → `hh_census_part_kernel` 一次发射出门普查结果
(与单发射版 hh_gate_census1_kernel 同式,结果逐位相同)和行表划分(判式、布局同 hh_bm2_part_kernel)。不多发射、不多同步,相对原样的
逐 bin 普查少 5 次发射。hash_product 里用 bm2_when(-1/0/1/2)统一分派,缓冲准备提成宏 BM2_PREP;默认不设任何开关时行为不变。
相对基线纯增 462 行。

**测量口径补充**:无侵入不再用两二进制三轮的官方 ab_pair 直接判——ab_arms.py 加了臂级二进制覆盖 `AB_BIN`,基线源码件作为一条臂
(off / aa / base)进同批交错电池,有效性看 A/A(待 B 端确认,dialog/qa-008)。

**作业 `cen_r1`**:冒烟三组 → S4a 门普查行为保持(约 70 阵,[census-stat] 行与 nnz 逐行相同)→ S4b 差分门 → K2 nsys 设备侧(off/cen/m2/cm2)
+ CUDA API 调用次数 → K3 ncu 探查 → T1 电池 off/aa/cen/m2/cm2 × 7 轮 → T1b 小阵探针 × 25 轮 → T2 无侵入(带对照臂)。

## 三十三、cen_r1 定谳:门普查单发射与"普查 + 划分合一"全绿;转入全 428 影子(11-26,外部模型)

**cen_r1(exit 0,674 分钟)**
- 行为保持:HH_CENSUS=0 / 1 / 合一 的 [census-stat] 行与 nnz 逐行相同,24 阵、75 个非空 (阵,bin) 组合、0 不一致
  (原计划约 70 阵:reference/mats_428.csv 是 CRLF 行尾,抽样的阵名带 `\r` 被静默跳过——脚本的错,全集比对并进 prof_p1 的签名)。
- 差分门全 PASS(含 GaAsH6 的 bin9、pre2 的合一 + legacy 形态)。
- 发射清单(nsys 逐核表,核名 + 次数,22 阵 × 4 臂):除普查 / 划分 / 两遍式 / hash_heavy 几族外,其余核逐名发射次数四臂完全相同;
  cen 对 off 每进程恰好 −10 次(每 run −5);cm2 −10(个别 −8);同一(阵,臂)两轮清单 88/88 相同。
- 设备侧 Σkernel 对 off:**cen GM 0.9772**(cage15 0.943、nlpkkt80 0.933、ldoor 0.950、Bump 0.959、Hook 0.964;最差 c-big 1.022,
  多出来的不在 HH 核族);**cm2 GM 0.9461**(最差 1.0001;有两遍式发射的 11 阵 0.919);m2 0.9690。
  门普查核族:cage15 7.0 ms/run → 0.24(单发射)/ 0.33(合一);Bump 4.25 → 0.15 / 0.33。
- 墙钟电池(A/A 认证;A/A GM 0.9960、非微级 0.977–1.021):**cen 0.9706**(非微级 GM 0.9768,最差 msc10848 1.022)、
  m2 0.9725(非微级 0.9823,最差 offshore 1.023)、**cm2 0.9605**(非微级 0.9630,最差 msc10848 1.038)。三臂都绿。
- 小阵探针(11 个微级阵 × 25 轮):第 1 次 A/A 1.0294 作废;整段重跑,第 2 次 A/A 1.0055 通过:cen 0.9193、cm2 0.9374。
- 无侵入(带对照臂,16 阵 × 7 轮):A/A 1.0011;base/off GM 1.0007,非微级 0.980–1.002 → 绿。
- ncu:/opt/nvidia/nsight-compute/2025.1.1/ncu 可用(application 重放);2022.4.1 那个不通。

**结论**:门普查那一刀成立——结果逐位相同,大阵省 4–6%,小阵每 run 少 5 次发射省 6–8%;合一形态在此之上叠加两遍式的收益,
且没有哪个阵的设备侧时长比基线差。msc10848 的墙钟读数与设备侧对不上(+2~4% 对 −1.4~3%),加轮复核。

**作业 `prof_p1`**:P0 影子脚本试跑 → O1 发射清单比对(基线件对本件开关全关)→ O2 官方 ab_pair 16×9(参考轨)→ O3 msc10848 × 21 轮
→ K3 ncu 机制计数 → **P1 全 428 影子**(off / aa / cen / cm2;METHOD=adaptive、生产默认 env;轮数按文件大小 5/3/2;每发带 [census-stat] 签名;
相位长表 → 默认形态地图与各池前 20)→ P2 旧口径(HASH_LHT23=0)地图单发。
全集上的 A/A 判据(开跑前写定,scripts/shadow_sum.py 文件头,dialog/qa-009):|GM−1| ≤ 1%(全体与非微级)且非微级阵超 ±5% 的不超过 2%
(列为不裁决阵);某臂绿 = A/A 干净 ∧ GM ≤ 1.02 ∧ 无可裁决的非微级阵 >1.10 ∧ nnz 逐阵相同。

**影子绿之后**(BRIEF 快速通道):先翻 `HH_CENSUS` 默认 1,再按 cm2 的读数决定 `HH_BM2=2` + `HH_BM2_LAZY=2`;两步分开提交;
翻之前做 routing_table.h 的菜单注册;翻完在 BRIEF 文件报备,confirm20 归 B 端。
