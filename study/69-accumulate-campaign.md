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
