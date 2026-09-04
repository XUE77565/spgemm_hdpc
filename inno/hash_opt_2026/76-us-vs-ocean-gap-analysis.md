# docs/76 我们 vs Ocean:差距全景与开枪顺序(2026-09-04,全部分相实测驱动)

> 结论一句话:**差距不在"选形的智慧"(我们 20 桶更细),在"被选中的形态本身"**
> (表 vs 零表、csort vs 直写、槽宽、块宽)。④ 先打,②③ 等验靶。

## 一、架构对照(骨架同款,差异在落点)

```
我们:  count_flop → MinHash(construct+merge) → est_scan → binning → scatter
        → accumulate(20 桶选形) → retry → compact+sort
Ocean:  analysis(乘积+ER+3%采样CR) → 工作流三选一(免估计/上界/HLL)
        → binning ×2(符号/数值) → numeric(ESC+hash+dense 梯) → epilogue(CR 门控)
```

| 维度 | 我们 | Ocean | 实测后果 |
|---|---|---|---|
| 前端信息采购 | count 0.49+est/bin 2.2ms | analysis+est ~2.5ms | **已打平,COUNT_TINY 反超 3×(7.6→0.49)** |
| 行选形钥匙 | est | 乘积数+est 双钥匙 | tiny 行殊途同归 |
| **accumulate 形态** | 表(哪怕 64 槽) | **零表 counting-sort** | **road 832 vs 314ns/行 = 残差 76%** |
| 输出有序性 | 表提取无序→csort | counting 天然有序→纯 copy | compact 6.12 vs 2.39 = 残差 23% |
| 表槽宽 | 符号+数值 12B | 符号 8B(key 压位)/数值 12B | 中行族候选根因(**未验靶**) |
| 块宽 | 固定 256/512 | 档位↔占用梯联动 | 同上候选根因 |
| 溢出兜底 | retry=flop 定表(重) | 专用 fallback kernel | cage15/est 欠估族 |
| compact 门控 | 每行必过 | CR 阈值 | 次要 |

## 二、分相对拍实录(road_usa,双方自家 profiler,同族口径)

| 相位 | 我们 v30 | Ocean(stats.json) |
|---|---|---|
| count | **0.49** | 1.45(product_calc) |
| est+bin+cnnz | 2.2 | ~1.5 |
| **accumulate** | **19.96** | **7.53(usparse_main = ESC 16-thread subwarp)** |
| **compact** | **6.12** | 2.39(epilogue.copy) |

- 残差 16.4ms = accumulate +12.4(76%)+ compact +3.7(23%)
- Ocean:avg 乘积 6.67(max 38),decision.workflow=2,ultra_sparse.kernel_size=16
- 我们:24M 行全落 bin[19]=BATCH_S(64 槽表)——v30 后 accumulate 19.96ms 即 BATCH_S 的账
- Ocean stats.json 用法:`ocean/spgemm <csr> config/bench_detail.json`(track_stage_time)

## 三、口径审计(2026-09-04,源码级验证)

| | 我们 | Ocean |
|---|---|---|
| 计时器 | HashProf 相位 cudaEvent(GPU 流 elapsed) | cuTimer cudaEvent 对(单括号包全程) |
| TOTAL | **Σ相位**(相位间 host 间隙不计) | 单括号(内部 host 空洞全计) |
| 排除 | h2d/d2h/mtx 读 | 同 |
| 统计 | 末轮+双 expand min | 10 warmup + 10 轮 mean |

三处不对称**全部偏向我们** → 1.3892× 为保守口径真实差距。
论文注记义务:对比表须注明"Σ相位 vs 单括号"结构差。

## 四、输赢全景(v29 口径,336 阵)

稳赢 2 / 赢 18 / 小胜 10 / 持平 9 / 小输 53 / **输 190(1.2-1.8×,六成 gap)** /
稳输 49 / 惨输 5。惨案已收口(road/c-big 有弹),宽输带 = ②③ 的账。

## 五、开枪顺序

### 第一枪(靶实锤,弹上膛):v31-④ ESC 零表短行内核
- **靶**:flop≤32 的行(road 24M 行 avg 乘积 6.7),替换其在 BATCH_S 的位置
- **弹**:csort 位次计数**前移到 accumulate**——每线程一乘积,位次 = popc(比我小的 key),
  leader 求和直写;零表零清零;输出天然有序(compact 的 csort 税同时消失)
- **血缘**:机制是我们自己的 csort,Ocean 只证明它在 accumulate 位置成立
- **预期**:accumulate 19.96→~9,compact 6.12→~3 → road_usa 28.7→~14-15(2.4×→1.2×),
  全族 ~12 阵,geomean −4~6%

### 第二枪(先验靶):pre2/TSOPF 分相对拍
同款 stats.json 对拍 → 确定中行族差距落在哪相(symbolic/numeric/epilogue)——
**这一枪的靶纸决定 ②③ 的真实形状**(12B→8B?块宽?还是别的)。没有这张靶纸不开枪。

### 第三枪:②③ 按对拍结果实施(8B 符号槽 / 占用梯块宽)

### 第四枪:溢出兜底轻量化(cage15 类 = largest 精确管线集成)

## 六、明令禁开(证伪/错靶存档)

1. **管线遍数合并**(v31-① 降级):前端已反超,遍数税假设被分相对拍杀死
2. **merge3 打 tiny 行**:K 桶在 k≈2.4 行上退化,武器错配;merge3 的战场 = first100 整阵
3. **更多 bin 细分**:粒度(20 vs 9)从来不是问题,落点形态质量才是
4. **dispatcher 调参**:337 上 0/337 选 merge3 是正确行为,无 geomean ROI

## 七、当前状态

- v30 全刷挂负载门(gem5 灾变);v30 后:first100 复测(#18)→ ④ 实施+电池
- v29 官方:1.3892×/赢 34;v30 预期 ~1.32×;v31 目标 ~1.15×
- 本文档为 v31 的执行依据,更新随战报追加

## 八、v31-⑤ warp 本地预合并散射(2026-09-04 用户提议)

**提议**:hash 同槽多次放存(+3/+7)合并为一次(+10)→ 省 CAS/atomicAdd 次数。

**机理定位**(分析入档):
- 重复只来自**不同链**(CSR 链内有序无重复);同槽命中**跨时间**(不同迭代/不同 G-group)
  → warp 票决聚合(同指令同地址)抓不到,必须 warp 本地 SMEM 缓冲
- 可实现形:warp 攒一段乘积 → 本地排序+游程合并 → 每 distinct 一次 CAS+一次 Add
- **经济学**:省 (dup−1)×原子,付 O(w·log w)/段;dup≥2 即赚,dup≈1 纯亏(须门控)
- 生态位 = 中行族 hash 桶(est 65-16K,dup 2-8)= 宽输带;与 ②(槽宽)③(块宽)正交
- 家族关系:④(ESC,flop≤32)与 merge3 是"完全体"(连 scatter 都省);⑤ = 中行域的增量形
- 执行:④ 落地划走小行域后,mid bin 单桶试验(env PRECOMB=1),任务 #23

## 九、v31 终版复盘 + ESC 门四连修(09-04 深夜)

- v31 = 1.3643×/34(ESC 收益被 est≤16 行回归对冲);c-big 再 −25% 至 103ms(1.72×)
- **门演化**:v2(无门,全 flop≤32)→ v3(est>16,误杀 germany 的 ESC 赢面:germany 行 est≤8)
  → **v4(dup≤2,终判据)**:counting 成本∝flop、表成本∝est,盈亏平衡 dup≈2
- **重大定性:BIN_ULTRA 是死代码且"正确的死"**——第二链 fallback 无条件覆写 ultra 位
  (守卫只跳 largest/ESC);复活实验(守卫补 ultra)实测 bloweya ~8× 慢 → est≤16 行
  在 BATCH_S/ESC 里确实比 ultra 线扫快,死代码回退保留,定性入档
- germany(dup1.4)vs patents(dup≥2)的分界 = counting/表 成本模型的教科书实证

## 十、第二枪靶纸(09-04 晨,phase_shootout 中行族)——②③ 原假设部分证伪!

| 阵 | 我们 TOTAL | Ocean | 主凶相位 |
|---|---|---|---|
| pre2 | 45.77 | 15.25 | **compact+sort 26.2(57%!)vs epilogue.sort 2.5 = 10×排序税** |
| TSOPF_b2383 | 29.76 | 10.38 | **dense_direct 22.2(75%)——路由错:同行 Ocean 走 hash numeric 5.6,我们错送 BIN_DITER 窗 = 4×** |
| bcsstk30 | 3.16 | 2.52 | accumulate 1.25× + compact/retry 小税(②③原靶,次要) |

**攻击序列重排**:①' compact 排序税(中行族大行输出有序化)→ ②' diter 路由门修正(TSOPF 形态回 hash)
→ ③' 原②③(accumulate 常数)。前端全平(再证伪遍数税);retry 3.45 vs overflow 0.11 顺带靶。

## 十一、①' SEGRTS 电池定谳(09-04 晨)——排序税假设【部分证伪】

- 实现:bin≥7(后扩 ≥6)csort → cub 分段基数排序,6/6 nnz 精确,两修(段内核索引域/scan 尾)
- 电池:pre2 −0.6~−1.0 / c-64 +2.7~+5.1 / F2 +1.1 / crankseg_2 +0.1 = **全 wash**
- **结论:csort 本身不是 pre2 compact 26.2ms 的主体**——换线性时间排序零变化 ⇒ 26ms 在
  compact 相位的其他成分(copy 群/margin/retry 善后/同步),需先做相位内分解再开下一枪
- SEGRTS 保持门控(中性偏负,默认关);#24 状态 = 假设部分证伪,待分解

## 十二、②' diter 路由门三轮实验(09-04 晨)——单行特征无法分离,存档开放

- **电池(DITER off)**:TSOPF×2 −34.3%(hash 赢)✓ / **c-64 +98%(diter 赢)**✓ / Ga/crankseg/nd24k/3Dspec wash
- 饱和门(2e≥cap):TSOPF 19.6 ✓ 但 c-64 43.9(把 e=4096∧dup2.6 的 diter 赢家行踢了)
- dup 门(flop≥2e):TSOPF 19.4 ✓ 但 c-64 26.8(e=8192∧dup1.6 行也被踢)+ Ga 58.1(dup1.57 行要窗)
- **结论**:c-64/Ga(diter)与 TSOPF(hash)的边界不在任何已测单行特征(est/dup/span·n 组合均反例),
  疑似矩阵级行分布形态(同 e=4096 的行在 c-64/Ga 要窗、在 TSOPF 要表)→ 需矩阵级判据或
  per-row 采样成本比较(如 OC 采样式);TSOPF −34% 收益锁在抽屉,待判据

## 十三、compact 相位分解终稿(09-04,#24 前置完成)

- pre2 compact 26.0ms = bin6 csort_fused **15.3ms** + bin7 **9.8ms** + 小桶 0.7 + retry 段 ~0.2
  (首轮插桩误落 accumulate 循环,13.3≈accumulate 13.84 自洽后才定位)
- **SEGRTS wash 悖论解开**:bin6 ≈ 3.5 亿项,cub 分段排序需 gather+sort+scatter 三过数据 ≈ csort 同价;
  Ocean epilogue.sort 仅 2.5ms(323M nnz)是【表内槽序提取 + 全库单趟 radix】设计 —— 不是换排序器
  能追的,是要让 accumulate 的提取天然按序(其 hash 表布局使桶序=列序的邻接)
- **#24 正确形态修正**:不是"换排序器",是 Ocean 式"表布局→提取序"改造(hash 梯桶内 key 存储按
  探测序?或 table 分 slot-band 使提取近似列序 + 单趟全库 radix 收尾)—— 工程量大,单列 v32+ 议题

## 十四、Ocean epilogue 终审(09-04,源码读毕)

- epilogue.sort = **逐行 sortOutputFused(cub BlockRadixSort)**,排序位 `0..IDX_BITS` =
  压缩列位宽(13-20 位)非全 32 位;`my_elements ≤ NUM_HASH_INPLACE_SORT_THRESHOLD → no sort just copy`
- 其 hash = 标准 open addressing(murmur%SIZE),**提取非序**——"表布局序"假设不成立(§13 修正);
  2.5ms 的真来源 = ①压缩位宽砍 radix passes ②多数行走 no-sort-copy 阈值(小表行在表内近乎有序?)待核
- 我们:radix 内核 `Sort(col,val,0,32)` 全 32 位(NOCSF 实测 61.9ms 反劣:radix+额外 copy);fused 计数 26ms 已是两者较优
- **可动刀口**:csort_radix 压缩位版(Sort(col,val,0,log2(A_cols)) 或 bin 级 IDX_BITS)—— 与 valid_bits
  参数同源,估计 bin6/7 15.3+9.8 → 5-8ms(直逼 Ocean);工程小,净赢 8-12ms/pre2 族

## 十五、csort 位压缩/IPT 两连试(09-04)——未能复现 Ocean 2.5ms,存档收线

- 位压缩(Sort 0..20 位):61.9→61.2ms 无感;bin6-7 换 512×8 小 IPT:61.7ms 无感
  ⇒ 我们 radix 路径的瓶颈不在 pass 数/寄存器配置,计数 fused(26ms)仍是我们最优
- **重新对账 pre2 全账**:Ocean numeric.hash 5.7 + epilogue.sort 2.5 = 8.2ms,而我们 accumulate
  13.8 + compact 26 = 39.8ms —— **他们的 accumulate 也比我们快 2.4×**,不止 epilogue!
  ⇒ 剩余差距不在"换排序器"层,在其 numeric 内核产出布局(疑似直写 CSR + 行内近序)——
  需 ncu 对拍两内核的指令/访存画像才能定位,列入下 session 首项
- 本轮结论:fused 计数排序保持默认;radix 位压缩代码保留(valid_bits 参数化,中性)

## 十六、ncu/nsys 双画像终稿(09-04,#26)——机制地图定案

**Ocean pre2 每轮(≈14ms)内核分解**:
| 内核 | ms | 含义 |
|---|---|---|
| **denseNumericKernel<8>** | **4.2** | **pre2 大块行量走 DENSE!** |
| hashNumeric<8,1327> | 3.4 | hash 只收剩余(压缩表 1327 槽) |
| sortOutputFused<16,192,**20 位**> | 1.7 | IDX_BITS=20 压缩 radix |
| hashNumeric<9,2719>/<6,331> | 1.2+1.0 | 两个更小 hash 档 |
| sortOutputFused<16,96,21 位> | 0.78 | |
| dense/hash Symbolic ×3 | 1.6 | |

**我们 pre2**:hash_spa 全收(accumulate 13.8)+ csort_fused(26)。

**三个机制结论**:
1. **pre2 的行量被 Ocean 分给 dense(4.2ms 免排序!),hash 只收边角** —— 我们全喂 hash(bin6/7
   24 万行,est 1-4K)。**新杠杆 = dense 窗路线扩到 est 1024-2048 行**(现 diter 门 est≥2048 挡住
   bin6;diter 的 span/dup 门它多半已过)
2. sortOutputFused 确证 20-21 位压缩 radix(§14 推断落实)
3. Ocean numeric.hash 快的部分原因是**它收的行少**(dense 分流),非内核本身神速 ——
   §15 "accumulate 2.4×" 的归因修正:分流 + 每行成本共同作用
**下一步(实锤刀口)**:diter 门 est 下限 2048→1024(或 Ocean 式 dense_bin 梯),
电池 pre2/c-64/Ga 守卫族。

## 十七、dense 分流杠杆两试(09-04 终)——wash,关线

- DITER_MIN_FLOP 4096→2048:pre2 −0.1 / in-2004 −5.7(单项)/ crankseg −2.1 / 守卫稳
- DITER_EST_MIN→1024(BINS_DBG 确认 bin6 行拿到资格):pre2 +0.2 全 wash
  ⇒ bin6 行(est 1-2K、大 span)进了 diter 资格也不赢窗——**spanf 门(16flop≥span)仍拦**或
  多窗重扫吃掉收益;Ocean 的 dense 优势在其 **span 级窗口内核本身**(symbolic 1-bit/numeric 9B/列
  + 档位窗宽),非我们现有 21B/列 diter 窗的门控可及
- **§16 杠杆正式关线**;DITER_EST_MIN 参数化保留(默认不变,中性)
- **本 session 中行族战果总结**:pre2 差距 45.8 vs 15.3 的结构地图完整(docs/76 §10-17),
  可动刀口收敛到【Ocean 式 span 窗内核族 = 位图 symbolic + 9B/列 numeric + 档位窗宽】=
  v32+ 的版本级工程(即 docs/70 的 BCNT/DSTRUCT 路线升级版)

## 十八、v32 span 窗内核族 Phase 1a 电池(09-04 终)

- DSTRUCT=1(位图 symbolic count):pre2/in-2004 wash,c-64 **+36%**(atomicOr 窄 span 争用,
  c-73/gupta3 同款教训),crankseg +6% —— **净负,Phase 1a 否决**
- 至此中行族全部已知杠杆(排序器/门控/位图化)逐一电池定谳:wash 或负
- **终局判断**:Ocean dense 优势 = 其 symbolic+numeric+binning 三者**咬合的整体**(行按
  span/乘积双维分 7 档,每档窗口/并发/布局协同),移植必须整体走,单件移植每件都被
  我们的行分布打回(c-64 窄 span 争用就是证据)。= 多日级版本工程,非电池可调
- **session 收官账**:v28 1.388→v30 1.3686(五件套)→v31/v32 噪声带;ESC/SEGRTS/diter×3/
  csort×2/DSTRUCT 六役全电池定谳;docs/76 十八节 = 追平 Ocean 的完整作战地图

## 二十、单窗帽第四试 + 最终解剖(09-04)

- 单窗帽(win=917K 列)后 c-64 仍 +43% ⇒ 病因非多窗重扫,是 **atomicOr 争用本身**
  (行乘积 1-3 万次打 ~1568 位图字);且 win 帽太宽(917K)形同虚设
- **pre2 wash 的真因解剖**:我们 pre2 的 diter 行只有 2 行!DSTRUCT 只换了 2 行的 count 内核,
  大队人马(est 1-4K 的 38 万行)全在 hash 桶 —— **缺的是路由,不是 count 内核**
- Ocean 侧:pre2 大队走 denseNumeric<8>(4.2ms,nsys 实证)。我们的 diter 窗(21B/列多窗)
  §17 实测无赢 ⇒ **唯一没试过的组合 = 方案5 管线(count→cnnz→直写)里把 21B 窗内核换成
  Phase 1b 的 8.25B/列 dense_struct_numeric(已存在),路由 est 1-4K ∧ span 单窗行进方案5**
  —— 这就是"五件咬合"在我们代码里的精确落点,下 session 一步可建
