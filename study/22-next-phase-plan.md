# 下一阶段计划(2026-09-17 定稿,用户钦定三分法;compact 前冻结版)

## 战场态势(全部实测)

对 ocean hash kernel:轻行赢 20-30%(F2 0.802/c-64 轻带 0.713);重行输(crankseg 1.355/c-64 中带 1.775);
系统级 ④交织B 已入账 −1.9%。五雷全清,11/11 阵 17 位值级+有序全绿(diter 修好默认开)。

关键物理(两天换来的):
- L1 命中:我们 43.8%(B16off)/29.6%(on) vs ocean 15-16% —— 但含占用比率假象成分;G 拓扑线关闭
- 合并 atomicAdd:四代(v1/v2.2/v2.3/v3WPM)全值级正确、全性能负 6-15% —— RED fire-and-forget 本廉,
  dup 是时间性分散非并发 —— **判词:负期望杠杆,是物理**;nsparse/spECK 无先例 = negative result 论文素材
- 压表 U 谷:single 点无解(1.2 悬崖/1.4 结构化);**三分法才是正解(用户钦定)**

## 三分法计划(用户已钦定)

```
重行(dense 候选)      → 送 dense:修争用常数 CONTENTION_BYTES_PER_DUP_MILLI 12B→实测~0-2B
                         (出处:四代 merge 物理;crankseg 型行裸账 dense 86KB<hash 98KB 本就该赢)
轻/中行(hash 原住民)  → 保大表 2.7×(F2 0.80/c-64 轻带 0.713 赢面根基,不压)
重行但非 dense 候选    → 行级 expand(est 平滑函数:重表行→1.4,轻表行→1.8;σ=1.04/√k 叙事)
                         评估 Robin Hood/双选择探测作压缩解锁器(治高 LF 方差)
```

## 执行队列(09-17 夜 用户钦定三阶段路线:纯 kernel → 新 kernel 覆盖 → 路由设计)

**① 纯 kernel 优化(dense/hash/cursor 各自内部做硬,不碰路由)**

- **第 0 刀画像已判(09-17 深夜,链假说证伪+真凶定位)**:
  - 链直方图仪器入库(HASH_PROBE_HIST,33 桶):默认表配 **avg probe/product 仅 0.07-0.15**(F2 0.116/crankseg 0.134/web-G 0.151/c-43 0.096/msc 0.072;91.5% 首发命中)——**链不是默认配下的延迟主项**,Robin Hood 优先级大降
  - ×1.22 表每行快 12-25% 的真凶 = **块梯耦合**:HASH_BLK_LADDER 表档硬绑块宽(bin3:64…bin8:1024)= 每行 warp 数 G,表跳档 = 行内并行度平方级变化;"占用对消"本质 = **行内并行 vs 行间并行的交换**
  - 块梯 wall 扫描:全256 crankseg +26.7/web-G +9.7 但 c-43 −8.8;L2(3-4:256) F2 −6.3/msc −10.4/crankseg −2.1 但 web-G +22;L3(仅 bin8 512) msc −13.4/F2 −4.5/crankseg −1.3
  - bin×flop 画像定根源:web-G = 91.6 万微行海(bin0-4,flop q50 1-266,要小块);F2 = 胖行(bin4-8,flop 2K-25K,要大块)→ **块宽最优由 flop 决定,表档只是相关非因果**
  - occgate12 值级 4/4 全绿收官(c-64 = 已知 artifact;停靠刀验证完整)
- **〔待定〕flop 感知块梯(09-17 深夜,污染窗口教训)**:crankseg/msc/c-43 稳态 ±1% 判无效应(基线稳定 61.6/3.5/2.6,可信);**F2 类未定**——两次"净窗"×5 给 −0.1% 与 −4.4% 互斥,后者经用户情报确认撞共租(xiatengfei 用卡,off 基线 10.6 vs 干净带 10.0);**净窗守卫已挂**(两卡 <2GB 连续 3 分钟自动跑 F2 ×7 定案,/tmp/ab_f2_ladlext_final.log)。F2 基线本身 10.0-10.7 漂移 = 10ms 级阵噪声地板 ±5%,效应若真也在 0-5% 单类收割带。
  - **物理判读(若 F2 复测归零则判死)**:线程预算恒等式——每行一切成本(乘积/清表/extract)随 B 重分配对消,唯一不对消项 warp 空转与同步皆负向 → 块宽只有"配对/配错"两态
  - 行级 −40~−60% 跳档效应保留为真(机理成立)但 wall 中性 = 不可收割
  - **尸检新红线:测量环境三态**(干净/共租在场/共租撤离余波);电池前后查卡 + off 基线对已知干净带(≤±3% 方可用);单发 ctime 屏幕 = 无效证据;10ms 级阵噪声地板 ±5%,判定需净窗 ×5+
  - ①阶段 hash 线剩余:ncu 停顿归因(每乘积成本构成→IPC/访存效率轴)/extract 融合(减总功,唯一能破恒等式的方向)/B16 逐行归因/口径 C 重拍 ocean
- dense/cursor:窗税 30μs/窗 clear+prefix 优化;exact-fit 窗/跳空窗(聚簇对齐);档位梯签名瘦身
- 实验口径纪律:**固定行种群 + kernel 内 clock64(HASH_ROWTIME/DD_ROWTIME)单变量**;wall 只作终验
**② 新 kernel 设计(覆盖无主行型)**:c-64 中带(span 50K/est 1-2K,1.775 输家群)/in-2004 超档轻行(est<2048∧span>14543 被地板挡)
**③ 路由设计**:军火库齐后;occgate 刀已停靠不翻默认(发现全入档,OCCGATE_COEFF env 留;coeff10 数字:crankseg −10 净窗/nd24k −24.7/c-64 +2.9)

0. **〔已判死〕旧第一刀"争用常数修正→重行流 dense"——行分布重放证伪**(commit 见 git log)
   - 重放三阵 hash 行(/tmp/rt_cur_*.txt,按默认门参数离线过 v4 判定):
     crankseg 51,312 hash 行 = occgate 挡 69%(est×16<span,行 span 中位 **58K** 近全宽)+
     超档 est 地板挡 30%(span>14543 ∧ est 1024-2047)+ **dup 门只挡 576 行(1.1%)**;
     c-64 7,214 hash 行 = 地板挡 96%(dup 中位 1.3,争用无关);F2 dup_mid 18% 行 span 70K,dense 裸账 570KB≫hash 73KB,挡对了
   - "dense 86KB<hash 98KB"裸账是**种群混用**:10620 是 dense 行的 span,hash 行 span 中位 58K → 实账 471KB≫98KB,字节模型自己也说留 hash
   - occgate 默认开 = 当年 17 阵电池 crankseg **−16%** 的实测来源;超档地板 = v28-v32 验证域。两门皆实证,**重行→dense 早已是现状**(crankseg 中位行 est1701/flop46719 本就过 dup_ok 第二支 flopq≥16K∧dup<64)
   - 残值:12B 常数在 SROUTE=4(默认 0,死路);四代 merge 物理结论(RED 本廉)不变,但它**不翻译成路由改动**
1. **收 sweep2 尾**:α110 前 8 阵干净落地(c-43/F2 被 15:43 二进制空窗打断,已重跑;组合档整体重跑中):
   - **α110(仅轻带 est≤4096:2.7×→1.98×,重带不动)**:crankseg **−6.3**(比 1.4 档的 −5.3 还好 → crankseg hash 行 est≤4096,红利在轻带压缩)/web-G −1.9/in-2004 −1.5/nd24k −1.6/c-64 −0.6;输 Zd +5.5/msc +1.9/Ga3 +1.2
   - **nd24k 对照**:1.4 档 −15.4 vs α110 档 −1.6 → nd24k 红利在重带(est>4096,1.8×→1.4×)
   - **两档互补 = 行级函数的实证形态**:轻带 ~2.0×(crankseg/web-G 赢)、重带 1.4×(nd24k 赢)、微行保 2.7×(c-43 pow2 吸附悬崖,待尾档确认)
   - REXP 默认锚点 1.8@256→1.4@4096 的插值恰逼近三带(2.7×/2.3-2.1×/1.4×),电池裁决
   - F2@1.98× 是否守得住 = 轻带端点定数(681 行@2.1× 的悬崖上沿),等尾档
2. **〔判死〕新第一刀:行级 expand——两模式全负,负结果入档(09-17 晚定谳)**
   - **模式 1(耦合,估计器乘 f(E))**:电池 crankseg −8.5/nd24k −12.6 但 in-2004 **+58.7**;机制解剖=est 全局耦合轮盘:crankseg 赢面实为 occgate 门槛迁移(est 4300→3440,×16<span → dense 候选被踢回 hash,hash 本是赢家),in-2004 灾难实为 dense_direct 人口 10,380→20,990 翻倍(DD_ROWTIME 实锤,dense_direct 相位 6.6→51.2ms)
   - **模式 2(解耦,est/路由恒等,仅表长目标折减)**:crankseg −0.1/nd24k +0.1(**纯压表收益≈0,全部大赢面确系路由轮盘**);in-2004 仍 +48.3 但换通道:accumulate 21→49.5ms = E>4096 怪物行表压至 0.78×est,est +30% 误差带 LF→0.85+,dup 25 行 15 万插入 ×4.5 链长——**重带压缩被插入流量物理否决**
   - **中带单独筛(解耦,E 256-4096 → 2.0×)**:crankseg −3.4(真收益但薄)/F2 +4.1/c-43 +13.7/in-2004 +52.5(重带又被牵连)→ 净负,停靠
   - **结构性教训(论文级)**:①表长不是瓶颈,U 谷无全局点的真因 = 各档差异实为门跨越;②est 是喂五类门(occgate/地板/direct/heavy 阈/bin 梯)的全局耦合变量,任何 est 侧改动 = 同时开五个轮盘;③真杠杆 = 路由门本身;④REXP=1 值级 4/4+有序 4/4 全绿(正确性无恙,纯性能问题),两模式代码在库默认关
   - **转向(已兑现)**:直接调门 OCCGATE_COEFF(env 已暴露,默认 16)——**×5×10 电池定谳:coeff12 = crankseg −7.4/nd24k −15.8/c-43 −6.5/web-G +0.1/Ga3 −0.2/F2 −0.1 vs in-2004 +4.5/c-64 +2.9/Zd +2.2/msc +1.0,geomean −2.1%,时间加权更大(nd24k 省 29ms)**;coeff8 单发(crankseg −10.3/nd24k −25.9 但 c-64 +8.6)——coeff10 电池在跑定边界
   - **coeff10 电池(×5×10)**:in-2004 **+0.3 归零**/nd24k **−24.7**/Ga3 −2.5/Zd −1.5/web-G −0.5/c-64 +1.5/F2 −0.3 vs msc +1.8/c-43 +1.5;crankseg off 侧异常(78.8 vs 各档 ~60,Δ−30.5% 存疑)但 ON 侧 54.8 跨档单调一致(8档53.3<10档54.8<12档55.9)→ 真实赢面 ~−9%,净窗×7 复核中。**当前倾向 coeff10 优于 12**(in-2004 归零+c-64 更好+nd24k 更深),待净窗与值级双确认
   - **LADDER=1 对照(5 阵)**:crankseg −8.1/nd24k **−21.8**(今日最大干净赢面)/in-2004 +2.6/F2 +0.4,但 **c-64 +1071% 灾难**——成本律对鲸鱼行(est 8K,过 occgate)算 t_hash(200K)<t_dense(454K) → 把本该 dense 的鲸鱼错误踢回 hash → hash 撞占用悬崖(表 8K+ 槽 → 1 块/SM)。**模型缺占用悬崖项**(表>4K 槽 t_hash 超线性;docs/67 "est 8K/占用 16%" 残差类正是此)——LADDER 修复 = 后续结构化工作,occgate 先行
   - **值级守门收官(09-17 深夜)**:coeff12 与 coeff10 **双双 4/4 绿**(17 位口径 v2 + 有序 4/4;c-64 = 已验证全零的 20504 显式零 artifact)。停靠刀记录完整:性能(coeff10: crankseg −10.0 净窗×7/nd24k −24.7/in-2004 +0.3/c-64 +2.9)+ 值级双验,③阶段任一系数一行启用
3. **Retry 2.0**(设计已侦察定稿,09-17):
   - **现解剖**:溢出触发 = 探测链绕满全表(probes≥ht_size,表 100% 满才开始丢);retry 定表 = pow2(2×flop) 且**下限 32768 槽全局表**(retry_prep_kernel:2418)——est 745/flop 13434 的行直接被推进 L2 全局表重路径(冷原子+大表 init),retry 1.33μs/行 = 主遍历 4-5 倍的真源
   - **改法(两级)**:① SMEM 快重试:表 = 原表 pow2 上探一档(2048→4096 槽=48KB 装得下),走同一个 hash_spa kernel 二次发射(est=新表),只在 SMEM 装不下时才落全局;② 仍溢出的极尾行再走现有全局 flop 路径(链式兜底,终止性不变)
   - 可选精修:水位判死(probes≥0.85×ht_size 提前触发,乘积零丢失,表内容可信可搬)——但 rehash 搬家复杂度高,先做两级定表(纯 host/定表改动,kernel 不动),已含大部分收益
   - 验收:F2@EXPAND=1.4 的 681 行 retry 段 1.5ms → <0.3ms;F2@1.2 的 6034 行 8.03ms → <2.5ms
   - Robin Hood 评估顺延至此刀之后(同为压缩解锁器;若两级重试已把轻带端点解开,RH 的边际价值再议)
4. **c-64 中带签名瘦身**(B16/WM/uo 参数收进模板或拆变体;中带 1.775→目标<1.2)
4. extract/排序融合(kernel 内 O(count²)→下游 radix;窗内排序税 5.9% 实测;口径 B 仪器在)
5. 补测:ocean occupancy 同口径(归因 L1 44% 的真伪)/ c-64 25.7 vs 18.4 残差(SMEM 5HT 嫌疑)
6. 全部落定后:系统级 vs ocean 重刷(docs/76 地图,全正确代码首次)+ 论文叙事整理

## 对拍基建(复用)

- ocean 格式矩阵:/tmp/oc_{F2,c-64,crankseg_2}.csr(转换:./ocean/convert <mtx> <csr> double)
- ocean 仪器化二进制:ocean/spgemm(env ROWCYCLES=path);配置 ocean/config/bench.json
- 我方逐行:HASH_ROWTIME=path;口径 A(对齐彼窗,不含排序):HASH_UO_XTRACT=1;块梯配平 HASH_BLK_LADDER="3:64,4:64,5:128,6:256,7:512,8:1024"
- 值级:make WRITE_MTX=1 构建 + truth17.mtx(cuSPARSE 小阵/scipy 大阵 17 位)+ scripts/compare_mtx_values.py + 行内有序扫
- ncu:/opt/nvidia/nsight-compute/2025.1.1,**必须 METHOD=adaptive**;sanitizer 用 /usr/local/cuda-12.8 版

## commit 地图(本阶段)

f3e6548 转正刀 / b7b9e5b 三雷 / 6861b3c 精度 / 67cea7d diter止血 / 04e0c04 diter根除 / 001a36b+8f04d30 study19+①收官 /
db21731 ④交织B入默认 / ff91e45+35ef7ba+c3aecd3 ⑤四代+判词 / ded46de study21 对比重刷 / c7203b1 规格79

## 创新点总账(论文定位,09-17 盘点)

### 已收割(库里的贡献,可写)
1. **连续估计驱动表配**(v38 est 连续化):行行 LF 均匀 0.37 → 短链,轻行赢 ocean 20% 的根基
2. **交织 B 布局**(④,db21731):col/val 两窄流→16B 单股,请求 −18%,系统 geomean −1.9%
3. **合并 atomicAdd 四代经济学**(negative result):H100 SMEM RED 本廉 + dup 时间性分散 = 负期望杠杆;nsparse/spECK 无先例 = 首个系统测明的设计空间分析
4. **read-first 机理自有化**(转正刀):对拍归因→自有形态,引 nsparse 公共先例

### 待收割(还有肉的,按性价比)
1. **Robin Hood / 双选择散列 = 压缩解锁器**(最有趣的):治高 LF 链长方差 → 敢把 LF 推 0.55-0.6 → 行级压表空间翻倍;ocean 没有;与②行级 expand 天生一对
2. **行级置信度余量模型**(σ=1.04/√k):sketch 状态定每行 expand——"estimation continuity → per-row confidence → adaptive margin" 叙事干净,估计器信息二次利用
3. ~~争用常数修正 + 三分法路由~~ **已证伪(09-17 重放)**:dup 门只挡 crankseg 1.1% 行,残余 hash 行是 span 58K 宽行(dense 付 5 倍);重行→dense 已是现状。物理结论(RED 本廉)保留为 merge negative result 的一部分,不再当路由杠杆
4. **extract/排序融合**:kernel 内 O(count²) → 下游 radix,窗内排序税 5.9% 实测

### 关闭线(测过,勿重开)
原子归并(四代判词)/ G 拓扑(L1 已 44% 含假象,基本关闭)/ int4 清表 / 素数梯 / PRIV / Lemire / subwarp8 / bsearch方案1

### 定位一句话
hash kernel 机理创新期结束,剩余空间在**kernel 与系统接缝处**(探测变体解锁压表/extract 融合/行级余量/路由常数);最大创新在体系层(dispatcher 生态位);护城河 = 自己测出的物理(RED 经济学/L1 口径/U 谷结构)——这些实测本身就是论文弹药。

## ①阶段补充判决(09-17 深夜二批)
- **F2 块梯定案(净窗守卫×7)**:on 侧三会话恒稳定 ~10.1ms,off 漂移 10.0-10.6;2/3 会话(含最干净)给 −4.4/−4.7% → **bin3-5@256 对 F2 类中等行带真实 0-5% 收获**;但 web-G 同形状 +22 → 全局翻默认不可,按阵选形=路由 → 停靠③阶段(env 留)
- **LLB 重开案判负(×5 净基线)**:crankseg +0.4/F2 +2.4/c-64 +1.9 → 动态 G 不是 crankseg 答案(当年 Ge99 判关 + 今日对比阵重测双重洗清;ocean 优势别处)
- **读 ocean 代码两收获**(AccumulatorHash.cuh):①localLoadBalance 逐行动态 G(我们 LLB 已移植已测=中性)②溢出=飞行中切全局表+合并零重扫(Retry 2.0 同构,他们已实现;人口少不解释 1.355)
- **crankseg 1.355 存活嫌疑仅剩三**:口径C 块宽混杂/ncu B拓扑停顿/两遍制(symbolic+numeric) vs 单遍制深层结构差

## ①阶段 ncu 停顿归因判决(09-17 深夜三批,crankseg 1.355 破案级)
- **方法**:我方 hash_spa 与 ocean hashNumeric 同指标 ncu 对照(crankseg;METHOD=adaptive;净卡)
- **B 拓扑嫌疑洗清**:每行 B 扇区几乎同(ocean 37.1K vs 我 38.4K);long_scoreboard 16-20% 两边相当
- **块宽混杂实锤现形**:crankseg 主力行群 ocean 给 1024 线程(素数表 5449 挂高档)vs 我方 512(bin7);按线程预算恒等式 wall 对消,非胜负手
- **真凶候选:每乘积指令数 我≈5.2 vs 彼≈4.1(+27%)**——指令=总功,不随占用对消,与 1.355 量级吻合
  - 嫌疑指令:①B16 解包(uint4+__hiloint2double;crankseg B16off 快 8% 早已对上!)②upper_tri 运行时分支(可模板化)③read-first 双查路径
  - ocean 副产物:SMEM 停顿 36%(LF 0.57 素数表链长)仍赢 → 停顿不是胜负手,总功才是;素数梯 673/1327/2719/5449 = 无空档表长(我方 pow2 梯 512↔1024 空档早有记录)
- **F2 侧判决**:barrier 停顿 21-28%(count-sort 排序税停顿化现身)vs crankseg 10% → 刀②(extract 两级排序)靶心已可视化
- **下一刀(新头号):内循环指令瘦身**——目标 5.2→4.5 inst/product 以下;A/B 仪:ncu smsp__inst_executed.sum/product;变体:B16 off/on × upper_tri 模板化 × 探测路径精简

## 指令经济性定谳(09-18 凌晨,人口公平修正版)
- **修正**:原"5.2 vs 4.1"系跨阵污染(我方数取自 F2);修正后人口公平(双方全 hash launch/同乘积总数):
  crankseg **我 3.66 vs 彼 3.23(+13%)**;F2 **我 5.33(含kernel内排序!) vs 彼 6.71(不含,彼排序独立kernel) = −21% 带 handicap**
- **双向定律:指令经济性预测胜负**——F2 我们瘦 21%(背排序)赢 20%;crankseg 胖 13% 输 35%
- SMEM 停顿机理定谳:探测依赖链(slot n+1 地址依赖 slot n 值,每跳整条 SMEM 延迟线程内不可重叠);彼 LF 0.57 链 2-3 跳 → 停顿 36% vs 我 21%,但 warp 并行藏延迟,用停顿换指令 = "停顿非胜负手总功才是"活例
- crankseg 特有 +0.4 inst/product 随 dup(27 vs 6.6)或大表缩放(逐乘积常数项已被 F2 反证排除);嫌疑短名单待 SASS diff/变体 A/B

## 口径 A+ 窗对齐仪器(09-18 凌晨,用户钦定修正)
- HASH_RT_NOEXTRACT=1:行计时截在探测循环尾(不含 extract/输出搬运)——对齐 ocean ROWCYCLES 窗
- 欠账背景:此前逐行对拍我方窗多付 extract(crankseg 行 5-15% 行时间),1.355 含此不对称;用户多次提醒对齐窗口,此仪器补齐
- 瘦身刀 v1 电池(共租窗但基线带内+重复性过):crankseg −0.7~−2% 小赢/其余平,无回退;值级 msc PASS,余 3 阵跑中

## 口径 A+ 对齐重拍定谳(09-18 凌晨,GPU1 净窗,历史性)
- 仪器:HASH_RT_NOEXTRACT=1+HASH_UO_XTRACT=1(窗=探测循环,无排序无 extract,对齐 ocean ROWCYCLES)+瘦身 v1
- **完整记分牌:F2 0.802→0.356(胜率 94.1%!) / c-64 1.117→0.906(翻盘赢!57.1%) / crankseg 1.355→1.107(41.8%)**
- 判读:①c-64 的输整个是窗口假象(我方窗多付 extract)②F2 悬殊优势现形③crankseg 剩 10.7% 且仍含块宽混杂(彼 1024 线程 vs 我 512,逐行延迟口径利彼)——真实代码差可能在个位数
- 系统级警示:对齐窗不含 extract ≠ extract 免费;系统赛(全管线)双方各自付自己的 epilogue,成绩另算;此表是【kernel 代码质量】的公平度量
- 教训升级:比较窗边界能凭空制造 1.5× 差距(1.571→1.107 中 42% 是幻影)——口径学本身是论文贡献素材

## hash kernel vs ocean 终局定谳(09-18 凌晨,口径 C 全对齐)
- **块宽对齐(bin7/8→1024)+口径A+:crankseg 中位 0.581,胜率 91.1%**
- **三阵全胜的同配置记分牌:F2 0.356(94.1%)/c-64 0.906(57.1%)/crankseg 0.581(91.1%)**
- 原 1.355 完整解剖 = 窗口假象(extract)+指令税(已修 −15%)+块宽混杂(彼 1024 vs 我 512)三层,无一层是代码质量
- 诚实注脚:①0.581=逐行代码质量口径;系统 wall 上我方 512 选择仍是对消最优(全1024 曾 +9.5%),两口径答不同问题②系统级残余差距(若有)在路由/估计器/管线=③阶段战场
- 值级:msc/crankseg PASS+有序;c-64 FAIL 复核=旧 artifact 形状(common 154,985,537),in-2004 比对中

- 瘦身刀 v1 值级收官(09-18):4/4 绿(口径v2;c-64 流式复验 20,504 条全零=既往 artifact 逐数一致)+有序 4/4;完整验证链=值级+指令3.66→3.10+wall无回退,代码已入默认(2b01232)
- ①阶段剩余:瘦身 v2(probe_hist 模板化/探测环检查重构,~0.2 inst/product)/extract 两级排序(规格81)/B16 逐行/系统级全管线 vs ocean 重刷(新记分牌下)

## 系统级战略表(09-18 凌晨,A级:GPU0 净窗 14MiB 前后,双方新二进制)
- **F2 9.84 vs 7.66(1.284)/ c-64 24.93 vs 20.57(1.212)/ crankseg 58.77 vs 27.29(2.153)**
- 终局战略图景:kernel 代码质量=我赢(同配置三阵全胜);系统吞吐=彼赢(1.21-2.15)
- **差距源定位:crankseg 2.15×=路由**(彼留重行于 hash 60K行/3.5G乘积,我送慢 dense/cursor = 停靠的 occgate coeff10 刀 crankseg −10% 正是此解!);**F2 1.28=管线**(全行 hash 双方,差距在 MinHash 两链 ~4.5ms/45% + compact+sort 税 vs 彼 HLL 管线)
- 逐行 clock64 与系统吞吐背离 = 占用对消恒等式再现(彼慢行高并发=快wall);两口径答不同问题,论文各占一节
- ②③阶段战场全部量化:extract融合+估计器提速(F2/c-64 线)/occgate+重行回流 hash(crankseg 线)
