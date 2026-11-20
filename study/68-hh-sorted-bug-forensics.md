# study/68:HH_SORTED 有序直写正确性破案 + Flex 数值取证(11-19)

> 触发:师兄 Thor 平台发现 bcsstk17 开 HH_SORTED 时输出 nnz 对/部分列号错,
> 428 全局关停该项;嘱我方查根因。Flex 数值问题("顺便")一并取证。
> 纪律:全程别名构建(spgemm_wmtx,WRITE_MTX 仅包 main.cu 的 dump 块,
> 内核 codegen 与认证件一致),未触碰 spgemm_test,未改生产代码。

## 一、一句话定谳

**Gv3v2 段内位移 rank 的 `__shfl_xor` 蝶式 5 轮不传播——每 lane 只与 5 个
固定伙伴(L^1/2/4/8/16)各比一次,而非全段 31 键两两比较。段内位次系统性
错误 → rank 碰撞(两条目同位互覆)+ 空洞(位置无人写)→ 输出行键集合损坏
(nnz 恰好不变,因为计数来自 Pass A 的 ballot,是对的)。**
 bins4/5/7/8/9 全部 HH 人口中招,与 bin 配置无关;"有序直写"机制从根上
不成立(即便 rank 对,输出序也是(段序,段内列序)≠ 全局列序)。

## 二、取证链(全部可复现)

1. **静态读码**(src/spgemm_kernel_hash.cu:4557-4603,hash_heavy_kernel):
   - Pass A 段计数(ballot)✓ 正确;seg_pre 独占前缀 ✓ 正确;
   - Pass B 段内 rank:`for off in {1,2,4,8,16}: other=shfl_xor(c,off);
     if(other<c) lt|=bit` —— c 每轮都是原值,蝶式不传播,**单值蝶式只能做
     归约(min/sum),数学上做不了 31 元素两两位次**。注释"5 轮广播同段键"
     的设想自始不成立。
2. **微复现器**(tools/hh_repro.cu,提取代码逐行拷贝+宿主播种):
   四档 bin 配置(HT_P=512/2048/4096/8192 × 128/256/512/1024T)
   **394-400/400 trial 失败**;诊断输出:kernel_r ∈ [0,5] ≈ 5 伙伴 popc 上限,
   seg_pre 全对 → 实锤段内 rank 错。
3. **生产实证**(H100 卡0,bcsstk17):
   | 配置 | L1 集合错行 | L2 乱序 | 结论 |
   |---|---|---|---|
   | 默认(HH_SORTED=1) | 9201/10974 | — | 污染;行1=38写入+109残零 |
   | HH_SORTED=0 | 0 | 0 | 干净 ✓ |
   | H45=H55=H69=0 | 0 | 0 | 干净(HH 全关) |
   | 只关 H69 | 9201 | — | 同污染(bins4/5 在阵上为主人口) |
   | COMPACT_FLAT=0 | 9201 | — | 与 FLAT 无关 |
   - F2(§六十五"收口"用阵):**23633 行集合错**;
   - c-64(bin7/8 人口):3614 行集合错+10248 零抵消软判,
     **保留项数值全对(max rel 2.1e-11)→ 丢项不错值**,与空洞机理吻合。
4. **认证链盲区审计**:
   - scripts/ab_pair.py 与 confirm 系判据 = 正则 `Result C:.*nnz = (\d+)`
     ——**只比 nnz**;§六十一承诺的"行内有序显式检查"在 §六十五收口时被
     "rank 数学保证"(本次证伪)+"15 阵三方 nnz + memcheck 0"替代;
     **WRITE_MTX 文件级检查"挂钩未走通挂 backlog"却照常转正** ——
     守门三件套之"真值级对比"在 Gv3v2 链上从未执行。四雷剧本第五例
     (nnz 对/检查器盲区/注释当验证)。
   - 全库 shfl_xor 审计:其余 19 处均为正确蝶式归约(`v+=shfl(v)` 型,
     值轮间传播);比较累积型仅此一处,病灶孤立。

## 三、影响面(待用户裁决的处置在 §五)

- **bins4-9 全部 HH 行人口输出不正确**(confirm16 默认形态:H45/H55/H69 开)。
- 记分 0.8054:计时真实,但**带病**(输出不正确的系统上测的);
  bcsstk17 单阵 HH_SORTED=1→0 = 2.62→3.54ms(+35%),聚合回退量待重测。
- Thor 处置(HH_SORTED=0 全局)**正确且必要**;Thor 的 1.19×/2.62×
  = 干净系统成绩,与我方 0.8054 不同口径(差一项优化)。
- 创新点①(study/64 v4.1)"有序直写"部分机制错误:修复+重验前不得引用;
  hash_heavy 本体(累积+无序提取+下游排序)与 MinHash/LLB/merge3 无恙。
- 师兄 Thor 报告的 bcsstk17 症状(nnz 对/列号错/关闭即好)全部复现并解释。

## 四、Flex 数值取证(师兄问题②)

1. **自检代码实锤**(external_sota/FlexSpGEMM/src/external/cusparse/
   spgemm_cusparse.h:441):值比对被注释 `//|| h_csrValC[j] != ...`,
   两处检查实例同病——"Column idx & values: PASSED"**只查列号,结构性
   查不出值错**(师兄判断在本地源码确认)。
2. **本地 H100 复现**(取证构建 = 恢复值比对的双精度 m16n16,sm_90,
   /usr/local/cuda-12.8 nvcc;生产树已还原):
   - arc130(general 阵,对称双计假说排除):tau=0 → **12583/15631=80.5%
     值错**;tau=1 → 30.7%;nnz/rowptr/列号全对;
   - 模式:值系统性偏大 1.07-3.4×,错误率随 τ 单调 → 与混合精度
     (fp16/TC 投射)路径覆盖度强相关;高 τ 仍残留 30% → 值累加路径
     本身亦有缺陷(不只是精度切换);
   - 师兄 Thor 案例(row22 col97:-9.7e4 vs 2.6e9)量级更极端,
     或含未初始化读,未深挖(其代码内部定位超出本轮"顺便"范围)。
3. **结论**:Flex 数值错误为其自身缺陷(非平台/非转换/非我方),
   其全部"自检通过"不可信。我方 448 账 Flex 对比为计时口径不受影响;
   论文引用 Flex 成绩时应加正确性脚注(与 TileSpGEMM OOB 读并列为
   该基准族正确性缺陷证据)。
4. 工具:/tmp/flex_forensic(取证件);构建钥匙=**必须用
   /usr/local/cuda-12.8/bin/nvcc**(PATH 里的 12.0 EDG+gcc12 头是坑)。

## 五、修复与处置选项(等裁决,未动手)

- **A 修 rank(治本)**:段内 32 键落 smem 全 lane 扫描计数(或奇偶置换
  排序;或复用 hash.cu:1125-1147 现成列位图前缀 rank)。修后还需正视
  全局序问题——段前缀是表序,直写行序 ≠ 列序,要么段定义换成列域分段
  (counting-sort 语义),要么退回下游排序;
- **B 短期止血**:HH_SORTED 默认翻 0(逃生门在),428 重认证出干净记分
  (预计 GM 回退:bin9 79 阵受益 + b45 有序加速部分蒸发);
- **C 认证协议补洞**(无论如何都做):scripts/verify_sorted_truth.py
  (L1 集合/L2 序/L3 数值三层+零抵消软判)入守门三件套;
  WRITE_MTX 管线挂钩修复(§六十五挂账项);confirm 判据从 nnz 升级
  为值级抽查。

## 六、Gv4 修复(11-19 晚,用户开工令"把这个优化复活")

**设计:机制替换,目标保留**——槽域蝶式 rank → **列域位图前缀 rank**:
位图 bit=列(原点=docs/24 跨度窗 span_lo),位次=组前缀+组内词 popc+词内低位
popc;表内键无重(CAS 去重)→ 位图位无重 → 位次是 [0,k) 双射,构造性无碰撞
无空洞。目标不变:免 counting-sort 载体趟,compact 回落 copy 级。

**实现**(src/spgemm_kernel_hash.cu):
- hash_heavy_kernel 模板加 BM_WORDS(位图词数);入口 span 门(span_len>
  BM_COLS → 注册 ovf 交 retry,零双算;unsigned 比较兜空行);提取段整体
  替换(清位图→置位 atomicOr→组计数→组独占前缀→位次直写;4 栅栏 3 走表);
- 发射侧 BM_WORDS 图:bins4-7=2048(64K 列)/bin8=4096(128K)/bin9=8192
  (256K;195+32KB=227.8KB≤232448B 极限压线,单块/SM 本就如此;bin4
  6+8=14KB→16 块=线程帽不降);
- **bin 级门普查**(hh_gate_census_kernel):门控行 Σflop>64M 帽则整 bin 退
  hash_spa LHT——retry 按 flop 分配槽位(≈12B/flop),人口级门控行打爆显存
  (Hook_1498 实证 9.6G flop→115GB cudaMalloc OOM);普查与内核门同数据同判据;
- **accumulate/compact 契约**:ctx 出参 hh_used + 函数层 h_hh_used[bin],
  compact 侧 bins4/5/7/8/9 六处判据改"env 开 ∧ 该 bin 实际走了 HH"——
  修普查翻 bin 后 compact 仍直拷的次生雷(F2 一度 9177 行乱序,实测抓获修复)。

**验证链**:
1. 微复现器 tools/hh_repro2.cu(五档配置 1925 用例含边界/聚簇/近满表/门界)
   ALL-GREEN——判据比 v1 严:输出必须恰为升序键序列;
2. 生产值级(verify_sorted_truth.py 三判):
   | 阵 | 路径 | L1 集 | L2 序 | 判定 |
   |---|---|---|---|---|
   | bcsstk17 | bins4 位图 | 0 | 0 | ✓(L3=抵消噪声,同二进制自对拍同签名证实) |
   | F2 | bin7 普查退 LHT | 0 | 0 | ✓(L3=47 行抵消噪声,带病时代同量级) |
   | c-64 | bin7/8 位图 | 0 | 0 | PASS(max rel 2.1e-11) |
   | Stanford_Berkeley | 门控行→ovf/retry | 0 | 0 | **PASS(max rel=0.000e+00,78M 项逐位精确)** |
   | Hook_1498 | bins4/5 普查退 LHT | 0 | 0 | ✓(128187 行 benign=结构零口径) |
3. **非确定性噪声定谳**:同二进制两次运行自对拍 bcsstk17 = 2856 行 1e-9 级
   差/最大 1.684——atomicAdd 浮点序 × 灾难抵消(值域 1e14 阵输出 1e-6),
   与 hhfix-vs-hh0 对拍(2865 行/1.684)同签名 → L3"数值差"非缺陷;
4. **Hook nnz 口径定谳**:truth(scipy)304.5M vs 我方 312.4M(=认证账原值)——
   小例证 scipy 对精确抵消零剪枝(A@A=[[2,0],[0,2]] 返回 nnz=2),我方/ocean
   保留结构零;多出 7.9M 全为 |v|≈0 项,benign 软判正确归类。非缺陷,账面口径。
5. 性能:bcsstk17 2.55ms(带病版 2.62/HH=0 版 3.54)——正确形态收回大部分
   收益;F2 44.6ms;Hook 284ms(普查退 LHT 形态);16 阵×3 电池+428 重认证
   待跑(见 §七)。

## 七、Gv4.1 路由调整(11-20 凌晨,用户"是不是得调整一下路由")

**三轮电池的定谳链**:
| 电池 | B 臂 | GM | 关键 |
|---|---|---|---|
| v1(hhfix) | 带病认证件 A | 1.4418 红 | retry 风暴家族(web-Google 5.98×) |
| v2(hhfix2:+5%比例判据+span界定清零) | 带病 A | 1.1923 红 | 风暴塌(web-Google→1.20),但翻 bin 家族仍 1.1-1.9 |
| v3(hhfix3:+翻bin回FLAT) | **正确基线 A(HH_SORTED=0)** | ~1.04(13/16 阵) | **换正确基线**(带病版速度=错序偷来,正确性修复刀对它不可能绿);F2 0.770 赢;翻 bin 家族 1.11-1.23 |

**v3 暴露的路由错**:普查翻 bin 后退到"**LHT 累积+FLAT**",而正确基线同 bin 形态是
"**HH 累积(无序)+载体**"(HH_SORTED 只关门提取,不关累积——基线的重行累积优势
还在,web-Google/ldoor/TSOPF/pre2/Hook 全是这条税)。

**Gv4.1 修法(机制性,非调参)**:普查翻 bin → **不退 LHT**,改发
`hash_heavy_kernel` 带 `force_legacy=1`:legacy 无序提取(内核 else 分支现成)+
载体 compact,发射免位图 smem(几何与基线逐位一致)——**翻 bin 代价构造性归零**,
翻与不翻只在"提取相要不要位图"上分岔。compact 侧 FLAT 旁路回退(legacy 行无序,
必须载体)。hh_used/hh_flip 契约不变。

预期终账:翻 bin 家族(web-Google/ldoor/TSOPF/pre2/Hook)→ ~1.00;F2 0.770、
bcsstk17 0.965 保留;kept-bin 大阵(cage15/Bump)待 v3 收尾读数;GM ≈ 0.99-1.00。

## 八-2、终版电池定谳(11-20 午,Gv4.1=hhfix4 vs 正确基线 HH_SORTED=0)

**GM=1.0094(绿,≤2% 带内),nnz 0 异常**。逐阵:翻 bin 家族全部构造性归零
(ldoor 1.915→1.018/pre2 1.229→1.015/Hook 1.220→1.034/web-Google 1.110→1.033/
cage15 1.228→1.044/Bump 1.147→1.037);F2 **0.761**/arc130 0.946/1138_bus 0.895 赢;
平带 case9/rgg/bloweybq/StocF ≈1.00。**两项豁免记录**(commit 门透明账):
- bcsstk17 电池 1.186 → **10× 交错定谳 0.984**(3 轮中位抓到热瞬态;微阵纪律兑现);
- TSOPF 1.216 → 诊断=**绝对值 1.15ms**(dense 阵:dense_direct 269ms 主导,hash
  路径基数仅 5.3ms;机理=census 同步税)→ 已知代价,Gv4.2(普查上提到 h_cnt 读回点,
  消 per-bin sync)对症排队;
- cage15 一臂瞬态 FAIL(600s 超时)→ 独立双跑全过(nnz 929023247 ✓×2)。
- 事故记录:WRITE_MTX 转储写满 /home(101GB,results/matrices/)致 ENOSPC,清理后
  101G 空闲——后续带 WRITE_MTX 的批量跑须转储定向到 /(root 盘)。

## 九、confirm17 官方账(11-20 晚,commit 28ea718 形态,428+ocean 同卡配对)

**GM(我方/ocean)= 0.9289(n=427,赢面 206/426≈48.4%),零 TIMEOUT/NOPROF。**
带病 confirm18 同共同集 = 0.8137/赢面 69.8% → **Δ0.115 = 还掉的假赢面**。
对照:confirm17hh0(HH=0 消融,ocean 复用)在跑——判"组件 vs census 税"的终审。
退化解剖:大阵=HH 人口还债(ldoor 电池 HH0=39.1≈confirm17=40.2 双证);
**小阵翻倍族(FEM_3D_thermal2 2.1→5.5/pkustk10 2.5→5.2/offshore 3.0→4.7)= 嫌疑
门普查流同步税(每 HH bin 一次 sync,小阵 compute 2-5ms 被吃 20-100%;16 阵
电池全 HH 重阵,稀释了此税=抽样偏差)**——若 hh0 批证实,Gv4.2(普查上提至
h_cnt 读回,零新增同步)当天可修。恢复路线:Gv4.2 census 税 → Gv4.3 BM_WORDS
旋钮(bins4/5 64K→128K 列,ldoor/nlpkkt/hangGlider"刚过门"人口)→ Gv5 长.span
位图(SB/c-big 类)。0.8054 正式退役,新底座=0.9289。

## 九-2、Gv4.2 正式账(11-20 晚,commit 7662d4c,confirm18)

**GM=0.9251,赢面 212/427(49.6%),ocean 列同日同卡复用 confirm17。**
四行总账:带病 0.8137(退役)→ HH=0 消融 0.9278 → Gv4.1 0.9289 → **Gv4.2 0.9251(现役)**。
Gv4.2 净进 0.0038(census 税的账面权重=0.4%,小于受税带逐阵比值的直观外推——
校准教训:逐阵大比值≠大账面,人口权重决定)。位图赢人口(cfd2/s3d 族 34-42%)
与 FEM_3D_thermal2 类 kept-bin 交换税并存,下一档优化=Gv4.3(BM_WORDS 旋钮,
bins4/5 门 64K→128K 接"刚过门"人口)+普查性能化(位图 vs 载体预测入判据)。

## 八、待办(修复收口)

- [ ] v3 电池收尾(cage15/Bump)→ hhfix4 值级快验(legacy-flip 四代表)→ 终版电池
      (A=正确基线 vs B=hhfix4)→ 绿则 commit 五件套;
- [ ] 428 confirm 重认证(新记分);
- [ ] 值级抽查入 confirm 判据(协议补洞,C 项);
- [ ] 通知师兄:Thor 可重开 HH_SORTED(需同步 hhfix4 构建);
- [ ] 残留:span>256K 大阵人口(Hook/cage15 类,位图物理装不下)→ Gv5 候选
      (全局内存位图/两级位图);普查帽与 BM_WORDS 为电池可调旋钮。

资产:scripts/verify_sorted_truth.py(L1/L2/L3 三判+零抵消软判)、
tools/hh_repro.cu(v2 蝶式病灶存档)/hh_repro2.cu(v4 位图验证)、
spgemm_wmtx/hhfix 系别名构建、/tmp 各 dump(根分区,重启即清)、
ab_pair.py 新增 A 臂专属 env 参数(argv[6],正确性修复刀的基线对照)。
GPU 全程卡0;sanitizer 不可用(注入库缺失),以微复现器替代定位。

---
*版本:11-20 v3(Gv4.1 路由调整+三轮电池定谳)。关联:study/57 §五十-六十五、
study/64(创新点①)、study/65(红线)、correctness-first 铁律。*

## 十、测量分级纪律(11-20 晚,用户钦定"做成纪律")

两案底:bcsstk17 电池 3 轮中位 1.186"回归"→10× 交错定谳 0.984;bcspwr01 实为
n=39 的 0.16ms 微阵,confirm 单发 1.47 vs bcspwr02 0.82 被误作"自然实验"。
**规则**:中位 <1ms=微级(不作单阵裁决,只进 GM 聚合);1-10ms=中级(≥5×交错);
≥10ms=稳级(3×中位);交错发射硬要求。已入工具:ab_pair 判读自动分级
(commit 0c1671a);待落地:confirm 系脚本微阵段 10× 中位。K7 三探教训并入:
矩阵级粗特征(span/a_len/dup)分不开 K7 输赢人口,真判据=逐行竞争力对比
(K7 vs 现任路径),bcspwr04 赢=从慢路径抢行/bcsstk13 输=抢走 warp 表甜点行。

## 九-3、K7 全账终案(11-20 晚,T5c)

**K7=1 全账 GM=0.9389(n=427,ocean 复用)vs 现役 0.9251 → 净负 1.5%,三度批级
证伪,K7 维持 opt-in 归档。**分级洗除:微带 94 阵(单发不作裁决)洗掉后可裁决段
仍 1.0163——噪声解释不了输面。真赢带 8 阵(thermal2 0.77/vsp_model1 0.88/F2
0.88/c-66b 0.89/c-67b 0.91/circuit_4 0.93/vsp_south31 0.93/pattern1 0.93,
全收仅 +0.22% GM);真输带 32 阵(exdata_1 3.25/bloweya 1.55/TSC_OPF 1.48/
rajat 族)。**结构性定谳**(瓶颈地图佐证):位图家族猎场=高密度行,而高密度行
大头住 dense_direct(多为赢阵);输阵 accumulate 大头=宽 span 低密度,位图物理
进不去——K7 非门没找对,是猎物不在。三次战役共同教训=**资格门≠竞争力门**,
此原则入新叙事 step④ 指派表设计。另案底:12 阵探针曾用 TOTAL 口径混 compute-only
(case9 假赢),测量纪律再+1。
