# study/60 — A 线 v2:行签名路由体系战役书(10-29 凌晨定稿)

> 承接:study/59(输面攻坚 + b45 收官,confirm16 = 0.8054)+ 当夜架构讨论(用户-师兄:
> 行压缩特征做路由)。**核心命题:行签名一个表示,贯通测量(标定台人口轴)、
> 路由(三级决策)、自适应(决策表漂移更新)三层。**
> 病根动机:现 binning 单特征(est)——sparsine 10× 病理行对 est-binning 不可见;
> multirow 三修三败 = 一维决策切多维人口。

## 架构(定稿)
```
行签名(4-8B/行,现成量打包):[log2 est | log2 max_B_len | log2 dup | log2 span]
L1 方法级 dispatcher(现有) → L2 多维签名 binning(新,替单维 est bin)
→ L3 核内自路由(签名/免费寄存器特征)→ ⊕编译层档(PTX 梯,已有)
菜单(代码,人注册)与决策表(模型生成,自动重出)分离;
新 kernel = 注册 + 标定台粗扫 + 表自动重出 + confirm 验证(无"猜阈值"步)。
```

## 切片序(按依赖)
**S0|sparsine 病理 × 签名分维验证(先行,一个实验两份产出)**:
nsys/bins/逐行(HASH_ROWTIME)定 10× 熔断机理 + 回答"哪维签名(log2 max_B_len/
dup/span)把熔断行与正常行分开"——签名概念的第一个实证或证伪。
**S1|签名基建(1 天)**:count_flop 阶段打包 4 字节签名(零额外 kernel,并读现量)
→ d_signature 数组;多维 binning 原型(先二维 est×maxbl,compute_bucket 扩维);
dbg 签名直方图(人口画像工具)。
**S2|标定台签名空间化(1 天)**:calib_station 改造——变体 × 签名格交叉面
(变体:batched/multirow 各档/LHT 梯/HH 档;签名格:粗分箱防组合爆炸);
产出=每变体赢面签名带(带置信)。
**S3|multirow v2(1-2 天)**:判决按 S2——A.同质发射路(多维 binning 分层,
launch 内零分化,首选);B.核内自路由路(签名查表 if);sanitizer+三件套+
三阵 A/B+15 阵电池(全人口门纪律:S2 签名带=S0 教训的制度化)。
**S4|决策表生成 + 在线层(1-2 天)**:离线=物理项+学习残差拟合 → 表(生成头文件/
symbol 热更新);在线=bandit 微调(生产运行=免费测量);**漂移演示**=故意换核
→ 表自动更新 → 恢复(论文的机制图实验)。
**S5|手门审计**:全码标量门过堂,逐个换签名交叉或标工程不变量。
**S6|四臂对照 + 500 confirm(里程碑)**:手门(今晚失败组)/静态交叉/离线 ML
(Flex 同款 LightGBM)/在线+行级——428 主账 + 72 held-out(d978488 规则已封)
两列;**四臂赢=论文主张兑付日**。

## 平行线
C2 基线(HSMU 跑中 247/428;TileSpGEMM/bhSparse/MMSpGEMM 随后;spECK/AC/nsparse 补)
| C3 第二卡 | C6 论文骨架(设计章底稿=本夜架构讨论,可先写)。

## 查新定位(预注册)
sketch 做定容(ocean MinHash/HLL)=已有,必引;DB sketch-guided execution 思想先例
必引;**行签名→多维分桶→kernel 族路由 + 决策表自动生成 + 漂移自适应 = 组合空位**。
IA-SpGEMM/Wang2025(逐阵选)/FlexSpGEMM(ML 选 tile+TC)对表;我们=粒度(行)
+判据(物理+学习,可解释)+维度(编译层)+漂移处理,四点正交。

## 纪律(继承+新增)
全套继承(GPU 保护/同会话 A-B/交错重采/三件套/三检门/清场按 PID);
**新增:一切新门=签名交叉(S2 产出),禁手拍;默认翻前人口带验证;电池绿≠428 绿;
四臂对照前禁宣称"解决根本问题"(期票纪律)。**

## S0 定谳(10-29 午):sparsine"12× 病理"= 幻影
**证据链**:①sparsine 逐行特征(py 直读 mtx)——全行 flop≥581/kcnt≥21/maxbl≥46,
bin0 仅 **1 行**(`0:1`);②multirow 以 n=1 发射,grid=(1/2+7)/8=**0 非法**;
③nsys 对拍 on/off:GPU kernel 层 **on≈off(2.4ms)**,multirow 根本不在表;
④API 摘要:两配置**共有** 2×~2.3s cudaMallocHost(系统级 pinned 卡顿,疑外来
RL 作业内存压力;独立观察项——每进程 4.5s!)。**结论:三修三败之第三败=边界
bug+host 噪声合成物,multirow 人口病理证据链重审**;第一二败(128/104 阵)真实,
但机理归因(B行长/表容)未证。**S0 签名验证数据源改向:confirm17 首跑 CSV 的
真实恶化 128 阵**(名单+数据在档)。grid ceil 已修+CHECK 待补。

## S0 签名验证(10-29 午后):第一维实锤 = est 量级
**数据**:confirm17 首跑 105 恶化阵(1.05-1.49×)取样 16 阵 × py 直算特征 vs 改善组:
| 组 | bin0 行中位 kcnt | 中位 maxbl | **中位 flop** |
|---|---|---|---|
| WORSE(16 阵) | 3.5 | 8.0 | **~15**(bcsstm 族=1!) |
| BETTER(atmosmodj 类) | ~7 | ~7 | **~49** |
**定谳**:①分维=**est/flop 量级**(交叉带 ~15-49 之间待 S1 合成定标);②**B 行长假说死**
(两组 maxbl 同 ≈8——昨晚第一败的归因也错);③bcsstm 族 flop=1 的空壳行是重灾区
(浮点开销占比型)。**S1 起手=合成格网扫(kcnt×blen×flop)钉死交叉点**,
multirow 门 = est 带门(签名交叉,非手拍)。

## S1 首轮(10-29 午后):格网部分成功 + 归因疑云升级
**格网**(19 格双正则阵,kcnt×blen,flop/行精确可控):①est_expand 1.15 把
flop≈64 的行推离 bin0(仅 8x8 巧合全落);②**8x8(est≈55,100k 行)干净格:
multirow −13% 赢** ✓;③flop≤8 的行走特殊 bin(19-21,BATCH_S/ESC)——
**multirow 根本不碰**;④我的格网解析粗糙(多行 bins 输出截断),部分格数据无效。
**归因疑云升级**:S0 取样的 16 个恶化阵 bin0 行 flop 1-36——flop≈15 的行为什么
在特殊 bin/bin0 边界?**confirm17 首跑 105 恶化阵里,可能相当部分根本不是
multirow 造成**(bcsstm 空壳行 flop=1 走特殊 bin,multirow 不接触)——
需逐阵 bin0 人口审计重归因。**S1 修正路线**:①105 恶化阵逐阵 bins 审计
(bin0 人口>0 才可能是 multirow 责任);②格网重跑(修解析,est 直方图 DBG);
③8x8 型高 est 格加密(est 30-55 加细)。
