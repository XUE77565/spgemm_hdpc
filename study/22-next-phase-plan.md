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

## 执行队列(09-17 下午修订:第一刀证伪,行级 expand 升位)

0. **〔已判死〕旧第一刀"争用常数修正→重行流 dense"——行分布重放证伪**(commit 见 git log)
   - 重放三阵 hash 行(/tmp/rt_cur_*.txt,按默认门参数离线过 v4 判定):
     crankseg 51,312 hash 行 = occgate 挡 69%(est×16<span,行 span 中位 **58K** 近全宽)+
     超档 est 地板挡 30%(span>14543 ∧ est 1024-2047)+ **dup 门只挡 576 行(1.1%)**;
     c-64 7,214 hash 行 = 地板挡 96%(dup 中位 1.3,争用无关);F2 dup_mid 18% 行 span 70K,dense 裸账 570KB≫hash 73KB,挡对了
   - "dense 86KB<hash 98KB"裸账是**种群混用**:10620 是 dense 行的 span,hash 行 span 中位 58K → 实账 471KB≫98KB,字节模型自己也说留 hash
   - occgate 默认开 = 当年 17 阵电池 crankseg **−16%** 的实测来源;超档地板 = v28-v32 验证域。两门皆实证,**重行→dense 早已是现状**(crankseg 中位行 est1701/flop46719 本就过 dup_ok 第二支 flopq≥16K∧dup<64)
   - 残值:12B 常数在 SROUTE=4(默认 0,死路);四代 merge 物理结论(RED 本廉)不变,但它**不翻译成路由改动**
1. **收 sweep2 尾**:读 /tmp/ab2_HASH_TGT_ALPHA_110.log + /tmp/ab2_HASH_EXPAND_1.3_HASH_TGT_ALPHA_120.log(α110 约 16:00/组合约 17:30 落)
   (前两档已判:1.4 = nd24k−15.4/Zd−13.6/Ga3−9.4/crankseg−5.3/in-2004−4.8 赢,c-43+42.4/F2+17.2 输,geomean+1.6% 不翻默认;1.2 悬崖死)
   有效表倍数换算:expand×α/100,默认 1.8×1.5=2.7×;1.4 档=2.1×;α110=1.98×;组合=1.56×
2. **新第一刀:行级 expand**(原第二刀升位;三分法可执行余部 = "hash 分级表配")
   - 植入点:估计器 kernel 的 est_expand 标量(spgemm_kernel_hash.cu:3428/3452 调用,est 逐行相乘处)→ 改 est 的函数
   - 形态(等 sweep2 尾定系数):重 est 行→~1.4,轻 est 行→1.8(=2.1×/2.7× 有效),log 线性平滑;
     叙事 = σ=1.04/√m_eff(轻行触及寄存器少→误差大→余量大;重行 σ 收敛→敢压)
   - 守门:值级 17 位(msc/crankseg/c-64/in-2004)+ 有序 + ×5 电池 10 阵 + **retry/ovf 行数对比**(压重行表的代价先看溢出)
   - Robin Hood 评估(若 LF 0.55+ 方差爆,上 Robin Hood 再压更深)
3. **c-64 中带签名瘦身**(B16/WM/uo 参数收进模板或拆变体;中带 1.775→目标<1.2)
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
