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
