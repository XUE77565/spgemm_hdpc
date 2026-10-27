# study/58 — 编译层红利兑现 + hash 核复兴线战役计划(v2,10-27 定稿)

> 承接 study/57 §五十三交接清单 + 上一窗口估值定谳。
> **破案:b45 差距全源 = compute_61→sm_90 编译路径(§四十四)**;E1 翻案确立
> **"死刑是编译条件性的"**——且按上一窗口警告:**不只活了两个,study/57 全部
> 旧判决(15-19 刀)理论上都可能翻案,理论地位整体降级为"sm_90 条件下的
> 局部最优"("肥肉即保护"的承重结构是编译器搭的,换编译路径承重墙位置就变),
> 至今没有系统性重审**。本计划 = 起手三步 + 系统性重审程序 + 新优化 + 守门 + 时间线。
> 记分链未动:union 428 GM **0.8873**(confirm14,10-23)。

## 〇、银行与价值账(开工前对账)

| 项 | 状态 | 量级 | 入账通道 |
|---|---|---|---|
| 刀 F(compute_61 gencode,纯 flags) | 已验待入账 | hash 核 compute +9%(Bump acc 78.6→71.7) | confirm15 |
| 刀 A(HASH_LHT23 bins2-3 常量折叠) | 已验待入账 | 九阵 −1.2~−4.4% 全绿 | confirm15 |
| 刀 G(hash_heavy 接管 bins4/5) | acc 穿天花板(−16~−31%)但载体税倒挂,四阵净亏 | 生死系于 Gv3v2 | Gv3v2 存活才入束 |
| E1 复活(hash_heavy bin4) | Bump 65.9=天花板85% / Hook 90% | 刀 G 的核基础 | 同刀 G |
| E6/E7 移植核 | 天花板仪器 −13~−17%(Bump 61.6/cage15 69.3);"互植不转移"旧判已废 | 永不转正(用户钦定 09-15/§四十七) | 无 |
| 旧判决 15-19 刀 | **全部待系统性重审**(env 门基本都在,便宜) | 未知 | 重审幸存者 → confirm16 |

**价值定位(上一窗口)**:这波发现不是一把刀,是**换了地基**——一个死刑翻案
(E1)+ 一个全新武器族(编译层选择,PTX 梯已扫:61 孤峰选型关闭)+ 全部旧判决
待重审。目前只兑付了刀 F 的 bench 验证,**记分一分未动**。完整兑现 =
**Gv3v2(载体悖论正解)+ per-file 落地,再捆绑 confirm**。

工具资产:`spgemm_test_sm90`/`spgemm_test_c61` 双二进制 + `tools/{realdata_bench,
theirflags_bench,calib_station,replay_bench}` 四测量台。GPU 双卡空闲健康(Xid 清)。

工作树卫生:src/spgemm_kernel_hash.cu 未提交改动 = E3 修复(8× WM_T=0 opt-in)+
Gv3v1 判死代码(HH_SORTED 块,env 默认关)。**P1 动刀前先 commit 此态作基线**;
两个未跟踪二进制 gitignore 化并档案注明。

---

## 一、起手三步(§五十三按序,三夜主干)

### P1. per-file 三层实施(刀 F 的生产工程形态)— 夜 1

**问题**:整二进制 c61 有 TSOPF 回归(+4.2%,定位=hash_spa 小表实例 accumulate
+30%,与 bin4/5 大表反向 = c61 分裂 codegen)。目标=只让 hash 大表实例吃 c61
红利,其余全保 sm_90。

**三层**:①大表(hash_spa/hash_heavy 的 bins≥4 实例)→ compute_61;②小表实例
(bins0-3)→ sm_90 原生;③其余八源文件全 sm_90;门控层运行时按 bin 分派,
env 逃生门 `HASH_C61=0` 全回 sm90。

**实施(先试 A)**:
- **A. 同源双 TU 宏分离**:`spgemm_kernel_hash.cu` 编两遍——`-DHASH_TB=61`
  (`-gencode=arch=compute_61,code=sm_90`)只实例化 bins≥4 大表 launcher,
  `-DHASH_TB=90` 只实例化 bins≤3 小表+通用路径;入口宏改名
  (`hash_product_c61`/`hash_product_sm90`),dispatcher 留一个 TU 调两入口;
- **B. 物理拆文件**(A 符号侵入过大时):大表实例块抽
  `spgemm_kernel_hash_c61.cu` + 私有共享头(分支上下文要复制,弃平价,非首选)。

**工程雷区(预标)**:`__device__` 全局(g_xbit_d/g_hh_sorted_d)只许一个 TU
定义——c61 TU 开关量改走 kernel 参数,**避免 -rdc=true**;`cudaMemcpyToSymbol`
符号归属唯一;opt-in 实例块照抄 WM_T=0 修复形态(§四十六 tick2 雷不再踩);
**纯 gencode 不带 fast_math**(tick1:贡献=0,零 est 语义风险)。

**验收(Phase 0 同协议,15 阵 × 交错 2 轮 + nnz 全同)**:TSOPF accumulate 回
4.26ms 级;Bump/Hook 大表 +9% 保持(F2 −3.1% 保持);merge3/first100 零回归;
垫片税≈0 复验;值门三件套(nnz 逐位/行内有序/memcheck 0 错);附带 loop 遗留项
**bins0-3 行占比统计**入档(小表人口边界)。

### P2. Gv3v2:hash_heavy 有序直写(ballot+popc 位移 rank + cap 守卫)— 夜 2 主攻

**v1 死因复盘(§五十)**:①段指派 bug——`s=tid*32` 步进,HT_P=2048/256T 时仅
tid<64 进环(1/4 线程);②逐槽 O(32) 串行表内 rank;③cap2=INT_MAX 无守卫
→ 16 分钟病理卡死。

**v2 四件**:①连续段认领(段=warp序+k×nwarp,每段 32 槽,全体 warp 进环);
②段内位移 rank(lane 持键 c,5 轮 `__shfl_xor` 广播同段键,累"小于我"掩码,
rank=popc——全并行,禁逐槽比较);③跨段前缀(每段有效计数入 smem,单 warp
扫前缀,段序=表序保行内有序;总数写 row_nnz);④cap 守卫(rank ≥ 行 est 槽宽
→ overflow 走既有 retry 语义,禁裸写)。表内键天然无重,rank 直写=行内有序,
下游免排序载体趟。

**判据(预注册,同会话同卡 A/B,Bump/Hook/Serena/cage15 四阵,相位账列表核算)**:
- B 臂 compact 回落 ≤11~17ms 基线级 → Σ(acc ~60-67 + compact ~11-17)< 基线
  83.5 → **刀 G 复活**,hash_heavy 接管 bins4/5;
- 不回落 → 刀 G 整体判死(不接管 bins4/5),只留 bin 拆分小刀(bin4 刀G/
  bin5 留 LHT)作最后小收;
- 纪律:compute-sanitizer 先行;病理卡死即杀;值门三件套+空行/满表/超帽覆盖。

### P3. 捆绑 confirm15_428(银行入账通道)— 夜 3

- **前置**:P1 验收绿;刀 A 默认翻(HASH_LHT23 0→1);P2 存活则 HASH_H45/H55
  默认翻;否则仅 F+A。**bank first:Gv3v2/重审批不阻塞 F+A 入账**;
- `scripts/confirm_adpt15_428.py` 依 confirm13 模板:428 阵(337 ocean ∪
  first100)+ **ocean 同夜同卡配对** + compute-only 口径(WARMUP_N=3);
- 预注册:nnz 逐位全 428;hash 核 +9% 兑现到 TOTAL(警惕 d2h 稀释——Bump d2h
  410ms 占 83%,计分口径见真章);逐阵尾审计(聚合 GM 掩盖逐阵尾,FLAT 冤案);
- 记分:GM 0.8873 起算,新值盖章入总账(paper/hash-kernel-wins/01)。

---

## 二、系统性旧判决重审程序(上一窗口警告的兑现,新增主段)

**原则**:重审不是逐刀重写论文,是一次设计好的批量实验——**载体=P1 产出的
per-file 生产形态构建**(大表 c61/小表 sm90:生产相关的 codegen,不是整二进制
c61 台架)。按"死因机制"分三档,先验=死因离 codegen 越近翻案概率越高。

### T1 批(死因=codegen 本身;env 门在,一晚扫)— 夜 2 深夜,confirm15 前完成

| 刀 | env 门 | sm_90 旧判(Bump acc) | 重审理由 |
|---|---|---|---|
| E4 unroll | 一行 pragma 重放(已回滚,≈env 级成本) | +10% 恶化("支架税是正收益") | **支架税就是 codegen**——c61 下 nvcc 展开/调度全变,承重墙位置移动 |
| D1×2 熔断换位 | `HASH_PROBE2` + `D1TIER` | +7.5%/+5% | 探测环形态×指令选择耦合 |
| E2×2 VG 值域原子 | `HASH_VG45`(前置 `HYB_TIER` 同开) | +4.5%/+4.9% | 死因=smem CAST.SPIN **模拟税**,模拟税是 codegen 敏感的;c61 下 RED.E.ADD.F64 vs 模拟差价重定价;若复活,与刀 G 竞争 bin4/5 接管权 |
| G×2 交错粒度 | `GFORCE` | +4.7%/+21.7% | G 分法×块布局耦合(c61 改布局) |
| SPEC99_OCH/ROT(hash_heavy 旋钮) | env 在 | 中性/混合 | hash_heavy 是新核**从未在 c61 下扫过** |
| WM(低先验,对照组性质) | `HASH_WM` | +53% 惨败 | 算法级差价(归约网络 vs 模拟原子),预期仍死,便宜带上 |
| **HT_SHIFT(阴性对照)** | `HASH_HT_SHIFT` | ±0.05% 轴不存在 | **实验设计关键**:sm_90 下完全无感化的轴,若 c61 下它动了=系统性位移证据 → 触发 T2 |

**协议**:锚阵 Bump/Hook/Serena/cage15(+TSOPF 小表 sm90 层对照)× 交错 2 轮
× 每刀 on/off;nnz 逐位;相位账(acc/compact/Σ)列表核算;**HT_SHIFT 阴性对照
先行判读**——它动 → 停下来先补 T2 代码刀重放,它不动 → 只有 T1 点状幸存。
**幸存者判据**:单阵 >3% 且 ≥2 阵同向 → 入 confirm16 候选束。

### T2 档(死因=形态×编译器耦合;代码刀,git 史在)— 条件开启

C2×2(块宽 128/64T,+18%/+78%)/C1(拷贝 ILP +6%)/C3(封包形态 +23%)。
重开成本=中(git 重放+适配)。**开启条件:T1 批出 ≥2 幸存 或 HT_SHIFT 阴性
对照翻动**(=c61 造成几何最优系统性位移的证据)。否则不开,防 churn。

### T3 档(死因=算法/非编译;不重开,防 churn——逐条有档案)

- E5 bitonic extract:+180%,45 阶×块栅栏的算法级灾难,非编译条件;
- D5H 精确计数流:工作流实测死刑(count 46.7=symbolic 3.3×),非内核判决;
- mh_merge:批级证伪+崩溃三态,池关;
- E6/E7 转正:用户钦定只作天花板仪器(§四十七);
- PTX 梯:61/70/80 已扫,61 孤峰,选型关闭(§五十二)。

---

## 三、可能新的 hash_kernel 优化

| # | 优化 | 内容 | 依赖 |
|---|---|---|---|
| N1 | **compact 相位融合收尾** | Gv3v2 成活后 bin4/5 的 csort 善后整趟免掉,compact 只剩 CSR 定位;若行级融合 copy 再瘦,Bump Σ 目标 83.5→~72 级 | 刀 G 复活 |
| N2 | **小表侧 TSOPF 补救边界** | bins0-3 行占比统计定人口;若 c61 小表回归不止 TSOPF 一阵,备选=小表行走 batched/merge3 路由 | P1 附带 |
| N3 | **hash_heavy 人口值门三件套预注册化** | 每扩一档(4→5→6→7):nnz 逐位+行内有序+memcheck+est 帽/overflow retry 覆盖(空行/满表/超帽) | R2 逐档 |
| N4 | 刀 A 延伸 bins0-1 | 预期收益小,低优先 | P3 后 |
| N5 | (裁决项)b45 瘦内核重写 spec99 | 若刀 G+扩档后 hash_heavy 距天花板(61.6)仍 >5%,重写议题回桌面;≤5% 议题消解 | **用户裁决** |

## 四、守门纪律(每刀,不例外)

1. **GPU 保护最高优先**:报错即停/sanitizer 先行/病理即杀/Xid 哨兵/降 churn;
2. **同会话 A/B** 才算刀级证据;跨夜单阵不作人群归因;ocean 配对同夜同卡;
3. **相位账列表核算**,结论行禁止心算(§四十九补教训);
4. 默认翻门必须捆绑 confirm;**confirm15 前记分链不动**;重审幸存者入
   confirm16,不阻塞 confirm15;
5. 新数据落未验证人群=补边界≠推翻(结论稳定纪律);语言分三档;
6. 档案:本计划 study/58(v2 已并入上一窗口估值:复活从个案升级为系统性重审,
   E4/D1 从 v1 保守"不复活清单"移入 T1 批——v1 站位被定谳纠正);执行段回写
   study/57 追加(§五十四起);commit 节奏照旧。

## 五、时间线(三夜主干 + 后续滚动)

- **夜 1(今)**:工作树基线 commit → **P1 per-file** → 15 阵验收(+bins0-3
  占比统计)。顺利则 P2 v2 实现落码(env 默认关,不当夜裁决)。
- **夜 2**:**P2 Gv3v2 四阵判生死**(主攻,相位账)→ 深夜 **T1 重审批一扫**
  (env 全开一晚,HT_SHIFT 阴性对照先行判读)。
- **夜 3**:**P3 confirm15_428**(同夜同卡 ocean 配对;F+A+G幸存)→ 记分+
  逐阵尾审计+总账盖章。
- **后续滚动(confirm16 束)**:T1/T2 重审幸存者 + R2 hash_heavy 扩 bins6/7
  (素数梯 2719/5449/16253 档现成)+ N1-N3;N5 裁决项提请用户。
- **弹性**:Gv3v2 病理/判死 → P3 不等,F+A 先入账(bank first);G 线转
  E2-VG(T1)/扩档线,P3 后补束 confirm16。T2 只有位移证据出现才开。
