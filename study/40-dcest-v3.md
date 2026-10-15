# study/40:DCEST V3——est 泄漏修正 + 竞态根除 + 溢出证书(10-12)

> 任务书:docs/83 第七部分第一步 / study/39 Phase 1b——修 est_expand 的 dense 行泄漏。
> 前作:V1(a81dc1b,直接换标签失败)→ V2(df6ccb6,gapped+fixup,c-64 +22%)→ 本篇 V3。

## 一、V2 尸检:三个带病事实(代码证据)

1. **fixup 核是 no-op 白付 6.8ms**。dense 数值核(梯子 `dense_struct_numeric_kernel`
   :199、2r :309、D3 merge、cursor :1346)**行内连续直写**(base 起 bitmap/cursor 前缀,
   无洞)且**自报精确 row_nnz**。fixup 在数值核之后跑,读到的 `row_nnz[row]` 已是 actual
   → 扫描区间坍缩为 [base, base+actual) 全真实数据,compaction 一次都不发生。
   sentinel 核**从未 launch** 而值门通过,本身就是"尾洞不存在"的反证。
2. **in-place 全局 shift 有两类竞态**(V2 值门通过 = 调度运气,不可复用):
   - 行内:dst=[new, new+cnt) 与 src=[old, old+cnt) 在 0<d<cnt 时重叠,跨线程无序写读;
   - 跨行:B(>A)的新区可落在 A 未读的旧尾 [new_B, old_A+cnt_A) 内,并行 block 无序。
   单调左移的 in-place 并行安全需要全序/网格同步——结构性不可修,只能换数据流。
3. **溢出无防护靠 2.16× 蛮力**:V2 的 est×1.2 隐含 raw×1.8×1.2=2.16 覆盖;收紧 margin
   后梯子核(写位无界)会踩邻行,cursor 核(rank≥cap 截断+drop_cnt,仅 cnf 接线)自截。

## 二、V3 设计(零 kernel 形态改动;数据流 + 证书)

```
dcest 相位:alloc = (est==flop ? est : est/expand) × margin     ← fill 核参数化
    └ flop 封顶行是精确上界【不除 expand】——除则 dup≈1 行必炸(V2 隐含修复)
dense_direct(照旧写 gapped dC,自报 actual row_nnz)
DCEST 修正链(移到 k7/D5H 之前 → 此刻 dC 只有 dense 行有数据):
    ① 证书核:actual > scan-1 槽宽(alloc)→ 计数 + ratio 直方图
       触发 → C_nnz=-1 判废 → adaptive 回退 merge3(正确性无条件)
    ② 二次 scan(exact)→ dC_rp
    ③ 双 buffer 紧凑拷贝(仅 dense 行,grid=dense_nr,不动行早退)
       读写分离 → 竞态结构性根除
k7_direct / D5H(链后发射 → 写 exact 基址,免搬)
compact+sort / d2h(消费 exact 布局)
```

要点:
- **k7/D5H 从链前挪到链后**:链前写会落在 gapped 位置且不在 dense_rows 里被拷贝漏掉;
  链后写 = exact 基址零搬运。行集互斥,无依赖(重排对 DCEST=0 路径等价)。
- **margin 时间中立**:数值核只写 actual 条、拷贝只搬 actual 条、d2h 只下 exact——
  margin 只放大 allocation 间距(内存),不花时间 → 按"证书零触发"选,不是越小越好。
  (docs/83 "216%→10%" 的预期修正:10% 时间上没意义,内存上证书不允许)
- env:`DCEST=1` 开;`DCEST_MARGIN`(默认 200);`DCEST_OVF=1` = 溢出仅报告(采数探针,
  输出可能损坏);与 CNT_FUSE 互斥(自动关)。

## 三、margin 实测(证书直方图,10-12,GPU0)

| 阵 | diter 行 | m=110 欠配 | m=125 | m=150 | m=200 | 直方图形状 |
|---|---|---|---|---|---|---|
| c-64 | 19444 | 451 (2.3%) | **0** | 0 | 0 | 主桶 [1.0,1.25),尾 ≥0.75@110 |
| pre2 | 4 | 2 | **0** | 0 | 0 | 4 行全部富余 |
| in-2004 | 21067 | 573 (2.7%) | — | 2 | **0** | 2 行结构性欠估(actual=raw×1.5-2) |

- MinHash 在 dense 行中位误差 ≈ ±10%(σ_eff≈4-5%),但**左尾到 0.5**(in-2004 2 行)
  = 结构性欠估,不是 σ 噪声 → margin 必须按最差尾选。
- **默认 200**(= 最差观测需求;≈V2 的 2.16 内存量,时间赢面全保留)。
- 125-150 对 c-64/pre2 已零触发 → **per-matrix margin 是 L0 的天然决策维度**(Phase 2)。

## 四、电池(AB_RUNS=5 交替,compute-only 中位,nnz 全一致)

| 阵 | diter 行 | off(ms) | on(ms) | Δ% |
|---|---|---|---|---|
| nd24k | 49978 | 140.06 | 119.27 | **−14.8** |
| in-2004 | 21067 | 51.40 | 46.05 | **−10.4** |
| Ga3As3H12 | 7149 | 39.97 | 38.32 | **−4.1** |
| c-64 | 19444 | 19.56 | 22.29 | +14.0 |
| pre2 | 4 | 29.42 | 32.91 | +11.9 |

注:电池采于早退修复前二进制(17:0x,机器低载窗)——修复的时序增量 = src==dst 的
头几个 dense 行补拷(µs 级,构造性);终版二进制重跑被同租户高载(loadavg 14)拖至
2h+ 已砍,数字以 17:0x 为准。

c-64 相位账:dcest 0.9 替 dense_count 2.7(−1.8),链 3.9 → 净 +2.1ms(V2 +22% → V3 +14%)。
pre2:4 行巨鲸,链的 scan-2(66 万行)+ 4 行亿条拷贝无 dense_count 可抵。
**赢亏判据 = dense_count(O(dense flop))vs 链费(O(dense_nnz) 拷贝 + O(A_rows) scan)
——L0 特征维度直接可算。**

## 五、守门

- memcheck(compute-sanitizer,pre2 + c-64,DCEST=1):**0 errors**(重排前后各一轮)。
- nnz:c-64 155,006,041 / pre2 323,445,872 / nd24k / in-2004 / Ga3As3H12 五阵 off=on。
- **值门第一轮 FAIL → 揪出 V3 自埋雷**(10-12 晚,教科书级):
  - 指纹:四阵 col_mismatch(1781/3447/793/1336)+ nnz 全对 + memcheck 0 err;
    首错行 ON 侧整行 `(row,1,0)` 重复 = **新 buffer 零页原样 dump**。
  - 根因:拷贝核的 `if (src == dst) return;` 早退——这在 V2 的**原地** shift 语义下
    成立(不动=已就位),在 V3 的**双 buffer**语义下是错的:不动的行也要从旧 buffer
    搬到新 buffer。受害行 = 前方无 padding 累积的头几个 dense 行。
  - 教训:**改数据流不变式时,微优化早退必须重审**(in-place→two-buffer 的语义迁移);
    nnz 门+memcheck 双双瞎(合法内存+计数来自数值核),值级指纹才破案。
  - 另:dump 为全精度 17 位,SMEM 原子累加序噪声在 text_diff 上可见属预期;
    判据看 col_mismatch=0 + worst_rel(off-vs-on 与 off-vs-off 同阶)。
- **构建系统雷(同晚,连踩两次)**:Makefile 不跟踪 flag 变化,`make DBG=1 WRITE_MTX=1`
  不重编未改源的 main.o → 首轮"PASS"实为**陈旧 dump 与自身对比的空对照**(nd24k 文件
  是 10-05 的);二次踩坑:`git checkout` 恢复 tracked 旧 .o 后,make 视其新于源不重编
  → 又链进无 WRITE_MTX 的 main.o(dump 静默消失)。防雷三条:①守门前 `strings
  spgemm_test | grep self_product_hash` 数出现次数;②dump mtime 必须晚于运行时刻;
  ③换 flag = `make -B` 强制全量。已入 correctness-first-gate memory(第五类盲区)。
- **off-vs-off 对照(c-64)**:col_mismatch=0 / text_diff=8,388,159 / worst_rel=6.9e-17
  → 基线列集完全确定;8.4M 文本差 = SMEM 原子累加序噪声(17 位 dump 可见),非缺陷。
  值门判据据此明确:**col_mismatch=0 且 worst_rel 在 1e-16 级**(text_diff 计数仅参考)。
- 值门第三轮(早退修复+make -B 全量重编后)——**四阵全绿**:

| 阵 | col_mismatch | unordered | text_diff | worst_rel | |
|---|---|---|---|---|---|
| Ga3As3H12 | 0 | 0 | 10,477,963 | 1.045e-15 | PASS |
| in-2004 | 0 | 0 | **0** | **0.0** | PASS(逐位一致;dup≈1 确定性累加) |
| c-64 | 0 | 0 | 8,366,336 | 1.383e-16 | PASS(text_diff=off-vs-off 底座 8.39M) |
| nd24k | 0 | 0 | 98,723,941 | 7.531e-16 | PASS(该阵噪声底座;雷轮同量级) |

  雷轮的 col_mismatch(1781/3447/793/1336)与 worst_rel(1e-3 级)全部归零 →
  早退修复实证,text_diff 大数=17 位 dump 下累加序噪声(off-vs-off 同量级)。
- memcheck(最终版二进制,pre2 DCEST=1):0 errors,C_nnz 精确。
- **组合门全绿**(链后直写者与 DCEST 共存,各配 DCEST 0/1 对照):
  - in-2004 [D5H=1](606,224 D5H 行 + 21067 diter 行):col_mismatch=0 / text_diff=0 /
    worst=0.0 —— **逐位一致**,链位重排(D5H 必须在链后发射)实证;
  - pre2 [K7=1](297,206 k7 行 + 4 diter 巨鲸):col_mismatch=0 / worst_rel=2.9e-18 PASS。
  - 排雷附记:组合门首两轮也是空对照——bash 的 `$X` 展开词在命令前缀位**不被识别为
    环境赋值**而成为 argv[1](程序把 "D5H=1" 当矩阵文件名秒退无 dump;单侧文件缺失
    的 gate 会假 FAIL,双侧缺失会假 PASS)→ `env $X` 修;mv 失败哨兵(cannot stat
    过滤进 Monitor)从此后列入 gate 循环标配。
- 行内有序:dense 窗口升序直写 = 天然有序;值门脚本的 unordered 计数同查。

## 六、边界与遗留

- **Ga41 异象破案(10-13 晨,相位表+dbg 实锤)**:OFF 臂 `方案5 规模门:跳过
  (Σest=17.45M/418.5M=4.17%<5%)` → 2516 行 diter(dup 塌缩 **27×**:flop 465M→est 17M)
  被弹回 hash accumulate(64.8ms CAS);DCEST 无此门 → 走梯子(7.2ms),accumulate
  34.7ms。**规模门按 dense_count 税时代标定,est 份额是错误分母**(按 flop 这些行是
  全阵大头)。Flan_1565 同机制(dc=0.03 却赢)。这是"旧门在新管线失准"第三例
  (SiO2 翻面/Flan/此例)→ **解析门必须加 dup 塌缩项**:
  `diter_dup=Σflop/Σest > 阈值 → 无条件开(CAS 规避);否则 dc vs 链`。两量 binning
  后零成本。屏的 dc_ms 列在门弹回阵上严重低估 off 臂真实成本——前哨 −0.6~−0.9%
  的外推因此偏保守(65 中毒阵里同机制的阵会更多受益)。
- **dense_win/dense_mode(矩阵级)不激活**:dense_rows=null 是 DCEST 门槛(混合 BIN_DITER
  专属)。brainpc2(13802 行)落在此界外——矩阵级路径要 DCEST 需另开刀(全体行语义)。
- bcsstk30 diter=0(路由已变迁,旧注释人口不存在)。
- 链可再省:证书 D2H 与 C_nnz D2H 两个同步点可合一(省 ~0.2-0.5ms);c-64 的 3.9ms 里
  拷贝 ~1ms、scan/alloc/sync ~2.9ms——若 pre2 类亏损要修,先做同步合并再做 alloc 池化。
- **in-2004 两行欠估计证完成(10-13 晨,ovf 探针)——定性反转:不是盲区,是统计尾部**:
  row=815546/320392,raw_est≈620 vs act=1008/936(**1.63×/1.51×**),但形态完全普通
  (a_len 56/85,flop 2762/3946,无可路由特征)。反 Gamma 尾部:K=128 每行
  P(欠估 x)≈e^(−128(1/x−1+ln1/x)),x=1.63 ≈ 1e-4/行 → 数千万 diter 行总量下出现
  几个是**应然**;两行确定性重现(列集-hash 固定交互)。定谳:不改估计器,证书+margin
  即正确响应;1a 阈值设计不受影响。副产品:**margin 解析先验 = 1+√(2lnN)·1.04/√K_eff**
  (N=diter 行数)——状态机从"首轮 200"升级为解析起点+直方图修正;margin 与人群大小
  反向(行少 → 300+ 白给,内存 ∝ Σalloc 与行数耦合)。

## 七、结论

1. 第一步任务书目标达成方式与 docs/83 预期不同但更真:泄漏修了(flop 封顶守卫 + ÷expand),
   fixup 归零(删除而非降费),shift 竞态根除(双 buffer);margin 按"证书零触发"实测
   定 200,不是纸面 1.1。
2. **溢出证书(L2)从 Phase 3c 提前落地**——没有它,任何 <2.16 的 margin 都是静默损坏。
3. nd24k −14.8% 首次实测"dense_count 启动税消灭"在重税阵上的兑现;部署决策交给 L0。

---

## 10-16 追记:dense_count 取舍定谳(批 437 权威数据,用户记忆触发复核)

按 dense_count 份额分桶的 CD/C 配对 GM:
```
0-5%: 1.0056(n=227) | 5-15%: 1.0202(n=36) | 15-25%: 1.1348(n=27) | >25%: 1.1155(wb-edu 51%)
```
**定谳:dc 份额越高,DCEST 输得越狠**——est 配额的 margin 膨胀+紧凑拷贝税随 dense 人口
放大,吃掉 count 节省还倒贴。用户 10-12"不偷生态"判断被权威数据钉死:
**dense_count 的 12% 中位份额是 dense 梯子零原子直写快的价格,已正确支付**。
同桶 A/C=0.91 → 高 dc 阵的真出路 = A 路由(调度层),非相位消灭。
**论文素材**:"精确计数最贵处恰是消灭它最亏处"——取舍悖论入证书/取舍节。
**后续动作**:①DCEST 解析门现形复核(69-75 阵主战场分桶归属待查)②L0 配置维度
把"A 路由"纳入;①②并入 study/46 训练时用本表数据重定界。
