# 2026-09-18 深夜战报:count-sort 两级化实施之夜(守门体系首战)

> 接续 09-18 凌晨测量解剖战报。本晚主线:dense 开线战略决策 → spec 81 ②号刀完整实施 →
> 守门体系实战首秀连拦两雷 → 双位点全绿入库。commit `e2c2dfc`。

---

## 0. 当晚净账

| 类型 | 内容 | 级别 |
|---|---|---|
| **战略决策** | dense 优化"先进诊断门,不先动刀"(用户批准)——c-64 1.21× 拆解决定 dense 线第一刀 | — |
| kernel 落地 | spec 81 count-sort 两级化:O(d²)→O(d+1024),双位点(MODE=0 tmp/MODE=2 D5H),默认关 | C 全绿 |
| **守门首战** | 值级 4/4 + 有序 4/4 + D5H 位点双绿;**拦截两个真 bug**(其一为静默型) | — |
| 尸检 | SMEM 指针算术雷/所有权布局雷/大阵 WM 姿势雷,三雷全录 spec 81 | — |
| 基建 | A 级电池净窗守卫(自动等共租撤离触发,×5 交替) | 挂账中 |
| loop | 更新含 dense 诊断线(1581325e;后按用户指示停过夜) | — |

**诚实注脚:性能收益未证实。** 口径 B 电池(排序税 5.9%→目标 <2%,F2 2.46→<1.5ms)
等净窗守卫触发,今晚不宣称性能赢面。

---

## 1. 时间线

1. 用户问"是不是要开始做 dense 优化了?" → 意见:**是,但先进诊断门**
   (三阵全胜是 hash 行口径;dense 行群双方 dense 谁优从未测过 = 口径 C 盲区;
   crankseg 不走此门——重行是 hash 形态,occgate ③ 才是解)
2. dense 诊断线入 study/22 + loop prompt 更新
3. 编码:Pass0 min/max(SMEM+warp 归约)→ 逐行自适应 SH(桶≤1024,聚簇不退化)→
   Pass1 直方图 → Pass2 排他前缀 → Pass3 桶分组重排(寄存器 staging 4 槽/线程)→
   Pass4 桶内 O(b²) 位次直写;门槛 ht∈[512,1024];hist 12KB host 侧条件追加
4. 首轮守门:F2 **illegal access**(雷①);msc 假 PASS(比对用了 T1 文件——雷③的前奏)
5. 修雷① → 二轮:值级双 PASS 但**有序违规 85,963/15,111**(雷②,命门拦截)
6. 修雷② + 发现雷③(大阵缺 METHOD=hash 门控 → T1 cuSPARSE insufficient resources 假死)
7. 三轮:**全绿**(下表);D5H=1 F2 hash_direct 6.5ms 执行实证补 MODE=2 位点双绿
8. commit `e2c2dfc`;c-64 相位解剖开跑即被叫停(用户收工指令);电池守卫挂起

---

## 2. 守门终局表

| 阵 | 值级(vs truth17 @1e-9) | 有序扫描 |
|---|---|---|
| F2 | PASS | **SORTED-OK** |
| msc10848 | PASS | **SORTED-OK** |
| crankseg_2 | PASS | **SORTED-OK** |
| c-64 | FAIL=已知 artifact(only_test=20,504 与既往逐数一致,common=truth 全量,误差 2e-16) | **SORTED-OK** |
| F2@D5H=1(MODE=2 位点) | PASS | **SORTED-OK** |

---

## 3. 两雷尸检(本晚最大收获:守门体系的价值实证)

### 雷① SMEM 指针偏移砸 sh_val(崩溃型)
- 症状:F2 `[hash] accumulate` 后 illegal access at 发射检查
- 根因:hist 起点写 `smem + 2*ht_size`(int 单位)——但表布局 = sh_col(ht×4B) + sh_val(ht×8B)
  = **12×ht 字节**,hist 应在 `smem + 3*ht_size`。2×ht 落进 sh_val 区 → 值被直方图清零覆写 +
  计数读成野 double 位型 → 落位越界
- 教训:**SMEM 多段布局的指针算术必须逐字节核算每段**;int* 基址下 double 段占 2 int 槽/元素

### 雷② 跨步所有权前缀(静默型——最危险的一类)
- 症状:**值级全部 PASS**(nnz 对、多重集对、误差 1e-9 内)但有序违规 85,963(F2)/15,111(msc)
- 根因:Pass2 线程跨步持桶(bin b 归线程 b%blockDim)+ 段基址公式——公式假设"前面线程的
  全部桶排在我之前",跨步布局下该顺序假设不成立 → h_start 错 → 桶区间重叠 → 落位重排成
  乱序(但元素多重集恰好守恒,故值级不报)
- 教训:①块级前缀和的 **ownership 布局必须与段基址公式的顺序假设一致**(修复=连续所有权
  [t·ib,(t+1)·ib));②**"值级 PASS ≠ 正确"**——这正是守门三件套里行内有序检查存在的意义,
  若无有序扫描,此 bug 将静默产出错序结果混入电池
- spec 81 预言自证:"这刀直接动序,有序检查是命门"

### 雷③ 大阵 WM 守门姿势(环境型)
- 症状:crankseg/c-64 两轮"莫名失败"(0 输出)——T1 cuSPARSE `insufficient resources` 假死
- 根因:大阵 in-binary cuSPARSE 本就资源不够(truth17 大阵历来 scipy 生成);守门跑大阵必须
  `METHOD=hash` 单方法门控跳过 T1;且比对目标是 **self_product_hash.mtx** 而非 T1 的
  self_product.mtx(首 轮 msc 的"PASS"实为 cuSPARSE 自比对,无效证据)
- 教训:WM 守门跑大阵 = METHOD=hash + 对 hash/adaptive 输出比对;另注意 cuSPARSE 报错被
  管道 grep 吞掉后 exit code 仍 0——**exit 0 不是完成证据,mtx mtime 才是**

---

## 4. 价值评估(回答"今晚的工作有没有价值")

| 价值类型 | 状态 | 依据 |
|---|---|---|
| **方法论价值(守门体系实战验证)** | ✅ 已超额落袋 | 首战拦截两雷,其一为"不崩+值全对+仅序错"静默型——若无有序守门即漏网。正确性优先铁律(correntness-first-gate)的活例,论文方法论节素材 |
| **正确性基建闭环** | ✅ 落袋 | 代码入默认旁路(HASH_EXTRACT=2),双位点验证链完整,三雷入档不复发 |
| **减总功方向的实质推进** | ✅ 结构性落袋 | 占用对消恒等式下唯一可收割方向(spec 81 定位)的第一刀真实落地;O(d²)→O(d+1024) 复杂度减 500×(d=1024)/50×(d=256) 的算法替换完成且正确 |
| **性能收益** | ⏳ 待判决 | A 级口径 B 电池挂净窗守卫;排序税 5.9%→<2% 与 F2 2.46→<1.5ms 均未测;**不提前宣称赢面**(单发 ctime 无效的教训) |
| **战略价值(dense 决策)** | ✅ 定方向 | "先进诊断门"避免盲动刀;c-64 1.21× 的三条判决分支已预设好 |

**一句话:今晚买到的最大资产不是这把刀,而是证明"我们的守门体系能在无人盯守下拦住
静默型错误"——这对后续每一刀都生效。**

---

## 5. 明日队列(按 ROI)

1. 看 `/tmp/ab_c2_wall.log`:电池若已自动完成 → 判决(HASH_EXTRACT=2 是否翻默认:全绿或小赢即翻)
2. 重跑 c-64 相位解剖(METHOD=hash + DD_ROWTIME + HASH_ROWTIME,卡净即跑)→ dense 诊断判决分支
3. counting-vs-radix 交叉实测(d≤2048,bin6 47.9K 行载体,净窗)
4. dense 行群口径 C 对拍(DD_ROWTIME vs ocean denseNumericIter 同行群逐行)

## 6. 状态快照

- HEAD = `e2c2dfc`(spec81 落地+守门全绿+三雷入档)
- 二进制:spgemm_c2 = 两级排序 WM 版(默认 off,行为=旧默认)
- loop:已按用户指示停过夜;重建按 study/22 队列
- 电池守卫:pid 见 /tmp/ab_c2_wall.sh(自动触发,日志 /tmp/ab_c2_wall.log)
- 卡况:卡 0 干净;卡 1 共租 69GB(xiatengfei 夜班)
