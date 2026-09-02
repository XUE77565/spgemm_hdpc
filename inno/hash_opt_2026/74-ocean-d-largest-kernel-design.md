# docs/74 Ocean-D largest-kernel 设计(c-big 4.76× 解锁;任务 #8)

## 缺口实锤(v29)

c-big = 头号单阵 loser(4.76×;345K 行,448M C_nnz,log-gap 占比 ~1%)。
docs/61 结论:19578 条 DIM2 候选行中 19491 挂在 v4 span 门(16×flop ≥ span 不成立,
即密度 < 1/16)——这些是"straggler 行":est 大(千级)∧ span 更大(10 万级)。

## 为什么我们的 dense 吃不下这些行,Ocean 吃得下

| 设计 | 我方 dense_direct | Ocean DENSE_NUMERIC_LARGEST |
|---|---|---|
| SMEM/列 | val 8B + bitmap 0.125 + prefix 0.125 = **8.25B** | **0.25B**(仅 bitmap+prefix) |
| val 存储 | SMEM 数组 | **全局双缓冲流过**(NUM_SM×3 buffer) |
| 单窗 span 上限 | 114KB/8.25 ≈ **14K 列**(DENSE_MAX_N=14966) | 227KB/0.25 ≈ **900K 列** |
| 发射 | flag 扫描 + 前缀 | 位图字 popcount 前缀 + 精确偏移直写 |

straggler 行 span ~10-30 万 → 我方需 20-50 窗(每窗全量重扫 A 行),Ocean **1-2 窗**。
密度 1/16-1/64 的行走 SMEM-val dense 是纯亏(窗内 val 数组 93% 空),但走
bitmap-SMEM + val-global:SMEM 只付 0.25B/列,val 只付 est×16B(实际输出)——
字节模型上这才是这类行的正确形态。

## 字节模型(路由判据,OCCGATE 同框架)

```
dense_largest_bytes = span × 0.25B(SMEM 位图+前缀)+ est × 16B(val 全局读+写+key)
hash_bytes          = ht × 12B + est × 24B(compact 税,SROUTE 同式)
```
密度阈值:span×0.25 + est×16 ≤ 2est×12 + est×24 → span ≤ 80×est。
即 **est/span ≥ 1/80** 的 straggler 行 largest-dense 赢 hash(Ga 族 occ<1/16=1/16
仍正确留 hash —— 两门不打架,OCCGATE 阈 1/16 ≪ 1/80?注意方向:1/80 比 1/16 更宽,
即 largest 只偷 OCCGATE 判 hash 的行里 occ∈[1/80,1/16) 的窄带 —— Ga 族 occ 3.6%=1/28
落在此带内!⚠ Ga 教训:低占用行进窗付 O(span) 窗税。但 largest 的窗税 = 0.25B/列
(比旧窗 8.25B/列 降 33×)——docs/67 的 Ga 判据(占用轴)需用新字节模型重推:
Ga41 est/span?occ 3.6% → span×0.25 = 28×est×0.25 = 7×est SMEM 字节 vs hash 24×est
→ largest 仍赢字节。但 Ga 回归实measured 过(旧窗 8.25B/列 = 231×est>24×est 输)。
新形态下 Ga 行为需电池重验,不可纸面定案。

## 内核草图(dense_largest_numeric_kernel)

```
行 r,窗 [c0, c1)(宽 = SMEM 预算/0.25B,~900K 列,通常 1 窗):
  pass1: 遍历 A 行 k 链 × B 行列 → atomicOr(bitmap, col)   (flop 次,流式读 B)
  pass2: 位图字 popcount 扫描 → prefix(SMEM)→ 行内精确偏移 + row_nnz
  pass3: 再遍历乘积 → 读 prefix 偏移 → 写全局 (key,col)|(val) 到行输出区   (flop 次,再流式读 B)
成本 ≈ 2×flop 流读 + span/32 字位图操作;vs hash 的 est 次 CAS + compact。
```
双遍历 = dup 行(flop≫est)付 2× 读;Ocean 同款(largest kernel 也是 re-walk)。

## 部署位

- 新 BIN_LARGEST(bin 20?):DIM2 路由改用上面字节判据(est/span ≥ 1/80 ∧ est ≥
  EST_DENSE_MIN);OCCGATE 门保持(对 largest 分支用新字节模型重推后决定去留)
- 输出直写 tmp_key/tmp_val(与 diter 路径同消费)或原地 —— 复用 compact 管线
- 验收:c-big nnz 逐位 + c-big/Ga41/3Dspec2/crankseg_2 电池(OCCGATE 17 阵全守卫)

## 排队

优先级 = 三件套电池之后(road 族 ~10 阵 > c-big 1 阵);工程量 ≈ DSTRUCT Phase 1b 同级(~2h)。

---

## 首战战报(2026-09-03 凌晨,净窗电池)

**c-big −60.5%**(289.4→114.4ms;ratio 4.76×→1.91×),26453 行进 BIN_LARGEST(straggler 全捕获),
nnz 447991461 逐位一致。

### 实现三坑(全部 BINS_DBG/诊断打印实证定位)

1. **heavy 字节模型**:首版 ht 被 HASH_CAP cap(16384)→ hash 代价被低估;heavy 行真表 =
   全局 2×est 槽(α=0.5)→ `ht_eff = e ≤ HASH_CAP ? ht : 2e`
2. **n_cols 传参**:首版沿用 dim2 的守卫 `n_cols > 0`,而 host 只在 DIM2=1 时传 A_cols
   (默认关)→ 门恒假;改 `(g_dim2 || g_largest) ? A_cols : 0`
3. **路由链覆写**:largest_ok 置 bid 后,第二路由链(`if(false){} else if v4...else if
   e>HASH_CAP→HEAVY`)无条件覆写;链头改 `if (bid == BIN_LARGEST) {}` 跳过

### 字节模型终版(毫字节,全具名)

```
dense: span×250 + est×16000 ≤ hash: ht_eff×12000 + est×24000
其中 ht_eff = e≤HASH_CAP ? SMEM梯表 : 2×est(heavy 全局表)
⟺ 常规档 span ≤ 48ht+32est;heavy 档 span ≤ 128×est(密度 ≥ 1/128 即赢)
```

cage15 类 est 欠估族:走 flop-retry 兜底(正确但吃内存;精确管线集成 = 下一步)。
守卫电池(OCCGATE 族)在跑。
