# 地雷 #3 全案:ESC 自加倍 + batched SMEM 重叠(2026-09-16 排雷日)

> 一句话:**pre2 首次完整值级验证揪出双雷——ESC 同键归约自加倍(14,773 点 2.0×/1.5×)+ batched kernel SMEM 布局重叠(696 点整型比特 denormal);两雷皆生于 T4f 空对照时代,活三周;一日双杀,python 逐指令模拟 + TMPSCAN 相位仪器定位。**

背景:pre2(OOB 触发阵)雷修后首次跑通完整值级 → FAIL:模式完美(3.23 亿项位置全对)、值错 15,469 点。判别链:RETRY_SMEM/HASH_PROBE_RF/DSTRUCT/A 本体重复 全部排除(逐位同失败)→ 确定性 bug。

## 雷 #3-a:hash_esc_kernel 同键归约自加倍

- **机理**:② 相 counting-sort fused 的循环 `for k in [0,SUBW)` **含 k==t(自己)**:`sum = my_val` 起步,`ok == my_key` 在 k==t 恒真再 `sum += ov` → 自己加两遍
- **代数**:单贡献键 = 2.0×;双贡献 {v1,v2} = (2v1+v2)/(v1+v2),v1≈v2 时 = **1.5×**(两个比值簇全解释;col 72 的 2.13651e6 = 2×1.916e4 + 2.098e6 逐位吻合)
- **定位法**:python 逐指令模拟 kernel 语义跑 row 618 → CPU 上复现 GPU 的 2× → 算法雷(GPU 语义无关)
- **修复**:`if (k != t) sum += ov`(一行)
- **触发面**:flop≤32 ∧ dup 小的微小行(ESC8/16/32);pre2 3.2 万行命中;msc/crankseg/F2 的 ESC 人口为零故历史"干净"

## 雷 #3-b:hash_spa_batched_kernel SMEM 布局重叠

- **机理**:`bsmem[WPB][4×HT]` 中 `t_val = (double*)(bank+HT)` 是 double 要占 **2×HT 个 int**([HT,3HT)),而 `pack_col = bank+2HT` 落在 [2HT,3HT) —— **extract 写 pack_col[pos] 时半改写高槽位(≥HT/2)条目的 val** → denormal 垃圾,两 int 字=两个列号(0x70e/0x60e=1806/1550 逐位解码)
- **受害者画像**:batched 行(est≤64)且哈希槽 ≥ HT/2 → 696 点/648 行(0.0002%),est≤32 行占不满半表所以其他阵全绿
- **定位法**:INITDBG memset 分叉(垃圾保留=主动写)→ TMPSCAN 相位仪器(accumulate 后即脏 → bin0)→ 比特解码 + 邻域 5 槽 dump。**教训:bit 解码的 (hi,lo) 先按"列,列"解释再按"行,列"——本轮 hi 被误读成行号 1806 带偏一轮;另 mtx 行号 1-based vs kernel 0-based 的 off-by-one 让 join 错位一格,仪器扫错行**
- **修复**:SMEM 4×HT→**5×HT**,pack_col→+3HT、pack_slot→+4HT(bsearch_symbolic 同款布局一并修,虽彼处 t_val 未用属良性);SMEM/块 8KB→10KB,occupancy 影响由电池观察
- 顺带:`hash_spa_batched_kernel` 是模板(`batched<HT,WPB>`),裸名 grep 会漏发射点

## 验证(按正确性优先铁律)

- 4 轴 memcheck(pre2 default/storm、msc default[含 diter]、crankseg default)全 0 errors(#3 修复不改全局访存,此前扫描仍有效)
- TMPSCAN 复扫 row 1486:全相位 clean ✓
- **全 11 阵值级 + 行内有序总扫(DITER off)**:结果见本篇末尾追加表
- 新增仪器(env 门控,默认零开销):TMPSCAN=<row>(相位时间线)、INITDBG=1(tmp_val 清零分叉)

## 方法论沉淀

1. **值级指纹三件套**:比值分布(2.0/1.5/0)→ 行/列聚集 → 比特解码,本案三层全用上
2. **python 逐指令模拟 kernel** 是区分"算法雷 vs GPU 语义雷"的最快刀
3. **set 对比序盲 + nnz 模式盲 + 空对照**——三盲区家族第四、五号成员;正确性优先铁律([[correctness-first-gate]])即由此立

## 验证总表(2026-09-16 深夜;口径 = WRITE_MTX 值级 set 对比 + 行内有序,DITER off)

| 阵 | 值级 | 乱序 | 备注 |
|---|---|---|---|
| pre2 | ✅ PASS(3.23 亿项,rel 2e-25) | 0 | 雷触发阵,史上首次完整值级 |
| web-Google / Zd_Jac2_db / msc10848 / c-43 / F2 | ✅ PASS | 0 | |
| crankseg_2 / in-2004 / nd24k / Ga3As3H12 / c-64 | ⏳ 待补 | — | 两轮均撞他方 VLA 作业 OOM(非代码失败,manual 单发通过);净窗补验,crankseg 最优先(ESC 重灾区) |
| diter 路径(bin13) | ❌ 已知未修 | — | 全表 DITER off 口径 |

另:本轮追加 **HASH_UO_XTRACT 口径开关**(MODE=0 强制无序 extract,对齐 ocean ROWCYCLES 窗口径;F2 实测窗内排序税 5.9%)——口径 A(机理对照)与口径 B(有序产出)从此可分;仪器禁入电池默认。
