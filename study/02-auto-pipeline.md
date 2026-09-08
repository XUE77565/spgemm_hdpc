# 我们 Auto 模式的计算流程(小白版脉络讲解)

> 我们的程序叫 Auto:它上面有个**调度器**,先看矩阵长相,决定整阵用 hash 管线、
> merge3 还是 cuSPARSE(大阵 hash 主场,小阵 merge3 主场)。
> hash 管线是主战场,本文讲它的脉络——以及它和 Ocean 最大的不同:
> 我们有**三条计算路径**和一串历史上长出来的"门"。

---

## 全景图

```
调度器(dispatcher):整阵选 hash / merge3 / cuSPARSE
        ↓ hash 管线
① 估计层:count_flop → MinHash 构建+合并 → est_scan(每行 est + span + flop)
        ↓
② 路由层:compute_bucket 逐行判 → BIN_DITER(dense 路)或 hash 梯(bin 0-10)/
        heavy / batch / ESC / …(一长串门,见下文" honest 清单")
        ↓
③ 路内:
   dense 路:方案5 精确 count → §22d 分流 → 六档位梯(结构内核)/ 老窗口 cursor 内核
   hash 路:20 档表 → accumulate → retry → compact+sort
```

---

## 第①层:估计

- `count_flop`:数每行的乘积总数(flop)
- **MinHash**:采样估计每行**去重后**的输出个数 est(hash 管线的特色,一次构建反复用)
- `span`:每行输出列的左右边界差(从乘积的列范围取)
- 三个量(est / span / flop)是后面所有决策的原料

## 第②层:路由(我们的痛点所在)

现状是一条**门链**(v34final 提交态):

1. ESC 门:flop≤32 ∧ dup≤2 → 零表 counting-sort 短行
2. LARGEST 门:straggler 行 → 位图窗内核
3. 小跨度门:span≤256 → 数组直址
4. v4 主门:flop≥2048 ∧ est≥1024 ∧ 16×flop≥span ∧ dup<8 ∧ occgate(est×16≥span)…→ dense 路
5. 超档行再按结构化地板(4096/2048)收口
6. 进了 dense 路还有:§28 矩阵级密度门(中位 est/span<0.5 全走老内核)、
   §29 惰性分流(免 host 同步税)、ALEN 门(a_len≤32 → 老内核)

每一道门都有自己的故事(某次电池里某族矩阵赢了),但**它们互相叠加后的行为空间
已经没人能完整说清**——这是我们被批评"打补丁"的原因。

## 第③层:路内(我们已经学到了 Ocean 的形)

**dense 路的六档梯**(与 Ocean 同公式,预算÷9B−4,外加第 6 档):

| 档 | 窗口(列) | 块宽 | 并发 |
|---|---|---|---|
| 0-4 | 451…7277 | 64…1024 | 32…2 |
| 5 | 14543 | 1024 | 1 |

- **结构内核**(dense_struct_numeric):位图 symbolic + popcount 前缀发射(Ocean 式)
- **老窗口内核**(hash_dense_direct):多窗数据驱动 + cursor——**Ocean 没有的第三条路**,
  对 c-64 类"全幅行"(est 16K ∧ span≈n)是实测赢家
- 分流在 host 做(按档位索引稳定排序,保行号序 = 保 CSR 访存局部性)

**hash 路**:20 档表 → SMEM 开放寻址累加 → 溢出 retry → compact+sort
(我们的 compact 排序税在中行族占比可观,是历史靶点)

---

## 我们有、Ocean 没有的东西(资产清单)

| 机制 | 一句话 | 状态 |
|---|---|---|
| MinHash 估计器 | 一次构建,重复使用,免重复扫描 | ✓ 在用 |
| 老窗口 cursor 内核 | 第三条路,全幅行/怪物行赢家 | ✓ 在用 |
| merge3 自适应分桶 | warp 竞标赛归并,小阵赢家(first100 0.74×) | ✓ 调度器在用 |
| h2d 重叠 / 方案5 直写 / H2DOVL | 管线级搬运优化 | ✓ 在用 |
| ESC 零表短行 | flop≤32 免表 | ✓ 在用 |
| 逐行周期采数器(DSN_RT/DD_ROWTIME/DUMP_ROUTING) | 成本模型的"仪器" | ✓ 基建 |

## 一句话总结

**①③两层我们与 Ocean 同构(估计+分档联动都学来了);差别全在第②层——
Ocean 是一次比较,我们是一串门。** 下一版的方向就是把门链换成
"三内核 × 足迹成本表 × 一次 argmin"(v36 已试第一轮,证伪后定位到缺人口项,待续)。
