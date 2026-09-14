# 三刀审计:α 表配 / LLB / read-first(2026-09-14,study/10 后续)

> 一句话:**三把快刀两死一存档;实验反向证伪"表太大"朴素版,把 bin5/7 行集输的真凶逼到"2 幂梯没有合身档"(Ocean 素数梯 673 恰好卡在我们 512↔1024 的空档里)。**

背景:study/10 matched 块逐行对拍定谳 F2 平价(45.9% 胜率,中位 1.047),输带 = bin5 行集 1.30×/bin7 行集 1.45×。

## 刀 1:α 表配因子(HASH_TGT_ALPHA,已入代码默认 200=legacy)

- 假说:`target=2×est` ∧ est 高估 1.63× → 表 ≈ 真需求 3.3×,init 税大
- 结果:**中间 α(150/125)全阵 no-op**——est 被量化成 2 的幂(F2 中位恰 1024),target 在 (1,2)× 区间不跨 2 幂界 → 行不挪档
- **α=100 全线崩**:F2 accumulate 5.7→10.5ms、crankseg 22.3→30.0、nd24k 49.2→78.5(×1.35-1.84)
- **机理反转**:小表不是省 init,是探测链爆(est 尾部欠估行 LF→1);2× 余量 = 承重墙
- 附:ADAPT_EXPAND 状态机(1.15↔1.4)与 α 有交互,α=100 的崩是探测串行化而非 retry 增(retry 计数 1319 未变)

## 刀 2:LLB 动态 G(Ocean localLoadBalance 同款,env 已有)

- matched 块 + LLB=1 逐行:总胜率 44.6% vs 静态 45.9%;bin5 9%→7%、bin7 19%→19% —— **wash,排除**

## 刀 3:read-first(见 study/10 + compare/rowlevel_F2/readfirst_probing.patch)

- dup≥7 赢 3.5-4.7% / dup≤2 输 15% / F2 wash → 无净赢,补丁存档,复活路径 = bin 级 dup 条件化(上限 3-5%)

## 真凶收敛:梯粒度(合身档缺失)

证据链:
1. bin5 行集输比 1.35× ≈ 1024/673(我们的表/他们的表)的 init 量比
2. α=100 证明"更小"会爆、α∈(1,2) 挪不动 → 问题不是"太大"是"没有合身档"
3. Ocean 素数梯 331/673/1327/2719/5449 每档 ~2× 且贴合 est;我们 2 幂梯在 512↔1024 等处有空档

**下一把刀(真手术)**:梯上加中间档(如 …512, 640, 768, 1024, 1280, 1536, 2048, 2560…),探测从 `& mask` 改 wrap(Ocean 同款 `slot+1>=ht?0`),ht 与 bin 号解耦(BIN_HT[] 查表)。涉及:compute_bucket 梯、hash_spa 探测、compact/csort 档配、hblk 梯索引。预期:bin5/7 行集 → 0.85-0.9×,F2 系统 ~4-6%。

## 资产

- 代码:HASH_TGT_ALPHA 旋钮(compute_bucket 五处 target + 签名/launch,默认 200 零行为差)
- 实验:F2/crankseg/nd24k α 扫 + F2 matched±LLB rowtime(/tmp/rowtime_F2_matched{,_llb}.txt)
