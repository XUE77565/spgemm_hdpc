# docs/75 v31 进攻路线(Ocean 残差解剖 → 四项攻击;2026-09-04 用户钦定入 roadmap)

## 残差分解(v30 五件套定版后,按族)

| 族 | v30 实测 | Ocean | 残差 | 主因 |
|---|---|---|---|---|
| road/OSM(~10 阵) | road_usa 28.4 / germany 11.5 | 11.9 / 5.3 | 2.2-2.4× | 管线遍数税 + 短行表税 |
| pre2/TSOPF/in-2004 | 46.2 / 29.8 / 86 | 15.2 / 10.2 / 32 | 2.6-3.0× | 符号槽 12B vs 8B + 块宽不随档 |
| c-big | 107 | 60 | 1.79× | largest 已攻,余量 = 管线/压缩 |
| 34 赢阵 | ≤1× | — | — | dispatch 层不输 |

geomean 残差 ~1.32× ≈ 前两族贡献。目标:v31 → ~1.15×,v32 → 冲 1.0×。

## Ocean 六件结构性优势(代码实据)

1. **一遍管线**:analysis 一遍出全(乘积数+sketch+binning 信息),估值 7%;我们 count_flop→mh_construct→mh_merge→est_scan→binning→scatter 六步,road 族 24M 行 × 每步 0.5-1.5ms
2. **符号哈希 8B/槽**:key 压位(32−log2 表宽)与 value 打包(HASH_NUMERIC_IDX_BITS);我们 12B → 容量/占用 1.5× 差
3. **七档占用梯**:CONCURRENT_BLOCKS_PER_SM={32..2} → 档位↔块宽↔SMEM 三联动,symbolic/numeric 两套;我们块宽固定
4. **ESC counting-sort 零表吃短行**(avg 乘积<64):countSortCompactFused = 位次计数 + ballot,零表零清零;≡ 我们 csort 的机制,但他们当独立累加器
5. **成熟溢出兜底**:HLL 32/64 寄存器溢出率 1.2%/0.3% + fallback kernel;我们 retry=flop 定表(重)
6. **CR 阈值化 compact**:只在压缩比划算时 compact;我们每行必过

## 四项攻击(任务 #19-22,ROI 排序)

| # | 项 | 攻击对象 | 工程量 | 预期 |
|---|---|---|---|---|
| ① | 管线遍数合并:count_flop×mh_construct 融合 → est_scan×binning×scatter 融合 | road 族 | 大 | road −20~30% |
| ② | 8B 符号槽位压缩(key 位压缩打包) | 中行族 | 中 | 中行族 −15~25% |
| ③ | 占用梯联动块宽(64T..512T 随档) | 全体 hash 档 | 小 | 组合② |
| ④ | ESC 式零表短行内核(est≤16) | road 甜区 | 小 | ultra 档 −10~20% |

执行原则(不变):env 门控 → 小口子 → 净窗电池(nnz 红旗+中位)→ 赢者翻默认;
全参数结构化具名(milli-byte/位宽推导),零裸数字铁律。

## 顺延队列

- v30 出终版(负载门)→ first100 复测(#18)→ merge 边界 v3 电池 → bhSparse 重行头对头
- largest 精确管线集成(cage15 类 flop-retry 内存饥渴)
- CR 阈值化 compact(第 6 件,远期)
