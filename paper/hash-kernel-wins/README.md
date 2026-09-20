# Hash kernel 赢点与创新点总结（2026-09-18 定格）

> 范围：`src/spgemm_kernel_hash.cu` 全家族（hash_spa / hash_dense_direct(cursor+dense) /
> csort 族 / LARGEST / ESC / batched / retry / 分箱路由）——对 ocean(ICS'26) 的全部实测赢面
> 与可辩护创新点。论文/PPT 素材入口。
>
> **一句话结论：我们守住的是"机制性聪明"（游标、分组、判据、无偏估计、占用物理），
> ocean 占住的是"结构性便宜"（槽窄、档位多、记号小）；09-18 三把占用刀之后，
> c-64 已系统级翻盘（0.981×），F2 距跨线一步（1.045×）。**

## 文件索引

| 文件 | 内容 |
|---|---|
| [01-wins-ledger.md](01-wins-ledger.md) | 赢点台账：系统记分牌 / ×7 加固 / 版本弧线 / 关键单刀 / 口径层赢面，全部带证据指针 |
| [02-innovation-claims.md](02-innovation-claims.md) | 创新点：机制创新 / 方法论创新 / 借鉴后超越 / 论文定位与查重防线 |
| [03-paper-narrative.md](03-paper-narrative.md) | **论文级叙述**：主论题 + 英文 contributions 块 + 六条贡献全文（Challenge→Insight→Mechanism→Evaluation→Novelty）+ 头条结果 + 新颖性防御矩阵 + 叙事弧线 |

## 记分牌速览

**最新定谳层①：kernel 级 hash vs ocean 三阵全胜【09-18 凌晨，口径 C（窗=探测循环+块宽对齐+逐行 clock64）】**

| 阵 | 中位逐行比 | 胜率 |
|---|---|---|
| F2 | **0.356** | 94.1% |
| c-64 | **0.906** | 57.1% |
| crankseg_2 | **0.581** | 91.1% |

（同日系统吞吐层仍 1.21-2.15 输——路由+管线+占用对消，两口径答不同问题，详见 01 §〇）

**系统级（09-18 终态，相位和口径，双边新数据）**

| 阵 | 我方 | ocean | 比值 | 状态 |
|---|---|---|---|---|
| c-64 | 20.41 | 20.80 | **0.981** | ✅ **系统级首胜**（×5；官宣值 0.979，×7 轮 21.20 挂星号待净窗） |
| F2 | 8.01 | 7.66 | 1.045 | retry 刀（0.58ms）即跨线 |
| crankseg_2 | 50.03 | 27.29 | 1.832 | ③ occgate 阵地 |
| msc | 3.02 | — | — | 前端 count 反超 3×（7.6→0.49ms） |

全弧线 ×7 加固（无刀 c2t → 全刀 c78t）：**msc −14.3 / c-64 −18.0 / crankseg −15.7 / F2 −21.9，四阵全胜**。

## 证据索引

- 三刀规格+判决：`inno/hash_opt_2026/82-launch-bounds-c64-dense.md` / `83-launch-bounds-hashspa.md` / `84-csort-occupancy-design.md`
- 追平作战地图（v28→v34 全程）：仓库根 `76-us-vs-ocean-gap-analysis.md` §1-27
- 机制对照（小白版）：仓库根 `77-us-vs-ocean-dense-hash-explained.md`
- 08-25~26 首役（4.04× 起步）：`inno/hash_opt_2026/README.md` + 01-30 系列规格
- 相关工作定位：`study/23-harmonia-survey-20260918.md` §4/§6
