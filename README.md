# APPT SpGEMM Challenge — GPU 稀疏矩阵乘的自适应加速

SuiteSparse 真实矩阵上的 SpGEMM(C = A·A 与 C = A·Aᵀ),NVIDIA H100 / CUDA C++。
**当前认证分:428 阵并集 GM 0.8054**(相对基线整体快 ~20%,confirm16,二进制冻结)。
体系 = 三形态内核族(hash 累积 / dense 直写 / merge3)× 三层路由(方法/人口/编译档)
× 物理先验+学习残差的自适应路由器(AI Router,漂移自适应)。

## 新会话/久未跟进,从这四份开始

1. **`study/61-ultimate-plan-v4.md`** — 终极计划 v4(总路线:打扫→判决→扩建→签名→收口)
2. **最新 handoff**(study/handoff-*.md,取日期最大者)— 冻结区/已知雷/开工第一刀
3. **`study/routing-decision-taxonomy.md`** — 132 个 env 旋钮的四类分类法
4. **`paper/hash-kernel-wins/01-wins-ledger.md`** — 刀族总账(每把刀的出处与数字)

## 目录地图

| 目录 | 内容 | 入库 |
|---|---|---|
| `src/` | CUDA 源码。**冻结纪律:记分二进制的源,改动须过电池+confirm** | ✅ |
| `papers/` | 参考论文 PDF(Harmonia / IA-SpGEMM / spECK) | ✅ |
| `study/` | **主档案屋**:NN 编号战役档(01-61)+ 论文草稿/图 + `legacy-inno/`(07~10 月旧设计笔记 94 份)+ `report-1021/`(10-21 汇报材料)+ day-reports/ | ✅ |
| `docs/` | 编号讲解文档(76-81,小白向:merge3/kernel 版图/PVD 等) | ✅ |
| `paper/` | 论文工程:DATE 初稿、hash-kernel-wins 总账、创新点 | ✅ |
| `fig/` | 旧论文/答辩图(plot_*.py 生成) | ✅ |
| `scripts/` | 运行/对比/训练脚本(~158,`archive/` 已归档 38) | ✅ |
| `tools/` | 测量台 .cu 源(realdata_bench / calib_station / phase_ablation 等;编译产物已 gitignore) | 源✅ |
| `compare/` | 对比数据。**`batch437/` = 428 主账**(我方 confirm 链 + HSMU/TileSpGEMM 基线);`d1_oracle/` = L0 时代 2.6G 逐阵数据(活引用,不动) | ✅ |
| `results/` | 实验产出(csv/json/日志) | ❌ gitignore |
| `data/` | 基准矩阵 .mtx(96+;ocean337/first100 等) | ❌ |
| `test/` `include/` | 自生成阵测试 / 公共头 | ✅ |
| `ocean/` `spECK/` `nsparse/` `external_sota/` `suitesparse_crawl/` `ref/` `SOTAsummary/` | 外部 SOTA 基线源码与爬取产物(口径对齐用) | ❌ |
| 根目录二进制 | `spgemm_test`(**0.8054 记分冻结,勿重建勿删**)+ 电池锚点(c61/sm90/437 等,配方见 .gitignore 注释) | ❌ |

## 冻结区(动之前必须用户点头)

- 记分二进制 `spgemm_test` 及 master/paper 生产路径;六个承重旋钮 T4a/T4c/T4e/T6/DSTRUCT/DIRECT5
- GPU 保护纪律:报错即停 / Xid 哨兵 / 带病二进制退役 / 清场按 PID(pkill 会自杀)
- **禁改写 git 历史**(有 GitHub remote);测量数据即档案,不删 compare/

## 编译

```bash
make                    # → spgemm_test(主二进制;⚠️ 会覆盖冻结记分二进制,重建前确认!)
make dense              # → spgemm_dense(tiled64 基线)
make cublas             # → spgemm_dense_cublas(cuBLAS FP64 基线)
```

## 计时口径

compute-only = cudaEvent `TOTAL − h2d − d2h`;各基线同口径详见 `scripts/compare_methods.py`。
实验纪律:同会话同卡配对 A/B;微阵(<1ms)10 次交错取中位(CV~19%);刀级裁定只认配对。

## 分支约定

- **paper = 实际主干**(守门后逐刀提交);master 名存实亡不再使用
- 计时里程碑打 timed-* tag,测量二进制只从 tag 冻结
