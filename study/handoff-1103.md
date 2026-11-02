# 交接档:打扫→判决→扩建战役开工前(11-02 深夜)

> 新会话起手读本档 + study/next-phase-plan.md。
> 状态:**等用户指令开工,未开始任何打扫操作。**

## 冻结区(碰之前必须用户点头)

- 记分二进制 spgemm_test(GM 0.8054)/ master 与 paper 分支的生产路径
- 六个承重旋钮不删:T4a/T4c/T4e/T6/DSTRUCT/DIRECT5(电池要翻转它们)
- GPU 保护纪律照旧(报错即停/Xid 哨兵/带病退役)

## 开工第一刀(用户发令后)

```
第 0 步 审计标注(半天,零风险):
  grep -oP 'getenv\("\K[A-Z_0-9]+' 各 .cu → 全量清单
  逐个标 Ⅰ/Ⅱ/Ⅲ/Ⅳ + 交叉核对 scripts//tools/ 引用
  产出 results/env_audit.csv(132+ 行处置表)
红线自检: T4a/T4c/T4e/T6/DSTRUCT/DIRECT5 标 Ⅰ(电池后再判);
          测量台在 set 的旋钮标"工具引用"不可删
```

## 已知雷(新会话必读)

1. **Makefile 无头文件依赖**:改 ai_router.h 后必须 `touch src/spgemm_kernel_hash.cu`
2. ai_router.h 导出已加 std 下限 1e-3(防 0*inf=NaN)
3. 大 smem 核 opt-in 是逐模板实例的(E3 家族);动发射链后必跑 check_disjoint.sh
4. 电池 env 直跑会挂 cuSPARSE(老坑):跑 spgemm_test 用标准 env 组
   (METHOD=adaptive WARMUP_N=3 USE_MEMPOOL=1 MP_HOST_MB=8192 D6EXP=0)
5. pkill 模式会自杀:清场用 PID
6. 微阵(<1ms)CV~19%:配对判读必须看显著性,不看单对比值

## 昨日资产(都在 git)

- 完整战役链 commit e01ec98..9730baa(训练修复→漂移→部署→held-out→§5→图→整理→计划)
- AI Router 已部署:AIROUTER=1 opt-in(插点 :6374 决策块/:5497 覆写)
- 可复跑实验:scripts/ai_drift_dual.py(GPU~25min)/ heldout_router_ab.py(~30min)
- 汇报讲稿要点在 11-02 对话;论文 §5=study/paper-draft-sec5.md(v2)

## 并行未了项(非阻塞,记着)

- 基线电池产物未提交:baseline_hsmu.csv(改)/baseline_tilespgemm.csv(新)/phase_ablation(HSMU/TileSpGEMM 线)
- 外部 72 大阵下载 403(泛化上限钥匙,可重试)
- 漂移实验加固(N 14→40 + bin9 点,~1h GPU)
- dispatcher 影子评估(离线,数据在 variant_combo)
