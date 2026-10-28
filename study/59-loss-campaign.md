# study/59 — A+ 线战役书:输面攻坚 + b45 收尾(10-28 定稿)

> 起点:confirm15 终态 **0.8311 / ocean337 主场 0.9801 / 赢面 69.3%**(§六十七)。
> 杠杆排序:输面 131 阵(>1.10 者 103)≫ 旗舰尾(Bump 1.056/cage15 1.072)> 微账。
> 纪律:全套继承(GPU 保护/同会话 A/B/d2h 交错重采/三件套/预注册/三检构建门/清场按 PID)。

## P0|输面取证分桶(先行,半天)
131 输阵逐个采 **路由([adapt] 裁定行)× bins 人口 × 相位账** → 分桶表:
- 已知桶:T5 预门误 route(bloweybq 3.07)/T6 diter 窗口族(case9 1.94/atmosmodj 1.83)/
  旧 known-cost(Stanford_Berkeley 1.86);
- 新面孔:hangGlider_3 2.35/lowThrust_4 2.33——人群未审;
- bcsstk01/02/STM07 1.3-1.5× 微阵回退;
- 产出:`results/loss_audit_1028.csv` + 每桶→靶刀映射,定 P1/P2 主次。

## P1|dense/diter 旧队列线(预期 GM 最大收益,按 P0 分桶开刀)
T5b k7 人口门(bloweybq 专靶,四夜稳定误 route)→ T3 ESC 覆盖带(usparse 对位
33 阵@1.95×,flop 33-128)→ T4b 内环瘦身(StocF 残余)。每刀预注册 + 同会话 A/B。

## P2|b45 旗舰收尾(平行小刀)
- bins6/7 素数档扩容(2719/5449,HH_LAUNCH 加档,cage15 专靶,env HASH_H67 试点);
- **Bump non-acc 相位分解**:compute-only 84.67 中 acc 66+compact 11.5=77.5,
  ~7ms 住 count/est/binning/retry/cnnz——逐相位列账 vs 差额定位(kernel 外赢面)。

## P3|账目清零
bcsstk 微阵定位 + **路由时效性检查**(dispatcher/L0 门 = 旧相位成本训练;
hash_heavy 转正后 bins4/5 成本已变,边界重估可能有免费赢面)。

## P4|confirm16(全捆绑入账)→ 总账/MEMORY 更新。

## P5(平行)|论文线
方法论五件 + 主场翻面 + per-file 架构;**FlexSpGEMM(ACA-Lab 2026-07,ML 选
tile/硬件路径 + TC 混执,基线=cuSPARSE/spECK/TileSpGEMM/HSMU)进 related work
对表**——选择空间正交(我们=形态族+编译层),需占位;AA/AAT 双轨=同赛道信号,
保持监控。

## 时间线
- 半日 1:P0 取证(脚本后台)+ Bump 相位分解(平行)→ 分桶定主次;
- 其后按桶开刀(P1 或 P2 主攻),每刀绿即 commit,攒 confirm16;
- 收官:confirm16 + 总账盖章。

## P0/P2 中途发现(10-28 晚)
**Bump non-acc 相位分解(P2 第二项,已完成)**:compute-only 84.65 全相位对齐——
count_flop 1.34 + **mh_construct 1.22 + mh_merge 4.23** + est_scan 0.07 + binning 0.25
+ acc 65.94 + cnnz 0.10 + compact 11.49 = 84.65 ✓。
**定谳:Bump 对 ocean 的 4.5ms 缺口 ≈ mh 定容机器 5.45ms 开销**(MHSAMP=0 不关机器,
它喂 binning/tmp 定容,裸门违鲁棒铁律)。**P2 新刀=定容管线降成本**:pregate 强判
hash 的阵跳过 MinHash sketch,est 退 cheaper 上界(需守 tmp 定容/retry 语义);
预期 Bump 84.65→~79 < ocean 80.19 翻面。ocean total_time 含其自家 sizing——
对比合法,我方 sizing 更贵是真实差额。
