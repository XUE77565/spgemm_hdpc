# 双电池终审 + 正确性收尾(2026-09-17)

> 一句话:**重锚电池(RF/CAS)系统级 geomean ≈ +0.05% = 平——机理级赢(accumulate −4~−13% 与 hash 重阵系统级兑现:nd24k +4.5/crankseg +2.7/msc +2.1)被相位稀释与噪声带拉平;v40 电池系统级中性(全阵 ±1.1%)——v38 已坍掉重试体积,重试相位仅占 1-5%,v40 的回报点在压表解锁(③)。**

## 重锚电池(ab_env "HASH_PROBE_RF=0" ×5,10 阵;off=RF 默认,on=CAS=v38 语义,同排序合同)

nd24k +4.5✓ / crankseg_2 +2.7✓ / msc10848 +2.1✓ / Ga3 +0.6✓ / c-64 −0.3 / Zd −1.0 / F2 −1.3 / web-Google −1.7 / in-2004 −2.5 / c-43 −2.8;**geomean ≈ +0.05%**。

**论文口径终审**:v39 原"系统级 −2.4%"声明作废三重:①T4f 空对照 ②免排序水分 ③本电池 = 干净重测为零。诚实叙事 = "read-first 机理 kernel 级赢、hash 重阵系统级兑现、整体系统平"。

## v40 电池(ab_env "RETRY_SMEM=0" ×5,11 阵含 pre2)

全阵 ±1.1% 噪声带,geomean ≈ 0。相位级 retry −84~92% 真实但 v38 后该相位体积已小。**v40 定位 = 保险经济学解锁器(压表前置)+ 鲸鱼路排雷载体,非直接系统赢**。

## 正确性收尾(路线①)

- cuSPARSE 对稠密行大阵恒 error 11 → crankseg/in-2004/nd24k/Ga3/c-64 改用 **scipy 17 位真值**(16-38s/阵)
- 值级终态:pre2/msc/c-43/F2/web-Google/Zd/in-2004/**crankseg_2(17 位 PASS,ESC 修复验收)** = 8 绿;nd24k/Ga3/c-64 的 FAIL 经裁定 = **100% 6 位舍入假象**(坏点全部比值=1.0),17 位复验中
- writer 6 位精度雷修复(setprecision(17),commit 6861b3c):历史全部 PASS 实为 6 位口径——粗错类捕获(2×/1.5×/denormal)仍有效
- diter 雷#1:数值相 rank 塌缩竞态(DDUMP 探针:out==cap 但尾槽 0、轮间漂移;search/cursor 同败=共享 prefix/emit 链);**默认关止血**(67cea7d),根因排队

## 纪律沉淀

- 电池脚本 stdout 块缓冲:后台跑时结果退出才落盘,勿误判卡死
- 真值文件要核对生成器年代(msc"仍 FAIL"乌龙 = 6 位陈旧真值)
