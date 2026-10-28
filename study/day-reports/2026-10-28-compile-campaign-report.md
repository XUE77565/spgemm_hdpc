# 战役报告:编译层发现 → 记分链 0.8873 → 0.8311(2026-10-27~28)

> 覆盖:study/57 §四十四~六十七,commits 81d838c→159b1df(20 个),GPU 零事故,nnz 全程逐位对拍。
> 一句话:**两天之内,b45 差距从"归因悬案"变成"主场翻面"——ocean337 主场 GM 1.099→0.9801,
> 总记分 0.8873→0.8311(−6.3%),赢面 45%→69.3%,Hook 对 ocean 正式翻面(0.990)。**

---

## 一、结果总账(confirm15_428,428 阵并集,同夜同卡 ocean 配对,compute-only 口径)

| 口径 | confirm14(10-23) | **confirm15(10-28)** | Δ |
|---|---|---|---|
| 配对 GM(n=427) | 0.8873 | **0.8311** | **−6.3% / 单日** |
| **ocean337 主场 GM** | 1.099 | **0.9801** | **首次整体翻面(<1.0)** |
| first100 GM | ~0.45 | 0.4521 | 碾压保持 |
| 赢面(赢 ocean 阵占比) | ~45% | **69.3%**(296/427) | +25pp |

**旗舰阵对 ocean 比值**(我方 compute-only / ocean total_time):

| 阵 | 10-27 晨 | **10-28 终** | 备注 |
|---|---|---|---|
| Hook_1498 | 1.154 | **0.990** | **翻面** |
| Bump_2911 | 1.105 | **1.056** | −12%,剩 5.6% |
| cage15 | 1.257 | **1.072** | −20%,剩 7.2% |
| SiO2 | 0.409 | 0.420 | 大赢保持 |

**kernel 单挑账**(对移植核天花板仪器,ocean flags 同编译):小中阵已大幅超越
(c-64 类 0.36ms vs 1.74 = **4.8×**);b45 大表阵 acc 相位已穿对方核(hash_heavy
Bump 60-66 < 天花板 61.6-71.9),差值余量在 non-acc 相位与 bins6/7 未扩档。

---

## 二、技术主线

### 2.1 破案(10-27):b45 差距全部来源 = 编译 flags
ocean Makefile 的 `-gencode=arch=compute_61,code=sm_90`(Pascal 级 PTX 生成 sm_90
SASASS,ptxas 走完全不同的指令选择/布局路径)+ fast_math(后证贡献=0)。
三日悬案"harness 提款魔力"结案;PTX 梯扫描:compute_61 是孤峰(70/80 双回落)。

### 2.2 per-file 三层架构(P1,生产形态)
- **TU-61**(compute_61):generic bins2-9 hash 主路径全部实例,**TB_TAG 模板参
  打签**(第 9 参,纯签名隔离);
- **TU-90**(sm_90 原生):special bins(bin0/批处理/esc/diter/重行)+ compact/
  extract 全家 + retry 簇;
- **三检体系**(进 `make all`):`check_disjoint`(两 TU 模板实例集零相交,
  防弱合并)+ `check_optin`(大表 tagged 实例 opt-in 覆盖,防 invalid argument)
  + 符号审计首跑即抓 22 个遗漏发射点。

### 2.3 刀 G 复活(Gv3v2):载体悖论的正解
- hash_heavy(c61 下复活)acc 穿透 ocean 核天花板,但无序写需要 counting-sort
  善后,载体税全额倒挂(四阵净亏)——刀 G 曾判死;
- **Gv3v2 = 核内有序直写**:warp 连续段认领 + 5 轮 `__shfl_xor` 位移 rank
  (popc 位掩码)+ smem 段前缀(表序)+ cap 守卫(超帽走既有 retry 语义);
  表内键无重 → rank 直写 = 行内有序 → compact 侧免排序载体(bin4/5 落 copy/FLAT,
  bin7/8 sorted 感知门 → copy_warp);
- 四阵 A/B 全绿:**Serena Σ−10.9% / Hook −15.9% / cage15 −18.5%(acc −23%)/
  Bump −7.2%**,nnz 四阵逐位同,memcheck 0 错;
- 13 阵 retry 体检无风暴 → **全部转正为生产默认**。

### 2.4 今日入账刀链(全默认形态)
```
per-file 三层(c61 tagged 大路径 + sm90 特殊路径 + 三检)
+ 刀A(bins2-3 常量折叠,九阵银行 −1.2~4.4%)
+ 刀F(compute_61,Bump acc 79.1→71.9,+9%)
+ 刀G 转正(hash_heavy bins4/5 + Gv3v2 有序直写)
+ bin7/8 有序直拷(counting 对有序输入 6.4× 病理的修复,三阵反超 v3)
+ 27 实例 opt-in 全家块(bin9 大表 196KB 雷,40 阵修复)
```

---

## 三、逐级性能证据链(全部同会话同卡,nnz 逐位对拍)

| 层级 | 证据 |
|---|---|
| kernel 单挑 | Bump 三构建 acc:sm90 79.12 / per-file 71.87 / 整c61 71.92(Δ0.07% 无损);F2 −31%/bcsstk17 −47%/c-64 −37% vs 整c61 |
| 15 阵电池(v4) | acc 全平或赢(cage15 −19.5ms/Hook −7.3/Bump −5.3/ldoor −4.9);零回归 |
| 刀G 四阵 | Σ −7.2~−18.5%,compact 回落 copy 级(20.7→7.3 等) |
| 428 记分 | GM 0.8311,主场 0.9801,赢面 69.3% |

---

## 四、方法论收获(论文级)

1. **链接层探针三定谳**(tools/arch_probe):两 TU 非模板 kernel 同名=硬链接错;
   **模板同参实例=链接过但 stub 弱合并,A 的发射静默跑 B 的 codegen**(比链接错误
   致命——构建绿数字假);`static __global__` 两 TU 各自独立=解。
2. **第五例"编译器承载一切",升至正确性级**:web-Google 的 TU-90 untagged LHT8
   smem 越界,双参考构建同 env 不崩——同一源码同 flags,实例**正确性**随编译
   单位内容(TU 边界、tag 参存在性)漂移。性能轮盘之上还有正确性轮盘。
3. **E3 类雷二捕**:新 TU=新符号=opt-in 必须镜像(bin9 tagged 196KB,40 阵批量
   invalid argument;15 阵电池无此人口故全绿)——**人口覆盖审计的必要性实证**,
   confirm 试跑自己当了审计员。
4. **口径纪律三课**:per-file 首轮 ~6ms 模块冷态 ≠ 稳态(计时一律末轮);
   TOTAL 含 d2h = 共享机噪声放大器(超线一律交错重采);`pkill/pgrep -f` 自匹配
   自杀(四踩,SOP=括号模式或 PID)。
5. **有序数据的排序载体病理**:counting sort 对(某类)有序输入 6.4× 慢
   (bin7,nsys 定位 0.59→3.78ms)——有序输出应跳过排序载体直拷,修复反超基线。

---

## 五、工程资产

- 双 TU 构建 + 三检体系进 `make all`(符号/覆盖,构建期拦截两类雷);
- `scripts/p1_battery.py`(15 阵三构建稳态电池,断点续跑,nnz 三方即断言);
- `scripts/confirm_adpt15_428.py` + `compare/batch437/confirm15_428.csv`(记分链);
- `tools/arch_probe/`(探针三件 + check_disjoint + check_optin);
- 逃生门三件:HASH_LHT23=0 / HASH_H45=0·HASH_H55=0 / HH_SORTED=0(诊断用)。

## 六、剩余空间(下轮菜单)

1. **Bump 剩 5.6% / cage15 剩 7.2%**:hash_heavy 扩 bins6/7 素数档(2719/5449
   已备,HH_LAUNCH 加档);Bump 差额部分在 non-acc 相位;
2. **输面 131 阵**(>1.10 者 103):头部 = dense/diter 路由族(bloweybq 3.07/
   hangGlider_3 2.35/lowThrust_4 2.33/TSOPF_FS 1.94/case9 1.94/atmosmodj 1.83)
   ——T6/T3 旧队列线的人口,非本战役 hash 回退;bcsstk01/02/STM07 微阵 1.3-1.5×
   小回退待单阵定位;
3. backlog:WRITE_MTX 挂钩(文件级行内有序检查)/ opt-in 审计增强 / 论文方法论
   章素材已齐(探针 P2 + ballot-sync 雷 + 第五例)。

## 七、档案索引

- 战役全档:study/57 §四十四(破案)~ §六十七(confirm15 终账);计划:study/58;
- 关键 commits:0964aa8(探针)/ea43b0c(cutover)/11c6b10(第三条路)/
  0abb8cd(刀G复活)/69f6e54(转正)/5aac3e8(opt-in)/159b1df(终账);
- 本报告:study/day-reports/2026-10-28-compile-campaign-report.md。
