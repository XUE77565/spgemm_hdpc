# T10 广域电池终报(52 阵 vs ocean,10-05 22:05 收官)

## 口径声明(双声明,用户钦定)

- **我方**:HashProf 相位和(17 相位全清单:count_flop/mh_construct/mh_merge/est_scan/binning/
  accumulate/retry/cnnz_scan/compact+sort/dense_count/dense_direct/hash_count/hash_direct/
  oc_rescale/oc_samp),默认 warmup(warm),**禁 WARMUP_N=0 冷 JIT**
- **彼方**:ocean bench 迭代 2-10 中位(ms)
- **已知口径税**:相位和 vs 净窗(认证 T2 口径)平均高 **7.8%**(锚阵校准:c-64 +16/F2 ±0/
  pre2 −28,非统一但中位 +7.8%)
- **事故与修复**:①K1 残留布局污染(P2 commit 6f13753,前 8 阵作废重跑)②解析器两轮
  漏相位修正(11→17 相位)③delaunay_n21 历史问题阵 illegal access 剔除(09-13 档案背书)
- 哨兵 ECC 0 首尾;零 rc 错误(delaunay 除外)

## 终局比分

**52 阵几何均值 1.653 | 赢 4 平 2 输 46**

### 赢区(4)
3Dspectralwave 0.684 / dielFilterV3clx 0.831 / invextr1_new 0.952 / bone010 0.978

### 病根分类(全 52 阵)

| 类 | 阵数 | 几何均值 | 代表 |
|---|---|---|---|
| **dense 主导(dd>40%∧dd>acc)** | **23** | **2.073** | c-43 3.64/pli 3.76/Si10H16 3.38/bloweybl 2.97/opt1 3.03 |
| **hash 主体(acc>40%)** | **29** | **1.381** | af_2 1.96/nd24k 1.86/FEM_3D 1.72/crankseg 1.69 |

### 大输阵病理解剖(抽样)

| 阵 | 比值 | 绝对值 | dense_direct 占比 | 形态 |
|---|---|---|---|---|
| c-43 | 3.64 | 6.2 vs 1.7ms | 65% | 小阵(123K nnz) diter 3.2K avg_B 5395 |
| bloweybl | 2.97 | 12.5 vs 4.2ms | 59% | 小阵 diter 10K avg_B 3349 |
| email-Enron | 2.46 | 8.3 vs 3.2ms | 57% | 小阵 diter 2.6K avg_B 1404 |
| Si10H16 | 3.38 | 7.4 vs 2.1ms | 63% | 小阵 diter 3.5K avg_B 6574 |

**共同病理:小阵 + diter 行走 dense_direct + 不摊销的固定开销(launch/SMEM/估计器链)**

### 路由修正模拟

- 若 23 个 dense 主导阵的 dense 部分改判 hash(按 nd24k 错配税经验 hash≈dense×0.6):
  **几何均值 1.653 → 1.404**
- 再除口径税(÷1.078):**→ 1.303**

## 锚阵核对

| 锚 | T10 相位和 | T2 净窗认证 | 偏差 |
|---|---|---|---|
| c-64 | 1.134 | 0.974 | +16% |
| nd24k | 1.856 | 1.734 | +7% |
| F2 | 1.009 | 1.013 | ±0 |
| msc23052 | 1.527 | 1.171 | +30% |
| crankseg_2 | 1.692 | 1.560 | +9% |
| pre2 | 1.338 | 1.847 | −28% |

偏差非统一 → 相位和与净窗测的不是完全同物(相位和含全部 GPU 相位,净窗 = TOTAL−h2d−d2h,
两者在小阵/大 d2h 阵上分歧最大)。**方向自洽,绝对值有 ±15% 口径差**。

## 关键发现

1. **广域输的主体 = 23 个 dense 主导小阵(GM 2.07)**——不是 kernel 质量问题,是
   **小阵 dense 路由固定开销不摊销**(diter 行才 2-10K,绝对值几 ms)
2. hash 主体阵 GM 1.381——与 7 锚认证一致(其中已知输区:nd24k 型自旋/pre2 型截距)
3. **"优化前没这么差"的校准**:v29 时代 1.35 = 自家版本对比,非 vs ocean;这是**第一次
   诚实广域 vs ocean 的基线**
4. 路由修正后估计 ~1.30-1.40(若 dense 主导小阵改判 hash 成功)

## 数据与工具

- 数据:/tmp/t10_our_{m}_{r}.err + /tmp/t10_oc_{m}_{r}.log(52 阵 ×3 双侧)
- 解析:/tmp/t10_analyze.py(17 相位修正版)
- 样本:/tmp/t10_sample.txt(52 阵分层;wb-edu 崩溃/delaunay_n21 illegal access 剔除)
