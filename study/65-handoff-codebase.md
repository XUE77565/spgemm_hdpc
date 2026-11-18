# 交接文档:SpGEMM 项目代码结构与整理指南(11-18)

> 给师兄及其 agent:本文档包含项目全貌、运行流程、以及代码整理的具体任务书。
> 配套阅读:`README.md`(目录地图)+ `study/64`(设计点总账)+ `study/63`(总计划)。

---

## 一、项目一句话

GPU 上的稀疏矩阵自乘(C = A·A 和 C = A·Aᵀ),NVIDIA H100,CUDA C++。
核心思想:**不同形状的行用不同 kernel**——按行人口分派到多种累加形态。

## 二、目录地图

```
spgemm-challenge/
├── src/                    ← 全部 CUDA 源码(整理对象)
│   ├── spgemm_kernel_hash.cu   7645 行 ★最大文件,整理重点
│   ├── spgemm_merge.cu         1202 行  merge3(列域分桶归并)
│   ├── att_tiered.cu            970 行  A·Aᵀ 三层实现
│   ├── spgemm_kernel_formulations.cu 483 行 ESC 基线
│   ├── main.cu                  466 行  入口+命令行
│   ├── spgemm_kernel_manual.cu  425 行  手工基线
│   ├── structural_params.cuh    408 行  编译期常量(表长/档位/阈值)
│   ├── spgemm_kernel_cusparse.cu 316 行 cuSPARSE 基线
│   ├── spgemm_kernel_struct.cu  222 行  dense 档位核
│   ├── spgemm_adaptive.cu       166 行  方法层 dispatcher(logit)
│   ├── ai_router.h               40 行  AI Router v3 权重(opt-in)
│   ├── routing_table.h           61 行  bin 路由查表
│   └── ocean_hash_port.cuh      188 行  ocean 移植核(仅测量用,不生产)
├── data/                   ← 基准矩阵(.mtx,不入库)
│   ├── ocean/square/           337 个(ocean 挑的)
│   └── first100/               100 个(SuiteSparse 经典)
├── compare/batch437/       ← 记分和基线数据(核心产出)
├── scripts/                ← 运行/分析脚本(~170 个)
├── tools/                  ← 测量台(.cu 源码)
├── study/                  ← 战役档案(01-64,全部设计决策的出处)
├── results/                ← 实验产出(gitignored,审计表除外)
├── ocean/                  ← ocean 基线源码(含 convert+spgemm 二进制)
├── external_sota/          ← 四套外部基线(HSMU/TileSpGEMM/FlexSpGEMM/MMSpGEMM)
├── Makefile                ← 构建入口(见下文)
└── spgemm_test             ← 认分二进制(GM 0.8054,勿重建!)
```

## 三、构建与运行

```bash
# 构建(注意:会覆盖根目录的 spgemm_test!)
make                                    # → spgemm_test (sm_90 + per-file c61 双 TU)
make DBG=1 TARGET=spgemm_test_alias     # → 别名构建(不覆盖,打扫战役的标准做法)

# 运行单阵(标准 env 组)
METHOD=adaptive WARMUP_N=3 USE_MEMPOOL=1 MP_HOST_MB=8192 D6EXP=0 \
  ./spgemm_test data/ocean/square/Bump_2911.mtx

# 全量 428 confirm(认证跑)
# 见 scripts/confirm_cleanup_428.py

# 配对 A/B 电池(16 阵×N 轮,改代码后的守门)
# 见 scripts/ab_pair.py
```

### 关键构建细节

- **per-file 双 TU**:hash.cu 编两遍(`-DHASH_TB=90` 走 sm_90,`-DHASH_TB=61` 走 compute_61),
  大表实例在 c61 下快 9%(刀F,study/57 §四十四)。
- **Makefile 无头文件依赖追踪**:改 `.h` 后必须 `touch` 对应 `.cu`,否则旧 `.o` 假象。
- **符号审计门**:`make` 自动跑 `check_disjoint.sh + check_optin.py`,防同参实例弱合并。

## 四、运行时全流程(一图)

```
./spgemm_test matrix.mtx
    │
    ▼
┌─ main.cu:读 .mtx → CSR ─→ GPU ─┐
│                                   │
▼                                   ▼
┌─ spgemm_adaptive.cu:方法层 dispatcher ─┐
│  特征:n, nnz, maxrow, skew           │
│  logit(训练分离)→ score               │
│  score<0 → hash | score≥0 → merge3    │
│  预门:n≥5000 → 强制 hash(省自税)      │
└──────────────────────────────────────┘
    │                        │
    ▼                        ▼
┌─ hash 路径(主路)──┐  ┌─ merge3 路径 ─┐
│                     │  │ 列域分桶归并  │
│ ①估计相:            │  └───────────────┘
│   MinHash 无偏估计   │
│   → est[]/行         │
│                     │
│ ②分箱相:            │
│   est→bin 0-10      │
│   (h_cnt[]人口)     │
│                     │
│ ③逐bin发射:         │
│   LHT/hash_spa梯    │
│   hash_heavy(素数)  │
│   batched/DIRECT5   │
│   csort/sorted直拷  │
│                     │
│ ④紧致相:            │
│   FLAT/csort/copy   │
│                     │
│ ⑤溢出→retry         │
└─────────────────────┘
    │
    ▼
┌─ 提取/输出:CSR 格式 C ─┐
└────────────────────────┘
```

### 三种累加形态(论文创新点的物理载体)

| 形态 | 内核 | 何时用 |
|---|---|---|
| 哈希表 | hash_spa 梯 + hash_heavy | 中重行(bin 1-9) |
| 稠密直寻 | dense_struct + DIRECT5 | 稠密行(span 门) |
| 归并 | merge3(列域分桶) | 小阵/straggler |

### 路由层次(从粗到细)

1. **矩阵级**:dispatcher logit → hash/merge3
2. **人口级**:est→bin → 每bin配内核/表长/块宽
3. **行级**:CS2 d门(rank vs count-sort)、cursor/search(avgB)、DSTRUCT 档位(span)
4. **编译级**:per-file 双 TU(大表走 c61)

## 五、⚠️ 红线(动之前必读)

1. **认证二进制**:根目录 `spgemm_test` = GM 0.8054,**勿用 `make` 覆盖**(用 `TARGET=别名`)。
2. **六个承重旋钮**:T4c/T4e/T6/DSTRUCT/DIRECT5/LLB,勿删勿改默认值。
3. **护栏类旋钮**:溢出保护/retry 语义/TB_TAG/opt-in,永不路由。
4. **改代码后守门**:`make TARGET=alias` → `scripts/ab_pair.py`(16阵×3轮)→ 绿才入账。
5. **GPU 纪律**:报错即停;微阵(<1ms)10次交错取中位;同日配对。

## 六、整理任务书(师兄的活)

### 任务 A:hash.cu 按相位拆分(7645 行 → ~5 个文件)

当前 hash.cu 是一个 7645 行的单体,包含全部 hash 路径的 82 个内核+发射逻辑+守卫。
建议拆为:

```
src/
├── hash_estimate.cu      ~800 行  ①估计相(MinHash/DCEST/est_expand)
├── hash_binning.cu       ~600 行  ②分箱+路由(bin ladder/RTABLE/AI Router)
├── hash_accumulate.cu   ~3000 行  ③累积核(hash_spa 梯+hash_heavy+batched)
├── hash_compact.cu       ~800 行  ④紧致+提取(csort/sorted/FLAT/CS2)
├── hash_overflow.cu      ~400 行  ⑤溢出/retry 语义
├── hash_dispatch.cu     ~1200 行  发射逻辑+per-bin 循环(host 侧主管线)
└── hash_common.cuh       ~800 行  共享类型/宏/设备函数
```

**约束**:
- 拆分是**物理移动**(不改语义),但会重洗编译布局 → **必须整链电池+confirm**;
- 每次 `make` 自动跑符号审计(check_disjoint),拆分后仍需通过;
- 两个 TU 编译(-DHASH_TB=90/61)的分区必须保持(大表实例在 c61 侧)。

### 任务 B:残余死码清除(~92 行 + 休眠层)

已知残余(打扫战役 b1-b4 后,study/64 §四):

| 类 | 内容 | 行数 | 处置 |
|---|---|---|---|
| 孤儿内核 | count_flop_range_tiny(16行)/hash_row_bcnt(51)/elig_span(9)/bin_alen_stat(8) | 84 | 直接删(零发射) |
| 恒零变量 | h_t3_end/h_t3_w(声明不写)/h_d3_n/d_d3_rows(恒零比较) | ~20 | 删声明+摘除3处恒真条件 |
| 惰性守卫 | g_dtier_single/g_segrts 的 decl/guard 残对 | ~8 | 删 |
| 休眠层 | WMA族/PHT/SPEC99_OCH/ROT/参数穿引族~30旋钮 | ~500 | **标记`[DORMANT]`注释,暂不删**(逃生门) |

### 任务 C:每个 if-else 挂出处注释

```cuda
// [出处] 类型:测(电池标定) | 档:study/53 T4e | 验证:confirm14-16 | 论文槽:§5/G4
if (skew >= 100.0 && skew <= 1900.0) ...
```

纯注释,零 codegen 变化,不需要电池。格式见 `study/64 §四` 的 G1-G7 映射。

### 任务 D:II 类旋钮烤死(8 件,需用户逐件点头)

默认≈oracle 已实测证明的 8 个(HH_SORTED/LHT23/CSORT_CNT/BLK_LADDER/COMPACT_FLAT/
est_expand档/EST_CONT/DIRECT5)→ 转正为代码常量+删 env。

### 完成后的守门流程

```
1. make TARGET=spgemm_clean_v2     (别名构建)
2. check_disjoint + check_optin    (自动)
3. python scripts/ab_pair.py spgemm_test spgemm_clean_v2 results/ab_v2.csv 3 1
   → nnz 16/16 逐位同 + GM≤1.02 = 绿
4. 绿 → git commit;红 → 回滚
5. 全部完成后 → 428 confirm 重认证
```

## 七、当前数据资产(勿删)

| 文件 | 内容 |
|---|---|
| `compare/batch437/confirm_cleanup428.csv` | 我方 428 阵当前时间(cleanup 后) |
| `compare/batch437/baseline_flex.csv` | Flex 428 阵时间 |
| `compare/batch437/baseline_hsmu.csv` | HSMU 428 阵时间 |
| `compare/batch437/baseline_tilespgemm.csv` | TileSpGEMM 428 阵时间 |
| `compare/batch437/confirm18_428.csv` | confirm18 我方+ocean 同日配对 |
| `compare/flexbench142/pair.csv` | Flex 自家 bench 我方+Flex 对决 |
| `compare/flexbench142/ocean.csv` | Flex 自家 bench ocean 时间 |
| `results/env_audit.csv` | 150 旋钮生死簿 |
| `results/kernel_audit.csv` | 内核生死簿 |

## 八、论文现状快照(11-18)

- **创新点四条**(查新封版,study/64 v4.1):有序直写+hash_heavy / 无偏 MinHash /
  LLB 行内粒度 / merge3 列域分桶
- **成绩**:448 阵四基线 vs ocean GM=0.822/69% vs Flex 0.361/88% vs HSMU 0.235/97%
- **投稿**:SC 2027(CCF-A,~4月截稿,artifact track)
- **进行中**:A线成本公式(G1-G7)推导 / AC 代码三批 / fig3 策略面板

---

*文档版本:11-18;上次全量更新:11-16(Flex bench 三方对决入账)。*
*如有疑问查 `study/64`(设计点)+ `study/63`(计划)+ `README.md`(目录地图)。*
