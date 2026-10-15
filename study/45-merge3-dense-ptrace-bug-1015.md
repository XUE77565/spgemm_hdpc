# study/45:merge3 dense 分支双雷(dfl/weight 别名 + smem 出处丢失)——10-15 审计抓获并修复

> 发现路径:SASS fire-and-forget 审计(study/44 §六 #1b)追"多余的 ATOM.E.ADD.F64"
> → 溯源到 bucket_merge(_flop)_kernel dense 分支 → 最小复现定位机理 → 读源码发现第二雷。
> **运行期守门未跑(GPU 满载),修复目前只有编译期验证**——见 §五验证清单。

## 一、两雷同根:一个指针表达式

```c
// 08-31 落地(b9137a6)以来的原码,两个 numeric kernel 同病:
unsigned char *dfl = (unsigned char*)(smem + 2 * num_k);                    // 雷①
double *dval = (double*)(((uintptr_t)(smem + 2 * num_k) + L + 7) & ~(uintptr_t)7);  // 雷②
```

**雷①(dfl 与 weight 别名=正确性)**:`weight = (double*)(smem + 2*num_k)` 同址!
dense 分支先 `dfl[c]=0` 清 L 字节 = 把 weight[0..L/8) 清零,再 `w = weight[p]` 重读
→ 污染段乘积贡献全为 0;L < 8·num_k 时 dval 区还压进 weight 尾部,累加互相践踏。
**nnz 模式仍对**(dfl 照常置 1)→ nnz 检查/计时全不报 → 典型"值盲"检查器盲区
(守门三件套 09-16 才立,此分支 08-31 落地)。
**雷②(uintptr_t 遮罩丢 smem 出处=性能)**:最小复现(smemtest.cu)实测:
- 纯 int 指针算术 → `ATOMS.CAST.SPIN`(片上)✓
- uintptr_t 位运算遮罩 → `ATOM.E.ADD.F64...STRONG.GPU`(每乘积一次 **L2 全局原子往返**)✗

即 dense 分支的设计意图(smem 累加,L≥128 地址地板防争用)从未真正生效过——
08-31 的 2.26× numeric 战报是**付着全局原子税**打出来的。

## 二、修复(10-15,两 kernel 同刀)

```c
unsigned char *dfl = (unsigned char*)(smem + 4 * num_k);                      // weight 区之后
double *dval = (double*)(smem + 4 * num_k + (((L + 7) >> 3) << 1));           // 纯 int 算术保出处
```

- 预算复核:新布局末端 16·num_k + align8(L) + 8L ≤ 16num_k+9L+7 < host 预算
  `smem_merge + 9L + 16` ✓(num_k ≤ max_row_nnz);
- **编译期验证(已过)**:重编 TU → bucket_merge / bucket_merge_flop 两 kernel
  `ATOM.E.ADD.F64 = 0`,`ATOMS`(片上 CAS)= 1 ✓;对照复现证明两形态语义等价。

## 三、count kernel 排查:干净

bucket_count_kernel 的 flags 分支:无 weight 数组、纯写无原子 → 无雷。✓

## 四、爆炸半径评估(结论稳定纪律逐项对账)

| 数据/主张 | 是否受染 | 依据 |
|---|---|---|
| 337 权威记分牌(C 1.2714/A 1.2301/oracle 1.1846,10-13/14) | **否** | 337 大阵 merge3 无生态位(09-03 定谳),三臂批(C/CD/A)无 merge3 臂;权威批复核时再确认路由日志 |
| first100 0.6224(七月 auto0727) | **否** | dense 分支 08-31 才落地,七月二进制无此代码 |
| 09-03 复测 0.614×/59 阵 | **计时可信,值级待重验** | post-dense;计时不受值污染影响(性能 bug 只会更慢);该批复测若只验 nnz/时间则未暴露 |
| merge3 dense 直写战报(count 5.2×/numeric 2.26×,08-31) | **计时成立且是带税跑的** | 修复后 dense 分支预期更快(隐藏利好) |
| **437 权威批(未跑)** | **必须带本修复跑** | merge3 路由臂在批内;带病跑=权威数据作废——**此雷在权威批之前被抓纯属侥幸** |

## 五、验证清单(GPU 释放即跑,承 study/42 10-15 队列)

- [ ] 值级门:dense-eligible 矩阵(L≥128 桶存在)修复前后 out_val 逐位对比 +
  与真值(A·A 参考实现)对比;预期修复前**值错模式必现**(全零/部分零贡献);
- [ ] nnz 门:修复前后模式逐桶一致(应零差异——bug 不影响模式);
- [ ] memcheck 0 err(新 smem 布局);
- [ ] first100 小电池计时:merge3 dense-eligible 阵的 numeric 相位预期可观下降
  (全局原子→片上);0.614 有机会再降;
- [ ] 权威批前置条件:本修复进 timed tag。

## 六、方法论入账

1. **性能审计挖出正确性雷**:两雷共享一个根(指针表达式),追多余的原子的
   "不划算感"直接把别名翻了出来——审计线索无大小,只有跟不跟到底;
2. **检查器盲区第 5 形态坐实**(值盲:路径不在值级门的矩阵集里);
3. **复现定谳流程**:主 binary SASS → 定位 kernel → 上下文解码 → 最小复现
   (redtest/smemtest 各 10 行)→ 机理定谳 → 修复 → 编译期验证 → 运行期守门挂队列
   ——全程零 GPU,可复用为标准流程。

## 七、共卡预跑值级门的安全裁定(10-15 晚,用户问"安全吗")

**裁定:不跑,等释放。**依据:①驱动仍为 570.195.03(GPU0 wedge 确凿根因版本,未打补丁);
②触发器未定,任何"小 run 应该没事"是概率话术;③收益半天 vs 赔注=楔死+弄死同租户
4+1 个进程+违反"卡坏不起"钦定纪律,不对称。翻转条件:驱动升级/deadline 逼急/同租户释放。
