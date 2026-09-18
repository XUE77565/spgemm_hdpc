# hash_spa 占用抬升刀:__launch_bounds__(1024,2)(09-18 午后,spec 82 姊妹刀)

## 证据链
1. cuobjdump:hash_spa_kernel<0/2,*> REG:40 → 1024thr 档 65536/(1024×40)=1 块/SM(50%);
   512thr 档 3 块(75%)——crankseg 主发射(31,770 块×512thr,20.7ms)ncu 实测 **74.72%**
2. 刀:`__launch_bounds__(HASH_BLOCK_STRUCT*4, 2)`(=1024,2)→ nvcc REG:40→**32**(STACK 0→16 温和溢出)
   → 1024 档 2 块 / 512 档 4 块(100%) / 256 档 8 块(100%)

## 守门(全绿)
值级 4/4(msc/F2/crankseg PASS,c-64 artifact 形状又逐数一致 only_test=20,504)+有序 4/4 SORTED-OK

## 电池判决(×5 相位和,off=lb2t=dense 刀后基线,off 与 lb2 预期值全部吻合)
- msc 3.252→3.091(**−5.0%**)/ c-64 21.010→20.408(**−2.9%**)/ crankseg 54.318→51.267(**−5.6%**)/
  **F2 9.916→8.334(−16.0%!**——accumulate 6.2ms 75%→100% ≈ −1.5ms 自洽 + MODE=1 计数遍同受益)
- **四阵全赢,无一回退 → 刀入默认(与 spec82 dense 刀同为无条件代码)**

## 累计系统记分牌(09-18 午后,双边新数据,ocean c-64=20.804 新跑)
| 阵 | 上午(c2t) | 现在(hbt=两刀) | ocean | 比值 |
|---|---|---|---|---|
| c-64 | 25.16 | 20.41 | 20.80 | **0.981 胜** |
| F2 | 9.95 | 8.33 | 7.66 | **1.088**(fork CSORT_CNT −9.6% 若入默认 → ~0.99 再跨线) |
| crankseg | 59.35 | 51.27 | 27.29 | 1.879(③occgate 阵地) |

## 泛化启示(物理台账)
**占用悬崖刀系**:凡 1024thr 块 + REG>32 = 半占用;cuobjdump REG 审计 = 一行破案工具。
已修:hash_dense_direct(50→32)/hash_spa(40→32)。待审计:其余热 kernel(largest 37/batched 31…)。
LADDER 验尸的"占用悬崖项"今日两次兑现——③阶段路由成本模型应纳入寄存器占用项。

## 二进制
spgemm_hb(WM)/spgemm_hbt(非WM)= 两刀全量;对照 spgemm_lb2t(仅 dense 刀)/spgemm_c2t(无刀)

## 全场 REG 审计(09-18 午后,cuobjdump,>32 的非 CUB kernel)
- csort 家族:compact_sort<256,64>=**255**/<256,16>=127/<128,16>=113;csort_fused<256,64>=229/<256,16>=80;
  csort_inplace<256,64>=255/<512,8>=79 → 256thr 下 1-2 块/SM(12.5-25% warps)——**下一片占用雷区**
  (crankseg compact+sort 相位载体;⚠ radix 高 REG 是算法内在,硬 launch_bounds 会灾难溢出,
  刀法=块形/IPT 再设计或 counting 变体替换——与队列③/fork 的 CSORT_CNT 汇流)
- 次要:sspan2<512,128>=40/t_merge_flop=40(边缘相位)
- 已修:hash_dense_direct 50→32 ✓/hash_spa 40→32 ✓
