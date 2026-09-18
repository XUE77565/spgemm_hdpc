# csort 占用雷区:刀法设计(09-18 午后,spec 83 审计的后续)

## 雷区画像(cuobjdump + 相位数据)
| kernel | REG | 块形 | 占用 | 载体 |
|---|---|---|---|---|
| compact_sort<256,64> | 255 | 256thr | 1 块=12.5% | 大行(16K items) |
| csort_fused<256,64> | 229 | 256thr | 1 块=12.5% | 大行 |
| compact_sort<256,16> | 127 | 256thr | 2 块=25% | **bin7(d 1-4K)** |
| csort_fused<256,16> | 80 | 256thr | 3 块=37.5% | **bin6/7** |
| csort_counting(fork) | **40** | 256thr | **6 块=75%** | bin6(已默认) |

- crankseg 51K hash 行集中 bin6/7 → compact+sort 相位在 25-37% warps 下跑 = 下一块肉
- 大行变体(<256,64>)低占用是行少所致,非悬崖,不治

## 刀法(radix 高 REG 是算法内在,不硬压)
1. **counting 扩展到 bin7**(fork 遗留半成品的方向):`csort_counting<512, 4096>`——
   SMEM = sc 16K + sv 32K + hist 12K = **60KB > 48KB 需 opt-in**(house 模式,cudaFuncSetAttribute);
   512thr:65536/(512×40)=3.2→3 块=75%;REG 40 级 → 可再试 __launch_bounds__(512,4) 压 32(4 块=100%,溢出风险小坑,电池裁决)
2. **bin6 counting 微压**:(256,7) bounds → ≤36 regs → 7 块=87.5%(从 40 压 4 个,温和)
3. 验收:F2 compact+sort 全相位 1.54(现状,bin6 only)→ bin7 扩展后 crankseg compact+sort 相位
   为主判(c-64/F2 顺带);值级+有序守门(counting 动序,命门同 spec81)
4. **retry 固定成本 0.58ms**(study/22 件 3,F2 线下一刀)独立推进:SMEM 快重试两级定表

## 归属注记
csort_counting 内核 = fork 会话遗产(b4be6ab 转正默认);fork 已退出(09-18 午后确认),
本线由主会话接管;其 F2 −9.6% 的机理判读 = 指令瘦 + 占用 6× 双修(占用物理的第三次兑现)

## bin7 扩展判决(09-18 15:20,×5 相位和,hbt vs c78t)
- msc +0.6 / c-64 +0.0(无种群,中性✓)/ **crankseg −3.3(51.73→50.01)/ F2 −5.7(8.49→8.01)** → 刀成立已在默认
- 记分更新:F2 8.01 vs 7.66 = **1.045**(retry 固定成本 0.58ms 是跨线刀)/ crankseg 1.832 / c-64 不变
- 守门:值级 4/4(msc/F2/crankseg PASS + c-64 artifact 逐数一致)+ 有序 4/4(动序刀,命门绿)
