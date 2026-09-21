# Retry 2.0 档位制快重试(F2 跨线刀,09-20 立项;设计基础=study/22 件3+本轮代码解剖)

## 现状解剖(09-20 读码定谳,src/spgemm_kernel_hash.cu)
1. **v40 SMEM 重试车已存在**(4816-4861,RETRY_SMEM 默认 1):溢出行按 `flop≥1024 ∧ 2×flop≤HASH_CAP`
   分流上 hash_spa 二次发射——但配表 = **批量统一 ht=pow2≥2×max(flop)**(一个肥行抬全批)
2. **准入缝隙 = study/22 解剖的病灶**:est 745/flop 13434 型行(小 est 大 flop)因 2×flop 超 HASH_CAP
   被推全局路;flop<1024 的小溢出行也全落全局
3. **全局路配表**:retry_prep(2543)h = pow2≥2×min(flop,ncols) 且**下限 2×HASH_CAP=32K 槽**
   ——小溢出行也配 32K 槽全局表(384KB arena/行)= 冷原子+大表 init,retry 1.33μs/行=主遍历 4-5×
4. **F2 固定成本 0.475ms(0 溢出行也付,study/22 待查)**:run_retry(4761)每轮先 blocking
   cudaMemcpy 读 d_ovf_cnt = accumulate 后首个 host 同步点 = 全管线 drain 落在 retry 相位账上

## 刀(三件,R0→R2 单刀序)
- **R0 固定成本**:①插桩定位(cudaEvent 细分 memcpy/scan/launch 或 nsys 一发)②若=drain:
  pinned 异步读回+event 等(只等 accumulate 不等全机)③相位和口径下验证 0 溢出行时 retry 相位归零
- **R1 档位制配表**(主刀):SMEM 车配表从"批量统一 2×max(flop)"改"**原表 pow2 上探一档 ht2=2×ht**"
  ——溢出仅意味着 distinct>ht(est 误差通常小,σ=1.04/√k),一档 2× 几乎总够;ht2>HASH_CAP 或
  再溢出 → 上探下一档(链式,≤2 SMEM 档)→ 仍败才全局 flop 路(终止性:distinct≤min(flop,ncols) 有界)
- **R2 准入放宽**:地板 flop≥1024 取消(flop<1024 的溢行走 ht2 小表 SMEM 车更便宜);
  鲸鱼(flop 大∧ht2 超帽)维持全局路不动

## 守门(三件套全量)
值级(truth17+compare_mtx_values.py;大阵 WM=METHOD=hash+self_product_hash.mtx)+
行内有序+memcheck(retry 是 study/17 双卡楔死案发地——positional 子集重建的未初始化尾段雷,
改动触碰 rows_g 补集逻辑必须 memcheck);c-64 artifact(20,504)豁免

## 电池(A 级,需净窗;本轮 09-20 双卡全占被阻塞)
- 主判 F2(retry 段 0.58ms→<0.3ms;系统 8.01→目标 ≤7.66=跨线);守卫 crankseg/c-64/msc
  +扩 bench 新阵(in-2004/nd24k/wb-edu/web-G/c-43/3tube,先建 off 带)
- off=c78t,新刀独立命名;×5 起步,跨线判定点 ×7
- env 旋钮:RETRY2=1 开关(默认开前守门全绿);RETRY_SMEM 保留兼容

## 预期与风险
- 预期:F2 retry 段 −0.3ms+;顺带解锁 est 收紧经济学(3689 注释:小阵被迫 EXPAND 1.4 就是为躲
  retry 固定成本——R0 修好后紧 est 可再试,bcsstk30 教训重评)
- 风险:①ht2 再溢出率若高(est 误差重尾)→链式档变慢,电池判;②分流逻辑改动触碰 study/17 雷
  (必须复用 rows_g 本体);③0.475 若非 drain 而是 PCIe RTT+host wake → pinned 只减半,记录实测定量

## 状态
- 09-20 11:53 loop 第 1 轮:spec 立;环境=GPU0 44.5GB+GPU1 78.8GB/98%(双卡全占)→实施可写码
  编译,C 级值级可跑(GPU0 闲置 SM),A 级电池等净窗

## ×7 加固+跨阵中性:刀候默认(09-21 07:4x,单发 loop#8)
- **F2 ×7**(/tmp/ab_f2_retry2_x7_20260921.log):off 中位 7.952 / **on 中位 7.582**,
  配对 −1.081/−0.320/−0.111/−0.236/−0.213/−0.249/+0.173 = **6/7 同号,中位 −3.5%**;
  与 ×5(4/5,−4.3%)复现一致——**刀硬化完成**
- **跨阵中性(单发)**:msc23052 1.240(零溢出,构造中性✓;vs off 1.212 的 +2.3% 在小阵噪声地板内)/
  c-64 20.861(**带内** 20.3-22.0;有溢出行走新路径)/crankseg_2 48.810(带下缘,有溢出行,疑似小赢 −2%)
- **刀候默认清单**:①×7 绿✓ ②跨阵中性✓(单发) ③crankseg 值级补验(其溢出行走新 kernel,
  F2 已绿但阵不同) ④用户过目 ⑤翻默认=RETRY2 默认 1+源码注释+二进制梯重切(新锚 r2t)
- 今夜全弧线:R0 解剖(0.556=编排队税)→R1 v1 负结果(thrust 临时分配真凶)→R1b fused(0.455)
  →R1c 拆段 S=5(0.418)→R1d arena 持久化(**0.204,−58%)→×5/×7 双电池胜+跨阵中性;
  副产物:mempool 8赢2输决策备忘、三发 shell 雷入档、ocean F2 时效课

## R1d 终局:刀完成+电池×5胜+ocean F2 时效课(09-21 07:1x,单发 loop#7)
- **实施**:arena 持久化(rk/rv/rtc/rtv+d_rslot/d_roff,hash_product 函数级 static 惰性扩容,
  三处 free 加 s_r2_arena_active 卫兵)+分流 pinned 合批(门回 g_retry2——**曾违规渗 off 路径致
  off 0.79-0.97 出带,门回后 off 0.486-0.503 复原,纪律救刀**)
- **判决:刀完成**——F2 retry 相位 **0.51→0.204-0.210ms(−58%)**,验收线<0.25 过;off 不回归
- **守门全绿**:0↔1 PASS+truth17 PASS+memcheck 0(arena 复用的陈旧数据风险由 row_nnz 界定+守门证实)
- **电池 ×5(净窗 14MiB,/tmp/ab_f2_retry2d_20260921.log)**:off 中位 7.961 / **on 中位 7.618**
  (配对 −0.650/−0.230/−0.963/+0.005/−0.392 = 4/5 同号)——刀胜
- **⚠ ocean F2 时效课(口径纪律实录)**:09-18 值 7.66 → **今日复测 7.410(−3.3%)**——若按旧值
  宣官即成假跨线(7.618 vs 7.66=0.995 假);**今日真值 7.618 vs 7.410 = 1.028,未跨但大步逼近**
  (历史 1.045→1.028);教训:跨线官宣必须同日双侧净窗
- **叠屏**:mempool+RETRY2+R1d = 7.460(单发)——比单独 mempool 7.73 好,刀间近加性;
  vs ocean 7.410 = **1.007,一步之遥**(mempool 翻默认仍待用户拍板,两输家阵在案)
- **待办**:×7 加固;跨阵中性验证(msc/c-64/crankseg h_ovf=0 阵构造上中性,值级快验即可);
  翻默认决议;crankseg retry 人口画像(其 retry 相位占多少待查)

## R1c 判决:拆段全局路成立+守门全绿(09-21 06:0x,单发 loop#6,r2t)
- **实施**:原 hash_global 每块自含 init+probe+extract(__syncthreads 假设单块独占行)→ 拆三段:
  memsetAsync init / retry2_gprobe(grid.y=S 切 k 值域,共表原子 CAS)/ retry2_gextract(单块/行,流序);
  S=min(8,ceil(mf_g/4096))(分流循环顺带 mf_g);F2 实跑 S=5;**仅 retry 路径,原 kernel/RETRY2=0 不动**
- **判决**:retry 相位 0.455→**0.418**(GPU 尾 ~198→~121µs);验收<0.32 未达——剩余账定谳:
  **allocs ~200µs(d_rslot/d_roff A_rows 级 + arena 2.4MB×4,legacy 分配器)= R1d 最后一块肉**
  +分流段 2×D2H 36µs
- **守门全绿**:值级 0↔1 PASS + truth17 PASS + 39,648,131 SORTED-OK + memcheck 0 errors
  (probe/extract 拆段后 extract 的 cap 实参=d_rslot(min(flop,ncols)≥distinct),原版同映射,语义不变)
- **R1d(下轮)**:arena 持久化(rk/rv/rtc/rtv static 惰性扩容,5022/5063/5468 三处 free 加旗卫兵)
  + 分流 2×D2H 合批;预期 0.418→**~0.25**(allocs 200µs 几乎全砍);完成后 ×5 电池+跨线判定
  (F2 当前 7.855 − retry 段 0.17 ≈ 7.69,与 mempool 叠加 7.5x——跨线在望,口径须分明)

## mempool 守卫阵屏判决+R1c 设计(09-21 05:18,单发 loop#5,×2 轮内一致,/tmp/ab_mempool_guard_20260921.log)
- **10 阵全表:8 赢 2 输,输家灾难级 → 不翻默认,升级为用户决策备忘**
  赢:F2 −6.4%(8.0-8.4→7.67-7.73,贴 ocean 线 1.002)/c-64 **−6.2%(21.0→19.7,vs ocean 20.75=0.95 加深)**/
  crankseg −2~6%/msc23052 −2.5%/3dtube −2%/c-43 −5%/in-2004 −1%/nd24k −0.5%
  **输:wb-edu +21%(166→201)/web-Google +30%(5.34→6.96)**
- 选项(给用户):①按阵选分配器=dispatcher 新维(特征候选:alloc 次数/arena 规模)②查 MP_HOST_MB=8192
  对两输家是否太小(池退化>裸 cudaMalloc)③维持 OFF(口径不动)
- 尸检红线+1(第三雷):**env 前缀不能经 $var 展开传递**(赋值识别在词法期,展开词只当参数——
  v1 屏把 USE_MEMPOOL=1 当命令名执行了全 0.000);修法=分支展开或 env $VARS
- **R1c 设计(hash_global 行内并行)**:现 1 block/行(blockIdx.x=行号,块独占行全局表);
  改 grid(n, S):blockIdx.y 切该行 k 值域段,同表探测已是原子 CAS/Add(块内并发本就依赖),
  跨 block 共表成立;S=min(ceil(flop/段定额),8)(d_flop 在手);溢出语义不变(probes≥表→原子旗);
  争用预算:鲸鱼 flop≥16K/distinct~8K/表 65K 槽,碰撞低。**arena 持久化**:5022/5063/5468 三处
  无条件 free 加 static 旗卫兵,容量惰性扩容
- R1c 预期:GPU 尾 190µs→~50µs(S=4-8 段并行),retry 相位 0.455→~0.3;配合 allocs 治理逼近 <0.25

## R1b fused 判决:刀成立(未完成)+三件套全绿+mempool 重磅屏(09-21 04:25,单发 loop#4)
- **fused kernel(单 block cub::BlockScan:排名+双 scan+scatter+尾值,杀全部 thrust)入 r2t**:
  RETRY2=1 retry 相位 0.67→**0.455**(fused 段自身仅 15µs);RETRY2=0 不回归(0.543 带内)
- **守门三件套全绿**:RETRY2 0↔1 输出 PASS(1e-9)+truth17 PASS+**39,648,131 条 SORTED-OK**
  +memcheck 0 errors(study/17 雷绿);c-64 artifact 未涉及
- **电池 ×3(GPU0 净窗 14MiB,/tmp/ab_f2_retry2_20260921.log)**:off 8.046 / on 7.855 中位,
  配对差 −0.162/−0.191/−0.664 **全同号**——刀方向确立;验收线<0.25 未达,**RETRY2 保持 env 门默认关**
- **残余账(=R1c)**:GPU 尾 ~198µs(hash_global 3 blocks 串行,行内多 block 并行化)+
  allocs ~120µs(arena 持久化——rk/rv 在 5022/5063/5468 三处无条件 free,须 null 卫兵)+
  分流段 2×D2H ~40µs(可上设备侧)
- **⚠ mempool 系统级屏(重磅,非本刀)**:USE_MEMPOOL=1 F2 相位和 8.26→**7.73(−6.4%)**,
  全管线效应非仅 retry;叠加屏 7.719(近加性);**距 ocean 7.66 仅 1.009**——但历史电池全
  mempool OFF,**翻默认=口径变更**,须全守卫阵电池+off 带重标+用户拍板,单独立项(勿混入本刀)

## R1 v1 判决:负结果+分配器真凶定谳(09-21 03:26,单发 loop#3,r2t,F2 净窗)
- **RETRY2=1(子集 sort+scan+scatter+合批读)反而更慢**:0.670/0.678 vs 旧路 0.586/0.605——
  **thrust::sort/exclusive_scan 在 n=3 时每次调用做临时 cudaMalloc**(legacy 分配器=同步+百 µs 级),
  吃掉全部子集收益;旧路的全 A_rows 扫描(71505)其实不是主凶,**分配器才是**(两次 thrust scan 各带临时分配)
- **分配器假说实测证实**:USE_MEMPOOL=1 MP_HOST_MB=8192 下旧路 retry 0.51→**0.424-0.451(−0.1ms 纯分配器)**;
  子集路+mempool 0.479-0.490(thrust sort 税 ~50µs 仍在)
- **R1b 正确形状(下轮)**:①杀 thrust——d_rtab scan(h_ovf 小)+子集 offset 合成**一个单 block kernel**
  (SMEM 内 gather+sort+scan+scatter+写尾值到 mapped pinned,host 零中间同步)②retry arena 持久化
  (进程内复用,免每次 dev_alloc)③GPU 尾 190µs=hash_global 3 blocks 行内并行化(多 block/行原子入全局表)
  ④USE_MEMPOOL 系统级效应另立单刀(A 级电池裁,**口径纪律:历史电池全 mempool OFF,翻默认=口径变更**)
- 预算重排:host 侧真可砍的=thrust×2+allocs+mempool 差(≈0.42-0.15 目标差全在 fused kernel+arena);
  RETRY2 env 默认关零伤害,r2t 已带两路径
- 电池脚本/插桩二进制:r2t(rtprof 标记常驻);F2 配方=METHOD=hash ./spgemm_r2t data/ocean/square/F2.mtx

## R0 编排分解定谳(09-21 02:39,单发 loop#2,r2t 插桩二进制,dbg 时戳差,3 轮一致)
- host 编排 ~300-320µs:**rslot 全 A_rows(71505) inclusive_scan+D2H#1 ≈118µs** ←最大项 /
  2×8B 阻塞 D2H+arena alloc ≈79µs / prep+全 A_rows memset+rslot ≈77µs / 发射 ≈11µs;
  GPU 侧余 ≈190µs = hash_global **3 blocks 尾延迟**(3 行 × 65536 槽全局表 init+串行累积)
- **人口定性修正(推翻本 spec 原设计前提)**:F2 默认 3 行溢出 = **鲸鱼行**(flop≥16K,
  arena=196608=3×65536 槽;2×flop>HASH_CAP)——**档位 SMEM 车(ht2 上探)救不了它们**,
  全局路是正确目的地;study/22 "est745/flop13434"是 EXPAND 实验人口非默认人口
- **R1 形状修正**:①host 快路=子集化 offset(排序 h_ovf 行+子集 scan+scatter d_roff,杀 118µs)
  +D2H 合批(杀 ~50µs)②R1b=hash_global 行内并行化(1 block/行 → 多 block/行原子进全局表,
  杀 ~190µs 尾延迟)③档位车保留给小 est 大行(flop<8K 人口,其他阵有用),F2 主判走 ①+②
- 验收不变:F2 retry 相位 0.51→<0.15ms(①预期 ~-170µs;②预期 ~-150µs)
- 扩 bench off 带已建(09-21 02:41 净窗 ×3,c78t 相位和,/tmp/offband_newbench_20260921.log):
  **nd24k 148.4 / in-2004 50.3 / web-Google 5.3 / wb-edu 167.6 / 3dtube 6.1 / c-43 2.2**(±1% 内)

## R0 复现与定性(09-21 01:35-01:49,单发 loop#1,GPU0 净窗 14MiB)
- F2 c78t 相位复现:retry=0.559/0.555ms,相位和 8.256ms 在带 ✓
- **定性修正:默认 F2 有 3 行溢出**(ovf_cnt=3,"row-retry: 3 行")——非"零行固定成本";
  0.556ms/3 行=185μs/行 = **编排队税主体**:d_rslot/d_roff 全 A_rows memset+thrust 扫(为 3 行扫全阵)、
  3 次阻塞 D2H(r_slots/last_h/rt_prev)、~8 次小 alloc、32K 槽全局 arena init——真计算仅数十 μs
- HashProf=事件对(include/hash_prof.h),retry 相位含 host 编排间隙 ✓
- **R0-fix 并入 R1 刀**:档位制 SMEM 车(3 行小表)+ offset 扫描子集化 + D2H 合批 → 目标 retry 相位 <0.15ms
- 电池脚本重建三雷(旧脚本库已死于 09-18 重启):①awk 正则缺右括号;②dbg/hash-prof 行走 **stderr** 须 2>&1;
  ③c-64 须 **METHOD=hash**(auto 先试 cuSPARSE→insufficient resources 失败)+ awk **每 tag 取末轮值**
  (否则 warmup+全轮加总=129ms 假值)。红线新增:**grab() 管线 smoke 单发验证通过才准跑电池**
