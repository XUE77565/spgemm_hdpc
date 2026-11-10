# dense_count 跨窗游标化刀(spec87,09-21 立项)

## 证据链(绝对时长驱动;c-64 dense_count 2.64ms 常规/83ms 强制,服务全 diter 行)
1. 停顿(09-21 双内核 ncu):hash_dense_count **long_scoreboard 5.66 独大**(direct 的一半但同族),
   short 1.08/wait 2.80/not_selected 2.30;Duration 2.99ms(ncu)/Compute 61.8%/占用 98.36%
2. **读码定谳(936-1004):count 每窗每 k 链做两次 `dev_lower_bound`**(窗界 c0 一次、c1 一次
   从 ks 续搜)——每窗每链 ~2×log2(链长) 次**依赖链全局读**=direct cursor 当年干掉的"每窗重搜"
   在 count 里的存活体;**count 从未游标化**
3. byte-flag 写无争用(atomicOr 教训已内化)+DCFUSE 向量化归约(v56 已入)——都不是瓶颈

## 刀
- **主刀:count 跨窗游标化**——窗 w 扫描 break 处 atomicMin 记游标(结构同 direct cursor 路:
  smap 镜像/复位 INT_MAX/G-group 续扫),w+1 免二分;首窗仍二分(冷启动)
- 附刀候选:①count→direct 游标直传(count 末窗 break 位=direct 首窗起点,免 direct 首窗二分)
  ②终极形态:count/direct 融合单遍(结构性,停靠列后)
- env COUNT_CURSOR=1 门,默认关

## 验收
- c-64 dense_count 2.64→**<2.0ms**;pre2 强制 83→**<60ms**;long_scoreboard 5.66→<4
- 守门三件套(值级 0↔1+truth17+有序+memcheck);c-64 常规 + pre2 强制栈双测
- 电池=A 级净窗 ×5(c-64 全相位和主判)

- 09-21 16:4x 单发#22 **修正版实施+判负**:csmem 动态扩段版落地(CCUR_CAP=1024,双发射点 g_ccur;
  首轮 typo 一次修复)+烟:**c-64 全相位和 22.791 vs 22.795(wash;且 span 50K<64K 窗=单窗,
  刀在 c-64 本是空转——首窗双二分保留,无窗可 carry)**;**pre2 强制 87.6→90.4(+3% 判负)**
- **机理定谳:二分搜索不是 count 的瓶颈**——省一半搜索(c0 lb)的收益 < 游标 SMEM 读写+扩段税;
  count 停顿的 long_scoreboard 主体是**流式 flag 写本身**的访存,不是窗界搜索
- **spec87 收线(判负,代码留库 env 门默认关)**:count 相位的真刀口只剩结构性件——
  **count/direct 融合单遍**(整个相位消失,终极形态)与窗界流水化并列,均多轮工程
- **今日决赛 A 阶段总账(三轮刀全负/平)**:B16(wash,请求数≠链延迟)/书签预取(负,书签=暖点)/
  count 游标化(负,搜索≠瓶颈)——**低垂果实时代结束,结构性工程时代开始**;这批负结果本身
  是"cursor/dense 内核已接近其形态内最优"的证据,论文 C2 的防守面

## 状态
- 09-21 15:5x 单发#20 立项(画像+读码定谳)
- 09-21 16:1x 单发#21 **首轮实施失败已回滚**(git checkout,spec86 完好):病灶=**静态
  `__shared__ int scur[1024]`(4KB 无条件)挤爆 count 内核发射**——csm 动态窗已近满,静态加 4KB
  超限,连默认路径(COUNT_CURSOR=0)都崩(静默死在 warmup);教训:**count 内核 SMEM 无余量,
  游标区必须走 csmem 动态扩段或全局 smap arena(direct 同款),不能静态 __shared__**
- 实施修正版(下轮):scur 放 csmem 尾部对齐扩段(发射 csm 加 4×nch_max 字节)或借 direct 的
  全局 d_smap arena(count 先跑可与 direct 分段共享);判据不变,烟流程不变

## 状态(旧)
