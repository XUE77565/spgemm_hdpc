# cursor 内核 B 链读延迟刀(B16 移植+书签预取;spec86,09-21 立项)

## 证据链(绝对时长驱动,用户口径:哪长打哪,ocean 只验收)
1. 停顿定谳(study/24 §2.2b):hash_dense_direct long_scoreboard **11.78 cyc/issue ≈55% 停顿**,
   short_scoreboard 3.64,not_selected 2.05(非吞吐瓶颈)——**访存延迟瓶颈**
2. 读码定谳(09-21):cursor 内核内环 `B_col_idx[q]`+`B_val[q]` **双流散载 stride 8**(1318-1330),
   **B16 交织从未移植**(signature 无 B16 参;HASH_BI 每轮已构建该 buffer 只是没传进来);
   无 __ldg/无预取
3. 机制先例:④交织B 在 hash_spa 实测请求 −18%(db21731 入默认);B16 分裂人格案例在案(kernel 级
   +16% vs 系统 −1.6%,系统电池为判)——移植后须电池裁决

## 刀(两变体,单刀序)
- **A(最简,先落):B16 移植**——kernel 加 `const uint4 *B16=nullptr` 参,非空时内环一次 uint4
  (col+val 同 16B 事务,`__hiloint2double` 解包),空走旧双流(默认关 env CURSOR_B16=1 门);
  call site 传 B16 buffer(已存在)
- **B(深刀,A 后):书签预取**——窗 w 计算期间,smap 游标**即下一窗预取地址**(书签=预取地址,
  机理优雅):__pipeline_memcpy_async/双缓冲 SMEM 装 w+1 的活跃 k 链段
- C(小项):__ldg 提示+合并度核查

## 验收
- c-64 常规态 dense_direct 13.6→**<12ms**;pre2 强制人口(栈见 study/24 §2.3b)316→**<280ms**
- DD_ROWTIME 中位同降;ncu 复查 long_scoreboard 11.78→**<8**
- 守门三件套(值级 0↔1+truth17+有序+memcheck);电池=A 级净窗 c-64/pre2 ×5(系统级判据,
  防 B16 分裂人格重演)

- 09-21 15:3x 单发#19 **变体 B 最简形判负+机理反转**:窗尾 touch 预取实施(smap 即下窗链头,
  prefix/extract 段掩护;CURSOR_PF env)+烟:c-64 off 13.36/13.40 vs PF 13.47/13.52(**+0.9%**)/
  PF+B16 13.44-13.47 wash 偏负;pre2 强制 313.4→313.3 wash。
  **机理反转定谳:书签=暖点非冷点**——链在本窗扫描中止处 break,该缓存行刚被读过,预取=重复加温;
  真冷失载在窗界(清窗/首窗/流推进新地),须**窗界流水化**(w 窗 extract 与 w+1 窗 scan 重叠)=
  结构性多轮工程,停靠列②候选
- **spec86 收线**:双变体(A wash/B 判负)皆负,但两机理入账(①请求数≠链延迟②书签=暖点);
  dense_direct 剩余攻击角度=窗界流水化+dense_count 侧;**转第二刀口 dense_count(2.64ms,位图
  符号形态,DSTRUCT 争用重审)开 spec87**;CURSOR_B16/CURSOR_PF env 留库默认关零伤害

## 状态
- 09-21 14:3x 单发#17 立项
- 09-21 15:0x 单发#18 **变体 A 判决:wash**——实施(签名参+站点1 search 路径/站点3=真 cursor 主路径
  1198-1206 移植;错打 subwarp8 内核一次已撤)+烟:c-64 off 13.831(带内✓)/B16=1 13.716(−0.8% 噪声级);
  pre2 强制 317.8→317.3(−0.15% wash)。**机理判读:B16 减请求数救不了依赖链上的延迟**
  (col 读→槽判定→CAS 串行链,占用并行藏不住链内等待)——与"书签省重搜不省等数"同族
- **变体 B 机理锐化**:预取的价值=装载提前与计算重叠(正打链上延迟),非减请求;设计重点=
  窗 w 计算期发 w+1 活跃链段的 async 装(书签=预取地址不变)
- 变体 A 代码保留 env 门默认关零伤害(CURSOR_B16;站点注记:1323 系是 subwarp8 遗留内核勿动)
