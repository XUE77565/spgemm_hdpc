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

## 状态
- 09-21 14:3x 单发#17 立项;变体 A 实施下一轮
