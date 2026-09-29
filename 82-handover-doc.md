# 82 · 交接文档(总纲+计划+纪律,2026-09-29)

> 给下一个会话的完整交接。新窗口第一步:读这份 + memory(自动加载)。

---

## 一、总纲(用户钦定,最高优先,一切工作以此为准)

```
① 孤立优化 kernel
        ↓
② 全面超过 ocean(所有 kernel)
        ↓
③ 拓展更多 kernel 覆盖更多矩阵行和矩阵类型(研究出创新的 kernel)
        ↓
④ 用创新办法训练出路由
        ↓
⑤ 产生我们的终极 SpGEMM
```

**顺序铁律:①②③全部完成之前(=每个载体超越 ocean 对应 kernel+创新 kernel 研究出来),
④路由一步不碰。①阶段只动 kernel 内部:改判/边界/回流/env 调 binning 全是④,违线即停。**

### 当前进度(09-25 定格)

| 战场 | 战况 | 靠什么 |
|---|---|---|
| hash 中轻行 | ✅ 七阵全胜 0.34-0.64 | est 连续化(37% 装填)+Knuth 2 条指令+档位定长表 |
| cursor 特宽行(c-64) | ✅ 系统级 0.88-0.96 | 书签游标+占用物理(REG 瘦身) |
| dense 档位 | ✅ 同档同块宽 1.3-2× | 自适应分组+座位直写+先读后挂 |
| **hash 重行** | ❌ 输 1.5-1.9× | 结构税(表 96KB→2 工位/SM vs 彼 32KB→4)——形态内定谳无解 |
| 系统级八阵 | c-64 0.967 胜/F2 1.012 平/四输 1.56-1.88 归④ | — |

**一句话:三路全胜,只剩重行一个缺口。②差这一个战场。**

### 重行缺口的解法(09-25 定向)

- **队首选 path A**:用 dense 位图打 nd24k 重行。22K 行全重行(flop 中位 107K/est 8K/
  dup 13×/n≈24K,密度 33% 在 dense 可行域);**dup 13× 对位图免费**。先 desk 查行 span
  分布(不碰卡),可行跑同行对拍三小任务(同窗档协议=study/25 §4d)。
- **path B(备)**:复刻 ocean 重行同构 kernel——不是抄代码,按定谳机理重实现(素数编译期
  常量表 magic-div+512T 块+值外置保 4 工位),目标追平+叠一遍制微赢。论文诚实写"对照形态"。
  **用户已听方案未正式批准,开工前须确认。**
- 已判死勿碰:WMA 归并累积器(warp 级串行骨架,27-166×劣)、两两归并树(带宽账 76GB>
  基线,spec98 尾节)、kbal(+7%)、CNT_FUSE(永久冻结,两次 Xid 31 前科)。

---

## 二、计划(接手队列,按序)

1. **验卡**:`nvidia-smi` 双卡干净才动(参考:5 次以上 GSP wedge 史,570.195.03 固件病,
   治本=管理员升驱动,已报告用户)
2. **队首:nd24k 重行 span 特征 desk 检查**——读 rt 行统计(重跑小任务或查档案),
   span 窄/密度高 → path A 对拍;span 宽 → 直接转 path B
3. **path A 对拍三小任务**(老二进制,值门有背书,小批量+Xid 哨兵):
   我方 dense_struct 强制行级计时(DD_ROWTIME)→ 彼 ROWCYCLES(/tmp/oc_nd24k.csr 需重建,
   /tmp 已被重启清)→ join 同行对比。验收=重行带中位 <1.0
4. path A 不赢 → **path B 复刻 spec** 写完交用户过目再实现
5. 低优:对称消融全套重配(六层常量网,需 227KB SMEM 档)、论文线(paper/hash-kernel-wins/01
   扩写,三大创新点已定稿:游标书签 9/10、占用物理 8.5/10、一遍制 7/10)

---

## 三、纪律(违者即停,全部用户钦定)

### ⚠ GPU 保护六条(09-25 用户明令"卡坏不起了",最高优先)

1. **报错即停**:任何 illegal access/Xid/段错误,当天熔断该特性不再重试,先诊断
   (kbal 三小时反复重试=三小时持续打 GSP,最大前科,绝不重演)
2. **sanitizer 先行**:新 kernel/含疑 kernel 首跑必须 compute-sanitizer 全绿才准裸跑性能
   (主线 spgemm_test 有值门背书可豁免)
3. **带病二进制退役**:spgemm_wml(Xid 31 前科)/含 kbal 调试版/CNT_FUSE 构建永不真阵
4. **Xid 哨兵**:每批 GPU 测试前后各查一次(`journalctl -k | grep -i xid`,无 sudo 时用
   nvidia-smi 响应正常性兜底),见自家 pid/name 立即停手
5. **降 churn**:电池进程内多轮,不每轮起进程;不跑高频 nvidia-smi 循环
6. 治本=管理员升驱动,已报告;此前六条是全部防线

### 总纲铁律

- ①②③完成前④路由一步不碰;改判/边界/回流/env 调 binning=④
- ocean 是超越标杆非优化边界:开刀依据=我方绝对时长画像,ocean 只用于验收
- 外部基线只吸收机理不 copy 形态(09-15 钦定);先查公开前作(nsparse/spECK)

### 正确性优先(09-16 钦定)

- **先门后屏**:值门三件套(truth17 值级对比+行内有序检查+memcheck)全绿才准性能屏
- 大赢面>15% 自动触发值级复审;nnz 对/模式对不等于值对

### 口径纪律

- 相位统计禁跨迭代求和;TOTAL 排 h2d/d2g 传输;相位和必排 TOTAL(GPU) 行
- 对拍声明窗口;跨阵数据不混用;测量环境三态(A 净窗计时/B ncu 比值/C 值级随时)
- 时效纪律:一律最新定谳层为准(paper/hash-kernel-wins/01 §〇=效果总账入口)

### 工程红线(事故累积,勿重蹈)

- make WRITE_MTX=1 后必 `rm src/main.o spgemm_test` 两删(.o 缓存污染)
- 无 assert 的 str.replace 禁用(缩进锚 no-op 静默失败);多 smem_flat 场景先数清循环数
- 新模板实例必查 opt-in;env 语义先读码(GFORCE=ctz 陷阱曾废两批结论)
- 相位和口径:hash-prof tag=$5 v=$(NF-1);rt 文件 cycles 在 $NF/$7,join 前验列位
- 换二进制独立名;电池脚本入库前 bash -n;表头与对照同步

---

## 四、关键文件索引

| 文件 | 内容 |
|---|---|
| memory/mmspgemm-pact25.md | 主线全档(事件簿/战果/负结果地图,自动加载索引在 MEMORY.md) |
| memory/gpu-safety-discipline.md | GPU 六条纪律+7 次 wedge 定责 |
| 81-three-kernels-and-roadmap.md | 三 kernel 赢点机理详解(小白版,含指令级对比) |
| study/24-dense-cursor-attack.md | dense/cursor 全战役档案(§3.1a-al) |
| study/25-dense-sota-study.md | ocean dense 解剖+同档对拍(§4d) |
| inno/hash_opt_2026/98-*.md | WMA 全套+归并树判死(尾节) |
| paper/hash-kernel-wins/01-*.md | 效果/创新点唯一总账 |
| src/spgemm_kernel_hash.cu | 主战场(~6000 行,hash+cursor+dense 三族) |
| ocean/ | 对拍基准(ICS'26,二进制 ocean/spgemm) |

**二进制梯(锚点制)**:c78t(全刀默认)/spgemm_test(主线,双刀默认后构建)/hbt/c2t 仪器档;
已退役:wm91/wml/r2t/r2w(带病或使命完成)。

---

## 五、恢复口令

新会话开场对用户说:"已读交接文档,当前队列=〈队首任务〉,先验卡。" 然后按 §二 执行。
