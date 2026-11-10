# 创新 kernel 线(spec90,09-22 立;总纲③——③是①的收尾半场:创新 kernel 研究出后 ①②才算全功)

## 候选 1:Robin Hood 散列(hash 线差异化武器)

### 机理
- 标准 open addressing:先到先得→高 LF 时链长方差爆炸(少数行运气差撞长链)
- **Robin Hood:探测时遇到"比我离家更近的同住者→我抢位它让路"**(rich steal from poor 的反义)
  → 所有键的探测距离趋均→**方差消减,平均链长在高 LF 下仍短**
- **双选择(breadth-first two-choice)**:每键两候选位取短者→平均链长 O(log log n)/n 级
- 我们的探测环已 instrument(HASH_PROBE_HIST 33 桶)→高 LF 方差现成可测

### 为什么值钱
- 当前默认 LF≈0.37(表半满=防撞):**敢推 0.55-0.6 = 表空间省 ~33%** →
  ①SMEM 装得下更多行(占用↑)②表档降一格(pow2 梯)③cold atomics 减
- est 连续化已给"行行均匀 LF"的基座;RH 治的就是"均匀 LF 下的残余方差"——正交互补
- ocean 没有;nsparse/spECK 无先例(查重入档)=**可 claim 的 kernel 级创新**

### 验收线
- 口径 C 三阵(F2/c-64/crankseg)不劣;高 LF 阵(LF 0.5+ 强制)probe_hist 方差可见降;
  SMEM 表档省一格的系统赢;三门守门
### 实施 sketch(probe 环插入位移)
- 插入环:探测中遇 occ 且 dist(它)>dist(我)→swap(我占位它续探);直写累积的 atomicCAS
  语义需改"占位-让位"两态;SMEM 原子下可行性先 spike(env ROBIN=1)

## 候选 2:二维梯解耦(dense 线,①→②的桥)

- 现状:档位=一维联动(窗宽∝span,块宽死跟窗档);ocean 同构一维
- **解耦:窗宽(span 档)×块宽(工作量 est/flop 档)独立选**——span 5K∧est 500 行给
  大窗小块(ocean 给大窗大块=浪费);§4d 顶档 1.3× 的收窄主因=固定 1024 块
- 验收:in-2004 顶档(1.3×档)拉回 1.8×+;20 阵守卫
- 出处注记:ocean CONCURRENT×BLOCK 一维联动(study/25);解耦=我方原创(查重 Harmonia 行方差
  特征同族=可引但形态不同)

## 候选 3(远):创新载体形态(无主行型覆盖——③的本职)
- c-64 中带(1.775 无主)/crankseg 宽中 est 的专用形态;待①②收官后人口普查驱动

## 状态
- 09-22 15:4x 单发#56 立项;RH spike(ROBIN=1 探测环改造)排首轮实施
