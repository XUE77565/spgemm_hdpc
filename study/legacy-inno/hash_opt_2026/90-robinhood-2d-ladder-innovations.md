
## 候选 2 实施设计(09-24 10:3x 日班#8 desk 定稿,待实现轮)

- **耦合点现状(compute_bucket_kernel 3080-3137)**:diter 档位=span 单维(span ≤
  DENSE_TIER_SPAN_MAX 档梯);块宽/窗宽由档位一维联动派生(dense_struct_numeric
  按档实例化)——ocean CONCURRENT×BLOCK 同构一维。
- **解耦形态(我方原创)**:档位键从单维 span 改双维 (span 档, est/flop 档):
  span 定窗宽(列覆盖),est·flop 定块宽(工作量车道)——span 5K∧est 500 行=大窗
  小块(ocean 给大窗大块=车道浪费);人口依据=§2.1 普查(c-64 宽行 est 均值 13.8K
  vs in-2004 窄 span est 3.4K——同窗档工作量差 4×,一维档必然错配一半)。
- **实现面**:compute_bucket 双维键(bkid 编码 span×wl 两轴,~3×7=21 档内);
  dense_struct_numeric 实例派发表加块宽维;发射点按双维选实例。改动集中=kernel 内部
  分派结构(①合法,不动行归属边界——diter 准入门不变,只变档内执行形态)。
- **验收**:in-2004 档位带(vs 彼七档 1.3-2× 基线再拉 1.3→1.8×+);20 阵守卫
  (K2 电池复用);值门三件套;查重注记=Harmonia 行方差特征同族可引但形态不同。
- **排期**:实现轮 1(双维键+派发表);轮 2(电池+对拍)。先于 RH 驱逐体(复杂度低
  且验收线现成)。
