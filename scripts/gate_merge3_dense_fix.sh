#!/bin/bash
# study/45 守门(10-15):merge3 dense 分支双雷修复的值级门
# 逻辑:
#   对 1 = MRG3_DENSE_SPAN=0(dense 关,纯 merge 路径,未受雷染)vs =1(dense 开,修复后)
#          → 修复正确性:两路输出应逐位一致
#   对 2 = merge3(dense-off)vs hash → 交叉真值(merge3 全路径 vs 久经验证的 hash 路径)
# 空对照防御:dense 开时 mrg3 的 count/merge 相位时间应显著变化 = 分支真开火;
#           相位时间雷同 → 疑未开火,门判 INVALID 而非 PASS。
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
G() { env CUDA_VISIBLE_DEVICES=0 WARMUP_N=0 "$@"; }   # 二进制放调用侧(10-12/10-15 两次 $X 位次雷)

for m in bloweybq vsp_befref_fxm_2_4_air02 bloweya; do
  f=data/ocean/square/$m.mtx
  rm -f results/matrices/$m/self_product_merge3.mtx results/matrices/$m/self_product_hash.mtx
  G METHOD=merge3 MRG3_DENSE_SPAN=0 ./spgemm_test "$f" > /tmp/g45_${m}_off.log 2>&1
  mv results/matrices/$m/self_product_merge3.mtx /tmp/g45_off.mtx
  G METHOD=merge3 MRG3_DENSE_SPAN=1 ./spgemm_test "$f" > /tmp/g45_${m}_on.log 2>&1
  mv results/matrices/$m/self_product_merge3.mtx /tmp/g45_on.mtx
  echo "== [$m] ① dense-off vs dense-on(fixed):"
  bash scripts/mtx_gate_fast.sh /tmp/g45
  # 开火证据:两次运行的 merge 相位时间
  t_off=$(grep -o 'merge[ =:]*[0-9.]*ms' /tmp/g45_${m}_off.log | head -1)
  t_on=$(grep -o 'merge[ =:]*[0-9.]*ms' /tmp/g45_${m}_on.log | head -1)
  echo "   merge phase: SPAN=0 [$t_off] vs SPAN=1 [$t_on] (应显著不同=开火)"

  # ② 交叉真值 vs hash
  G METHOD=hash ./spgemm_test "$f" > /tmp/g45_${m}_hash.log 2>&1
  h=results/matrices/$m/self_product_hash.mtx
  [ -f "$h" ] || h=results/matrices/$m/self_product.mtx
  cp "$h" /tmp/g45x_on.mtx
  G METHOD=merge3 MRG3_DENSE_SPAN=0 ./spgemm_test "$f" > /dev/null 2>&1
  cp results/matrices/$m/self_product_merge3.mtx /tmp/g45x_off.mtx
  echo "== [$m] ② merge3(dense-off) vs hash(真值):"
  bash scripts/mtx_gate_fast.sh /tmp/g45x
  echo
done
echo "gate_merge3_dense_fix done"
