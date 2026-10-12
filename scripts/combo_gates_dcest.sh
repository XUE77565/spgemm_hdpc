#!/bin/bash
# DCEST V3 组合守门(10-12):D5H/K7(链后直写者)与 DCEST 的组合正确性。
# 逻辑:D5H=1(DCEST 0/1)与 K7=1(DCEST 0/1)各配对值门——隔离 DCEST 增量。
# 靶子:in-2004(dup≈1 → D5H 资格 + 21067 diter 行)、pre2(K7 人口 + 4 diter 巨鲸行)。
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
G() { env CUDA_VISIBLE_DEVICES=0 METHOD=hash WARMUP_N=0 "$@"; }   # env:展开词($X 的 D5H=1)在前缀位才会被当赋值——裸 bash 会当 argv[1](10-12 空对照雷)

combo() {  # combo <name> <envs...>
  local m=$1; shift
  G DCEST=0 "$@" ./spgemm_test data/ocean/square/$m.mtx >/dev/null 2>&1
  mv results/matrices/$m/self_product_hash.mtx /tmp/cg_${m}_off.mtx
  G DCEST=1 "$@" ./spgemm_test data/ocean/square/$m.mtx >/dev/null 2>&1
  mv results/matrices/$m/self_product_hash.mtx /tmp/cg_${m}_on.mtx
  echo "== $m [$*] gate:"
  bash scripts/mtx_gate_fast.sh /tmp/cg_${m}
  rm -f /tmp/cg_${m}_off.mtx /tmp/cg_${m}_on.mtx
}

combo in-2004 D5H=1
combo pre2 K7=1
echo "combo gates done"
