#!/bin/bash
# P1 符号审计门(study/58 附录A.1 保命件4):断言 TU-90/TU-61 模板实例集不相交
# 用法: check_disjoint.sh [obj90] [obj61]   (探针P2:同参实例弱合并=静默错codegen)
O90=${1:-src/spgemm_kernel_hash.o}; O61=${2:-src/spgemm_kernel_hash_c61.o}
[ -f "$O90" ] || { echo "MISSING $O90"; exit 2; }
[ -f "$O61" ] || { echo "MISSING $O61"; exit 2; }
for OBJ in "$O90" "$O61"; do
    cuobjdump -symbols "$OBJ" 2>/dev/null | grep -oE '_Z[0-9]+[a-zA-Z_]*(hash_spa_kernel|hash_heavy_kernel|csort|hash_spa_packed|ocean_hash)[A-Za-z0-9_]*' 
done > /tmp/all_syms.txt
S90=$(cuobjdump -symbols "$O90" 2>/dev/null | grep -oE '_Z[0-9]+[a-zA-Z_]*(hash_spa_kernel|hash_heavy_kernel|hash_spa_packed|ocean_hash)[A-Za-z0-9_]*' | sort -u)
S61=$(cuobjdump -symbols "$O61" 2>/dev/null | grep -oE '_Z[0-9]+[a-zA-Z_]*(hash_spa_kernel|hash_heavy_kernel|hash_spa_packed|ocean_hash)[A-Za-z0-9_]*' | sort -u)
INTER=$(comm -12 <(echo "$S90") <(echo "$S61"))
echo "TU-90 模板实例: $(echo "$S90" | wc -l) 个;TU-61: $(echo "$S61" | wc -l) 个"
if [ -n "$INTER" ]; then echo "✗ 实例集相交(弱合并雷!):"; echo "$INTER"; exit 1; fi
echo "✓ 实例集不相交(探针P2 风险解除)"
