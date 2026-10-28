#!/usr/bin/env python3
"""§六十六制度化:opt-in 覆盖审计(第三检)。
扫描 spgemm_kernel_hash.cu 的 TU-61 区(HASH_TB==61 wrapper):凡 smem 可能 >48KB 的
hash_spa_kernel tagged 实例(LOG_HT≥12 → 表≥48KB;HT_P≥2719 素数档),必须有对应
cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...) 覆盖行,否则构建失败。
用法:python3 tools/arch_probe/check_optin.py
"""
import re, sys
src = open('src/spgemm_kernel_hash.cu').read()
i0 = src.index('#if HASH_TB == 61\n// TU-61 大表/托付形态发射器')
i1 = src.index('}\n#endif', i0)
seg = src[i0:i1]
launches = set()
for m in re.finditer(r'hash_spa_kernel<([^>]+)>', seg):
    args = [a.strip() for a in m.group(1).split(',')]
    if len(args) < 6: continue
    try:
        lht, htp = int(args[4]), int(args[5])
    except ValueError:
        continue
    if lht >= 12 or htp >= 2719:
        launches.add(m.group(1))
optins = set(m.group(1) for m in re.finditer(r'cudaFuncSetAttribute\(hash_spa_kernel<([^>]+)>,', seg))
missing = launches - optins
print(f"大表 tagged 发射实例: {len(launches)};opt-in 覆盖: {len(launches & optins)}")
if missing:
    print("✗ opt-in 缺失(E3 类雷,launch 将 invalid argument):")
    for x in sorted(missing): print("   hash_spa_kernel<%s>" % x)
    sys.exit(1)
print("✓ opt-in 全覆盖")
