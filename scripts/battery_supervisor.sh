#!/usr/bin/env bash
# 电池链监督者:等当前 netwindow_batteries.sh 退出(monitor 超时杀或自然完),
# 然后检查每场是否留下结果行(被杀的场 = 缓冲丢失零输出),补跑缺场。幂等。
set -u
cd /home/xueyizhou/spgemm-challenge
LOG=compare/netwindow_batteries.log

while pgrep -f 'netwindow_batteries.sh' > /dev/null 2>&1; do sleep 60; done

# 每场的结果行 = "matrix  xx.xxx  xx.xxx  -x.x%  nnz" 形态;场完成 ⟺ 头后有 ≥1 结果行
stage_done() {   # $1 = 场头标记
    awk -v h="$1" '
        index($0, h) {inhdr=1; next}
        inhdr && /^--- \[|=== 电池链完成/ {inhdr=0}
        inhdr && /%  [0-9]+$/ {found=1}
        END {exit !found}
    ' $LOG
}

guard() {
    local L=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
    [ "$L" -lt 8 ]
}

if ! stage_done "--- [1/3]"; then
    echo "--- [1/3·补] 三件套电池(接管) $(date +%H:%M:%S) ---" >> $LOG
    guard && .venv/bin/python -u scripts/ab_env.py "BATCH_S=1 COMPACT_T=1 COUNT_TINY=1" \
        road_usa germany_osm hugetrace-00010 hugebubbles-00000 web-Google pre2 bcsstk30 pwtk F2 Ga3As3H12 nd24k >> $LOG 2>&1
fi
if ! stage_done "--- [2/3]"; then
    echo "--- [2/3·补] H2DOVL 电池(接管) $(date +%H:%M:%S) ---" >> $LOG
    guard && .venv/bin/python -u scripts/ab_env.py "H2DOVL=1" \
        road_usa germany_osm pre2 hugetrace-00010 hugebubbles-00000 web-Google bcsstk30 pwtk Ga3As3H12 F2 nd24k >> $LOG 2>&1
fi
if ! stage_done "--- [3/3]"; then
    echo "--- [3/3·补] v29 回归复验(接管) $(date +%H:%M:%S) ---" >> $LOG
    guard && .venv/bin/python -u scripts/reverify_v29_regressions.py >> $LOG 2>&1
fi
echo "=== 电池链完成(监督者收尾) $(date +%H:%M:%S) ===" >> $LOG
