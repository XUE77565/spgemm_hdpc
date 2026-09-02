#!/usr/bin/env bash
# 净窗电池链(docs/13 v2 三件套 → H2DOVL → v29 回归复验)
# 由负载门自动触发:load<6 持续 3 个采样点(每 60s)才开跑;跑中每场前复查负载,>8 熔断。
set -u
cd /home/xueyizhou/spgemm-challenge
LOG=compare/netwindow_batteries.log
echo "=== 净窗电池链 $(date +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg) ===" >> $LOG

gate() {   # 负载门:连续 3 次 <6
    local ok=0
    for i in 1 2 3; do
        local L=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
        [ "$L" -lt 6 ] && ok=$((ok+1)) || ok=0
        [ "$i" -lt 3 ] && sleep 60
    done
    [ "$ok" -eq 3 ]
}
guard() {  # 每场前熔断
    local L=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
    if [ "$L" -ge 8 ]; then echo "⚠ 熔断:场前 load=$L" >> $LOG; return 1; fi
    return 0
}

MAT_TRIO="road_usa germany_osm hugetrace-00010 hugebubbles-00000 web-Google pre2 bcsstk30 pwtk F2 Ga3As3H12 nd24k"
MAT_H2D="road_usa germany_osm pre2 hugetrace-00010 hugebubbles-00000 web-Google bcsstk30 pwtk Ga3As3H12 F2 nd24k"

if ! gate; then echo "负载门未通过,退出" >> $LOG; exit 1; fi

echo "--- [1/3] docs/13 v2 三件套电池 $(date +%H:%M:%S) ---" >> $LOG
if guard; then
    .venv/bin/python scripts/ab_env.py "BATCH_S=1 COMPACT_T=1 COUNT_TINY=1" $MAT_TRIO >> $LOG 2>&1
fi

echo "--- [2/3] H2DOVL 电池 $(date +%H:%M:%S) ---" >> $LOG
if guard; then
    .venv/bin/python scripts/ab_env.py "H2DOVL=1" $MAT_H2D >> $LOG 2>&1
fi

echo "--- [3/3] v29 回归复验(v28 双 binary) $(date +%H:%M:%S) ---" >> $LOG
if guard; then
    .venv/bin/python scripts/reverify_v29_regressions.py >> $LOG 2>&1
fi
echo "=== 电池链完成 $(date +%H:%M:%S) ===" >> $LOG
