#!/usr/bin/env bash
# v30 全量刷新:默认翻开的四件套(BATCH_S/COMPACT_T/COUNT_TINY/H2DOVL 结构化门)出终版。
# 前提:确认电池(compare/confirm_battery.log)已通过。
# 流程:负载门 → cp v29→v30 → REFRESH=Auto 全刷 → DNF 补跑(timeout=1800,无 MP 前缀)→ 终版报告
set -u
cd /home/xueyizhou/spgemm-challenge
LOG=compare/v31_refresh.log

# 负载门:3×60s <6
ok=0
for i in 1 2 3; do
    L=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
    [ "$L" -lt 6 ] && ok=$((ok+1)) || ok=0
    [ "$i" -lt 3 ] && sleep 60
done
while [ "$ok" -ne 3 ]; do      # 持续等待(灾变期循环重试;上游 = nohup 生命周期)
    ok=0
    for i in 1 2 3; do
        L=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
        [ "$L" -lt 6 ] && ok=$((ok+1)) || ok=0
        [ "$i" -lt 3 ] && sleep 60
    done
done

cp compare/ocean337/methods_cmp_v29.csv compare/ocean337/methods_cmp_v31.csv
echo "=== v31 刷新启动 $(date +%H:%M:%S) ===" | tee -a $LOG
REFRESH=Auto OUT_CSV=compare/ocean337/methods_cmp_v31 bash scripts/run_method_cmp.sh >> $LOG 2>&1

echo "=== DNF 补跑 $(date +%H:%M:%S) ===" | tee -a $LOG
.venv/bin/python scripts/v31_dnf_fill.py >> $LOG 2>&1

echo "=== v31 完成 $(date +%H:%M:%S) ===" | tee -a $LOG
python3 - <<'EOF' >> $LOG 2>&1
import csv, math
rows = list(csv.DictReader(open('compare/ocean337/methods_cmp_v31.csv')))
def num(x):
    try: return float(x)
    except: return None
p = [(r['matrix'], num(r['Auto']), num(r['Ocean'])) for r in rows]
p = [(m,a,o) for m,a,o in p if a and o]
w = sum(1 for _,a,o in p if a<=o)
print(f"=== v31 终版:{len(p)} 有效阵,赢 {w},geomean {math.exp(sum(math.log(a/o) for _,a,o in p)/len(p)):.4f}× ===")
EOF
