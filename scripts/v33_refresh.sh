#!/usr/bin/env bash
# v33 全量刷新:档位梯 v33d 配方翻默认(六档梯 + DTIER_CMP + 双侧分流 + 结构化地板分治)。
# 前提:compare/v33d_battery.log 八阵全绿。
# 流程:负载门 → cp v32→v33 → REFRESH=Auto 全刷(默认 env,零环境变量)→ DNF 补跑(timeout=1800)→ 报告
set -u
cd /home/xueyizhou/spgemm-challenge
LOG=compare/v33_refresh.log

ok=0
for i in 1 2 3; do
    L=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
    [ "$L" -lt 6 ] && ok=$((ok+1)) || ok=0
    [ "$i" -lt 3 ] && sleep 60
done
while [ "$ok" -ne 3 ]; do
    ok=0
    for i in 1 2 3; do
        L=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
        [ "$L" -lt 6 ] && ok=$((ok+1)) || ok=0
        [ "$i" -lt 3 ] && sleep 60
    done
done

cp compare/ocean337/methods_cmp_v32.csv compare/ocean337/methods_cmp_v33.csv
echo "=== v33 刷新启动 $(date +%H:%M:%S) ===" | tee -a $LOG
REFRESH=Auto OUT_CSV=compare/ocean337/methods_cmp_v33 bash scripts/run_method_cmp.sh >> $LOG 2>&1

echo "=== DNF 补跑 $(date +%H:%M:%S) ===" | tee -a $LOG
.venv/bin/python scripts/v33_dnf_fill.py >> $LOG 2>&1

echo "=== v33 完成 $(date +%H:%M:%S) ===" | tee -a $LOG
python3 - <<'PYEOF' >> $LOG 2>&1
import csv, math
rows = list(csv.DictReader(open('compare/ocean337/methods_cmp_v33.csv')))
def num(x):
    try: return float(x)
    except: return None
p = [(r['matrix'], num(r['Auto']), num(r['Ocean'])) for r in rows]
p = [(m,a,o) for m,a,o in p if a and o]
ratios = sorted(a/o for m,a,o in p if o > 0)
geo = math.exp(sum(math.log(r) for r in ratios)/len(ratios)) if ratios else float('nan')
wins = sum(1 for m,a,o in p if a < o)
print(f"v33: 有效 {len(p)}/{len(rows)} 阵, geomean Auto/Ocean = {geo:.4f}, 胜 Ocean {wins}")
PYEOF
