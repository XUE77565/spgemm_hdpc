#!/usr/bin/env bash
# cen_rm2 —— 影子待复测阵的补测(上一轮作业的 R1 在第 8 个阵做到一半时驱动被结束,16 阵里完整做完 7 个)
#
# 上一轮 R1 的事实(results/…/r1.csv.gz、r1_run.log):c-45、bcsstk33、circuit_4、rajat18、rajat16、rajat17、rajat28 各 63 发全做完、0 失败,
#   rajat20 做了 14 轮(42 发);按写定的判法这 8 阵 cen 全部放行。rajat 族每发约 41 秒(C 有 8.9 亿非零,回传 11GB),7 个阵就用了 3 小时 15 分——
#   我当初把 R1 估成一两个小时是错的。没有哪一发挂死:驱动被结束前的 29 分钟里 rajat20 照常做了 42 发。
# 本作业:
#   A 组(21 轮):rajat25、Zd_Jac6_db(影子里的不裁决阵,必须补)、rajat20(重做完整 21 轮)、rajat31
#   B 组(9 轮):std1_Jac3、av41092、TSOPF_RS_b2052_c1、CurlCurl_4、Stanford_Berkeley(影子里 cen 在 1.02–1.04 的可裁决阵,不是必须,顺手核)
#   臂 off / aa / cen;env 与影子、confirm20 相同。每一发都包在 `timeout -k 30` 里,记墙钟秒数;驱动每发覆写心跳文件
#   $JOB_DIR/rm/r1a.hb / r1b.hb(阵、臂、轮、开始时刻、子进程 pid)——要判断是不是挂了,看它,别看驱动进程的年龄。
#   最后把上一轮已完成的 7 阵与本轮合并,按同一判法出 16 阵的总判定。
# 回传:OUT_DIR/SUMMARY.md + r1a_* / r1b_* / r1_all_remeasure.txt。
set -u
ROOT=$(pwd)
OUT=${OUT_DIR:?}
JD=${JOB_DIR:?}
ID=${JOB_ID:-cen_rm2}
DATA=/home/xueyizhou/spgemm-challenge/data
SUM=$OUT/SUMMARY.md
BASE_SHA=141fbc46a690225559b235d79a7da33a7eb7b7029795078e3e9f927e9875011c   # 基线 src/spgemm_kernel_hash.cu(70d1dea)
HDR_SHA=2fc25346de6deabc2eda6d49739ba0627a976969896b861e3798517bab59339b    # bm2_r4 / bm2_r5 的 S1 验过的 src/hh_bm2.cuh
LOAD_MAX=${LOAD_MAX:-6}
QUIET_WAIT_S=${QUIET_WAIT_S:-900}
BIN=./spgemm_m2
RUNENV="USE_MEMPOOL=1 MP_HOST_MB=8192 D6EXP=0"
MYUID=$(id -u)
mkdir -p "$OUT" "$JD/vl" "$JD/k2" "$JD/k3" "$JD/keep" "$JD/rm"
: > "$SUM"
say() { echo "$*" | tee -a "$SUM"; }
TS0=$(date +%s); TSL=$TS0
stage() { local now; now=$(date +%s); echo; echo "################ $* ################ (+$((now - TSL))s 上一段,累计 $((now - TS0))s)"; TSL=$now; date -u +%FT%TZ; }
SAMPLER_PID=""
sampler_start() { ( while :; do echo "$(date +%s) $(cut -d' ' -f1 /proc/loadavg) $(foreign_hot)"; sleep 60; done ) > "$1" 2>/dev/null & SAMPLER_PID=$!; }
sampler_stop() { if [ -n "$SAMPLER_PID" ]; then kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""; fi; }
clean_dumps() { rm -rf "$ROOT"/results/matrices/*/self_product*.mtx "$JD/vl" 2>/dev/null; }
cleanup() { sampler_stop; clean_dumps; rm -f "$JD/TIMING_ACTIVE" "$JD/WAITING_QUIET"; }
trap 'cleanup; echo "收到终止信号,已清转储"; exit 143' TERM INT
finish() {   # $1=退出码 $2=结论一句话
  cleanup
  say ""; say "## 结束"; say "- 结论: $2"; say "- 主机(结束时): $(host_line)"; say "- 磁盘: $(df -h "$JD" | tail -1)"
  say "- 结束时间: $(date -u +%FT%TZ),总用时 $(( ($(date +%s) - TS0) / 60 )) 分钟"
  exit "$1"
}

# ───────────────────────── 主机负载与负载门 ─────────────────────────
foreign_hot() {   # CPU 占用最高的 5 个进程里,不属于本作业用户且 >50% 的个数
  ps -eo pcpu,uid --sort=-pcpu 2>/dev/null | awk -v me="$MYUID" 'NR>1 && NR<=6 && $2!=me && $1+0>50 {n++} END{print n+0}'
}
apps_on() {       # $1=卡号 → 该卡上的计算进程数
  local uuid
  uuid=$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader 2>/dev/null)
  [ -n "$uuid" ] || { echo 0; return 0; }
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$uuid"
}
gpu_busy() {      # 我钉的卡上有别的计算进程,或另一张卡利用率 ≥10% → 返回 0(忙)
  local g u
  [ "$(apps_on "$GPU")" = 0 ] || return 0
  for g in 0 1; do
    [ "$g" = "$GPU" ] && continue
    u=$(nvidia-smi -i $g --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -dc 0-9)
    [ "${u:-0}" -ge 10 ] && return 0
  done
  return 1
}
host_line() { echo "load=$(cut -d' ' -f1-3 /proc/loadavg) 外来>50%CPU进程数=$(foreign_hot) 卡=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ';')"; }
quiet_once() {
  awk -v l="$(cut -d' ' -f1 /proc/loadavg)" -v t="$LOAD_MAX" 'BEGIN{exit !(l+0 < t+0)}' || return 1
  [ "$(foreign_hot)" = 0 ] || return 1
  gpu_busy && return 1
  return 0
}
quiet_twice() { quiet_once || return 1; sleep 60; quiet_once; }
wait_quiet() {    # $1=最长等待秒数;成功(连续两次安静)返回 0
  local t0 n=0
  t0=$(date +%s)
  while :; do
    quiet_twice && return 0
    [ $(( $(date +%s) - t0 )) -ge "$1" ] && return 1
    [ $((n % 6)) = 0 ] && echo "等安静窗口:$(host_line)(已等 $(( ($(date +%s) - t0) / 60 )) 分钟)"
    n=$((n + 1))
    sleep 300
  done
}
load_med() {      # $1=采样文件 → load1 中位(无采样回显空)
  local n
  n=$(wc -l < "$1" 2>/dev/null || echo 0)
  [ "${n:-0}" -gt 0 ] || return 0
  awk '{print $2}' "$1" | sort -n | sed -n "$(( (n + 1) / 2 ))p"
}
load_stats() {    # $1=采样文件 → 一行文字
  local n
  n=$(wc -l < "$1" 2>/dev/null || echo 0)
  [ "${n:-0}" -gt 0 ] || { echo "无采样"; return 0; }
  echo "load1 中位 $(load_med "$1") / 最大 $(awk '{print $2}' "$1" | sort -n | tail -1)($n 个采样,每分钟一次;其中 $(awk '$3>0{c++}END{print c+0}' "$1") 次见到外来 >50% CPU 进程)"
}


# ───────────────────────── S0 环境 ─────────────────────────
stage "S0 环境"
say "# cen_rm2 汇总(影子待复测阵:上一轮没测完的部分补完;每一发记墙钟秒数)"
say ""
say "## S0 环境"
say "- 主机: $(hostname)  开始: $(date -u +%FT%TZ)  快照: $ROOT  nproc=$(nproc 2>/dev/null)"
say "- nvcc(PATH): $(command -v nvcc) / $(nvcc --version | tail -1)"
HAVE_SCIPY=0; python3 -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)" > "$OUT/s0_py.log" 2>&1 && HAVE_SCIPY=1
say "- python3: $(python3 --version 2>&1) / $(tail -1 "$OUT/s0_py.log")"
HAVE_PATCH=0; command -v patch >/dev/null && HAVE_PATCH=1
say "- 磁盘: $(df -h "$JD" | tail -1)"

pick_gpu() {
  local g uuid n
  for g in 0 1; do
    uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader 2>/dev/null) || continue
    [ -n "$uuid" ] || continue
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$uuid")
    [ "$n" = 0 ] && { echo $g; return 0; }
  done
  return 1
}
GPU=""; SHARED=0
for try in $(seq 1 30); do
  GPU=$(pick_gpu) && break
  echo "两张卡都有计算进程,等 60s($try/30)"; sleep 60
done
if [ -z "$GPU" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
  SHARED=1
fi
export CUDA_VISIBLE_DEVICES=$GPU
say "- **用卡: GPU $GPU**$([ $SHARED = 1 ] && echo '(⚠ 等了 30 分钟仍有别的计算进程,与人共卡)')"
say "- 主机(开跑时): $(host_line)"

ln -sfn "$DATA" "$ROOT/data"
find_mtx() {   # $1=阵名(无 .mtx)→ 相对路径
  local d f
  for d in data/ocean/square data/first100 data/train_disp; do
    [ -f "$d/$1.mtx" ] && { echo "$d/$1.mtx"; return 0; }
  done
  f=$(find -L data -maxdepth 4 -name "$1.mtx" 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "$f"; return 0; }
  return 1
}
arm_env() {   # $1=臂名 → env 赋值串
  case $1 in
    m1)    echo "HH_BM2=1";;
    m2)    echo "HH_BM2=2";;
    m2e)   echo "HH_BM2=2 HH_BM2_LAZY=0";;
    m2p)   echo "HH_BM2=2 HH_BM2_P1=1";;
    m2L)   echo "HH_BM2=2 HH_BM2_L1=1";;
    cen)   echo "HH_CENSUS=1";;
    cenm2) echo "HH_CENSUS=1 HH_BM2=2";;
    cm2)   echo "HH_CENSUS=1 HH_BM2=2 HH_BM2_LAZY=2";;
    cm2L)  echo "HH_CENSUS=1 HH_BM2=2 HH_BM2_LAZY=2 HH_BM2_L1=1";;
    *)     echo "HH_BM2=0";;
  esac
}

# ───────────────────────── S1 微复现器 ─────────────────────────
stage "S1 微复现器"
say ""; say "## S1 微复现器"
S1=red
GOT_HDR=$(sha256sum src/hh_bm2.cuh | cut -d' ' -f1)
if [ "$GOT_HDR" = "$HDR_SHA" ]; then
  S1=green
  say "- src/hh_bm2.cuh 的 sha256 与 bm2_r4 / bm2_r5 的 S1 验过的相同(两种子 ALL-PASS,168480 行 bad=0)→ 沿用"
else
  say "- src/hh_bm2.cuh 已变($GOT_HDR),重跑 hh_repro5"
  if nvcc -O3 -arch=sm_90 -std=c++14 -Isrc -Iinclude -I/usr/local/cuda/include tools/hh_repro5.cu -o "$JD/hh_repro5" > "$OUT/s1_repro_build.log" 2>&1; then
    ( timeout -k 30 900 "$JD/hh_repro5" 40 42; echo "rc=$?"; timeout -k 30 900 "$JD/hh_repro5" 25 2026; echo "rc=$?" ) > "$OUT/s1_repro.log" 2>&1
    if [ "$(grep -c '^ALL-PASS' "$OUT/s1_repro.log")" = 2 ] && [ "$(grep -c '^rc=0' "$OUT/s1_repro.log")" = 2 ]; then S1=green; fi
    say "- $(grep -E '^(ALL-PASS|FAIL)' "$OUT/s1_repro.log" | tr '\n' ';')"
  else
    say "- **复现器构建失败**:"; tail -30 "$OUT/s1_repro_build.log" | tee -a "$SUM"
  fi
fi
say "- S1: $S1"

# ───────────────────────── S2 构建 ─────────────────────────
stage "S2 构建"
say ""; say "## S2 构建"
S2=red
T0=$(date +%s)
make -k -j4 -f Makefile -f "jobs/$ID/keep.mk" TARGET=spgemm_m2 > "$JD/build_m2.log" 2>&1; RC=$?   # -k:一个翻译单元报错后继续编另一个,报错一次出全
T1=$(date +%s)
tail -c 100000 "$JD/build_m2.log" > "$OUT/s2_build_tail.log"
say "- make TARGET=spgemm_m2: rc=$RC,用时 $((T1 - T0))s;审计: $(grep -E '✓|✗|模板实例' "$JD/build_m2.log" | tr '\n' ';')"
if [ $RC = 0 ] && [ -x spgemm_m2 ] && [ "$(grep -c '✓' "$JD/build_m2.log")" -ge 2 ] && ! grep -q '✗' "$JD/build_m2.log"; then
  if [ -f src/spgemm_kernel_hash.o ]; then rm -f src/main.o; MK="-f Makefile -f jobs/$ID/keep.mk"; else MK=""; fi
  make -j4 $MK WRITE_MTX=1 TARGET=spgemm_m2w > "$JD/build_m2w.log" 2>&1; RCW=$?
  say "- WRITE_MTX 件 spgemm_m2w: rc=$RCW($([ -n "$MK" ] && echo '只重编 main.cu' || echo '全量重编'))"
  [ $RCW = 0 ] && [ -x spgemm_m2w ] && S2=green
else
  say "- **构建或审计失败**。报错行(前 80 条,全文尾部在 s2_build_tail.log):"
  grep -nE "error|Error|✗|MISSING|undefined reference" "$JD/build_m2.log" | head -80 | tee -a "$SUM"
  say "- 日志尾:"; tail -25 "$JD/build_m2.log" | tee -a "$SUM"
fi
say "- S2: $S2"
[ $S1 = green ] || finish 11 "红:S1 微复现器未全绿(S2=$S2);后续阶段未跑"
[ $S2 = green ] || finish 12 "红:S2 构建/审计失败;后续阶段未跑"


# ───────────────────────── S3 冒烟 ─────────────────────────
stage "S3 冒烟"
say ""; say "## S3 冒烟(各臂 1 发:nnz 与失败检查)"
smoke() {   # $1=组名 $2=--only 列表 $3..=臂参数 → 回显 green / red
  local g=$1 only=$2 rc v=green; shift 2
  python3 scripts/ab_arms.py --bin "$BIN" --out "$OUT/s3_$g" --rounds 1 --gpu "$GPU" "$@" --only "$only" --stat > "$OUT/s3_${g}_run.log" 2>&1
  rc=$?
  [ $rc = 0 ] || v=red
  grep -qE "NNZ-MISMATCH|FAIL|不可测" "$OUT/s3_$g.txt" 2>/dev/null && v=red
  [ -s "$OUT/s3_$g.txt" ] || v=red
  echo $v
}
smoke_taken() { awk -F, 'NR>1{s+=$7}END{print s+0}' "$OUT/s3_$1_stat.csv" 2>/dev/null; }
S3=$(smoke g1 bcsstk17,F2,case9,c-64 --arm off: --arm cen:HH_CENSUS=1)
say "- off/cen,4 阵 → $S3"
if [ $S3 != green ]; then tail -25 "$OUT/s3_g1_run.log" | tee -a "$SUM"; tail -20 "$OUT/s3_g1_fail.log" 2>/dev/null | tee -a "$SUM"; finish 13 "红:S3 冒烟未过;后续阶段未跑"; fi

# ───────────────────────── R1 补测 ─────────────────────────
SIG='^\[census-stat\].*$'
shot_report() {   # $1=逐发 CSV → 一行:发数、失败数、单发秒数的中位 / 最长、超过 300 秒的发数
  awk -F, 'NR>1{n++; if($6+0<=0)f++; d[n]=$9+0; if($9+0>300)l++} END{if(!n){print "无数据";exit} m=0; for(i=1;i<=n;i++) if(d[i]>m) m=d[i]; printf "%d 发,失败 %d 发;单发墙钟最长 %.0f 秒;超过 300 秒的 %d 发", n, f+0, m, l+0}' "$1" 2>/dev/null
}
stage "R1a 补测 A 组"
say ""; say "## R1a 补测 A 组(rajat25 / Zd_Jac6_db / rajat20 / rajat31;off / aa / cen;21 轮;单发上限 600 秒,整段预算 270 分钟)"
{ echo mat; for n in Zd_Jac6_db rajat25 rajat20 rajat31; do echo "$n"; done; } > "$JD/rm/listA.txt"
say "- 主机(R1a 前): $(host_line);心跳文件 $JD/rm/r1a.hb"
: > "$JD/TIMING_ACTIVE"
sampler_start "$JD/load_R1a.log"
python3 scripts/shadow428.py --bin "$BIN" --list "$JD/rm/listA.txt" --roots data/ocean/square,data/first100 --out "$JD/rm/r1a" \
  --arm off: --arm aa: --arm cen:HH_CENSUS=1 --env HH_CENSUS_STAT=1 --method adaptive --rounds 21,21,21 --sig-regex "$SIG" \
  --timeout 600 --budget-min 270 --gpu "$GPU" > "$OUT/r1a_run.log" 2>&1
say "- shadow428 rc=$?;$(tail -1 "$OUT/r1a_run.log")"
sampler_stop
say "- 段内主机负载: $(load_stats "$JD/load_R1a.log")"
say "- 逐发统计: $(shot_report "$JD/rm/r1a.csv")"
grep -E "^\[ *[0-9]+/" "$OUT/r1a_run.log" | sed 's/^/    /' | tee -a "$SUM"
[ -s "$JD/rm/r1a.csv" ] && gzip -c "$JD/rm/r1a.csv" > "$OUT/r1a.csv.gz"
tail -40 "$JD/rm/r1a_fail.log" 2>/dev/null > "$OUT/r1a_fail_tail.log"

stage "R1b 补测 B 组"
say ""; say "## R1b 补测 B 组(std1_Jac3 / av41092 / TSOPF_RS_b2052_c1 / CurlCurl_4 / Stanford_Berkeley;off / aa / cen;9 轮;单发上限 900 秒,整段预算 180 分钟)"
{ echo mat; for n in std1_Jac3 av41092 TSOPF_RS_b2052_c1 CurlCurl_4 Stanford_Berkeley; do echo "$n"; done; } > "$JD/rm/listB.txt"
say "- 主机(R1b 前): $(host_line);心跳文件 $JD/rm/r1b.hb"
sampler_start "$JD/load_R1b.log"
python3 scripts/shadow428.py --bin "$BIN" --list "$JD/rm/listB.txt" --roots data/ocean/square,data/first100 --out "$JD/rm/r1b" \
  --arm off: --arm aa: --arm cen:HH_CENSUS=1 --env HH_CENSUS_STAT=1 --method adaptive --rounds 9,9,9 --sig-regex "$SIG" \
  --timeout 900 --budget-min 180 --gpu "$GPU" > "$OUT/r1b_run.log" 2>&1
say "- shadow428 rc=$?;$(tail -1 "$OUT/r1b_run.log")"
sampler_stop
rm -f "$JD/TIMING_ACTIVE"
say "- 段内主机负载: $(load_stats "$JD/load_R1b.log")"
say "- 逐发统计: $(shot_report "$JD/rm/r1b.csv")"
grep -E "^\[ *[0-9]+/" "$OUT/r1b_run.log" | sed 's/^/    /' | tee -a "$SUM"
[ -s "$JD/rm/r1b.csv" ] && gzip -c "$JD/rm/r1b.csv" > "$OUT/r1b.csv.gz"
tail -40 "$JD/rm/r1b_fail.log" 2>/dev/null > "$OUT/r1b_fail_tail.log"

# ───────────────────────── 合并判定 ─────────────────────────
stage "R1 合并判定"
say ""; say "## R1 合并判定(上一轮已做完的 7 阵 + 本轮 9 阵;rajat20 用本轮完整的 21 轮,上一轮那 14 轮不并入)"
PREV="$(dirname "$JD")/cen_rm1/out/r1.csv.gz"
{
  echo "mat,tier,size_mb,arm,round,co,nnz,sig"
  if [ -r "$PREV" ]; then zcat "$PREV" | awk -F, 'NR>1 && $1!="rajat20"' | cut -d, -f1-8; fi
  [ -s "$JD/rm/r1a.csv" ] && tail -n +2 "$JD/rm/r1a.csv" | cut -d, -f1-8
  [ -s "$JD/rm/r1b.csv" ] && tail -n +2 "$JD/rm/r1b.csv" | cut -d, -f1-8
} > "$JD/rm/r1_all.csv"
say "- 上一轮的逐发数据 $PREV:$([ -r "$PREV" ] && echo "读到,并入 $(zcat "$PREV" | awk -F, 'NR>1 && $1!="rajat20"' | wc -l) 发" || echo '**读不到**,下面只有本轮的 9 阵(上一轮 7 阵的判定见它自己的 r1_remeasure.txt)')"
python3 scripts/remeasure_sum.py "$JD/rm/r1_all" > "$OUT/r1_all_remeasure.txt" 2>&1
say "- remeasure_sum rc=$?;逐阵全文 r1_all_remeasure.txt"
say ""
say '```'
grep -E "^== |该阵 A/A|放行|仍红|^REMEASURE_RED|^# " "$OUT/r1_all_remeasure.txt" | tee -a "$SUM"
say '```'
say "- 判法(dialog/qa-011,本机侧已确认):稳定阵看中位比 ≤1.10;不稳阵(该阵自己的 A/A 超 ±5%)看最小值 / 下四分位 / 上四分位三档比 ≤1.05 / 1.05 / 1.10。"
say "- 9 轮的 B 组分位数比 21 轮的粗,它们在影子里本来就是可裁决且 ≤1.10 的阵,这里只是加核。"
finish 0 "S3 冒烟 $S3;R1 的 16 阵总判定看 r1_all_remeasure.txt 的 REMEASURE_RED 行;每发墙钟秒数在 r1a.csv.gz / r1b.csv.gz 的 dur_s 列"
