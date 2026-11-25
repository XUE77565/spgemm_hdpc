#!/usr/bin/env bash
# prof_k1 —— 设备侧 profiling(nsys 逐 kernel 时长 + ncu 机制计数)+ 带负载门的 v3 计时重测
#
# 背景:bm2_r5 的正确性段全绿(S1-S4、S8),但那一轮 C 的主机 CPU 被别的任务打满(dialog/qa-006),S6/S7 的计时读数
#   已预先作废为定谳依据。本作业分两类阶段:
#   K 段(不吃主机计时,随时可跑):
#     K1 构建          make TARGET=spgemm_k1(源码与 bm2_r5 的 spgemm_bm2 相同)
#     K2 nsys          22 阵 × off/m2/m2p/m2L × 2 轮,逐 kernel 的 GPU 侧累计时长(设备侧时间戳)
#                      → 两遍式的设备侧臂间比(主机吵时的主指标)+ 逐核族/逐实例排行(BRIEF §五 P3)
#     K3 ncu           s3dkq4m2 / cfd2 / msc10848 / bcsstk30 / TSOPF 上 hash_heavy 对 hash_bm2 的硬件计数(尽力而为;
#                      报 ERR_NVGPUCTRPERM 就写 NEED_USER.md,不绕)
#   T 段(吃主机计时,过负载门才跑):
#     T1 电池          off / aa(与 off 相同的 A/A 对照臂)/ m2 / m2p,24 阵 × 5 轮交错
#     T2 无侵入        基线源码件对本件(开关关),官方 ab_pair.py 16×3
#   负载门(B 端判据,dialog/qa-006):load1 < 6 连续两次(间隔 60s)且没有外来进程 >50% CPU(nproc=32);
#     另加:我钉的卡上没有别的计算进程、另一张卡利用率 <10%。开跑时就安静 → 先跑 T 段再跑 K 段;
#     否则先跑 K 段,之后等安静窗口(最多 WAIT_MAX_S 秒,默认 10 小时),等不到就不跑 T 段。
#   T 段期间后台每分钟采一次 load;A/A 臂的 GM 偏离 1 超 1% 或非微级阵偏离超 5% → 该段判污染(scripts/aa_check.py)。
#   等待期间作业目录有 WAITING_QUIET 文件,计时期间有 TIMING_ACTIVE 文件。
# 只评测与 profiling:不转储、不标定、不动默认。回传:OUT_DIR/SUMMARY.md(先看这个)+ k2_* / k3_* / t1_* / t2_* 文件。
set -u
ROOT=$(pwd)
OUT=${OUT_DIR:?}
JD=${JOB_DIR:?}
ID=${JOB_ID:-prof_k1}
DATA=/home/xueyizhou/spgemm-challenge/data
SUM=$OUT/SUMMARY.md
BASE_SHA=141fbc46a690225559b235d79a7da33a7eb7b7029795078e3e9f927e9875011c   # 基线 src/spgemm_kernel_hash.cu(70d1dea)
LOAD_MAX=${LOAD_MAX:-6}
WAIT_MAX_S=${WAIT_MAX_S:-36000}
BIN=./spgemm_k1
RUNENV="USE_MEMPOOL=1 MP_HOST_MB=8192 D6EXP=0"
MYUID=$(id -u)
mkdir -p "$OUT" "$JD/k2" "$JD/k3" "$JD/keep"
: > "$SUM"
say() { echo "$*" | tee -a "$SUM"; }
TS0=$(date +%s); TSL=$TS0
stage() { local now; now=$(date +%s); echo; echo "################ $* ################ (+$((now - TSL))s 上一段,累计 $((now - TS0))s)"; TSL=$now; date -u +%FT%TZ; }
SAMPLER_PID=""
sampler_start() { ( while :; do echo "$(date +%s) $(cut -d' ' -f1 /proc/loadavg) $(foreign_hot)"; sleep 60; done ) > "$1" 2>/dev/null & SAMPLER_PID=$!; }
sampler_stop() { if [ -n "$SAMPLER_PID" ]; then kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""; fi; }
cleanup() { sampler_stop; rm -f "$JD/TIMING_ACTIVE" "$JD/WAITING_QUIET"; }
trap 'cleanup; echo "收到终止信号"; exit 143' TERM INT
finish() {   # $1=退出码 $2=结论一句话
  cleanup
  say ""; say "## 结束"; say "- 结论: $2"; say "- 主机(结束时): $(host_line)"
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
say "# prof_k1 汇总(设备侧 profiling:nsys 逐 kernel + ncu;带负载门的 v3 电池与无侵入重测)"
say ""
say "## S0 环境"
say "- 主机: $(hostname)  开始: $(date -u +%FT%TZ)  快照: $ROOT  nproc=$(nproc 2>/dev/null)"
say "- nvcc(PATH): $(command -v nvcc) / $(nvcc --version | tail -1)"
say "- python3: $(python3 --version 2>&1)"
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
say "- 负载门: load1 < $LOAD_MAX 连续两次(间隔 60s)且无外来 >50% CPU 进程,且本卡无别的计算进程、另一张卡利用率 <10%;等待上限 $((WAIT_MAX_S / 60)) 分钟"

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
    m2)   echo "HH_BM2=2";;
    m2p)  echo "HH_BM2=2 HH_BM2_P1=1";;
    m2L)  echo "HH_BM2=2 HH_BM2_L1=1";;
    *)    echo "HH_BM2=0";;
  esac
}

# ───────────────────────── K1 构建 ─────────────────────────
stage "K1 构建"
say ""; say "## K1 构建"
T0=$(date +%s)
make -k -j4 -f Makefile -f "jobs/$ID/keep.mk" TARGET=spgemm_k1 > "$JD/build_k1.log" 2>&1; RC=$?
tail -c 100000 "$JD/build_k1.log" > "$OUT/k1_build_tail.log"
say "- make TARGET=spgemm_k1: rc=$RC,用时 $(( $(date +%s) - T0 ))s;审计: $(grep -E '✓|✗|模板实例' "$JD/build_k1.log" | tr '\n' ';')"
if [ $RC = 0 ] && [ -x spgemm_k1 ] && [ "$(grep -c '✓' "$JD/build_k1.log")" -ge 2 ] && ! grep -q '✗' "$JD/build_k1.log"; then
  say "- K1: green"
else
  say "- **构建或审计失败**。报错行(前 80 条):"
  grep -nE "error|Error|✗|MISSING|undefined reference" "$JD/build_k1.log" | head -80 | tee -a "$SUM"
  finish 12 "红:K1 构建/审计失败;后续阶段未跑"
fi

# ───────────────────────── K2 nsys 逐 kernel ─────────────────────────
NSYS_POPT=""; NSYS_REPORT=""; NSYS_OUTMODE=""
nsys_stats() {   # $1=报告文件 $2=输出 csv;用试点选定的 NSYS_REPORT / NSYS_OUTMODE;有表返回 0
  local rep=$1 out=$2 base=${2%.csv}
  rm -f "$out"
  if [ "$NSYS_OUTMODE" = stdout ]; then
    timeout 900 nsys stats --report "$NSYS_REPORT" --format csv --output - "$rep" > "$out" 2> "$base.stats.log"
  else
    timeout 900 nsys stats --report "$NSYS_REPORT" --format csv --output "$base" "$rep" > "$base.stats.log" 2>&1
    [ -f "${base}_${NSYS_REPORT}.csv" ] && mv "${base}_${NSYS_REPORT}.csv" "$out"
  fi
  grep -qi "total time" "$out" 2>/dev/null
}
do_K2() {
  stage "K2 nsys 逐 kernel"
  say ""; say "## K2 逐 kernel 的 GPU 侧累计时长(nsys;设备侧时间戳,不吃主机 CPU 争用)"
  say "- 主机(K2 前): $(host_line)"
  if ! command -v nsys >/dev/null 2>&1; then say "- **nsys 不在 PATH**,K2 跳过"; return 0; fi
  say "- nsys: $(command -v nsys) / $(nsys --version 2>&1 | head -1)"
  local pm o PREP="" R M ok=0
  pm=$(find_mtx bcsstk17) || { say "- 缺试点阵 bcsstk17,K2 跳过"; return 0; }
  : > "$OUT/k2_pilot.log"
  # 试点一:哪组 profile 选项能产出报告(版本差异:选项写法、报告扩展名)
  for o in "-t cuda -s none --cpuctxsw=none --force-overwrite=true" "--trace=cuda --sample=none --force-overwrite true" "--force-overwrite true" ""; do
    rm -f "$JD/k2/pilot".*
    env $RUNENV METHOD=hash WARMUP_N=1 HH_BM2=2 timeout 900 nsys profile $o -o "$JD/k2/pilot" "$BIN" "$pm" > "$JD/k2/pilot.run.log" 2>&1 < /dev/null
    echo "nsys profile [$o] rc=$?" >> "$OUT/k2_pilot.log"
    PREP=$(ls "$JD/k2/pilot.nsys-rep" "$JD/k2/pilot.qdrep" 2>/dev/null | head -1)
    if [ -n "$PREP" ]; then NSYS_POPT=$o; break; fi
    tail -5 "$JD/k2/pilot.run.log" >> "$OUT/k2_pilot.log"
  done
  if [ -z "$PREP" ]; then
    say "- **nsys profile 四组选项都没产出报告**,K2 跳过(k2_pilot.log 里有各组的日志尾与帮助摘录)"
    nsys profile --help 2>&1 | head -60 >> "$OUT/k2_pilot.log"
    return 0
  fi
  # 试点二:哪个报告名、哪种输出方式能出表
  for R in cuda_gpu_kern_sum gpukernsum; do
    for M in stdout file; do
      NSYS_REPORT=$R; NSYS_OUTMODE=$M
      if nsys_stats "$PREP" "$JD/k2/pilot.csv"; then ok=1; break 2; fi
      echo "nsys stats [$R / $M] 没出表" >> "$OUT/k2_pilot.log"
      tail -3 "$JD/k2/pilot.stats.log" >> "$OUT/k2_pilot.log" 2>/dev/null
    done
  done
  if [ $ok = 0 ]; then
    say "- **nsys stats 两个报告名都没出表**,K2 跳过(k2_pilot.log 里有日志尾与可用报告清单)"
    nsys stats --help-reports 2>&1 | head -80 >> "$OUT/k2_pilot.log"
    rm -f "$JD/k2/pilot".*
    return 0
  fi
  head -12 "$JD/k2/pilot.csv" > "$OUT/k2_pilot_head.csv"
  say "- 试点通过:profile 选项 [$NSYS_POPT],报告 $NSYS_REPORT(输出方式 $NSYS_OUTMODE),试点表 $(wc -l < "$JD/k2/pilot.csv") 行(表头样例在 k2_pilot_head.csv)"
  rm -f "$JD/k2/pilot".*
  local nrun=0 nfail=0 name path arm e rep pfx rp
  for name in bcsstk17 af23560 msc10848 bcsstk30 cfd2 s3dkq4m2 offshore FEM_3D_thermal2 TSOPF_RS_b678_c2 web-Google case9 bloweybq \
              StocF-1465 rgg_n_2_21_s0 c-64 nlpkkt80 pre2 ldoor Hook_1498 c-big Bump_2911 cage15; do
    path=$(find_mtx "$name") || { echo "K2 缺 $name"; continue; }
    for rep in 1 2; do
      for arm in off m2 m2p m2L; do
        e=$(arm_env $arm)
        pfx="$JD/k2/${name}__${arm}__${rep}"
        rm -f "$pfx".nsys-rep "$pfx".qdrep "$pfx".sqlite
        env $RUNENV METHOD=hash WARMUP_N=1 $e timeout 1500 nsys profile $NSYS_POPT -o "$pfx" "$BIN" "$path" > "$pfx.run.log" 2>&1 < /dev/null
        rp=$(ls "$pfx.nsys-rep" "$pfx.qdrep" 2>/dev/null | head -1)
        nrun=$((nrun + 1))
        if [ -n "$rp" ] && nsys_stats "$rp" "$pfx.csv"; then
          :
        else
          nfail=$((nfail + 1)); echo "K2 FAIL $name $arm $rep"; tail -3 "$pfx.run.log"
        fi
        rm -f "$pfx".nsys-rep "$pfx".qdrep "$pfx".sqlite "$pfx.run.log" "$pfx.stats.log"
      done
    done
    echo "K2 done $name(累计 $nrun 发,失败 $nfail)"
  done
  say "- 采数:$nrun 发(22 阵 × 4 臂 × 2 轮,WARMUP_N=1),失败 $nfail 发"
  python3 scripts/kern_sum.py --dir "$JD/k2" --out "$OUT/k2_kern" > "$OUT/k2_kern_run.log" 2>&1
  say "- kern_sum rc=$?;全文 k2_kern.txt,长表 k2_kern_long.csv.gz"
  if [ -s "$OUT/k2_kern_long.csv.gz" ] && [ "$(stat -c %s "$OUT/k2_kern_long.csv.gz")" -gt 1900000 ]; then
    mv "$OUT/k2_kern_long.csv.gz" "$JD/keep/"; say "- 长表超 1.9MB,改留 $JD/keep/k2_kern_long.csv.gz"
  fi
  say ""
  say '```'
  awk '/===== ②/{exit} /^# |===== ①/{p=1} p' "$OUT/k2_kern.txt" 2>/dev/null | tee -a "$SUM"
  say '```'
  say "- 说明:比值是设备侧 Σkernel 时长之比(含预热轮,各臂相同),不含主机侧的发射与同步开销;核族排行与逐实例排行见 k2_kern.txt 的 ②③"
  say "- 主机(K2 后): $(host_line)"
}

# ───────────────────────── K3 ncu 机制计数 ─────────────────────────
do_K3() {
  stage "K3 ncu 机制计数"
  say ""; say "## K3 ncu 硬件计数(hash_heavy 对 hash_bm2:缓存命中、显存吞吐、指令数、占用;尽力而为)"
  if ! command -v ncu >/dev/null 2>&1; then say "- **ncu 不在 PATH**,K3 跳过"; return 0; fi
  say "- ncu: $(command -v ncu) / $(ncu --version 2>&1 | head -1)"
  local pm KRE='hash_heavy_kernel|hash_bm2_kernel|hh_bm2_part_kernel' t0 t1 rc REPLAY=""
  pm=$(find_mtx bcsstk17) || { say "- 缺试点阵 bcsstk17,K3 跳过"; return 0; }
  t0=$(date +%s)
  env $RUNENV METHOD=hash WARMUP_N=0 HH_BM2=2 timeout 900 ncu --target-processes all -k "regex:$KRE" --launch-count 3 -f -o "$JD/k3/pilot" "$BIN" "$pm" > "$JD/k3/pilot.log" 2>&1 < /dev/null
  rc=$?; t1=$(date +%s)
  tail -15 "$JD/k3/pilot.log" > "$OUT/k3_pilot.log"
  if grep -q "ERR_NVGPUCTRPERM" "$JD/k3/pilot.log"; then
    say "- **ncu 报 ERR_NVGPUCTRPERM(GPU 性能计数器权限)**:K3 未做,已写 NEED_USER.md;没有绕过"
    { echo "# NEED_USER"; echo; echo "- prof_k1 的 K3:ncu 报 ERR_NVGPUCTRPERM(GPU 性能计数器需要管理员放开权限)。未绕过,等处理后重跑 K3。"; } > "$OUT/NEED_USER.md"
    return 0
  fi
  if [ ! -s "$JD/k3/pilot.ncu-rep" ]; then
    say "- ncu 试点没产出报告(rc=$rc,用时 $((t1 - t0))s;日志尾在 k3_pilot.log),K3 跳过"
    ncu --help 2>&1 | head -60 >> "$OUT/k3_pilot.log"
    return 0
  fi
  timeout 300 ncu --import "$JD/k3/pilot.ncu-rep" --csv --page raw > "$JD/k3/pilot.raw.csv" 2>> "$JD/k3/pilot.log"
  head -c 4000 "$JD/k3/pilot.raw.csv" > "$OUT/k3_pilot_raw_head.csv"
  say "- 试点(bcsstk17,HH_BM2=2,前 3 次发射):rc=$rc,用时 $((t1 - t0))s,原始表 $(wc -l < "$JD/k3/pilot.raw.csv") 行(表头样例在 k3_pilot_raw_head.csv)"
  if [ $((t1 - t0)) -gt 240 ]; then REPLAY="--replay-mode application"; say "- 试点偏慢 → 其余改用 --replay-mode application"; fi
  rm -f "$JD/k3/pilot".*
  local spec name path arm e pfx n=0 ok=0 k3t0
  k3t0=$(date +%s)
  for spec in "cfd2:off m2 m2p" "s3dkq4m2:off m2 m2p" "msc10848:off m2" "bcsstk30:off m2" "TSOPF_RS_b678_c2:off m2 m2p"; do
    name=${spec%%:*}
    path=$(find_mtx "$name") || { echo "K3 缺 $name"; continue; }
    for arm in ${spec#*:}; do
      if [ $(( $(date +%s) - k3t0 )) -gt 4500 ]; then echo "K3 已用 75 分钟,余下的组跳过"; break 2; fi
      e=$(arm_env $arm); pfx="$JD/k3/${name}__${arm}"; n=$((n + 1))
      rm -f "$pfx.ncu-rep"
      env $RUNENV METHOD=hash WARMUP_N=0 $e timeout 1200 ncu --target-processes all $REPLAY -k "regex:$KRE" --launch-count 16 --set full -f -o "$pfx" "$BIN" "$path" > "$pfx.log" 2>&1 < /dev/null
      if [ ! -s "$pfx.ncu-rep" ]; then
        echo "K3 $name $arm:--set full 没产出,退回默认指标集"; tail -4 "$pfx.log"
        env $RUNENV METHOD=hash WARMUP_N=0 $e timeout 600 ncu --target-processes all $REPLAY -k "regex:$KRE" --launch-count 16 -f -o "$pfx" "$BIN" "$path" > "$pfx.log" 2>&1 < /dev/null
      fi
      if [ -s "$pfx.ncu-rep" ] && timeout 600 ncu --import "$pfx.ncu-rep" --csv --page raw > "$pfx.raw.csv" 2>> "$pfx.log" && grep -q "Kernel Name" "$pfx.raw.csv"; then
        ok=$((ok + 1))
        gzip -c "$pfx.raw.csv" > "$OUT/k3_raw_${name}__${arm}.csv.gz"
        mv "$pfx.ncu-rep" "$JD/keep/k3_${name}__${arm}.ncu-rep"
      else
        echo "K3 FAIL $name $arm"; tail -5 "$pfx.log"
      fi
    done
  done
  say "- 采数:$n 组(阵 × 臂;整段预算 75 分钟,超了余下跳过),成功 $ok 组;原始宽表 k3_raw_<阵>__<臂>.csv.gz,ncu 报告留在 $JD/keep/"
  python3 scripts/ncu_sum.py --dir "$JD/k3" --out "$OUT/k3_ncu" > "$OUT/k3_ncu_run.log" 2>&1
  say "- ncu_sum rc=$?;挑选指标的并排表 k3_ncu.txt,长表 k3_ncu_long.csv($(head -1 "$OUT/k3_ncu.txt" 2>/dev/null))"
  say "- 主机(K3 后): $(host_line)"
}

# ───────────────────────── T 计时段(过负载门才跑) ─────────────────────────
T_DONE=0
do_T() {
  T_DONE=1
  : > "$JD/TIMING_ACTIVE"
  stage "T1 电池(带 A/A 对照臂)"
  say ""; say "## T1 只评测电池(同一二进制;off / aa=与 off 完全相同的对照臂 / m2=位图形态不设门 / m2p=同上+Pass1 读 B16;5 轮交错)"
  say "- 主机(T1 前): $(host_line)"
  local EXTRA="" name p AA med1 T1V
  for name in c-big cfd2 s3dkq4m2 nlpkkt80 FEM_3D_thermal2 offshore msc10848 bcsstk30; do
    p=$(find_mtx "$name") && EXTRA="$EXTRA --extra $name=$p:hash"
  done
  sampler_start "$JD/load_T1.log"
  python3 scripts/ab_arms.py --bin "$BIN" --out "$OUT/t1_battery" --rounds 5 --gpu "$GPU" \
    --arm off: --arm aa: --arm m2:HH_BM2=2 --arm m2p:HH_BM2=2,HH_BM2_P1=1 --stat $EXTRA > "$OUT/t1_battery_run.log" 2>&1
  say "- ab_arms rc=$?;逐阵表 t1_battery.txt,逐发 t1_battery.csv,逐 bin 普查 t1_battery_stat.csv"
  sampler_stop
  cp "$JD/load_T1.log" "$OUT/t1_load.log" 2>/dev/null
  say "- 段内主机负载: $(load_stats "$JD/load_T1.log")"
  python3 scripts/aa_check.py "$OUT/t1_battery.csv" > "$OUT/t1_aa.txt" 2>&1
  AA=$(grep -oE "^AA_VERDICT [a-z]+" "$OUT/t1_aa.txt" | cut -d' ' -f2)
  say "- A/A 对照臂: $(grep -E '^AA_(N|GM|NONMICRO_MIN|NONMICRO_MAX|VERDICT) ' "$OUT/t1_aa.txt" | tr '\n' ';')(逐阵见 t1_aa.txt)"
  med1=$(load_med "$JD/load_T1.log")
  T1V="干净"
  [ "${AA:-nodata}" = clean ] || T1V="污染(A/A 对照臂未过)"
  if [ -n "$med1" ] && ! awk -v l="$med1" -v t="$LOAD_MAX" 'BEGIN{exit !(l+0 < t+0)}'; then T1V="污染(段内 load1 中位 $med1 ≥ $LOAD_MAX)"; fi
  say "- **T1 计时判定: $T1V**$([ "$T1V" = 干净 ] || echo ' —— 下面的 GM 与裁定只作参考,不作定谳')"
  grep -E "^(GM|裁定|=====)" "$OUT/t1_battery.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"

  stage "T2 无侵入电池"
  say ""; say "## T2 无侵入(A=spgemm_base:反打补丁后的基线源码;B=spgemm_k1 默认 env=开关关;官方 ab_pair.py 16×3)"
  if [ $HAVE_PATCH = 1 ]; then
    rm -rf "$JD/base"; mkdir -p "$JD/base"
    cp -r Makefile src include tools "$JD/base/"
    rm -f "$JD"/base/src/*.o
    ( cd "$JD/base" && patch -p1 -R < "$ROOT/jobs/$ID/bm2_src.patch" ) > "$OUT/t2_patch.log" 2>&1
    local GOT RCB
    GOT=$(sha256sum "$JD/base/src/spgemm_kernel_hash.cu" | cut -d' ' -f1)
    if [ "$GOT" = "$BASE_SHA" ]; then
      say "- 基线源码 sha256 核对通过"
      ( cd "$JD/base" && make -j4 TARGET=spgemm_base ) > "$JD/build_base.log" 2>&1; RCB=$?
      say "- make TARGET=spgemm_base: rc=$RCB;审计: $(grep -E '✓|✗|模板实例' "$JD/build_base.log" | tr '\n' ';')"
      if [ $RCB = 0 ] && [ -x "$JD/base/spgemm_base" ]; then
        cp "$JD/base/spgemm_base" "$ROOT/spgemm_base"
        quiet_once || { say "- T2 开始前主机已不满足安静判据:$(host_line);再等最多 60 分钟"; wait_quiet 3600 || say "- 仍未安静,T2 照跑并标注"; }
        say "- 主机(T2 前): $(host_line)"
        sampler_start "$JD/load_T2.log"
        python3 scripts/ab_pair.py spgemm_base spgemm_k1 "$OUT/t2_ab_base_vs_k1off.csv" 3 "$GPU" > "$OUT/t2_ab_run.log" 2>&1
        sampler_stop
        sed -n '/===== 判读/,$p' "$OUT/t2_ab_run.log" > "$OUT/t2_ab_verdict.txt"
        grep -E "^(GM|裁定|微级)" "$OUT/t2_ab_verdict.txt" | sed 's/^/    /' | tee -a "$SUM"
        med1=$(load_med "$JD/load_T2.log")
        say "- 段内主机负载: $(load_stats "$JD/load_T2.log")$([ -n "$med1" ] && ! awk -v l="$med1" -v t="$LOAD_MAX" 'BEGIN{exit !(l+0 < t+0)}' && echo ' **→ 污染,读数只作参考**')"
      else
        tail -30 "$JD/build_base.log" | tee -a "$SUM"
      fi
    else
      say "- **基线源码 sha256 不符**(得 $GOT),T2 未跑;见 t2_patch.log"
    fi
  else
    say "- 跳过:C 上没有 patch 命令"
  fi
  rm -f "$JD/TIMING_ACTIVE"
  say "- 主机(T 段后): $(host_line)"
}

# ───────────────────────── 编排:安静就先计时,否则先 profiling 再等 ─────────────────────────
stage "负载门(开跑时)"
say ""; say "## 负载门"
if quiet_twice; then
  say "- 开跑时主机满足安静判据($(host_line))→ 先跑 T 段,再跑 K 段"
  do_T
else
  say "- 开跑时主机不满足安静判据($(host_line))→ 先跑 K 段(不吃主机计时),之后等安静窗口"
fi
do_K2
if [ $T_DONE = 0 ] && quiet_twice; then
  say ""; say "- K2 之后主机已满足安静判据($(host_line))→ 先插入 T 段,再做 K3"
  do_T
fi
do_K3
if [ $T_DONE = 0 ]; then
  stage "等安静窗口"
  say ""; say "## 等安静窗口"
  : > "$JD/WAITING_QUIET"
  TW0=$(date +%s)
  if wait_quiet "$WAIT_MAX_S"; then
    rm -f "$JD/WAITING_QUIET"
    say "- 等了 $(( ($(date +%s) - TW0) / 60 )) 分钟后主机满足安静判据($(host_line))→ 跑 T 段"
    do_T
  else
    rm -f "$JD/WAITING_QUIET"
    say "- **T 段未跑**:等了 $(( ($(date +%s) - TW0) / 60 )) 分钟,主机始终不满足安静判据(结束时 $(host_line))"
  fi
fi
if [ $T_DONE = 1 ]; then
  finish 0 "K 段与 T 段都跑了;T 段是否干净看 T1 的计时判定与 T2 的段内负载"
else
  finish 0 "K 段跑了;T 段因主机未恢复没有跑(计时重测待下一个作业)"
fi
