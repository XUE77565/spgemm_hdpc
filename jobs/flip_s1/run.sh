#!/usr/bin/env bash
# flip_s1 —— HH_CENSUS 翻默认之后的自检(十几分钟):确认默认形态确实走了单发射普查,而且产物与显式 HH_CENSUS=1 / =0 一致
#
# 只验三件事,不计时:
#   C1 冒烟:def(不设)/ c0(HH_CENSUS=0)/ c1(HH_CENSUS=1)三臂 nnz 相同、无失败
#   C2 签名:清单里最小的 40 个阵 + 电池里 4 个阵,三臂的 [census-stat] 签名与 nnz 逐阵相同
#   C3 发射清单(nsys):def 对 c1 逐项相同(= 默认就是单发射);def 对 c0 的差异只在门普查核(12 次 → 2 次)
# 任一不符 → exit 16,SUMMARY 写明;B 端的 confirm20 等这个作业绿了再开。
set -u
ROOT=$(pwd)
OUT=${OUT_DIR:?}
JD=${JOB_DIR:?}
ID=${JOB_ID:-flip_s1}
DATA=/home/xueyizhou/spgemm-challenge/data
SUM=$OUT/SUMMARY.md
BASE_SHA=141fbc46a690225559b235d79a7da33a7eb7b7029795078e3e9f927e9875011c   # 基线 src/spgemm_kernel_hash.cu(70d1dea)
HDR_SHA=2fc25346de6deabc2eda6d49739ba0627a976969896b861e3798517bab59339b    # bm2_r4 / bm2_r5 的 S1 验过的 src/hh_bm2.cuh
LOAD_MAX=${LOAD_MAX:-6}
QUIET_WAIT_S=${QUIET_WAIT_S:-900}
BIN=./spgemm_f1
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
say "# flip_s1 汇总(HH_CENSUS 翻默认后的自检:默认形态确实走单发射普查,产物与两种显式设置一致)"
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
make -k -j4 -f Makefile -f "jobs/$ID/keep.mk" TARGET=spgemm_f1 > "$JD/build_f1.log" 2>&1; RC=$?   # -k:一个翻译单元报错后继续编另一个,报错一次出全
T1=$(date +%s)
tail -c 100000 "$JD/build_f1.log" > "$OUT/s2_build_tail.log"
say "- make TARGET=spgemm_f1: rc=$RC,用时 $((T1 - T0))s;审计: $(grep -E '✓|✗|模板实例' "$JD/build_f1.log" | tr '\n' ';')"
if [ $RC = 0 ] && [ -x spgemm_f1 ] && [ "$(grep -c '✓' "$JD/build_f1.log")" -ge 2 ] && ! grep -q '✗' "$JD/build_f1.log"; then
  if [ -f src/spgemm_kernel_hash.o ]; then rm -f src/main.o; MK="-f Makefile -f jobs/$ID/keep.mk"; else MK=""; fi
  make -j4 $MK WRITE_MTX=1 TARGET=spgemm_f1w > "$JD/build_f1w.log" 2>&1; RCW=$?
  say "- WRITE_MTX 件 spgemm_f1w: rc=$RCW($([ -n "$MK" ] && echo '只重编 main.cu' || echo '全量重编'))"
  [ $RCW = 0 ] && [ -x spgemm_f1w ] && S2=green
else
  say "- **构建或审计失败**。报错行(前 80 条,全文尾部在 s2_build_tail.log):"
  grep -nE "error|Error|✗|MISSING|undefined reference" "$JD/build_f1.log" | head -80 | tee -a "$SUM"
  say "- 日志尾:"; tail -25 "$JD/build_f1.log" | tee -a "$SUM"
fi
say "- S2: $S2"
[ $S1 = green ] || finish 11 "红:S1 微复现器未全绿(S2=$S2);后续阶段未跑"
[ $S2 = green ] || finish 12 "红:S2 构建/审计失败;后续阶段未跑"


# ───────────────────────── C1 冒烟 ─────────────────────────
stage "C1 冒烟"
say ""; say "## C1 冒烟(def=不设 / c0=HH_CENSUS=0 / c1=HH_CENSUS=1;各 1 发)"
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
C1=$(smoke g1 bcsstk17,F2,case9,c-64,TSOPF,web-Google --arm def: --arm c0:HH_CENSUS=0 --arm c1:HH_CENSUS=1)
say "- 6 阵 × 3 臂:nnz 与失败检查 → $C1"
[ $C1 = green ] || { tail -25 "$OUT/s3_g1_run.log" | tee -a "$SUM"; tail -20 "$OUT/s3_g1_fail.log" 2>/dev/null | tee -a "$SUM"; }

# ───────────────────────── C2 签名 ─────────────────────────
stage "C2 签名"
say ""; say "## C2 门普查签名与 nnz(三臂逐阵相同?)"
C2=red
{ echo mat; for n in bcsstk17 cfd2 web-Google ldoor; do echo "$n"; done; } > "$JD/rm/bat.txt"
python3 scripts/shadow428.py --bin "$BIN" --list reference/mats_428.csv --roots data/ocean/square,data/first100 --out "$JD/rm/c2a" \
  --arm def: --arm c0:HH_CENSUS=0 --arm c1:HH_CENSUS=1 --env HH_CENSUS_STAT=1 --method adaptive --rounds 1,1,1 \
  --sig-regex '^\[census-stat\].*$' --limit 40 --timeout 300 --gpu "$GPU" > "$OUT/c2a_run.log" 2>&1
python3 scripts/shadow428.py --bin "$BIN" --list "$JD/rm/bat.txt" --roots data/ocean/square,data/first100 --out "$JD/rm/c2b" \
  --arm def: --arm c0:HH_CENSUS=0 --arm c1:HH_CENSUS=1 --env HH_CENSUS_STAT=1 --method adaptive --rounds 1,1,1 \
  --sig-regex '^\[census-stat\].*$' --timeout 600 --gpu "$GPU" > "$OUT/c2b_run.log" 2>&1
C2BAD=$( { tail -n +2 "$JD/rm/c2a.csv"; tail -n +2 "$JD/rm/c2b.csv"; } 2>/dev/null | awk -F, '
  { if ($6+0 <= 0) { fail++; next } k=$1; n[k]++; if (!(k in z)) { z[k]=$7; s[k]=$8 } else if (z[k]!=$7 || s[k]!=$8) bad[k]=1 }
  END { for (k in n) { m++; if (n[k] != 3) short++; if (k in bad) b++ } printf "%d %d %d %d", m+0, b+0, short+0, fail+0 }')
set -- $C2BAD
say "- 比对 ${1:-0} 阵(清单最小的 40 个 + bcsstk17 / cfd2 / web-Google / ldoor):签名或 nnz 三臂不全同的 ${2:-?} 阵;没凑齐三臂的 ${3:-?} 阵;失败发 ${4:-?}"
if [ "${1:-0}" -ge 30 ] && [ "${2:-1}" = 0 ] && [ "${3:-1}" = 0 ] && [ "${4:-1}" = 0 ]; then C2=green; fi
say "- C2: $C2"

# ───────────────────────── C3 发射清单 ─────────────────────────
stage "C3 发射清单"
say ""; say "## C3 发射清单(nsys;核名 + 发射次数)"
C3=red
NSYS_POPT="-t cuda -s none --cpuctxsw=none --force-overwrite=true"; NSYS_REPORT=cuda_gpu_kern_sum; NSYS_OUTMODE=stdout
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
if command -v nsys >/dev/null 2>&1; then
  for name in bcsstk17 cfd2 web-Google ldoor; do
    path=$(find_mtx "$name") || continue
    for arm in def c0 c1; do
      case $arm in c0) e="HH_CENSUS=0";; c1) e="HH_CENSUS=1";; *) e="";; esac
      pfx="$JD/k2/${name}__${arm}__1"
      rm -f "$pfx".nsys-rep "$pfx".qdrep "$pfx".sqlite
      env $RUNENV METHOD=adaptive WARMUP_N=1 $e timeout -k 30 900 nsys profile $NSYS_POPT -o "$pfx" "$BIN" "$path" > "$pfx.run.log" 2>&1 < /dev/null
      rp=$(ls "$pfx.nsys-rep" "$pfx.qdrep" 2>/dev/null | head -1)
      if [ -n "$rp" ] && nsys_stats "$rp" "$pfx.csv"; then :; else echo "C3 nsys FAIL $name $arm"; tail -3 "$pfx.run.log"; fi
      rm -f "$pfx".nsys-rep "$pfx".qdrep "$pfx".sqlite "$pfx.run.log" "$pfx.stats.log"
    done
  done
  python3 scripts/launch_diff.py --dir "$JD/k2" --a def --b c1 > "$OUT/c3_def_vs_c1.txt" 2>&1
  python3 scripts/launch_diff.py --dir "$JD/k2" --a def --b c0 > "$OUT/c3_def_vs_c0.txt" 2>&1
  V1=$(grep -oE "^LAUNCH_DIFF_VERDICT [a-z]+" "$OUT/c3_def_vs_c1.txt" | cut -d' ' -f2)
  N1=$(grep -oE "^LAUNCH_DIFF_N [0-9]+" "$OUT/c3_def_vs_c1.txt" | cut -d' ' -f2)
  say "- def 对 c1(显式 HH_CENSUS=1):比对 ${N1:-0} 阵,${V1:-nodata}(same = 默认形态的发射清单与显式开完全一致)"
  say "- def 对 c0(显式 HH_CENSUS=0)的差异项(应只有门普查核:原样的 hh_gate_census_off_kernel 多、单发射的 hh_gate_census1_kernel 少):"
  grep -E "^== |→" "$OUT/c3_def_vs_c0.txt" | sed 's/^/    /' | cut -c1-170 | tee -a "$SUM"
  OTHER=$(grep -E "→" "$OUT/c3_def_vs_c0.txt" | grep -vc "hh_gate_census")
  ONDEF=$(grep -E "→" "$OUT/c3_def_vs_c0.txt" | grep -c "hh_gate_census1_kernel")
  say "- def 对 c0:门普查核之外的差异项 $OTHER 条;hh_gate_census1_kernel 出现在差异里的阵次 $ONDEF"
  if [ "${V1:-x}" = same ] && [ "${N1:-0}" -ge 3 ] && [ "$OTHER" = 0 ] && [ "$ONDEF" -ge 3 ]; then C3=green; fi
else
  say "- 没有 nsys,C3 没做"
fi
say "- C3: $C3"
rm -rf "$JD/k2"
if [ $C1 = green ] && [ $C2 = green ] && [ $C3 = green ]; then
  finish 0 "绿:默认形态走单发射普查(发射清单与显式 HH_CENSUS=1 相同),签名与 nnz 三臂一致"
fi
finish 16 "红:C1=$C1 C2=$C2 C3=$C3 —— 翻默认的提交有问题,confirm20 先别开"
