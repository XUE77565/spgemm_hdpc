#!/usr/bin/env bash
# cen_rm1 —— 影子 428 之后的三件事:R1 待复测阵加轮复测(翻 HH_CENSUS 默认之前必须做完,dialog/qa-009);
#            R2 两遍式在生产入口(METHOD=adaptive)下的值级;R4 训练池按生产入口重做逐行标定(门值的合法出处);
#            R3 合一形态输家 / 赢家的画像(设备侧逐核、逐 bin 普查、ncu 计数)
#
# 影子(prof_p1)的结论:A/A 干净(全体 1.0013、非微级 1.0026,超 ±5% 的非微级阵 6 个 = 1.8%);nnz 与门普查签名 428 阵全同。
#   cen/off GM 0.9925(非微级 0.9889):按写定的判据是红——可裁决阵 rajat16 1.267,另有不裁决阵 rajat25 / rajat17 / rajat28 >1.10。
#     rajat 族是基线自身双峰(同一配置三发里就有 49ms 与 62ms 两种读数),三发中位说明不了问题 → R1 给每阵 21 发、逐阵判。
#   cm2/off GM 0.9490(非微级 0.9365,148 阵快 5% 以上):红——11 个可裁决阵 >1.10(fem_filter 1.45、torso1 1.42、nd12k 1.24 …),
#     是 accumulate 相位真的变慢。两遍式默认不翻;R2 / R3 为后续的门设计留证据(R3 只作画像,不用来定门值:428 是测试账)。
#   生产入口的 dispatcher 给 hash 路径设了 DITER_MIN_FLOP=999999999(密行不走 dense-iter,全进哈希梯),HH 行人口比 METHOD=hash 的电池大得多;
#     此前两遍式的值级都是在 METHOD=hash 下做的,所以补 R2。
# 回传:OUT_DIR/SUMMARY.md(先看这个)+ r1_* / r2 表 / r3_* 文件。
set -u
ROOT=$(pwd)
OUT=${OUT_DIR:?}
JD=${JOB_DIR:?}
ID=${JOB_ID:-cen_rm1}
DATA=/home/xueyizhou/spgemm-challenge/data
SUM=$OUT/SUMMARY.md
BASE_SHA=141fbc46a690225559b235d79a7da33a7eb7b7029795078e3e9f927e9875011c   # 基线 src/spgemm_kernel_hash.cu(70d1dea)
HDR_SHA=2fc25346de6deabc2eda6d49739ba0627a976969896b861e3798517bab59339b    # bm2_r4 / bm2_r5 的 S1 验过的 src/hh_bm2.cuh
LOAD_MAX=${LOAD_MAX:-6}
QUIET_WAIT_S=${QUIET_WAIT_S:-900}
BIN=./spgemm_m1
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
say "# cen_rm1 汇总(影子待复测阵加轮复测;两遍式在生产入口下的值级;合一形态输家 / 赢家画像)"
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
    ( "$JD/hh_repro5" 40 42; echo "rc=$?"; "$JD/hh_repro5" 25 2026; echo "rc=$?" ) > "$OUT/s1_repro.log" 2>&1
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
make -k -j4 -f Makefile -f "jobs/$ID/keep.mk" TARGET=spgemm_m1 > "$JD/build_m1.log" 2>&1; RC=$?   # -k:一个翻译单元报错后继续编另一个,报错一次出全
T1=$(date +%s)
tail -c 100000 "$JD/build_m1.log" > "$OUT/s2_build_tail.log"
say "- make TARGET=spgemm_m1: rc=$RC,用时 $((T1 - T0))s;审计: $(grep -E '✓|✗|模板实例' "$JD/build_m1.log" | tr '\n' ';')"
if [ $RC = 0 ] && [ -x spgemm_m1 ] && [ "$(grep -c '✓' "$JD/build_m1.log")" -ge 2 ] && ! grep -q '✗' "$JD/build_m1.log"; then
  if [ -f src/spgemm_kernel_hash.o ]; then rm -f src/main.o; MK="-f Makefile -f jobs/$ID/keep.mk"; else MK=""; fi
  make -j4 $MK WRITE_MTX=1 TARGET=spgemm_m1w > "$JD/build_m1w.log" 2>&1; RCW=$?
  say "- WRITE_MTX 件 spgemm_m1w: rc=$RCW($([ -n "$MK" ] && echo '只重编 main.cu' || echo '全量重编'))"
  [ $RCW = 0 ] && [ -x spgemm_m1w ] && S2=green
else
  say "- **构建或审计失败**。报错行(前 80 条,全文尾部在 s2_build_tail.log):"
  grep -nE "error|Error|✗|MISSING|undefined reference" "$JD/build_m1.log" | head -80 | tee -a "$SUM"
  say "- 日志尾:"; tail -25 "$JD/build_m1.log" | tee -a "$SUM"
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
S3=$(smoke g1 bcsstk17,F2,case9,c-64,TSOPF,web-Google --arm off: --arm cen:HH_CENSUS=1 --arm cm2:HH_CENSUS=1,HH_BM2=2,HH_BM2_LAZY=2)
say "- off/cen/cm2,6 阵 → $S3"
if [ $S3 != green ]; then tail -25 "$OUT/s3_g1_run.log" | tee -a "$SUM"; tail -20 "$OUT/s3_g1_fail.log" 2>/dev/null | tee -a "$SUM"; finish 13 "红:S3 冒烟未过;后续阶段未跑"; fi

# ───────────────────────── R1 待复测阵加轮复测 ─────────────────────────
stage "R1 待复测阵加轮复测"
say ""; say "## R1 待复测阵加轮复测(off / aa / cen;每阵 21 发、逐阵逐轮交错;env 与影子、confirm20 相同:METHOD=adaptive WARMUP_N=3 USE_MEMPOOL=1 MP_HOST_MB=8192 D6EXP=0)"
say "- 名单:影子里 A/A 超 ±5% 的 6 个不裁决阵、cen 臂 >1.10 的可裁决阵 rajat16、rajat 族其余成员,以及 cen 臂 >1.02 的非微级阵"
{ echo mat; for n in rajat16 rajat17 rajat18 rajat20 rajat25 rajat28 rajat31 Zd_Jac6_db bcsstk33 circuit_4 TSOPF_RS_b2052_c1 c-45 std1_Jac3 CurlCurl_4 av41092 Stanford_Berkeley; do echo "$n"; done; } > "$JD/rm/list.txt"
say "- 主机(R1 前): $(host_line)"
: > "$JD/TIMING_ACTIVE"
sampler_start "$JD/load_R1.log"
python3 scripts/shadow428.py --bin "$BIN" --list "$JD/rm/list.txt" --roots data/ocean/square,data/first100 --out "$JD/rm/r1" \
  --arm off: --arm aa: --arm cen:HH_CENSUS=1 --env HH_CENSUS_STAT=1 --method adaptive --rounds 21,21,21 --sig-regex '^\[census-stat\].*$' --gpu "$GPU" > "$OUT/r1_run.log" 2>&1
say "- shadow428 rc=$?;$(tail -1 "$OUT/r1_run.log")"
sampler_stop
rm -f "$JD/TIMING_ACTIVE"
say "- 段内主机负载: $(load_stats "$JD/load_R1.log")"
python3 scripts/remeasure_sum.py "$JD/rm/r1" > "$OUT/r1_remeasure.txt" 2>&1
say "- remeasure_sum rc=$?;逐阵全文 r1_remeasure.txt,逐发 r1.csv.gz,相位长表 r1_phases.csv.gz"
for f in r1.csv r1_phases.csv; do [ -s "$JD/rm/$f" ] && gzip -c "$JD/rm/$f" > "$OUT/$f.gz"; done
say ""
say '```'
grep -E "^== |该阵 A/A|放行|仍红|^REMEASURE_RED|^# " "$OUT/r1_remeasure.txt" | tee -a "$SUM"
say '```'
say "- 判法(跑之前写定,scripts/remeasure_sum.py 文件头;是否采用由 dialog/qa-011 定):稳定阵看中位比 ≤1.10;不稳阵(该阵自己的 A/A 超 ±5%)"
say "  看最小值 / 下四分位 / 上四分位三档比值依次 ≤1.05 / 1.05 / 1.10。"

# ───────────────────────── R2 生产入口下的值级 ─────────────────────────
vl_dump() {   # $1=name $2=path $3=臂名 → $JD/vl/<name>_<臂名>.mtx;回显 "nnz taken skip"
  local name=$1 path=$2 tag=$3 log
  log="$JD/vl/${name}_${tag}.log"
  rm -f "results/matrices/$name/self_product_hash.mtx"
  env $RUNENV METHOD=hash WARMUP_N=0 $(arm_env "$tag") HH_BM2_STAT=1 timeout 1800 ./spgemm_m1w "$path" > "$log" 2>&1
  local rc=$?
  if [ $rc != 0 ] || [ ! -s "results/matrices/$name/self_product_hash.mtx" ]; then
    echo "RUNFAIL rc=$rc"; tail -15 "$log" >&2; return 1
  fi
  mv "results/matrices/$name/self_product_hash.mtx" "$JD/vl/${name}_${tag}.mtx"
  local nnz taken skip
  nnz=$(grep -oE "Result C:.*nnz = [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+$")
  taken=$(grep -oE "taken_rows=[0-9]+" "$log" | cut -d= -f2 | awk '{s+=$1}END{print s+0}')
  skip=$(grep -oE "skip_rows=[0-9]+" "$log" | cut -d= -f2 | awk '{s+=$1}END{print s+0}')
  grep "bm2-stat" "$log" | sed "s/^/[$name $tag] /" >> "$OUT/s4_value_stat.log"
  echo "$nnz $taken $skip"
}
gate() {      # $1=name $2=tag(m1/m2)→ 回显 "PASS|FAIL 细节"(对 <name>_off.mtx)
  ln -sfn "$JD/vl/${1}_${2}.mtx" "$JD/vl/${1}_on.mtx"
  local g; g=$(bash scripts/mtx_gate_fast.sh "$JD/vl/$1" 2>&1)
  rm -f "$JD/vl/${1}_on.mtx"
  echo "$(echo "$g" | grep -oE "PASS|FAIL" | tail -1) $(echo "$g" | grep -oE "col_mismatch=.*" | head -1)"
}
VL_TAKEN=0; S4=green

gate_diag() {   # $1=name:三臂转储 → 差分门 + l3_diag
  local name=$1 path ok=1 mode tag gv ne ob verdict
  path=$(find_mtx "$name") || { say "| $name | - | - | - | 缺文件,跳过 | - | - | - |"; return 0; }
  local free_gb; free_gb=$(df -BG --output=avail "$JD" | tail -1 | tr -dc 0-9)
  [ "${free_gb:-0}" -ge 60 ] || { say "| $name | - | - | - | 磁盘余量 <60G,跳过 | - | - | - |"; return 0; }
  echo "---- 差分门+归一误差 $name ($path)"
  declare -A Q=()
  for tag in off m1 m2; do
    Q[$tag]=$(vl_dump "$name" "$path" $tag) || { say "| $name | $tag | ${Q[$tag]} | - | **运行失败** | - | - | 红 |"; S4=red; ok=0; }
  done
  [ $ok = 1 ] || { rm -f "$JD"/vl/${name}_*.mtx; return 0; }
  if [ $HAVE_SCIPY = 1 ]; then
    timeout 3600 python3 scripts/l3_diag.py "$path" "off=$JD/vl/${name}_off.mtx" "m1=$JD/vl/${name}_m1.mtx" "m2=$JD/vl/${name}_m2.mtx" \
      --list 30 --csv "$OUT/s4_l3diag_${name}.csv" > "$OUT/s4_l3diag_${name}.txt" 2>&1 || echo "l3_diag rc=$? ($name)"
    tail -12 "$OUT/s4_l3diag_${name}.txt"
  fi
  for tag in off m1 m2; do
    set -- ${Q[$tag]}
    ne=$(grep -E "^NERR_MAX $tag " "$OUT/s4_l3diag_${name}.txt" 2>/dev/null | awk '{print $3}')
    ob=$(grep -E "^OFFICIAL_BAD_ROWS $tag " "$OUT/s4_l3diag_${name}.txt" 2>/dev/null | awk '{print $3}')
    if [ $tag = off ]; then
      say "| $name | off | $1 | $2 / $3 | - | ${ob:-?} | ${ne:-?} | 对照 |"
      continue
    fi
    VL_TAKEN=$((VL_TAKEN + $2))
    gv=$(gate "$name" $tag)
    verdict=绿
    [ "${gv%% *}" = PASS ] || verdict=红
    [ "${Q[off]%% *}" = "$1" ] || verdict="红(nnz≠off)"
    if [ -n "$ne" ]; then
      case "$ne" in
        *inf*|*nan*) verdict="红(nerr=$ne)";;
        *) awk -v v="$ne" 'BEGIN{exit !(v+0 <= 1e-10)}' || verdict="红(nerr 超 1e-10)";;
      esac
    else
      [ "$verdict" = 绿 ] && verdict="绿(l3_diag 无输出,仅凭差分门)"
    fi
    case "$verdict" in 红*) S4=red;; esac
    say "| $name | $tag | $1 | $2 / $3 | $gv | ${ob:-?} | ${ne:-?} | $verdict |"
  done
  rm -f "$JD"/vl/${name}_*.mtx
}
gate_only() {   # $1=name $2..=臂名:只做差分门(特大阵)
  local name=$1; shift
  local path r0 r tag gv
  path=$(find_mtx "$name") || { say "| $name | - | - | - | 缺文件,跳过 |"; return 0; }
  local free_gb; free_gb=$(df -BG --output=avail "$JD" | tail -1 | tr -dc 0-9)
  [ "${free_gb:-0}" -ge 60 ] || { say "| $name | - | - | - | 磁盘余量 <60G,跳过 |"; return 0; }
  echo "---- 差分门 $name ($path)"
  r0=$(vl_dump "$name" "$path" off) || { say "| $name | off | $r0 | - | **基线臂运行失败** |"; S4=red; return 0; }
  for tag in "$@"; do
    r=$(vl_dump "$name" "$path" "$tag") || { say "| $name | $tag | $r | - | **运行失败** |"; S4=red; continue; }
    set -- $r
    VL_TAKEN=$((VL_TAKEN + $2))
    gv=$(gate "$name" $tag)
    [ "${gv%% *}" = PASS ] || S4=red
    [ "${r0%% *}" = "$1" ] || { S4=red; gv="$gv NNZ≠off(${r0%% *})"; }
    say "| $name | $tag | $1 | $2 / $3 | $gv |"
    rm -f "$JD/vl/${name}_${tag}.mtx"
  done
  rm -f "$JD/vl/${name}_off.mtx"
}
vl_dump_ad() {   # $1=name $2=path $3=臂名 → $JD/vl/<name>_<臂名>.mtx;回显 "nnz taken skip"
  local name=$1 path=$2 tag=$3 log
  log="$JD/vl/${name}_${tag}.log"
  rm -f "results/matrices/$name/self_product_adaptive.mtx"
  env $RUNENV METHOD=adaptive WARMUP_N=0 $(arm_env "$tag") HH_BM2_STAT=1 timeout 1800 ./spgemm_m1w "$path" > "$log" 2>&1
  local rc=$?
  if [ $rc != 0 ] || [ ! -s "results/matrices/$name/self_product_adaptive.mtx" ]; then
    echo "RUNFAIL rc=$rc"; tail -15 "$log" >&2; return 1
  fi
  mv "results/matrices/$name/self_product_adaptive.mtx" "$JD/vl/${name}_${tag}.mtx"
  local nnz taken skip
  nnz=$(grep -oE "Result C:.*nnz = [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+$")
  taken=$(grep -oE "taken_rows=[0-9]+" "$log" | cut -d= -f2 | awk '{s+=$1}END{print s+0}')
  skip=$(grep -oE "skip_rows=[0-9]+" "$log" | cut -d= -f2 | awk '{s+=$1}END{print s+0}')
  grep "bm2-stat" "$log" | sed "s/^/[$name $tag] /" >> "$OUT/s4_value_stat.log"
  echo "$nnz $taken $skip"
}
gate_only_ad() {   # $1=name $2..=臂名:只做差分门(特大阵)
  local name=$1; shift
  local path r0 r tag gv
  path=$(find_mtx "$name") || { say "| $name | - | - | - | 缺文件,跳过 |"; return 0; }
  local free_gb; free_gb=$(df -BG --output=avail "$JD" | tail -1 | tr -dc 0-9)
  [ "${free_gb:-0}" -ge 60 ] || { say "| $name | - | - | - | 磁盘余量 <60G,跳过 |"; return 0; }
  echo "---- 差分门 $name ($path)"
  r0=$(vl_dump_ad "$name" "$path" off) || { say "| $name | off | $r0 | - | **基线臂运行失败** |"; S4=red; return 0; }
  for tag in "$@"; do
    r=$(vl_dump_ad "$name" "$path" "$tag") || { say "| $name | $tag | $r | - | **运行失败** |"; S4=red; continue; }
    set -- $r
    VL_TAKEN=$((VL_TAKEN + $2))
    gv=$(gate "$name" $tag)
    [ "${gv%% *}" = PASS ] || S4=red
    [ "${r0%% *}" = "$1" ] || { S4=red; gv="$gv NNZ≠off(${r0%% *})"; }
    say "| $name | $tag | $1 | $2 / $3 | $gv |"
    rm -f "$JD/vl/${name}_${tag}.mtx"
  done
  rm -f "$JD/vl/${name}_off.mtx"
}
stage "R2 生产入口下的值级"
say ""; say "## R2 值级·生产入口(METHOD=adaptive;各臂对 off 的差分门)"
say ""
say "此前两遍式的值级都在 METHOD=hash 下做。生产入口把密行也送进哈希梯,HH 行人口不同,这里补验:影子里合一形态最快与最慢的阵各取几个。"
say ""
say "| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |"
say "|---|---|---|---|---|"
VL_TAKEN=0; S4=green
gate_only_ad case9 cen cm2
gate_only_ad bloweybq cm2
gate_only_ad net50 cm2
gate_only_ad hangGlider_4 cm2
gate_only_ad filter3D cm2
gate_only_ad pkustk12 cm2
gate_only_ad fem_filter cm2
gate_only_ad nd12k cm2
gate_only_ad Si41Ge41H72 cm2
rm -rf "$ROOT"/results/matrices/*/self_product*.mtx
R2=$S4
say ""
say "- R2: $R2;走两遍式的行数合计 $VL_TAKEN"
[ $R2 = green ] || say "- **R2 有红**:两遍式在生产入口下的值不对(见上表)——在修好之前 HH_BM2 不得进默认;R1 的门普查复测不受影响"

# ───────────────────────── R4 训练池按生产入口重做逐行标定 ─────────────────────────
stage "R4 训练池按生产入口重标定"
say ""; say "## R4 行门标定·生产入口(只用 data/train_disp;METHOD=adaptive;逐行配对周期;位图形态,off / m1 / m2)"
say "- 出处:bm2_r5 的标定是在 METHOD=hash 下做的(位图形态 34 阵全部 R<1,不需要门)。生产入口把密行也送进哈希梯,影子里输的那类行当时没进 HH bin。"
say "  这里换成生产入口重做,看训练池里有没有会输的行、(FX, CX) 门能不能把它们分出来——门值只能从这里或物理式来,不看 428。"
mkdir -p "$JD/cal"
NTR=0; : > "$JD/cal/list.txt"
if [ -d data/train_disp ]; then
  while read -r sz f; do
    b=$(basename "$f" .mtx)
    [ "$b" = af23560 ] && continue
    [ -f "data/ocean/square/$b.mtx" ] || [ -f "data/first100/$b.mtx" ] || [ -n "$(find -L data/flex142 data/external_72 -maxdepth 3 -name "$b.mtx" 2>/dev/null | head -1)" ] && continue
    echo "$b $f" >> "$JD/cal/list.txt"; NTR=$((NTR + 1))
    [ $NTR -ge 60 ] && break
  done < <(find -L data/train_disp -maxdepth 2 -name '*.mtx' -size +1M -size -800M -printf '%s %p\n' 2>/dev/null | sort -rn)
fi
say "- 训练池候选 $NTR 阵(1MB-800MB,非测试账同名,不含 af23560)"
NRUN=0; NFAIL=0
while read -r b f; do
  for arm in off m1 m2; do
    case $arm in off) e="HH_BM2=0";; m1) e="HH_BM2=1 HH_BM2_LAZY=0";; *) e="HH_BM2=2 HH_BM2_LAZY=0";; esac
    env $RUNENV METHOD=adaptive WARMUP_N=1 $e HH_BM2_STAT=1 HASH_ROWTIME="$JD/cal/$b.$arm.rt" timeout 600 "$BIN" "$f" < /dev/null > /dev/null 2> "$JD/cal/$b.$arm.log"
    rc=$?; NRUN=$((NRUN + 1))
    if [ $rc != 0 ] || [ ! -s "$JD/cal/$b.$arm.rt" ]; then NFAIL=$((NFAIL + 1)); echo "cal 无行计时 $b $arm rc=$rc(dispatcher 走了 merge3 的阵属正常)"; rm -f "$JD/cal/$b.off.rt"; break; fi
  done
done < "$JD/cal/list.txt"
say "- 采数:$NRUN 发;没出行计时的 $NFAIL 发(dispatcher 路由到 merge3 的阵没有哈希行,属正常);逐行周期文件 $(ls "$JD/cal"/*.off.rt 2>/dev/null | wc -l) 组"
if [ $HAVE_SCIPY = 1 ] && ls "$JD/cal"/*.off.rt >/dev/null 2>&1; then
  timeout 1800 python3 scripts/bm2_gate_cal.py --dir "$JD/cal" --out "$OUT/r4_cal" > "$OUT/r4_cal_run.log" 2>&1
  say "- bm2_gate_cal rc=$?;见 r4_cal.txt / r4_cal_cells.csv / r4_cal.env"
  grep -E "^(# bm2_gate_cal|=====|  全收|  → |  实例税|# auto|# 位图形态|# legacy)" "$OUT/r4_cal.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"
  [ -s "$OUT/r4_cal.txt" ] || { tail -20 "$OUT/r4_cal_run.log" | tee -a "$SUM"; }
else
  say "- 没有可用的逐行周期文件(或没有 scipy),标定没跑"
fi
rm -rf "$JD/cal"

# ───────────────────────── R3 合一形态输家 / 赢家的画像 ─────────────────────────
stage "R3 输家 / 赢家画像"
say ""; say "## R3 合一形态(cm2)在生产入口下的输家 / 赢家画像(只作画像;428 是测试账,不据此定门值)"
WIN="pkustk12 net50 filter3D hangGlider_4 hangGlider_5 net75 hangGlider_3 net100 conf5_4-8x8-15 case9 bloweybq TSOPF_FS_b9_c6"
LOSE="fem_filter torso1 nd12k pattern1 nd24k nemeth23 Si41Ge41H72 nemeth22 rajat25 nemeth25 nemeth24 Ga41As41H72"
: > "$OUT/r3_bm2stat.txt"
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
R3N=0; R3F=0
for name in $LOSE $WIN; do
  path=$(find_mtx "$name") || { echo "R3 缺 $name"; continue; }
  env $RUNENV METHOD=adaptive WARMUP_N=0 HH_CENSUS=1 HH_BM2=2 HH_BM2_LAZY=2 HH_BM2_STAT=1 timeout 900 "$BIN" "$path" 2>&1 < /dev/null | grep -a "bm2-stat" | sed "s/^/[$name] /" >> "$OUT/r3_bm2stat.txt"
  if command -v nsys >/dev/null 2>&1; then
    for arm in off cm2; do
      e=$(arm_env $arm); pfx="$JD/k2/${name}__${arm}__1"
      rm -f "$pfx".nsys-rep "$pfx".qdrep "$pfx".sqlite
      env $RUNENV METHOD=adaptive WARMUP_N=1 $e timeout 1500 nsys profile $NSYS_POPT -o "$pfx" "$BIN" "$path" > "$pfx.run.log" 2>&1 < /dev/null
      rp=$(ls "$pfx.nsys-rep" "$pfx.qdrep" 2>/dev/null | head -1)
      R3N=$((R3N + 1))
      if [ -n "$rp" ] && nsys_stats "$rp" "$pfx.csv"; then :; else R3F=$((R3F + 1)); echo "R3 nsys FAIL $name $arm"; tail -3 "$pfx.run.log"; fi
      rm -f "$pfx".nsys-rep "$pfx".qdrep "$pfx".sqlite "$pfx.run.log" "$pfx.stats.log"
    done
  fi
done
say "- 逐 bin 普查([bm2-stat]:每个 bin 走 / 未走两遍式的行数与 Σflop、SMEM 字数):r3_bm2stat.txt($(wc -l < "$OUT/r3_bm2stat.txt") 行)"
python3 scripts/kern_sum.py --dir "$JD/k2" --out "$OUT/r3_kern" > "$OUT/r3_kern_run.log" 2>&1
say "- nsys 设备侧(off 对 cm2,METHOD=adaptive,各 1 轮):$R3N 发,失败 $R3F 发;全文 r3_kern.txt,长表 r3_kern_long.csv.gz"
say '```'
awk '/===== ②/{exit} /===== ①/{p=1} p' "$OUT/r3_kern.txt" 2>/dev/null | tee -a "$SUM"
say '```'
rm -rf "$JD/k2"
NCU=/opt/nvidia/nsight-compute/2025.1.1/ncu
if [ -x "$NCU" ]; then
  KRE='hash_heavy_kernel|hash_bm2_kernel'
  K3N=0; K3OK=0; K3T0=$(date +%s)
  for name in fem_filter nd12k Si41Ge41H72 nemeth23 case9 net50; do
    path=$(find_mtx "$name") || continue
    for arm in off cm2; do
      if [ $(( $(date +%s) - K3T0 )) -gt 3000 ]; then echo "R3 ncu 已用 50 分钟,余下的组跳过"; break 2; fi
      e=$(arm_env $arm); pfx="$JD/k3/${name}__${arm}"; K3N=$((K3N + 1))
      rm -f "$pfx.ncu-rep"
      env $RUNENV METHOD=adaptive WARMUP_N=0 $e timeout 900 "$NCU" --target-processes all --replay-mode application -k "regex:$KRE" --launch-count 16 --set full -f -o "$pfx" "$BIN" "$path" > "$pfx.log" 2>&1 < /dev/null
      if [ -s "$pfx.ncu-rep" ] && timeout 600 "$NCU" --import "$pfx.ncu-rep" --csv --page raw > "$pfx.raw.csv" 2>> "$pfx.log" && grep -q "Kernel Name" "$pfx.raw.csv"; then
        K3OK=$((K3OK + 1))
        gzip -c "$pfx.raw.csv" > "$OUT/r3_ncu_raw_${name}__${arm}.csv.gz"
      else
        echo "R3 ncu FAIL $name $arm"; grep -E "==ERROR==" "$pfx.log" | head -3
      fi
    done
  done
  python3 scripts/ncu_sum.py --dir "$JD/k3" --out "$OUT/r3_ncu" > "$OUT/r3_ncu_run.log" 2>&1
  say "- ncu(off 对 cm2,6 阵,application 重放):$K3N 组,成功 $K3OK 组;逐发射的指令数 / 显存读字节 / 缓存命中见 r3_ncu_long.csv 与 r3_ncu.txt"
  if [ -s "$OUT/r3_ncu_long.csv" ] && [ "$(stat -c %s "$OUT/r3_ncu_long.csv")" -gt 1900000 ]; then gzip -f "$OUT/r3_ncu_long.csv"; fi
else
  say "- ncu 跳过:$NCU 不存在"
fi
finish 0 "S3 冒烟 $S3;R1 的逐阵判定看 r1_remeasure.txt 的 REMEASURE_RED 行;R2 值级 $R2;R3 画像见各文件"
