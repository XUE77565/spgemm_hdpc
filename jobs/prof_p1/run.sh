#!/usr/bin/env bash
# prof_p1 —— 全 428 影子(off / aa / cen / cm2)+ 默认形态相位地图(BRIEF §五 P1)+ 上一轮作业的三项补账 + ncu 机制计数
#
# 上一轮(cen_r1)的结论:门普查行为保持 24 阵 0 不一致;差分门全 PASS;nsys 设备侧 cen/off GM 0.9772、cm2/off 0.9461;
#   A/A 认证电池 cen 0.9706 / m2 0.9725 / cm2 0.9605(非微级最差 1.022 / 1.023 / 1.038,A/A 0.9960);小阵探针第 2 次 A/A 过:
#   cen 0.919 / cm2 0.937;带对照臂的无侵入 base/off 1.0007(A/A 1.0011)。源码自那一轮以来未变。
# 臂:off=默认;aa=同 off;cen=HH_CENSUS=1;cm2=HH_CENSUS=1,HH_BM2=2,HH_BM2_LAZY=2(普查 + 划分合一)。
# 阶段(前面是短的补账,最长的影子放最后):
#   S1/S2  沿用微复现器;构建 spgemm_p1 + 基线源码件 spgemm_base
#   S3     冒烟 off/cen/cm2/m2
#   P0     影子采数脚本试跑(8 个最小的阵 × 1 轮);不通就停(exit 15),不白占十几个小时
#   O1     发射清单比对(dialog/qa-008 的验证注):nsys 下基线件对本件开关全关,22 阵,核名 + 发射次数逐项相同?
#   O2     官方 ab_pair 16×9(参考轨:无对照臂,只入档,不作裁定)
#   O3     msc10848 加轮复核(上一轮它在 cen / cm2 臂是 1.022 / 1.038;21 轮,带 A/A)
#   K3     ncu 2025.1.1(上一轮探到可用,application 重放):hash_heavy 对 hash_bm2 的硬件计数,5 阵,60 分钟预算
#   P1     全 428 影子:清单 reference/mats_428.csv;METHOD=adaptive、生产默认 env;轮数按文件大小分档 5/3/2;
#          每发带 [census-stat] 签名(行为保持的全集证据);计时轮的全部相位落长表 → 默认形态相位地图与各池前 20
#   P2     地图对齐发:off 臂单发、HASH_LHT23=0(对齐 phase_decomp_current.sh 的旧口径,供与本机侧 atlas 对照;放最后,超时不伤主体)
# 计时口径:A/A 认证(dialog/qa-007、qa-008、qa-009);判据不过整段作废,不挑段。只评测:不标定、不动默认。
# 回传:OUT_DIR/SUMMARY.md(先看这个)+ p1_* / o1_* / o2_* / o3_* / k3_* 文件。
set -u
ROOT=$(pwd)
OUT=${OUT_DIR:?}
JD=${JOB_DIR:?}
ID=${JOB_ID:-prof_p1}
DATA=/home/xueyizhou/spgemm-challenge/data
SUM=$OUT/SUMMARY.md
BASE_SHA=141fbc46a690225559b235d79a7da33a7eb7b7029795078e3e9f927e9875011c   # 基线 src/spgemm_kernel_hash.cu(70d1dea)
HDR_SHA=2fc25346de6deabc2eda6d49739ba0627a976969896b861e3798517bab59339b    # bm2_r4 / bm2_r5 的 S1 验过的 src/hh_bm2.cuh
LOAD_MAX=${LOAD_MAX:-6}
QUIET_WAIT_S=${QUIET_WAIT_S:-900}
BIN=./spgemm_p1
RUNENV="USE_MEMPOOL=1 MP_HOST_MB=8192 D6EXP=0"
MYUID=$(id -u)
mkdir -p "$OUT" "$JD/vl" "$JD/k2" "$JD/k3" "$JD/keep" "$JD/p1"
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
say "# prof_p1 汇总(全 428 影子:off / aa / cen / cm2;默认形态相位地图;上一轮的三项补账;ncu 机制计数)"
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
make -k -j4 -f Makefile -f "jobs/$ID/keep.mk" TARGET=spgemm_p1 > "$JD/build_p1.log" 2>&1; RC=$?   # -k:一个翻译单元报错后继续编另一个,报错一次出全
T1=$(date +%s)
tail -c 100000 "$JD/build_p1.log" > "$OUT/s2_build_tail.log"
say "- make TARGET=spgemm_p1: rc=$RC,用时 $((T1 - T0))s;审计: $(grep -E '✓|✗|模板实例' "$JD/build_p1.log" | tr '\n' ';')"
if [ $RC = 0 ] && [ -x spgemm_p1 ] && [ "$(grep -c '✓' "$JD/build_p1.log")" -ge 2 ] && ! grep -q '✗' "$JD/build_p1.log"; then
  if [ -f src/spgemm_kernel_hash.o ]; then rm -f src/main.o; MK="-f Makefile -f jobs/$ID/keep.mk"; else MK=""; fi
  make -j4 $MK WRITE_MTX=1 TARGET=spgemm_p1w > "$JD/build_p1w.log" 2>&1; RCW=$?
  say "- WRITE_MTX 件 spgemm_p1w: rc=$RCW($([ -n "$MK" ] && echo '只重编 main.cu' || echo '全量重编'))"
  [ $RCW = 0 ] && [ -x spgemm_p1w ] && S2=green
else
  say "- **构建或审计失败**。报错行(前 80 条,全文尾部在 s2_build_tail.log):"
  grep -nE "error|Error|✗|MISSING|undefined reference" "$JD/build_p1.log" | head -80 | tee -a "$SUM"
  say "- 日志尾:"; tail -25 "$JD/build_p1.log" | tee -a "$SUM"
fi
say "- S2: $S2"
[ $S1 = green ] || finish 11 "红:S1 微复现器未全绿(S2=$S2);后续阶段未跑"
[ $S2 = green ] || finish 12 "红:S2 构建/审计失败;后续阶段未跑"

# S2b 基线源码件(供 O1 / O2;失败不挡别的阶段)
BASE_OK=red
if [ $HAVE_PATCH = 1 ]; then
  rm -rf "$JD/base"; mkdir -p "$JD/base"
  cp -r Makefile src include tools "$JD/base/"
  rm -f "$JD"/base/src/*.o
  ( cd "$JD/base" && patch -p1 -R < "$ROOT/jobs/$ID/bm2_src.patch" ) > "$OUT/s2_patch.log" 2>&1
  GOT=$(sha256sum "$JD/base/src/spgemm_kernel_hash.cu" | cut -d' ' -f1)
  if [ "$GOT" = "$BASE_SHA" ]; then
    ( cd "$JD/base" && make -j4 TARGET=spgemm_base ) > "$JD/build_base.log" 2>&1; RCB=$?
    if [ $RCB = 0 ] && [ -x "$JD/base/spgemm_base" ]; then cp "$JD/base/spgemm_base" "$ROOT/spgemm_base"; BASE_OK=green; fi
    say "- 基线源码件 spgemm_base:sha256 核对通过,make rc=$RCB"
  else
    say "- **基线源码 sha256 不符**(得 $GOT),spgemm_base 没建;见 s2_patch.log"
  fi
else
  say "- C 上没有 patch 命令,spgemm_base 没建"
fi

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
S3=$(smoke g1 bcsstk17,F2,case9,c-64,TSOPF,pre2,web-Google --arm off: --arm cen:HH_CENSUS=1 --arm cm2:HH_CENSUS=1,HH_BM2=2,HH_BM2_LAZY=2 --arm m2:HH_BM2=2)
say "- off/cen/cm2/m2,7 阵 → $S3;走两遍式的行数合计 = $(smoke_taken g1)"
if [ $S3 != green ]; then tail -25 "$OUT/s3_g1_run.log" | tee -a "$SUM"; tail -20 "$OUT/s3_g1_fail.log" 2>/dev/null | tee -a "$SUM"; finish 13 "红:S3 冒烟未过;后续阶段未跑"; fi

SH_ARMS="--arm off: --arm aa: --arm cen:HH_CENSUS=1 --arm cm2:HH_CENSUS=1,HH_BM2=2,HH_BM2_LAZY=2"
SH_SIG='^\[census-stat\].*$'

# ───────────────────────── P0 影子采数脚本试跑 ─────────────────────────
stage "P0 影子采数脚本试跑"
say ""; say "## P0 影子采数脚本试跑(清单里最小的 8 个阵 × 1 轮;只验脚本通不通,读数不用)"
python3 scripts/shadow428.py --bin "$BIN" --list reference/mats_428.csv --roots data/ocean/square,data/first100 --out "$JD/p1/pilot" \
  $SH_ARMS --env HH_CENSUS_STAT=1 --method adaptive --rounds 1,1,1 --sig-regex "$SH_SIG" --limit 8 --gpu "$GPU" > "$OUT/p0_pilot_run.log" 2>&1
RC0=$?
python3 scripts/shadow_sum.py "$JD/p1/pilot" > "$OUT/p0_pilot_sum.txt" 2>&1
RC1=$?
P0ROWS=$(awk -F, 'NR>1 && $6+0 > 0 {n++} END{print n+0}' "$JD/p1/pilot.csv" 2>/dev/null)
say "- shadow428 rc=$RC0,shadow_sum rc=$RC1;有效发 ${P0ROWS:-0} / 32;$(head -1 "$OUT/p0_pilot_run.log")"
say "- $(grep -E '^# shadow_sum|^# 分级' "$OUT/p0_pilot_sum.txt" | tr '\n' ' ')"
if [ $RC0 != 0 ] || [ $RC1 != 0 ] || [ "${P0ROWS:-0}" -lt 24 ] || ! grep -q "^AA_VERDICT" "$OUT/p0_pilot_sum.txt"; then
  tail -20 "$OUT/p0_pilot_run.log" | tee -a "$SUM"; tail -20 "$OUT/p0_pilot_sum.txt" | tee -a "$SUM"; tail -20 "$JD/p1/pilot_fail.log" 2>/dev/null | tee -a "$SUM"
  finish 15 "红:P0 影子采数脚本试跑不通;后续阶段未跑(修脚本后重推)"
fi

# ───────────────────────── O1 发射清单比对(基线件 对 本件开关全关) ─────────────────────────
NSYS_POPT="-t cuda -s none --cpuctxsw=none --force-overwrite=true"; NSYS_REPORT=cuda_gpu_kern_sum; NSYS_OUTMODE=stdout   # prof_k1 试点选定的组合
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
stage "O1 发射清单比对"
say ""; say "## O1 发射清单比对(nsys;核名含模板实参 + 发射次数;基线源码件 base 对 本件开关全关 off)"
if [ $BASE_OK = green ] && command -v nsys >/dev/null 2>&1; then
  O1N=0; O1F=0
  for name in bcsstk17 af23560 msc10848 bcsstk30 cfd2 s3dkq4m2 offshore FEM_3D_thermal2 TSOPF_RS_b678_c2 web-Google case9 bloweybq \
              StocF-1465 rgg_n_2_21_s0 c-64 nlpkkt80 pre2 ldoor Hook_1498 c-big Bump_2911 cage15; do
    path=$(find_mtx "$name") || continue
    for arm in base off; do
      if [ $arm = base ]; then B1=./spgemm_base; else B1=$BIN; fi
      pfx="$JD/k2/${name}__${arm}__1"
      rm -f "$pfx".nsys-rep "$pfx".qdrep "$pfx".sqlite
      env $RUNENV METHOD=hash WARMUP_N=1 timeout 1500 nsys profile $NSYS_POPT -o "$pfx" "$B1" "$path" > "$pfx.run.log" 2>&1 < /dev/null
      rp=$(ls "$pfx.nsys-rep" "$pfx.qdrep" 2>/dev/null | head -1)
      O1N=$((O1N + 1))
      if [ -n "$rp" ] && nsys_stats "$rp" "$pfx.csv"; then :; else O1F=$((O1F + 1)); echo "O1 FAIL $name $arm"; tail -3 "$pfx.run.log"; fi
      rm -f "$pfx".nsys-rep "$pfx".qdrep "$pfx".sqlite "$pfx.run.log" "$pfx.stats.log"
    done
  done
  python3 scripts/launch_diff.py --dir "$JD/k2" --a base --b off > "$OUT/o1_launch_diff.txt" 2>&1
  say "- 采数 $O1N 发(22 阵 × 2 件),失败 $O1F 发;$(grep -E '^LAUNCH_DIFF_(N|MATS|VERDICT) ' "$OUT/o1_launch_diff.txt" | tr '\n' ';')(明细 o1_launch_diff.txt)"
  grep -E "^# " "$OUT/o1_launch_diff.txt" | sed 's/^/    /' | tee -a "$SUM"
  python3 scripts/kern_sum.py --dir "$JD/k2" --out "$OUT/o1_kern" --base base > "$OUT/o1_kern_run.log" 2>&1
  grep -E "^GM_KERN" "$OUT/o1_kern.txt" 2>/dev/null | sed 's/^/    设备侧 Σkernel:/' | tee -a "$SUM"
  rm -rf "$JD/k2"
else
  say "- 跳过(spgemm_base 没建成,或没有 nsys)"
fi

# ───────────────────────── O2 官方 ab_pair(参考轨) ─────────────────────────
stage "O2 官方 ab_pair 16×9"
say ""; say "## O2 官方 ab_pair 16×9(参考轨:两个二进制、无对照臂;按 dialog/qa-008 只入档,不作裁定;裁定以上一轮 T2 的带对照臂版为准)"
if [ $BASE_OK = green ]; then
  say "- 主机(O2 前): $(host_line)"
  sampler_start "$JD/load_O2.log"
  python3 scripts/ab_pair.py spgemm_base spgemm_p1 "$OUT/o2_ab_base_vs_p1off.csv" 9 "$GPU" > "$OUT/o2_ab_run.log" 2>&1
  sampler_stop
  sed -n '/===== 判读/,$p' "$OUT/o2_ab_run.log" > "$OUT/o2_ab_verdict.txt"
  grep -E "^(GM|裁定|微级)" "$OUT/o2_ab_verdict.txt" | sed 's/^/    /' | tee -a "$SUM"
  say "- 段内主机负载: $(load_stats "$JD/load_O2.log")"
else
  say "- 跳过(spgemm_base 没建成)"
fi

# ───────────────────────── O3 msc10848 加轮复核 ─────────────────────────
stage "O3 msc10848 加轮复核"
say ""; say "## O3 msc10848 加轮复核(off / aa / cen / cm2 / m2,21 轮交错)"
if pmsc=$(find_mtx msc10848); then
  python3 scripts/ab_arms.py --bin "$BIN" --out "$OUT/o3_msc" --rounds 21 --gpu "$GPU" --no-battery --extra "msc10848=$pmsc:hash" \
    --arm off: --arm aa: --arm cen:HH_CENSUS=1 --arm cm2:HH_CENSUS=1,HH_BM2=2,HH_BM2_LAZY=2 --arm m2:HH_BM2=2 > "$OUT/o3_msc_run.log" 2>&1
  say "- ab_arms rc=$?"
  grep -E "msc10848" "$OUT/o3_msc.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"
else
  say "- 跳过(找不到 msc10848)"
fi

# ───────────────────────── K3 ncu 机制计数 ─────────────────────────
stage "K3 ncu 机制计数"
say ""; say "## K3 ncu 硬件计数(hash_heavy 对 hash_bm2;application 重放;60 分钟预算)"
NCU=/opt/nvidia/nsight-compute/2025.1.1/ncu
if [ -x "$NCU" ]; then
  say "- ncu: $NCU / $("$NCU" --version 2>&1 | grep -iE 'version' | head -1)"
  KRE='hash_heavy_kernel|hash_bm2_kernel'
  K3N=0; K3OK=0; K3T0=$(date +%s)
  for spec in "cfd2:off cm2" "s3dkq4m2:off cm2 m2p" "bcsstk30:off cm2" "msc10848:off cm2" "TSOPF_RS_b678_c2:off cm2 m2p"; do
    name=${spec%%:*}
    path=$(find_mtx "$name") || continue
    for arm in ${spec#*:}; do
      if [ $(( $(date +%s) - K3T0 )) -gt 3600 ]; then echo "K3 已用 60 分钟,余下的组跳过"; break 2; fi
      e=$(arm_env $arm); pfx="$JD/k3/${name}__${arm}"; K3N=$((K3N + 1))
      rm -f "$pfx.ncu-rep"
      env $RUNENV METHOD=hash WARMUP_N=0 $e timeout 900 "$NCU" --target-processes all --replay-mode application -k "regex:$KRE" --launch-count 16 --set full -f -o "$pfx" "$BIN" "$path" > "$pfx.log" 2>&1 < /dev/null
      if [ ! -s "$pfx.ncu-rep" ]; then
        echo "K3 $name $arm:--set full 没产出,退回默认指标集"; grep -E "==ERROR==" "$pfx.log" | head -3
        env $RUNENV METHOD=hash WARMUP_N=0 $e timeout 600 "$NCU" --target-processes all --replay-mode application -k "regex:$KRE" --launch-count 16 -f -o "$pfx" "$BIN" "$path" > "$pfx.log" 2>&1 < /dev/null
      fi
      if [ -s "$pfx.ncu-rep" ] && timeout 600 "$NCU" --import "$pfx.ncu-rep" --csv --page raw > "$pfx.raw.csv" 2>> "$pfx.log" && grep -q "Kernel Name" "$pfx.raw.csv"; then
        K3OK=$((K3OK + 1))
        gzip -c "$pfx.raw.csv" > "$OUT/k3_raw_${name}__${arm}.csv.gz"
        mv "$pfx.ncu-rep" "$JD/keep/k3_${name}__${arm}.ncu-rep"
      else
        echo "K3 FAIL $name $arm"; grep -E "==ERROR==" "$pfx.log" | head -3
      fi
    done
  done
  say "- 采数 $K3N 组(阵 × 臂),成功 $K3OK 组;原始宽表 k3_raw_<阵>__<臂>.csv.gz,ncu 报告留在 $JD/keep/"
  python3 scripts/ncu_sum.py --dir "$JD/k3" --out "$OUT/k3_ncu" > "$OUT/k3_ncu_run.log" 2>&1
  say "- ncu_sum rc=$?;挑选指标的并排表 k3_ncu.txt($(head -1 "$OUT/k3_ncu.txt" 2>/dev/null))"
  if [ -s "$OUT/k3_ncu_long.csv" ] && [ "$(stat -c %s "$OUT/k3_ncu_long.csv")" -gt 1900000 ]; then gzip -f "$OUT/k3_ncu_long.csv"; fi
else
  say "- 跳过:$NCU 不存在"
fi

# ───────────────────────── P1 全 428 影子 + 默认形态相位地图 ─────────────────────────
stage "P1 负载门"
say ""; say "## P1 全 428 影子(off / aa / cen / cm2;METHOD=adaptive,生产默认 env;轮数按文件大小分档:<10MB 5 轮 / 10–200MB 3 轮 / ≥200MB 2 轮)"
: > "$JD/WAITING_QUIET"
TW0=$(date +%s)
if wait_quiet "$QUIET_WAIT_S"; then
  say "- 等了 $(( ($(date +%s) - TW0) / 60 )) 分钟,主机满足安静判据($(host_line))"
else
  say "- 等了 $(( ($(date +%s) - TW0) / 60 )) 分钟,主机仍不满足安静判据($(host_line))。照跑:有效性由 A/A 对照臂认证,判据不过整段作废"
fi
rm -f "$JD/WAITING_QUIET"
: > "$JD/TIMING_ACTIVE"
stage "P1 全 428 影子"
sampler_start "$JD/load_P1.log"
python3 scripts/shadow428.py --bin "$BIN" --list reference/mats_428.csv --roots data/ocean/square,data/first100 --out "$JD/p1/shadow" \
  $SH_ARMS --env HH_CENSUS_STAT=1 --method adaptive --rounds 5,3,2 --tier-mb 10,200 --sig-regex "$SH_SIG" --gpu "$GPU" > "$OUT/p1_shadow_run.log" 2>&1
say "- shadow428 rc=$?;$(tail -1 "$OUT/p1_shadow_run.log")"
sampler_stop
rm -f "$JD/TIMING_ACTIVE"
gzip -c "$JD/load_P1.log" > "$OUT/p1_load.log.gz" 2>/dev/null
say "- 段内主机负载: $(load_stats "$JD/load_P1.log")"
python3 scripts/shadow_sum.py "$JD/p1/shadow" > "$OUT/p1_shadow_sum_run.log" 2>&1
say "- shadow_sum rc=$?;全文 p1_shadow_sum.txt,逐阵 p1_shadow_permat.csv,相位地图 p1_shadow_map.csv.gz,逐发 p1_shadow.csv.gz,相位长表 p1_shadow_phases.csv.gz"
cp "$JD/p1/shadow_sum.txt" "$OUT/p1_shadow_sum.txt" 2>/dev/null
cp "$JD/p1/shadow_permat.csv" "$OUT/p1_shadow_permat.csv" 2>/dev/null
for f in shadow.csv shadow_phases.csv shadow_map.csv; do
  [ -s "$JD/p1/$f" ] && gzip -c "$JD/p1/$f" > "$OUT/p1_$f.gz"
done
for f in "$OUT"/p1_shadow*.gz; do
  if [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 1900000 ]; then mv "$f" "$JD/keep/"; say "- $(basename "$f") 超 1.9MB,改留 $JD/keep/"; fi
done
tail -30 "$JD/p1/shadow_fail.log" 2>/dev/null > "$OUT/p1_shadow_fail_tail.log"
say ""
say '```'
awk '/===== 相位地图/{exit} {print}' "$OUT/p1_shadow_sum.txt" 2>/dev/null | cut -c1-900 | tee -a "$SUM"
say '```'
say "- 相位地图与各池前 20 阵(默认形态 = 臂 off;cen / cm2 的池总量差也在其中),全文在 p1_shadow_sum.txt:"
say '```'
awk '/===== 相位地图/{p=1} p' "$OUT/p1_shadow_sum.txt" 2>/dev/null | cut -c1-600 | head -60 | tee -a "$SUM"
say '```'

# ───────────────────────── P2 地图对齐发(旧口径 env) ─────────────────────────
stage "P2 地图对齐发"
say ""; say "## P2 地图对齐发(off 臂单发,HASH_LHT23=0:对齐 scripts/phase_decomp_current.sh 的旧口径,只为与本机侧 atlas 对照;不是生产默认)"
python3 scripts/shadow428.py --bin "$BIN" --list reference/mats_428.csv --roots data/ocean/square,data/first100 --out "$JD/p1/lht0" \
  --arm off: --env HASH_LHT23=0 --method adaptive --rounds 1,1,1 --gpu "$GPU" > "$OUT/p2_lht0_run.log" 2>&1
say "- shadow428 rc=$?;$(tail -1 "$OUT/p2_lht0_run.log")"
python3 scripts/shadow_sum.py "$JD/p1/lht0" > /dev/null 2>&1
awk '/===== 相位地图/{p=1} p' "$JD/p1/lht0_sum.txt" 2>/dev/null | cut -c1-600 > "$OUT/p2_lht0_map.txt"
[ -s "$JD/p1/lht0_map.csv" ] && gzip -c "$JD/p1/lht0_map.csv" > "$OUT/p2_lht0_map.csv.gz"
say "- 旧口径地图 p2_lht0_map.txt / p2_lht0_map.csv.gz(长表 mat,arm,phase,ms)"
finish 0 "S3 冒烟 $S3;P0 试跑通;O1/O2/O3、K3 见各段;P1 影子的裁定看 p1_shadow_sum.txt 里的 AA_VERDICT 与各臂的裁定行;P2 旧口径地图已出"
