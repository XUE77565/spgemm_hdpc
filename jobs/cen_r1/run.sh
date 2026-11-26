#!/usr/bin/env bash
# cen_r1 —— 门普查单发射版(HH_CENSUS=1)与"门普查 + 行表划分合一"(再加 HH_BM2=2、HH_BM2_LAZY=2):先证行为保持,再量收益;
#           并补 bm2_t1 的两项遗留(小阵探针 A/A 未过需整段重跑;无侵入三轮无对照臂被微级阵噪声判红,需带对照臂重做)
#
# bm2_t1 的结论:值级全绿(v3.1 回归、pre2、web-Google 强制宽跨度封条、bin9 两阵);A/A 认证的电池 m2/off GM 0.9862(A/A 1.0001,干净);
#   小阵探针 A/A GM 0.9816 → 按约定整段作废;无侵入(官方 ab_pair 16×3,无对照臂)GM 1.0296 判红,红在 1138_bus 1.32、F2 1.15 两个微级阵,
#   非微级 12 阵 GM 1.002;ncu 2022.4.1 对任何核都报 "Failed to prepare kernel for profiling"(四种组合都不通)。
# 臂:off=默认;aa=与 off 相同的对照;cen=HH_CENSUS=1;m2=HH_BM2=2(按需划分);cm2=HH_CENSUS=1,HH_BM2=2,HH_BM2_LAZY=2(合一);
#     m2e=HH_BM2=2,HH_BM2_LAZY=0;cenm2=HH_CENSUS=1,HH_BM2=2(单发射普查 + 按需划分);cm2L=cm2 再加 HH_BM2_L1=1(只做值级)。
# 阶段:
#   S1 微复现器    hh_bm2.cuh 未变 → 沿用
#   S2 构建        spgemm_c1(审计双绿)+ WRITE_MTX 件 + 基线源码件 spgemm_base(反打补丁,供 T2)
#   S3 冒烟        三组各自判:G1 两遍式(宏重构后的 v3.1 回归)off/m2/m2p/m2L/m2e;G2 单发射普查 off/cen/cenm2;G3 合一 off/cm2/cm2L
#   S4a 行为保持   约 70 阵(电池 24 + 428 清单每 9 个抽 1 个):HH_CENSUS=0 / 1 / 合一 各跑一发,[census-stat] 行与 nnz 逐行相同
#   S4b 值级       差分门:bcsstk17、cfd2、msc10848、GaAsH6(bin9)、web-Google、pre2(各臂见下)
#   K2  nsys       22 阵 × off/cen/m2/cm2 × 2 轮的逐 kernel 设备侧时长;6 个小阵另出 CUDA API 调用次数(发射数、同步数)
#   K3  ncu        只做探查:找机器上别的 ncu,逐个试点;都不通就写 NEED_USER.md(换版本属依赖安装)
#   T   计时       A/A 认证口径(dialog/qa-007 已定):先等安静最多 15 分钟,等不到照跑;A/A 判据不过整段作废,不挑段
#                  T1 电池 off/aa/cen/m2/cm2,24 阵 × 7 轮;T1b 小阵探针 off/aa/cen/cm2,25 轮(A/A 不过自动整段重跑一次);
#                  T2 无侵入:off/aa/base(基线源码件,AB_BIN 臂)同批交错,16 阵 × 7 轮
# 只评测:不标定、不动默认、门值不入源码。回传:OUT_DIR/SUMMARY.md(先看这个)+ s*_ / k2_* / k3_* / t1_* / t1b_* / t2_* 文件。
set -u
ROOT=$(pwd)
OUT=${OUT_DIR:?}
JD=${JOB_DIR:?}
ID=${JOB_ID:-cen_r1}
DATA=/home/xueyizhou/spgemm-challenge/data
SUM=$OUT/SUMMARY.md
BASE_SHA=141fbc46a690225559b235d79a7da33a7eb7b7029795078e3e9f927e9875011c   # 基线 src/spgemm_kernel_hash.cu(70d1dea)
HDR_SHA=2fc25346de6deabc2eda6d49739ba0627a976969896b861e3798517bab59339b    # bm2_r4 / bm2_r5 的 S1 验过的 src/hh_bm2.cuh
LOAD_MAX=${LOAD_MAX:-6}
QUIET_WAIT_S=${QUIET_WAIT_S:-900}
BIN=./spgemm_c1
RUNENV="USE_MEMPOOL=1 MP_HOST_MB=8192 D6EXP=0"
MYUID=$(id -u)
mkdir -p "$OUT" "$JD/vl" "$JD/k2" "$JD/k3" "$JD/keep" "$JD/cs"
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
say "# cen_r1 汇总(门普查单发射版与普查+划分合一:行为保持 → 设备侧时长 → A/A 认证的电池、小阵探针、带对照臂的无侵入)"
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
make -k -j4 -f Makefile -f "jobs/$ID/keep.mk" TARGET=spgemm_c1 > "$JD/build_c1.log" 2>&1; RC=$?   # -k:一个翻译单元报错后继续编另一个,报错一次出全
T1=$(date +%s)
tail -c 100000 "$JD/build_c1.log" > "$OUT/s2_build_tail.log"
say "- make TARGET=spgemm_c1: rc=$RC,用时 $((T1 - T0))s;审计: $(grep -E '✓|✗|模板实例' "$JD/build_c1.log" | tr '\n' ';')"
if [ $RC = 0 ] && [ -x spgemm_c1 ] && [ "$(grep -c '✓' "$JD/build_c1.log")" -ge 2 ] && ! grep -q '✗' "$JD/build_c1.log"; then
  if [ -f src/spgemm_kernel_hash.o ]; then rm -f src/main.o; MK="-f Makefile -f jobs/$ID/keep.mk"; else MK=""; fi
  make -j4 $MK WRITE_MTX=1 TARGET=spgemm_c1w > "$JD/build_c1w.log" 2>&1; RCW=$?
  say "- WRITE_MTX 件 spgemm_c1w: rc=$RCW($([ -n "$MK" ] && echo '只重编 main.cu' || echo '全量重编'))"
  [ $RCW = 0 ] && [ -x spgemm_c1w ] && S2=green
else
  say "- **构建或审计失败**。报错行(前 80 条,全文尾部在 s2_build_tail.log):"
  grep -nE "error|Error|✗|MISSING|undefined reference" "$JD/build_c1.log" | head -80 | tee -a "$SUM"
  say "- 日志尾:"; tail -25 "$JD/build_c1.log" | tee -a "$SUM"
fi
say "- S2: $S2"
[ $S1 = green ] || finish 11 "红:S1 微复现器未全绿(S2=$S2);后续阶段未跑"
[ $S2 = green ] || finish 12 "红:S2 构建/审计失败;后续阶段未跑"

# S2b 基线源码件(供 T2;失败不挡别的阶段)
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
    say "- 基线源码件 spgemm_base:sha256 核对通过,make rc=$RCB;审计: $(grep -E '✓|✗|模板实例' "$JD/build_base.log" | tr '\n' ';')"
  else
    say "- **基线源码 sha256 不符**(得 $GOT),spgemm_base 没建;见 s2_patch.log"
  fi
else
  say "- C 上没有 patch 命令,spgemm_base 没建"
fi

# ───────────────────────── S3 冒烟(三组各自判) ─────────────────────────
stage "S3 冒烟"
say ""; say "## S3 冒烟(各臂 1 发:nnz 与失败检查;单发的时间比不作读数)"
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
LAZY_OK=$(smoke g1 bcsstk17,F2,case9,c-64,TSOPF,pre2 --arm off: --arm m2:HH_BM2=2 --arm m2p:HH_BM2=2,HH_BM2_P1=1 --arm m2L:HH_BM2=2,HH_BM2_L1=1 --arm m2e:HH_BM2=2,HH_BM2_LAZY=0)
say "- G1 两遍式(宏重构后的回归)off/m2/m2p/m2L/m2e,6 阵 → $LAZY_OK;走两遍式的行数合计 = $(smoke_taken g1)"
[ $LAZY_OK = green ] || { tail -25 "$OUT/s3_g1_run.log" | tee -a "$SUM"; tail -20 "$OUT/s3_g1_fail.log" 2>/dev/null | tee -a "$SUM"; }
CEN_OK=$(smoke g2 bcsstk17,F2,case9,c-64,TSOPF,pre2,web-Google --arm off: --arm cen:HH_CENSUS=1 --arm cenm2:HH_CENSUS=1,HH_BM2=2)
say "- G2 单发射普查 off/cen/cenm2,7 阵 → $CEN_OK;走两遍式的行数合计 = $(smoke_taken g2)"
[ $CEN_OK = green ] || { tail -25 "$OUT/s3_g2_run.log" | tee -a "$SUM"; tail -20 "$OUT/s3_g2_fail.log" 2>/dev/null | tee -a "$SUM"; }
FUSE_OK=$(smoke g3 bcsstk17,F2,case9,c-64,TSOPF,pre2,web-Google --arm off: --arm cm2:HH_CENSUS=1,HH_BM2=2,HH_BM2_LAZY=2 --arm cm2L:HH_CENSUS=1,HH_BM2=2,HH_BM2_LAZY=2,HH_BM2_L1=1)
say "- G3 普查 + 划分合一 off/cm2/cm2L,7 阵 → $FUSE_OK;走两遍式的行数合计 = $(smoke_taken g3)"
[ $FUSE_OK = green ] || { tail -25 "$OUT/s3_g3_run.log" | tee -a "$SUM"; tail -20 "$OUT/s3_g3_fail.log" 2>/dev/null | tee -a "$SUM"; }
if [ $LAZY_OK != green ] && [ $CEN_OK != green ] && [ $FUSE_OK != green ]; then finish 13 "红:S3 冒烟三组都没过;后续阶段未跑"; fi
okarms() {   # 过滤掉冒烟没过的臂;回显空格分隔的臂名
  local a
  for a in "$@"; do
    case $a in
      m1|m2|m2e|m2p|m2L) [ $LAZY_OK = green ] && printf "%s " "$a";;
      cen|cenm2)         [ $CEN_OK = green ] && printf "%s " "$a";;
      cm2|cm2L)          [ $FUSE_OK = green ] && printf "%s " "$a";;
      *)                 printf "%s " "$a";;
    esac
  done
}
armspec() {  # 臂名 → ab_arms 的 --arm 参数值
  echo "$1:$(arm_env "$1" | tr ' ' ',')"
}

# ───────────────────────── S4a 门普查行为保持 ─────────────────────────
stage "S4a 门普查行为保持"
say ""; say "## S4a 门普查行为保持([census-stat] 行 = 每个 bin 的 rows / gated / total,连同结果 nnz;三种模式逐行比对)"
CS_N=0; CS_BAD=0; CS_SKIP=0; CS_NONDET=0; CS_BADLIST=""
cs_one() {   # $1=name $2=path $3=标签 $4..=env → $JD/cs/<name>.<标签>.txt;成功返回 0
  local name=$1 path=$2 tag=$3; shift 3
  env $RUNENV METHOD=hash WARMUP_N=0 "$@" HH_CENSUS_STAT=1 timeout 900 "$BIN" "$path" 2>&1 < /dev/null | grep -aE "^\[census-stat\]|Result C:.*nnz = " > "$JD/cs/$name.$tag.txt"
  grep -q "Result C:" "$JD/cs/$name.$tag.txt"
}
if [ $CEN_OK = green ]; then
  { for n in bcsstk17 case9 web-Google TSOPF_RS_b678_c2 bloweybq af23560 StocF-1465 ldoor c-64 pre2 rgg_n_2_21_s0 Hook_1498 cage15 Bump_2911 \
             c-big cfd2 s3dkq4m2 nlpkkt80 FEM_3D_thermal2 offshore msc10848 bcsstk30 GaAsH6 Ga3As3H12; do echo "$n"; done
    awk 'NR>1 && (NR % 9) == 2' reference/mats_428.csv 2>/dev/null; } | awk '!seen[$0]++' > "$JD/cs/list.txt"
  while read -r name; do
    [ -n "$name" ] || continue
    path=$(find_mtx "$name") || continue
    if ! cs_one "$name" "$path" c0 HH_CENSUS=0; then CS_SKIP=$((CS_SKIP + 1)); echo "S4a 跳过 $name(基线模式没跑出结果)"; continue; fi
    # 基线模式自己跑两遍:两遍都不同的阵(估计/分 bin 本身不确定)不可比,单列,不计入不一致
    cs_one "$name" "$path" c0b HH_CENSUS=0
    if ! cmp -s "$JD/cs/$name.c0.txt" "$JD/cs/$name.c0b.txt"; then CS_NONDET=$((CS_NONDET + 1)); echo "S4a 不可比 $name(基线模式两遍结果不同)"; rm -f "$JD/cs/$name".c0*.txt; continue; fi
    CS_N=$((CS_N + 1)); bad=""
    cs_one "$name" "$path" c1 HH_CENSUS=1; cmp -s "$JD/cs/$name.c0.txt" "$JD/cs/$name.c1.txt" || bad="$bad cen"
    if [ $FUSE_OK = green ]; then
      cs_one "$name" "$path" c2 HH_CENSUS=1 HH_BM2=2 HH_BM2_LAZY=2; cmp -s "$JD/cs/$name.c0.txt" "$JD/cs/$name.c2.txt" || bad="$bad cm2"
    fi
    if [ -n "$bad" ]; then
      CS_BAD=$((CS_BAD + 1)); CS_BADLIST="$CS_BADLIST $name($bad )"
      { echo "==== $name:$bad"; echo "-- HH_CENSUS=0"; cat "$JD/cs/$name.c0.txt"; echo "-- HH_CENSUS=1"; cat "$JD/cs/$name.c1.txt"; [ -f "$JD/cs/$name.c2.txt" ] && { echo "-- 合一"; cat "$JD/cs/$name.c2.txt"; }; } >> "$OUT/s4a_census_mismatch.txt"
    fi
  done < "$JD/cs/list.txt"
  rm -f "$JD"/cs/*.c0b.txt
  NZ=$(cat "$JD"/cs/*.c0.txt 2>/dev/null | awk '/census-stat/{split($3,a,"="); if (a[2]+0 > 0) n++} END{print n+0}')
  say "- 比对了 $CS_N 阵(另有 $CS_SKIP 阵基线模式没跑出结果、$CS_NONDET 阵基线模式自己两遍就不同,都不计入);其中非空的 (阵,bin) 组合 $NZ 个"
  say "- **不一致的阵: $CS_BAD 个**$([ $CS_BAD = 0 ] || echo ":$CS_BADLIST(明细在 s4a_census_mismatch.txt)")"
  say "- 合一形态$([ $FUSE_OK = green ] && echo '已一并比对' || echo '冒烟未过,没比')"
  [ $CS_BAD = 0 ] || finish 14 "红:S4a 门普查结果在新旧模式下不一致;后续阶段未跑"
else
  say "- 单发射普查冒烟未过,S4a 跳过"
fi
rm -rf "$JD/cs"

# ───────────────────────── S4b 值级 ─────────────────────────
vl_dump() {   # $1=name $2=path $3=臂名 → $JD/vl/<name>_<臂名>.mtx;回显 "nnz taken skip"
  local name=$1 path=$2 tag=$3 log
  log="$JD/vl/${name}_${tag}.log"
  rm -f "results/matrices/$name/self_product_hash.mtx"
  env $RUNENV METHOD=hash WARMUP_N=0 $(arm_env "$tag") HH_BM2_STAT=1 timeout 1800 ./spgemm_c1w "$path" > "$log" 2>&1
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
stage "S4b 值级"
say ""; say "## S4b 值级:各臂对 off 的差分门"
say ""
say "| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |"
say "|---|---|---|---|---|"
VL_TAKEN=0; S4=green
A=$(okarms m2 m2e cen cm2);  [ -n "$A" ] && gate_only bcsstk17 $A
A=$(okarms m2 cen cm2);      [ -n "$A" ] && gate_only cfd2 $A
A=$(okarms cm2);             [ -n "$A" ] && gate_only msc10848 $A
A=$(okarms cm2);             [ -n "$A" ] && gate_only GaAsH6 $A
A=$(okarms cen);             [ -n "$A" ] && gate_only web-Google $A
A=$(okarms cen cm2L);        [ -n "$A" ] && gate_only pre2 $A
rm -rf "$ROOT"/results/matrices/*/self_product*.mtx
say ""
say "- S4b: $S4;走两遍式的行数合计 $VL_TAKEN(GaAsH6 是 bin9 有行的阵;cm2L 是合一 + legacy 形态,只验值)"
[ $S4 = green ] || finish 14 "红:S4b 值级未过(见上表);后续阶段未跑"

# ───────────────────────── K2 nsys 逐 kernel(设备侧) ─────────────────────────
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
do_K2() {
  stage "K2 nsys 逐 kernel"
  say ""; say "## K2 逐 kernel 的 GPU 侧累计时长(nsys;设备侧时间戳,不吃主机 CPU 争用)"
  if ! command -v nsys >/dev/null 2>&1; then say "- **nsys 不在 PATH**,K2 跳过"; return 0; fi
  local KARMS nrun=0 nfail=0 name path arm e rep pfx rp
  KARMS="off $(okarms cen m2 cm2)"
  say "- nsys: $(nsys --version 2>&1 | head -1);臂: $KARMS;22 阵 × 2 轮,WARMUP_N=1"
  : > "$OUT/k2_api_launch.txt"
  for name in bcsstk17 af23560 msc10848 bcsstk30 cfd2 s3dkq4m2 offshore FEM_3D_thermal2 TSOPF_RS_b678_c2 web-Google case9 bloweybq \
              StocF-1465 rgg_n_2_21_s0 c-64 nlpkkt80 pre2 ldoor Hook_1498 c-big Bump_2911 cage15; do
    path=$(find_mtx "$name") || { echo "K2 缺 $name"; continue; }
    for rep in 1 2; do
      for arm in $KARMS; do
        e=$(arm_env $arm)
        pfx="$JD/k2/${name}__${arm}__${rep}"
        rm -f "$pfx".nsys-rep "$pfx".qdrep "$pfx".sqlite
        env $RUNENV METHOD=hash WARMUP_N=1 $e timeout 1500 nsys profile $NSYS_POPT -o "$pfx" "$BIN" "$path" > "$pfx.run.log" 2>&1 < /dev/null
        rp=$(ls "$pfx.nsys-rep" "$pfx.qdrep" 2>/dev/null | head -1)
        nrun=$((nrun + 1))
        if [ -n "$rp" ] && nsys_stats "$rp" "$pfx.csv"; then
          case "$rep:$name" in   # 小阵第 1 轮:另出 CUDA API 调用次数(发射数、同步数是确定量,不受主机负载影响)
            1:bcsstk17|1:af23560|1:msc10848|1:cfd2|1:case9|1:web-Google)
              timeout 300 nsys stats --report cuda_api_sum --format csv --output - "$rp" > "$JD/k2/api.csv" 2>/dev/null
              echo "$name $arm 发射=$(awk -F, '/cudaLaunchKernel/{s+=$3}END{print s+0}' "$JD/k2/api.csv") 流同步=$(awk -F, '/cudaStreamSynchronize/{s+=$3}END{print s+0}' "$JD/k2/api.csv") 设备同步=$(awk -F, '/cudaDeviceSynchronize/{s+=$3}END{print s+0}' "$JD/k2/api.csv") 同步拷贝=$(awk -F, '/cudaMemcpy_|cudaMemcpy,|"cudaMemcpy"/{s+=$3}END{print s+0}' "$JD/k2/api.csv")" >> "$OUT/k2_api_launch.txt"
              [ "$arm" = off ] && [ "$name" = bcsstk17 ] && head -40 "$JD/k2/api.csv" > "$OUT/k2_api_sample.csv"
              ;;
          esac
        else
          nfail=$((nfail + 1)); echo "K2 FAIL $name $arm $rep"; tail -3 "$pfx.run.log"
        fi
        rm -f "$pfx".nsys-rep "$pfx".qdrep "$pfx".sqlite "$pfx.run.log" "$pfx.stats.log" "$JD/k2/api.csv"
      done
    done
    echo "K2 done $name(累计 $nrun 发,失败 $nfail)"
  done
  say "- 采数:$nrun 发,失败 $nfail 发"
  python3 scripts/kern_sum.py --dir "$JD/k2" --out "$OUT/k2_kern" > "$OUT/k2_kern_run.log" 2>&1
  say "- kern_sum rc=$?;全文 k2_kern.txt,长表 k2_kern_long.csv.gz"
  if [ -s "$OUT/k2_kern_long.csv.gz" ] && [ "$(stat -c %s "$OUT/k2_kern_long.csv.gz")" -gt 1900000 ]; then
    mv "$OUT/k2_kern_long.csv.gz" "$JD/keep/"; say "- 长表超 1.9MB,改留 $JD/keep/k2_kern_long.csv.gz"
  fi
  say ""
  say '```'
  awk '/===== ②/{exit} /^# |===== ①/{p=1} p' "$OUT/k2_kern.txt" 2>/dev/null | tee -a "$SUM"
  say '```'
  say "- 比值是设备侧 Σkernel 时长之比(含预热轮,各臂相同),不含主机侧的发射与同步开销。门普查核族在各臂的累计时长见 k2_kern.txt 的 ②"
  say "- CUDA API 调用次数(6 个小阵,每臂一发 = 预热 1 次 + 计时 1 次;k2_api_launch.txt,表头样例 k2_api_sample.csv):"
  sed 's/^/    /' "$OUT/k2_api_launch.txt" | tee -a "$SUM"
  rm -rf "$JD/k2"
}
do_K2

# ───────────────────────── K3 ncu 探查 ─────────────────────────
stage "K3 ncu 探查"
say ""; say "## K3 ncu 探查(bm2_t1 里 /usr/bin/ncu 2022.4.1 四种组合都报 Failed to prepare kernel for profiling)"
say "- 驱动: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1);nsys: $(nsys --version 2>&1 | head -1)"
NCU_GOOD=""; NCU_SEEN=""
pm=$(find_mtx bcsstk17)
for c in $(ls -d /usr/bin/ncu /usr/local/bin/ncu /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu /usr/local/cuda*/nsight-compute*/ncu 2>/dev/null); do
  [ -x "$c" ] || continue
  r=$(readlink -f "$c")
  case " $NCU_SEEN " in *" $r "*) continue;; esac
  NCU_SEEN="$NCU_SEEN $r"
  rm -f "$JD/k3/pilot.ncu-rep"
  env $RUNENV METHOD=hash WARMUP_N=0 timeout 600 "$c" --target-processes all --replay-mode application -k "regex:hash_heavy_kernel" --launch-count 2 -f -o "$JD/k3/pilot" "$BIN" "$pm" > "$JD/k3/pilot.log" 2>&1 < /dev/null
  rc=$?
  say "- $c → $r($("$c" --version 2>&1 | grep -iE 'version' | head -1)):试点 rc=$rc,$([ -s "$JD/k3/pilot.ncu-rep" ] && echo '**出了报告**' || echo "没出报告:$(grep -E '==ERROR==' "$JD/k3/pilot.log" | head -1)")"
  if [ -s "$JD/k3/pilot.ncu-rep" ]; then
    NCU_GOOD=$c
    timeout 300 "$c" --import "$JD/k3/pilot.ncu-rep" --csv --page raw 2>/dev/null | head -c 4000 > "$OUT/k3_pilot_raw_head.csv"
    break
  fi
done
if [ -n "$NCU_GOOD" ]; then
  say "- 可用的 ncu: $NCU_GOOD(application 重放)。机制计数留到下一个作业做(本作业不再加长)"
else
  say "- **机器上现有的 ncu 都不通**,已写 NEED_USER.md(换与驱动匹配的版本属依赖安装,不自行安装)"
  { echo "# NEED_USER"; echo
    echo "- cen_r1 的 K3:本机现有的 ncu(见 SUMMARY 的 K3 段)对任何核都报 \"Failed to prepare kernel for profiling\",不是权限错(没有 ERR_NVGPUCTRPERM)。"
    echo "  多半是 ncu 版本与驱动 / H100 不匹配。需要装一个与驱动匹配的新版 ncu(属依赖安装,等用户或本机侧处理);在那之前 P2 的 ncu 深挖做不了,"
    echo "  机制判断先用 nsys 的设备侧时长。"; } > "$OUT/NEED_USER.md"
fi

# ───────────────────────── T 计时(A/A 认证口径,dialog/qa-007) ─────────────────────────
stage "T0 负载门"
say ""; say "## T0 负载门"
QUIET=0
: > "$JD/WAITING_QUIET"
TW0=$(date +%s)
if wait_quiet "$QUIET_WAIT_S"; then
  QUIET=1; say "- 等了 $(( ($(date +%s) - TW0) / 60 )) 分钟,主机满足安静判据($(host_line))"
else
  say "- 等了 $(( ($(date +%s) - TW0) / 60 )) 分钟,主机仍不满足安静判据($(host_line))。照跑:计时有效性由 A/A 对照臂认证,判据不过整段作废"
fi
rm -f "$JD/WAITING_QUIET"
: > "$JD/TIMING_ACTIVE"

stage "T1 电池(带 A/A 对照臂)"
TARMS="off aa $(okarms cen m2 cm2)"
say ""; say "## T1 只评测电池(同一二进制;臂: $TARMS;24 阵 × 7 轮交错)"
say "- 主机(T1 前): $(host_line)"
EXTRA=""
for name in c-big cfd2 s3dkq4m2 nlpkkt80 FEM_3D_thermal2 offshore msc10848 bcsstk30; do
  p=$(find_mtx "$name") && EXTRA="$EXTRA --extra $name=$p:hash"
done
AARGS="--arm off: --arm aa:"
for a in $(okarms cen m2 cm2); do AARGS="$AARGS --arm $(armspec $a)"; done
sampler_start "$JD/load_T1.log"
python3 scripts/ab_arms.py --bin "$BIN" --out "$OUT/t1_battery" --rounds 7 --gpu "$GPU" $AARGS --stat $EXTRA > "$OUT/t1_battery_run.log" 2>&1
say "- ab_arms rc=$?;逐阵表 t1_battery.txt,逐发 t1_battery.csv,逐 bin 普查 t1_battery_stat.csv"
sampler_stop
cp "$JD/load_T1.log" "$OUT/t1_load.log" 2>/dev/null
say "- 段内主机负载: $(load_stats "$JD/load_T1.log")"
python3 scripts/aa_check.py "$OUT/t1_battery.csv" > "$OUT/t1_aa.txt" 2>&1
AA=$(grep -oE "^AA_VERDICT [a-z]+" "$OUT/t1_aa.txt" | cut -d' ' -f2)
say "- A/A 对照臂: $(grep -E '^AA_(N|GM|NONMICRO_MIN|NONMICRO_MAX|VERDICT) ' "$OUT/t1_aa.txt" | tr '\n' ';')(逐阵见 t1_aa.txt)"
if [ "${AA:-nodata}" = clean ]; then
  say "- **T1 计时判定: A/A 认证通过**(下面的 GM 与裁定有效)"
else
  say "- **T1 计时判定: A/A 判据未过 → 整段作废**(下面的数字只留档,不引用;需整段重跑)"
fi
grep -E "^(GM|裁定|=====)" "$OUT/t1_battery.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"

stage "T1b 小阵探针"
PARMS="off aa $(okarms cen cm2)"
say ""; say "## T1b 小阵探针(臂: $PARMS;25 轮交错;看用不上两遍式的小阵上少 5 次发射值多少、合一形态有没有新增固定成本)"
PEX=""; PN=0
for name in bcspwr05 494_bus 662_bus 685_bus bcsstk01 bcsstk13 can_292 bcspwr10 bcsstm07 bcsstk17 af23560; do
  p=$(find_mtx "$name") && { PEX="$PEX --extra $name=$p:hash"; PN=$((PN + 1)); }
done
PARGS="--arm off: --arm aa:"
for a in $(okarms cen cm2); do PARGS="$PARGS --arm $(armspec $a)"; done
AAB=nodata
if [ $PN -gt 0 ]; then
  for attempt in 1 2; do
    sampler_start "$JD/load_T1b.log"
    python3 scripts/ab_arms.py --bin "$BIN" --out "$OUT/t1b_probe$attempt" --rounds 25 --gpu "$GPU" --no-battery $PARGS $PEX > "$OUT/t1b_probe${attempt}_run.log" 2>&1
    say "- 第 $attempt 次:ab_arms rc=$?($PN 阵);逐阵表 t1b_probe$attempt.txt,逐发 t1b_probe$attempt.csv"
    sampler_stop
    say "- 段内主机负载: $(load_stats "$JD/load_T1b.log")"
    python3 scripts/aa_check.py "$OUT/t1b_probe$attempt.csv" > "$OUT/t1b_aa$attempt.txt" 2>&1
    AAB=$(grep -oE "^AA_VERDICT [a-z]+" "$OUT/t1b_aa$attempt.txt" | cut -d' ' -f2)
    say "- A/A 对照臂: $(grep -E '^AA_(N|GM|NONMICRO_MIN|NONMICRO_MAX|VERDICT) ' "$OUT/t1b_aa$attempt.txt" | tr '\n' ';')"
    grep -E "^(GM|=====)" "$OUT/t1b_probe$attempt.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"
    if [ "${AAB:-nodata}" = clean ]; then say "- **T1b 第 $attempt 次: A/A 认证通过**"; break; fi
    say "- **T1b 第 $attempt 次: A/A 判据未过 → 这一次整段作废**$([ $attempt = 1 ] && echo ',整段重跑一次' || echo ';两次都没过,探针读数不引用')"
  done
else
  say "- 跳过(候选小阵一个都没找到)"
fi

stage "T2 无侵入(带对照臂)"
say ""; say "## T2 无侵入:off=本件默认 env / aa=同 off 的对照 / base=基线源码件(反打补丁),同批交错,16 阵 × 7 轮"
say "- 说明:bm2_t1 的官方 ab_pair(16×3,无对照臂)被两个微级阵的噪声判红。这里把基线件作为一条臂放进同一个交错电池,与 A/A 臂同批;"
say "  判读 = base/off 的 GM 与逐阵比(off 是分母,所以 >1 表示本件更快),有效性看 A/A。"
AAT=nodata
if [ $BASE_OK = green ]; then
  say "- 主机(T2 前): $(host_line)"
  sampler_start "$JD/load_T2.log"
  python3 scripts/ab_arms.py --bin "$BIN" --out "$OUT/t2_noint" --rounds 7 --gpu "$GPU" --arm off: --arm aa: --arm base:AB_BIN=./spgemm_base > "$OUT/t2_noint_run.log" 2>&1
  say "- ab_arms rc=$?;逐阵表 t2_noint.txt,逐发 t2_noint.csv"
  sampler_stop
  say "- 段内主机负载: $(load_stats "$JD/load_T2.log")"
  python3 scripts/aa_check.py "$OUT/t2_noint.csv" > "$OUT/t2_aa.txt" 2>&1
  AAT=$(grep -oE "^AA_VERDICT [a-z]+" "$OUT/t2_aa.txt" | cut -d' ' -f2)
  say "- A/A 对照臂: $(grep -E '^AA_(N|GM|NONMICRO_MIN|NONMICRO_MAX|VERDICT) ' "$OUT/t2_aa.txt" | tr '\n' ';')"
  if [ "${AAT:-nodata}" = clean ]; then say "- **T2 计时判定: A/A 认证通过**"; else say "- **T2 计时判定: A/A 判据未过 → 整段作废**(数字只留档)"; fi
  grep -E "^(GM|裁定|=====)" "$OUT/t2_noint.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"
else
  say "- 跳过:spgemm_base 没建成"
fi
rm -f "$JD/TIMING_ACTIVE"
finish 0 "S3 冒烟:两遍式 $LAZY_OK / 单发射普查 $CEN_OK / 合一 $FUSE_OK;S4a 比对 $CS_N 阵、不一致 $CS_BAD 阵;S4b 值级 $S4;A/A 判定:T1=${AA:-nodata} T1b=${AAB:-nodata} T2=${AAT:-nodata};K2、K3 见各段"
