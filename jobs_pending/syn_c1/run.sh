#!/usr/bin/env bash
# syn_c1 —— 两遍式行门 HH_BM2_FX 的标定:合成训练阵(flop/HT 铺到 1…192)+ 真实训练池,生产入口逐行配对周期 → R=1 的交点
#
# 出处(dialog/qa-013,B 端批准的方案 A):训练池 data/train_disp 的量程只到 flop/HT≈8(R 0.95–0.98,还没越过 1),428 里会输的行在 10–175。
#   门值不能看着 428 定,所以用受控结构的合成阵把量程补上。生成器 tools/train_synth/gen_synth.py(参数表与固定种子在脚本里),
#   阵现生成在 $JOB_DIR/train_synth/,作业结束时删掉(可按种子重生成),不进 git、不进测试账目录。
# 阶段:
#   S1/S2  沿用微复现器;构建 spgemm_y1 + WRITE_MTX 件
#   S3     冒烟
#   G0     生成 14 个合成阵(W ∈ {112,…,7200} × 步长 {1,16};约 1.3 亿非零,几分钟)
#   G1     合成阵逐行配对标定:off / m1 / m2(HH_BM2_LAZY=0,METHOD=adaptive,HASH_ROWTIME 落逐行块周期)→ bm2_gate_cal → g1_cal*
#   G2     真实训练池同样做一遍(与 cen_rm1 的 R4 同法,同一个二进制同一场)→ g2_cal*;两者在 flop/HT 1–8 的重叠段应当接得上
#   G3     交点:scripts/bm2_fx_cross.py 出各模式各 bin 的 R(flop/HT) 曲线、R=1 交点、建议门值(0.65 × 最小交点)
#   G4     合成阵上的值级(生产入口转储,m2 / cm2 对 off 的差分门)
# 全程不做墙钟计时判读(逐行周期是 GPU 侧计数),不碰 428 / flex142 / external_72。单卡;每个 GPU 进程都在 `timeout -k 30` 里。
# 估计 70–90 分钟。回传:OUT_DIR/SUMMARY.md + g0_* / g1_* / g2_* / g3_fx.txt。
set -u
ROOT=$(pwd)
OUT=${OUT_DIR:?}
JD=${JOB_DIR:?}
ID=${JOB_ID:-syn_c1}
DATA=/home/xueyizhou/spgemm-challenge/data
SUM=$OUT/SUMMARY.md
BASE_SHA=141fbc46a690225559b235d79a7da33a7eb7b7029795078e3e9f927e9875011c   # 基线 src/spgemm_kernel_hash.cu(70d1dea)
HDR_SHA=2fc25346de6deabc2eda6d49739ba0627a976969896b861e3798517bab59339b    # bm2_r4 / bm2_r5 的 S1 验过的 src/hh_bm2.cuh
LOAD_MAX=${LOAD_MAX:-6}
QUIET_WAIT_S=${QUIET_WAIT_S:-900}
BIN=./spgemm_y1
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
say "# syn_c1 汇总(两遍式行门 HH_BM2_FX 的标定:合成训练阵 + 真实训练池,生产入口逐行配对周期 → R=1 交点)"
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
make -k -j4 -f Makefile -f "jobs/$ID/keep.mk" TARGET=spgemm_y1 > "$JD/build_y1.log" 2>&1; RC=$?   # -k:一个翻译单元报错后继续编另一个,报错一次出全
T1=$(date +%s)
tail -c 100000 "$JD/build_y1.log" > "$OUT/s2_build_tail.log"
say "- make TARGET=spgemm_y1: rc=$RC,用时 $((T1 - T0))s;审计: $(grep -E '✓|✗|模板实例' "$JD/build_y1.log" | tr '\n' ';')"
if [ $RC = 0 ] && [ -x spgemm_y1 ] && [ "$(grep -c '✓' "$JD/build_y1.log")" -ge 2 ] && ! grep -q '✗' "$JD/build_y1.log"; then
  if [ -f src/spgemm_kernel_hash.o ]; then rm -f src/main.o; MK="-f Makefile -f jobs/$ID/keep.mk"; else MK=""; fi
  make -j4 $MK WRITE_MTX=1 TARGET=spgemm_y1w > "$JD/build_y1w.log" 2>&1; RCW=$?
  say "- WRITE_MTX 件 spgemm_y1w: rc=$RCW($([ -n "$MK" ] && echo '只重编 main.cu' || echo '全量重编'))"
  [ $RCW = 0 ] && [ -x spgemm_y1w ] && S2=green
else
  say "- **构建或审计失败**。报错行(前 80 条,全文尾部在 s2_build_tail.log):"
  grep -nE "error|Error|✗|MISSING|undefined reference" "$JD/build_y1.log" | head -80 | tee -a "$SUM"
  say "- 日志尾:"; tail -25 "$JD/build_y1.log" | tee -a "$SUM"
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
S3=$(smoke g1 bcsstk17,F2,case9,TSOPF --arm off: --arm m2:HH_BM2=2 --arm cm2:HH_BM2=2,HH_BM2_LAZY=2)
say "- off/m2/cm2,4 阵 → $S3"
if [ $S3 != green ]; then tail -25 "$OUT/s3_g1_run.log" | tee -a "$SUM"; tail -20 "$OUT/s3_g1_fail.log" 2>/dev/null | tee -a "$SUM"; finish 13 "红:S3 冒烟未过;后续阶段未跑"; fi

# ───────────────────────── G0 生成合成训练阵 ─────────────────────────
stage "G0 生成合成训练阵"
say ""; say "## G0 合成训练阵(tools/train_synth/gen_synth.py;固定种子;只在作业目录里)"
SYN="$JD/train_synth"
T0=$(date +%s)
python3 tools/train_synth/gen_synth.py --out "$SYN" > "$OUT/g0_gen.log" 2>&1
RC=$?
cp "$SYN/MANIFEST.tsv" "$OUT/g0_manifest.tsv" 2>/dev/null
NSYN=$(ls "$SYN"/*.mtx 2>/dev/null | wc -l)
say "- 生成器 rc=$RC,用时 $(( $(date +%s) - T0 ))s;得到 $NSYN 个阵,共 $(du -sh "$SYN" 2>/dev/null | cut -f1);清单 g0_manifest.tsv"
if [ $RC != 0 ] || [ "$NSYN" -lt 10 ]; then tail -20 "$OUT/g0_gen.log" | tee -a "$SUM"; rm -rf "$SYN"; finish 17 "红:合成阵生成失败;后续阶段未跑"; fi

# ───────────────────────── G1 / G2 逐行配对标定 ─────────────────────────
cal_run() {   # $1=标定目录 $2=清单文件(每行 "名字 路径")→ 逐阵 off / m1 / m2;回显 "发数 无行计时数"
  local dir=$1 list=$2 b f arm e rc nrun=0 nfail=0
  mkdir -p "$dir"
  while read -r b f; do
    [ -n "$b" ] || continue
    for arm in off m1 m2; do
      case $arm in off) e="HH_BM2=0";; m1) e="HH_BM2=1 HH_BM2_LAZY=0";; *) e="HH_BM2=2 HH_BM2_LAZY=0";; esac
      env $RUNENV METHOD=adaptive WARMUP_N=1 $e HH_BM2_STAT=1 HASH_ROWTIME="$dir/$b.$arm.rt" timeout -k 30 900 "$BIN" "$f" < /dev/null > "$dir/$b.$arm.out" 2> "$dir/$b.$arm.log"
      rc=$?; nrun=$((nrun + 1))
      if [ $rc != 0 ] || [ ! -s "$dir/$b.$arm.rt" ]; then
        nfail=$((nfail + 1)); echo "cal 无行计时 $b $arm rc=$rc:$(grep -a '\[adapt\]' "$dir/$b.$arm.out" | tail -1)" >&2
        rm -f "$dir/$b.off.rt"; break
      fi
    done
    { echo "== $b:$(grep -a '\[adapt\]' "$dir/$b.m2.out" 2>/dev/null | tail -1)"; grep -a "bm2-stat" "$dir/$b.m2.log" 2>/dev/null | tail -8; } >> "$dir/routes.txt"
  done < "$list"
  echo "$nrun $nfail"
}
stage "G1 合成阵标定"
say ""; say "## G1 合成阵逐行配对标定(METHOD=adaptive;off / m1=值在 SMEM / m2=值直写全局;同批划分)"
ls "$SYN"/*.mtx | while read -r f; do echo "$(basename "$f" .mtx) $f"; done > "$JD/syn_list.txt"
set -- $(cal_run "$JD/cal_syn" "$JD/syn_list.txt" 2> "$OUT/g1_cal_skips.log")
say "- 采数 ${1:-0} 发;没出行计时的 ${2:-0} 发(见 g1_cal_skips.log);逐行周期文件 $(ls "$JD/cal_syn"/*.off.rt 2>/dev/null | wc -l) 组"
cp "$JD/cal_syn/routes.txt" "$OUT/g1_routes.txt" 2>/dev/null
say "- 各阵的路由与逐 bin 普查(dispatcher 走 hash 还是 merge3;HH 行落在哪些 bin、收了多少):g1_routes.txt"
if [ $HAVE_SCIPY = 1 ] && ls "$JD/cal_syn"/*.off.rt >/dev/null 2>&1; then
  timeout 1800 python3 scripts/bm2_gate_cal.py --dir "$JD/cal_syn" --out "$OUT/g1_cal" > "$OUT/g1_cal_run.log" 2>&1
  say "- bm2_gate_cal rc=$?;g1_cal.txt / g1_cal_cells.csv"
  grep -E "^(# bm2_gate_cal|=====|  全收|  → )" "$OUT/g1_cal.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"
else
  say "- 没有可用的逐行周期文件(或没有 scipy),合成阵标定没出数"
fi
rm -rf "$JD/cal_syn"

stage "G2 真实训练池标定"
say ""; say "## G2 真实训练池逐行配对标定(data/train_disp;与 G1 同一个二进制、同一场;用来核对合成曲线在 flop/HT 1–8 的重叠段)"
NTR=0; : > "$JD/real_list.txt"
if [ -d data/train_disp ]; then
  while read -r sz f; do
    b=$(basename "$f" .mtx)
    [ "$b" = af23560 ] && continue
    [ -f "data/ocean/square/$b.mtx" ] || [ -f "data/first100/$b.mtx" ] || [ -n "$(find -L data/flex142 data/external_72 -maxdepth 3 -name "$b.mtx" 2>/dev/null | head -1)" ] && continue
    echo "$b $f" >> "$JD/real_list.txt"; NTR=$((NTR + 1))
    [ $NTR -ge 60 ] && break
  done < <(find -L data/train_disp -maxdepth 2 -name '*.mtx' -size +1M -size -800M -printf '%s %p\n' 2>/dev/null | sort -rn)
fi
set -- $(cal_run "$JD/cal_real" "$JD/real_list.txt" 2> "$OUT/g2_cal_skips.log")
say "- 训练池候选 $NTR 阵;采数 ${1:-0} 发;没出行计时的 ${2:-0} 发"
if [ $HAVE_SCIPY = 1 ] && ls "$JD/cal_real"/*.off.rt >/dev/null 2>&1; then
  timeout 1800 python3 scripts/bm2_gate_cal.py --dir "$JD/cal_real" --out "$OUT/g2_cal" > "$OUT/g2_cal_run.log" 2>&1
  say "- bm2_gate_cal rc=$?;g2_cal.txt / g2_cal_cells.csv"
  grep -E "^(# bm2_gate_cal|=====|  全收|  → )" "$OUT/g2_cal.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"
fi
rm -rf "$JD/cal_real"

# ───────────────────────── G3 交点 ─────────────────────────
stage "G3 交点"
say ""; say "## G3 R(flop/HT) 曲线与 R=1 交点(scripts/bm2_fx_cross.py;建议门值 = 0.65 × 各模式各 bin 交点的最小值)"
SRC=""
[ -s "$OUT/g1_cal_cells.csv" ] && SRC="$SRC 合成=$OUT/g1_cal_cells.csv"
[ -s "$OUT/g2_cal_cells.csv" ] && SRC="$SRC 真实=$OUT/g2_cal_cells.csv"
if [ -n "$SRC" ]; then
  python3 scripts/bm2_fx_cross.py $SRC > "$OUT/g3_fx.txt" 2>&1
  say "- bm2_fx_cross rc=$?;全文 g3_fx.txt"
  say '```'
  cut -c1-220 "$OUT/g3_fx.txt" | tee -a "$SUM"
  say '```'
else
  say "- 两份标定都没出格表,G3 没做"
fi

# ───────────────────────── G4 合成阵上的值级 ─────────────────────────
vl_dump() {   # $1=name $2=path $3=臂名 → $JD/vl/<name>_<臂名>.mtx;回显 "nnz taken skip"
  local name=$1 path=$2 tag=$3 log
  log="$JD/vl/${name}_${tag}.log"
  rm -f "results/matrices/$name/self_product_hash.mtx"
  env $RUNENV METHOD=hash WARMUP_N=0 $(arm_env "$tag") HH_BM2_STAT=1 timeout -k 30 1800 ./spgemm_y1w "$path" > "$log" 2>&1
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
  env $RUNENV METHOD=adaptive WARMUP_N=0 $(arm_env "$tag") HH_BM2_STAT=1 timeout -k 30 1800 ./spgemm_y1w "$path" > "$log" 2>&1
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
stage "G4 合成阵值级"
say ""; say "## G4 合成阵上的值级(METHOD=adaptive 转储;m2 / cm2 对 off 的差分门)"
say ""
say "| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |"
say "|---|---|---|---|---|"
find_mtx() { [ -f "$SYN/$1.mtx" ] && { echo "$SYN/$1.mtx"; return 0; }; return 1; }   # 此后只找合成阵
VL_TAKEN=0; S4=green
gate_only_ad syn_w224_s1 m2 cm2
gate_only_ad syn_w448_s16 m2 cm2
gate_only_ad syn_w1800_s1 cm2
rm -rf "$ROOT"/results/matrices/*/self_product*.mtx
say ""
say "- G4: $S4;走两遍式的行数合计 $VL_TAKEN"
rm -f "$SYN"/*.mtx
say "- 合成阵已删(清单与种子在 g0_manifest.tsv,可重生成);$(df -h "$JD" | tail -1)"
[ $S4 = green ] || finish 14 "红:G4 合成阵值级未过(见上表);标定读数不能用"
finish 0 "S3 冒烟 $S3;G1 / G2 标定与 G3 交点见各段(建议门值看 g3_fx.txt 的 FX_SUGGEST_PCT 行);G4 值级 green"
