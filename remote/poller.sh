#!/usr/bin/env bash
# exp-poller: runs on machine B.
#
# Polls a private git repo for jobs/<id>/run.sh on the jobs branch, ships a
# git snapshot of the repo to server C over SSH, starts run.sh there in the
# background, and pushes status, a log tail and small output files to the
# results branch. A heartbeat goes to a separate, force-pushed branch.
#
# B never executes repo content itself: it only reads files with git, copies
# a snapshot to C and starts run.sh on C. job.env is parsed, never sourced.
#
# Usage: poller.sh [-c config] once|loop|init|check
set -u
VERSION=1

usage() {
  cat <<'EOF'
usage: poller.sh [-c config] <command>
  once   poll once (for cron / Task Scheduler, every minute)
  loop   poll forever every POLL_SEC seconds (for tmux / nohup)
  init   clone the repo, create the results branch, mark jobs that
         already exist as "skipped" (they will not run)
  check  test git and SSH access and show what would run
config: -c FILE, or $EXP_POLLER_CONF, or ~/.exp-poller/poller.env
EOF
}

CONF="${EXP_POLLER_CONF:-$HOME/.exp-poller/poller.env}"
while getopts "c:h" o; do
  case $o in
    c) CONF=$OPTARG ;;
    h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
CMD="${1:-once}"

# ---------- defaults (override in the config file) ----------
REPO_URL=""                 # git@github.com:OWNER/REPO.git
GIT_KEY=""                  # deploy key for this repo only (empty = default ssh key)
WORK_DIR="$HOME/.exp-poller"
JOBS_BRANCH=main
RESULTS_BRANCH=results
HEARTBEAT_BRANCH=heartbeat
C_HOST=""                   # Host alias from ~/.ssh/config on B
C_BASE="exp-runs"           # job folders on C (relative to C home, or absolute)
SSH_BIN=ssh
SSH_MUX=1                   # reuse one SSH connection (set 0 on Windows)
MAX_PARALLEL=1
DEFAULT_TIMEOUT=24h
POLL_SEC=60
PROGRESS_MIN=10             # push log tail of running jobs at most this often
HEARTBEAT_MIN=10
LOG_TAIL_LINES=200
OUT_FILE_MAX_KB=2048        # out/ files larger than this are not uploaded
OUT_TOTAL_MAX_KB=10240      # total out/ upload per job
LAUNCH_RETRIES=5

if [ -f "$CONF" ]; then
  # The config belongs to B's owner, so sourcing it is fine.
  # shellcheck disable=SC1090
  . "$CONF"
elif [ "$CMD" != "-h" ]; then
  echo "config not found: $CONF" >&2
  exit 2
fi

REPO="$WORK_DIR/repo"
RES="$WORK_DIR/results-wt"
STATE="$WORK_DIR/state"
LOGF="$WORK_DIR/poller.log"
LOCK="$WORK_DIR/lock"
GIT_ID=(-c user.name=exp-poller -c user.email=exp-poller@localhost)

die() { echo "error: $*" >&2; log "error: $*"; exit 1; }
now() { date +%s; }
iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }

log() {
  mkdir -p "$WORK_DIR"
  if [ -f "$LOGF" ] && [ "$(wc -c < "$LOGF")" -gt 1048576 ]; then
    mv -f "$LOGF" "$LOGF.1"
  fi
  echo "$(iso) $*" >> "$LOGF"
}

valid_id() { [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; }

[ -n "$REPO_URL" ] || die "REPO_URL is not set in $CONF"
[ -n "$C_HOST" ] || die "C_HOST is not set in $CONF"
[[ $C_BASE =~ ^[A-Za-z0-9._/-]+$ ]] || die "C_BASE may only use letters, digits and . _ / -"
[[ $DEFAULT_TIMEOUT =~ ^[0-9]+[smhd]?$ ]] || die "bad DEFAULT_TIMEOUT"

if [ -n "$GIT_KEY" ]; then
  export GIT_SSH_COMMAND="ssh -i '$GIT_KEY' -o IdentitiesOnly=yes -o BatchMode=yes"
fi

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30)
if [ "$SSH_MUX" = 1 ]; then
  SSH_OPTS+=(-o ControlMaster=auto -o "ControlPath=$WORK_DIR/cm-%C" -o ControlPersist=10m)
fi

g() { git -C "$REPO" "$@"; }
r() { git -C "$RES" "$@"; }
cssh() { "$SSH_BIN" "${SSH_OPTS[@]}" "$C_HOST" "$@"; }

# ---------- per-job state on B: $STATE/jobs/<id>/<key> ----------
sget() { cat "$STATE/jobs/$1/$2" 2>/dev/null; }
sset() { mkdir -p "$STATE/jobs/$1"; printf '%s\n' "$3" > "$STATE/jobs/$1/$2"; }
jobs_in_state() {
  local d
  for d in "$STATE"/jobs/*/; do
    [ -d "$d" ] || continue
    d=${d%/}; d=${d##*/}
    [ "$(sget "$d" state)" = "$1" ] && echo "$(sget "$d" seq || echo 0) $d"
  done | sort -n | while read -r _ id; do echo "$id"; done
}

# ---------- scripts that run on C (sent over stdin, never stored in the repo) ----------
REMOTE_PREPARE='
set -eu
base=$1; id=$2
d="$base/$id"
if [ -e "$d/start" ]; then echo "job folder already used on C: $d" >&2; exit 3; fi
command -v timeout >/dev/null || { echo "timeout(1) missing on C" >&2; exit 4; }
rm -rf "$d/code"
mkdir -p "$d/code" "$d/out"
'

REMOTE_EXTRACT='
set -eu
d="$1/$2"
tar -xf - -C "$d/code"
'

REMOTE_LAUNCH='
set -eu
base=$1; id=$2; to=$3
d=$(cd "$base/$id" && pwd)
cat > "$d/runner.sh" <<"RUNNER"
#!/usr/bin/env bash
# written by exp-poller; runs one job on C
to=$1; id=$2
d=$(cd "$(dirname "$0")" && pwd)
cd "$d/code" || exit 1
export JOB_ID="$id" JOB_DIR="$d" OUT_DIR="$d/out"
echo $$ > "$d/pid"
date +%s > "$d/start"
timeout --kill-after=120 "$to" bash "jobs/$id/run.sh" > "$d/log.txt" 2>&1 < /dev/null &
tp=$!
echo "$tp" > "$d/tpid"
wait "$tp"
echo $? > "$d/exit"
date +%s > "$d/end"
RUNNER
if command -v setsid >/dev/null; then S=setsid; else S=; fi
nohup $S bash "$d/runner.sh" "$to" "$id" > /dev/null 2>&1 < /dev/null &
i=0
while [ ! -s "$d/tpid" ] && [ $i -lt 100 ]; do sleep 0.1; i=$((i + 1)); done
[ -s "$d/tpid" ] || { echo "job did not start" >&2; exit 5; }
cat "$d/pid"
'

REMOTE_COLLECT='
set -u
base=$1; id=$2; n=$3; per=$4; tot=$5
d="$base/$id"
[ -d "$d" ] || { echo "no job folder on C: $d" >&2; exit 3; }
t=$(mktemp -d); trap "rm -rf \"$t\"" EXIT
mkdir -p "$t/r/out"
for f in pid tpid start end exit; do [ -f "$d/$f" ] && cp "$d/$f" "$t/r/$f"; done
alive=0
if [ -s "$d/pid" ] && kill -0 "$(cat "$d/pid")" 2>/dev/null; then alive=1; fi
echo $alive > "$t/r/alive"
if [ -f "$d/log.txt" ]; then
  tail -n "$n" "$d/log.txt" > "$t/r/log_tail.txt"
  wc -c < "$d/log.txt" | tr -d " " > "$t/r/log_bytes"
fi
: > "$t/r/skipped.txt"
if [ -d "$d/out" ]; then
  sum=0
  (cd "$d/out" && find . -type f | LC_ALL=C sort) > "$t/list"
  while IFS= read -r f; do
    sz=$(wc -c < "$d/out/$f" | tr -d " ")
    kb=$(( (sz + 1023) / 1024 ))
    if [ "$kb" -gt "$per" ] || [ $((sum + kb)) -gt "$tot" ]; then
      echo "$sz ${f#./}" >> "$t/r/skipped.txt"
      continue
    fi
    sum=$((sum + kb))
    mkdir -p "$t/r/out/$(dirname "$f")"
    cp "$d/out/$f" "$t/r/out/$f"
  done < "$t/list"
fi
tar -cf - -C "$t/r" .
'

REMOTE_CANCEL='
set -u
d="$1/$2"; sig=$3
[ -s "$d/tpid" ] || exit 0
tp=$(cat "$d/tpid")
if [ "$sig" = KILL ]; then
  kill -KILL -- "-$tp" 2>/dev/null || kill -KILL "$tp" 2>/dev/null
else
  kill -TERM "$tp" 2>/dev/null
fi
exit 0
'

# ---------- git setup ----------
ensure_clone() {
  mkdir -p "$WORK_DIR" "$STATE/jobs"
  if [ ! -d "$REPO/.git" ]; then
    git clone -q --no-checkout "$REPO_URL" "$REPO" || die "git clone failed"
  fi
}

fetch() {
  g fetch -q --prune origin "+refs/heads/$JOBS_BRANCH:refs/remotes/origin/$JOBS_BRANCH" || return 1
  g fetch -q origin "+refs/heads/$RESULTS_BRANCH:refs/remotes/origin/$RESULTS_BRANCH" 2>/dev/null || true
  return 0
}

ensure_results_wt() {
  [ -e "$RES/.git" ] && return 0
  g worktree prune
  if g rev-parse -q --verify "refs/remotes/origin/$RESULTS_BRANCH" >/dev/null; then
    g worktree add -q -B "$RESULTS_BRANCH" "$RES" "origin/$RESULTS_BRANCH" || die "worktree add failed"
  else
    local empty c
    empty=$(g hash-object -t tree /dev/null)
    c=$(g "${GIT_ID[@]}" commit-tree "$empty" -m "init results branch") || die "commit-tree failed"
    g branch -f "$RESULTS_BRANCH" "$c"
    g worktree add -q "$RES" "$RESULTS_BRANCH" || die "worktree add failed"
    r push -q origin "$RESULTS_BRANCH:refs/heads/$RESULTS_BRANCH" || log "could not push new results branch"
  fi
}

# ---------- results files ----------
DIRTY=""
mark_dirty() { DIRTY="$DIRTY $1"; }

json_num() { if [[ ${1:-} =~ ^-?[0-9]+$ ]]; then echo "$1"; else echo null; fi; }
json_str() { if [ -n "${1:-}" ]; then printf '"%s"' "$1"; else echo null; fi; }

write_status() {
  local id=$1 m="$STATE/jobs/$1/meta" out="$RES/results/$1"
  mkdir -p "$out"
  cat > "$out/status.json" <<EOF
{
  "id": "$id",
  "state": "$(sget "$id" state)",
  "note": $(json_str "$(sget "$id" note)"),
  "commit": $(json_str "$(sget "$id" commit)"),
  "timeout": $(json_str "$(sget "$id" timeout)"),
  "queued_at": $(json_num "$(sget "$id" queued_at)"),
  "started_at": $(json_num "$(cat "$m/start" 2>/dev/null)"),
  "ended_at": $(json_num "$(cat "$m/end" 2>/dev/null)"),
  "exit_code": $(json_num "$(cat "$m/exit" 2>/dev/null)"),
  "log_bytes": $(json_num "$(cat "$m/log_bytes" 2>/dev/null)"),
  "updated_at": $(now),
  "poller_version": $VERSION
}
EOF
  if [ -d "$m" ]; then
    [ -f "$m/log_tail.txt" ] && cp "$m/log_tail.txt" "$out/log_tail.txt"
    if [ -s "$m/skipped.txt" ]; then cp "$m/skipped.txt" "$out/out_skipped.txt"; else rm -f "$out/out_skipped.txt"; fi
    rm -rf "$out/out"
    if [ -d "$m/out" ] && [ -n "$(ls -A "$m/out" 2>/dev/null)" ]; then cp -R "$m/out" "$out/out"; fi
  fi
  mark_dirty "$id"
}

# Pull status, log tail and small outputs of a job from C into $STATE/jobs/<id>/meta.
collect() {
  local id=$1 tmp
  tmp=$(mktemp -d "$WORK_DIR/tmp.XXXXXX") || return 1
  if ! cssh bash -s -- "$C_BASE" "$id" "$LOG_TAIL_LINES" "$OUT_FILE_MAX_KB" "$OUT_TOTAL_MAX_KB" \
       <<< "$REMOTE_COLLECT" > "$tmp/c.tar" 2> "$tmp/err"; then
    log "collect $id failed: $(head -c 300 "$tmp/err")"
    rm -rf "$tmp"; return 1
  fi
  mkdir "$tmp/x"
  if ! tar -xf "$tmp/c.tar" -C "$tmp/x" 2>> "$tmp/err"; then
    log "collect $id: bad tar stream (does a shell startup file on C print text?): $(head -c 300 "$tmp/err")"
    rm -rf "$tmp"; return 1
  fi
  rm -rf "$STATE/jobs/$id/meta"
  mv "$tmp/x" "$STATE/jobs/$id/meta"
  rm -rf "$tmp"
}

# ---------- poll steps ----------
queue_job() {
  local id=$1 env to pin commit seq
  if g cat-file -e "origin/$RESULTS_BRANCH:results/$id/status.json" 2>/dev/null; then
    sset "$id" state known
    return 0
  fi
  env=$(g show "origin/$JOBS_BRANCH:jobs/$id/job.env" 2>/dev/null || true)
  to=$(printf '%s\n' "$env" | sed -n 's/^TIMEOUT=\([0-9][0-9]*[smhd]\{0,1\}\)[[:space:]]*$/\1/p' | head -n 1)
  pin=$(printf '%s\n' "$env" | sed -n 's/^COMMIT=\([0-9a-f]\{7,40\}\)[[:space:]]*$/\1/p' | head -n 1)
  if [ -n "$pin" ]; then
    commit=$(g rev-parse -q --verify "$pin^{commit}") || {
      sset "$id" state error; sset "$id" note "COMMIT $pin not found"
      sset "$id" queued_at "$(now)"; write_status "$id"; log "$id: pinned commit $pin not found"; return 0; }
  else
    commit=$(g rev-parse "origin/$JOBS_BRANCH")
  fi
  seq=$(( $(cat "$STATE/seq" 2>/dev/null || echo 0) + 1 ))
  echo "$seq" > "$STATE/seq"
  sset "$id" seq "$seq"
  sset "$id" commit "$commit"
  sset "$id" timeout "${to:-$DEFAULT_TIMEOUT}"
  sset "$id" queued_at "$(now)"
  sset "$id" state queued
  write_status "$id"
  log "$id: queued at ${commit:0:10}, timeout ${to:-$DEFAULT_TIMEOUT}"
}

# New jobs are queued in the order their run.sh entered the jobs branch.
discover() {
  local list id c n new="" sorted
  list=$(g ls-tree -d --name-only "origin/$JOBS_BRANCH:jobs" 2>/dev/null) || return 0
  while IFS= read -r id; do
    [ -n "$id" ] || continue
    if ! valid_id "$id"; then
      grep -qxF -- "$id" "$STATE/bad_ids" 2>/dev/null \
        || { printf '%s\n' "$id" >> "$STATE/bad_ids"; log "ignoring bad job id: $id"; }
      continue
    fi
    [ -d "$STATE/jobs/$id" ] && continue
    g cat-file -e "origin/$JOBS_BRANCH:jobs/$id/run.sh" 2>/dev/null || continue
    c=$(g log -1 --format=%H --diff-filter=A "origin/$JOBS_BRANCH" -- "jobs/$id/run.sh")
    n=$(g rev-list --count "${c:-origin/$JOBS_BRANCH}")
    new="$new$n $id
"
  done <<< "$list"
  sorted=$(printf '%s' "$new" | sort -n -k1,1)
  while read -r n id; do
    [ -n "$id" ] && queue_job "$id"
  done <<< "$sorted"
  return 0
}

has_cancel() { g cat-file -e "origin/$JOBS_BRANCH:jobs/$1/CANCEL" 2>/dev/null; }

check_running() {
  local id req ex alive
  for id in $(jobs_in_state running); do
    req=$(sget "$id" cancel_req)
    if [ -z "$req" ] && has_cancel "$id"; then
      cssh bash -s -- "$C_BASE" "$id" TERM <<< "$REMOTE_CANCEL" >/dev/null 2>&1
      sset "$id" cancel_req "$(now)"; req=$(now)
      log "$id: cancel requested, sent TERM"
    elif [ -n "$req" ] && [ $(( $(now) - req )) -gt 150 ]; then
      cssh bash -s -- "$C_BASE" "$id" KILL <<< "$REMOTE_CANCEL" >/dev/null 2>&1
    fi
    collect "$id" || continue
    ex=$(cat "$STATE/jobs/$id/meta/exit" 2>/dev/null)
    alive=$(cat "$STATE/jobs/$id/meta/alive" 2>/dev/null)
    if [ -n "$ex" ]; then
      if [ -n "$req" ]; then sset "$id" state cancelled
      elif [ "$ex" = 0 ]; then sset "$id" state done
      elif [ "$ex" = 124 ] || [ "$ex" = 137 ]; then sset "$id" state timeout
      else sset "$id" state failed
      fi
      write_status "$id"
      log "$id: finished, exit $ex, state $(sget "$id" state)"
    elif [ "$alive" = 0 ]; then
      if [ -n "$req" ]; then sset "$id" state cancelled; else sset "$id" state lost; sset "$id" note "process gone without exit code (C rebooted or killed?)"; fi
      write_status "$id"
      log "$id: process gone, state $(sget "$id" state)"
    elif [ $(( $(now) - $(sget "$id" last_progress || echo 0) )) -ge $(( PROGRESS_MIN * 60 )) ]; then
      sset "$id" last_progress "$(now)"
      write_status "$id"
    fi
  done
}

cancel_queued() {
  local id
  for id in $(jobs_in_state queued); do
    if has_cancel "$id"; then
      sset "$id" state cancelled; sset "$id" note "cancelled before start"
      write_status "$id"; log "$id: cancelled before start"
    fi
  done
}

launch_one() {
  local id=$1 commit to err tmp
  commit=$(sget "$id" commit); to=$(sget "$id" timeout)
  tmp=$(mktemp "$WORK_DIR/tmp.XXXXXX")
  if cssh bash -s -- "$C_BASE" "$id" <<< "$REMOTE_PREPARE" 2> "$tmp" \
     && g archive --format=tar "$commit" | cssh "bash -c '$REMOTE_EXTRACT' extract $C_BASE $id" 2>> "$tmp" \
     && cssh bash -s -- "$C_BASE" "$id" "$to" <<< "$REMOTE_LAUNCH" > /dev/null 2>> "$tmp"; then
    sset "$id" state running
    sset "$id" last_progress "$(now)"
    rm -f "$tmp"
    collect "$id" || true
    write_status "$id"
    log "$id: started on C"
    return 0
  fi
  err=$(head -c 300 "$tmp"); rm -f "$tmp"
  local n=$(( $(sget "$id" launch_fail || echo 0) + 1 ))
  sset "$id" launch_fail "$n"
  log "$id: launch failed ($n/$LAUNCH_RETRIES): $err"
  if [ "$n" -ge "$LAUNCH_RETRIES" ] || [[ $err == *"already used on C"* ]]; then
    sset "$id" state error; sset "$id" note "launch failed: $err"
    write_status "$id"
  fi
  return 1
}

launch_queued() {
  local running id
  running=$(jobs_in_state running | wc -l | tr -d ' ')
  for id in $(jobs_in_state queued); do
    [ "$running" -lt "$MAX_PARALLEL" ] || break
    launch_one "$id" && running=$((running + 1)) || break
  done
}

commit_push() {
  local ids
  ids=$(echo $DIRTY | tr ' ' '\n' | sort -u | tr '\n' ' ')
  r add -A
  if ! r diff --cached --quiet; then
    r "${GIT_ID[@]}" commit -q -m "results: ${ids:-update}" || log "commit failed"
  fi
  local ahead
  ahead=$(r rev-list --count "refs/remotes/origin/$RESULTS_BRANCH..HEAD" 2>/dev/null || echo 1)
  [ "$ahead" = 0 ] && return 0
  if ! r push -q origin "HEAD:refs/heads/$RESULTS_BRANCH" 2>/dev/null; then
    g fetch -q origin "+refs/heads/$RESULTS_BRANCH:refs/remotes/origin/$RESULTS_BRANCH" \
      && r rebase -q "origin/$RESULTS_BRANCH" \
      && r push -q origin "HEAD:refs/heads/$RESULTS_BRANCH" \
      || { r rebase --abort 2>/dev/null; log "push to $RESULTS_BRANCH failed, will retry"; return 1; }
  fi
  g fetch -q origin "+refs/heads/$RESULTS_BRANCH:refs/remotes/origin/$RESULTS_BRANCH" 2>/dev/null || true
}

heartbeat() {
  local last t c_ok txt blob tree c
  last=$(cat "$STATE/last_hb" 2>/dev/null || echo 0)
  [ $(( $(now) - last )) -ge $(( HEARTBEAT_MIN * 60 )) ] || return 0
  if cssh true 2>/dev/null; then c_ok=yes; else c_ok=no; fi
  txt="time: $(iso)
poller_version: $VERSION
c_reachable: $c_ok
running: $(jobs_in_state running | tr '\n' ' ')
queued: $(jobs_in_state queued | tr '\n' ' ')
"
  blob=$(printf '%s' "$txt" | g hash-object -w --stdin) || return 1
  tree=$(printf '100644 blob %s\theartbeat.txt\n' "$blob" | g mktree) || return 1
  c=$(g "${GIT_ID[@]}" commit-tree "$tree" -m heartbeat) || return 1
  if g push -q -f origin "$c:refs/heads/$HEARTBEAT_BRANCH" 2>/dev/null; then
    now > "$STATE/last_hb"
  else
    log "heartbeat push failed"
  fi
}

poll() {
  DIRTY=""
  if ! fetch; then log "git fetch failed"; return 1; fi
  ensure_results_wt
  discover
  cancel_queued
  check_running
  launch_queued
  commit_push
  heartbeat
}

# ---------- lock: one poller at a time ----------
take_lock() {
  mkdir -p "$WORK_DIR"
  if ! mkdir "$LOCK" 2>/dev/null; then
    local op
    op=$(cat "$LOCK/pid" 2>/dev/null)
    if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then exit 0; fi
    rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null || exit 0
  fi
  echo $$ > "$LOCK/pid"
  trap 'rm -rf "$LOCK"' EXIT
  trap 'exit 130' INT TERM
}

case "$CMD" in
  once)
    take_lock; ensure_clone; poll ;;
  loop)
    take_lock; ensure_clone
    log "loop started (every ${POLL_SEC}s)"
    while true; do poll; sleep "$POLL_SEC"; done ;;
  init)
    take_lock; ensure_clone
    fetch || die "git fetch failed"
    ensure_results_wt
    n=0
    while IFS= read -r id; do
      valid_id "$id" || continue
      [ -d "$STATE/jobs/$id" ] && continue
      sset "$id" state skipped; n=$((n + 1))
    done <<< "$(g ls-tree -d --name-only "origin/$JOBS_BRANCH:jobs" 2>/dev/null)"
    echo "init done: repo at $REPO, results worktree at $RES, $n existing job(s) marked skipped" ;;
  check)
    ensure_clone
    echo "== git"; fetch && echo "fetch ok: origin/$JOBS_BRANCH = $(g rev-parse --short "origin/$JOBS_BRANCH")"
    echo "== C ($C_HOST)"
    cssh "echo ssh ok; uname -sr; for t in bash tar timeout setsid; do printf '%s: ' \$t; command -v \$t || echo MISSING; done; mkdir -p $C_BASE && df -h $C_BASE | tail -n 1"
    echo "== jobs not seen yet (would be queued by 'once')"
    while IFS= read -r id; do
      [ -n "$id" ] || continue
      if ! valid_id "$id"; then echo "  (bad id, ignored) $id"; continue; fi
      [ -d "$STATE/jobs/$id" ] || echo "  $id"
    done <<< "$(g ls-tree -d --name-only "origin/$JOBS_BRANCH:jobs" 2>/dev/null)"
    echo "== state on B"
    for s in queued running; do echo "  $s: $(jobs_in_state $s | tr '\n' ' ')"; done ;;
  *)
    usage; exit 2 ;;
esac
