#!/usr/bin/env bash
# exp_poller.sh -- run on B (Linux / macOS) in a terminal window:  bash exp_poller.sh
# Every INTERVAL seconds: git pull, find jobs/<id>/run.sh without results/<id>/,
# copy that version of the code to C, run it there, push the results back.
# Stop with Ctrl+C.

# ======== fill these in ========
REPO_URL=""                # GitHub repo, e.g. git@github.com:OWNER/REPO.git
C_HOST=""                  # how B reaches C with ssh, e.g. user@1.2.3.4 or a Host alias
# ===============================
C_DIR="exp-runs"           # folder on C (relative to home) for the jobs
BRANCH="main"
INTERVAL=60                # seconds between checks
REPO_DIR="$(cd "$(dirname "$0")" && pwd)/exp-repo"   # local clone on B

[ -n "$REPO_URL" ] && [ -n "$C_HOST" ] || { echo "Please fill in REPO_URL and C_HOST at the top of this script."; exit 1; }

g() { git -C "$REPO_DIR" "$@"; }
ts() { date '+%Y-%m-%d %H:%M:%S'; }

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR" || exit 1
fi
g config core.autocrlf false
g config user.name >/dev/null || g config user.name "exp-poller"
g config user.email >/dev/null || g config user.email "exp-poller@localhost"

run_job() {
  local id=$1 dir="$C_DIR/$1" res="$REPO_DIR/results/$1" tar commit start code
  commit=$(g rev-parse HEAD)
  start=$(ts)
  echo "[$(ts)] ===== job $id (commit ${commit:0:8}) ====="

  # 1. send this version of the code to C
  tar=$(mktemp "${TMPDIR:-/tmp}/exp-$id.XXXXXX")
  g -c core.autocrlf=false archive --format=tar -o "$tar" HEAD
  if ! scp -q "$tar" "$C_HOST:exp-$id.tar"; then
    echo "[$(ts)] upload to C failed, will retry next round"
    rm -f "$tar"; return 1
  fi
  rm -f "$tar"

  # 2. unpack and run on C; output shows here and is saved to log.txt on C
  ssh "$C_HOST" "mkdir -p $dir && cd $dir && rm -rf code out exit_code && mkdir code out && tar -xf ~/exp-$id.tar -C code && rm -f ~/exp-$id.tar || exit 254; cd code; export JOB_ID=$id OUT_DIR=\$(cd ../out && pwd); { bash jobs/$id/run.sh; echo \$? > ../exit_code; } 2>&1 | tee ../log.txt; exit \$(cat ../exit_code)"
  code=$?
  if [ "$code" = 255 ] || [ "$code" = 254 ]; then
    echo "[$(ts)] could not reach C or set up the job there (code $code), will retry next round"
    return 1
  fi

  # 3. bring back log.txt and out/, record status, push
  mkdir -p "$res"
  scp -q -r "$C_HOST:$dir/log.txt" "$C_HOST:$dir/out" "$res/" 2>/dev/null
  printf 'exit_code=%s\nstart=%s\nend=%s\ncommit=%s\n' "$code" "$start" "$(ts)" "$commit" > "$res/status.txt"
  g add "results/$id"
  g commit -q -m "result $id (exit $code)"
  g pull -q --rebase origin "$BRANCH" && g push -q origin "HEAD:$BRANCH" \
    || echo "[$(ts)] push failed, will retry next round"
  echo "[$(ts)] job $id finished, exit code $code"
}

echo "[$(ts)] polling $REPO_URL every ${INTERVAL}s; jobs run on $C_HOST:$C_DIR. Ctrl+C to stop."
while true; do
  if g pull -q --rebase origin "$BRANCH"; then
    # push results that failed to push earlier
    [ "$(g rev-list --count "origin/$BRANCH..HEAD")" = 0 ] || g push -q origin "HEAD:$BRANCH"
    for job in "$REPO_DIR"/jobs/*/; do
      [ -f "$job/run.sh" ] || continue
      id=$(basename "$job")
      [ -e "$REPO_DIR/results/$id" ] && continue
      if [[ ! $id =~ ^[A-Za-z0-9._-]+$ ]]; then echo "skip bad job name: $id"; continue; fi
      run_job "$id"
    done
  else
    echo "[$(ts)] git pull failed, will retry"
  fi
  sleep "$INTERVAL"
done
