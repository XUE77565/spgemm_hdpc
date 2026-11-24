# exp_poller.ps1 -- run on B (Windows) in a PowerShell window:
#   powershell -ExecutionPolicy Bypass -File .\exp_poller.ps1
# Every $Interval seconds: git pull, find jobs/<id>/run.sh without results/<id>/,
# copy that version of the code to C, run it there, push the results back.
# Needs git and the Windows OpenSSH client (ssh, scp). Stop with Ctrl+C.

# ======== fill these in ========
$RepoUrl = ""              # GitHub repo, e.g. git@github.com:OWNER/REPO.git
$CHost   = ""              # how B reaches C with ssh, e.g. user@1.2.3.4 or a Host alias
# ===============================
$CDir     = "exp-runs"     # folder on C (relative to home) for the jobs
$Branch   = "main"
$Interval = 60             # seconds between checks
$RepoDir  = Join-Path $PSScriptRoot "exp-repo"   # local clone on B

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not $RepoUrl -or -not $CHost) {
    Write-Host "Please fill in `$RepoUrl and `$CHost at the top of this script."
    exit 1
}

function Ts { Get-Date -Format "yyyy-MM-dd HH:mm:ss" }

if (-not (Test-Path (Join-Path $RepoDir ".git"))) {
    git clone -b $Branch $RepoUrl $RepoDir
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
git -C $RepoDir config core.autocrlf false
git -C $RepoDir config user.name *> $null
if ($LASTEXITCODE -ne 0) { git -C $RepoDir config user.name "exp-poller" }
git -C $RepoDir config user.email *> $null
if ($LASTEXITCODE -ne 0) { git -C $RepoDir config user.email "exp-poller@localhost" }

function Run-Job($id) {
    $dir = "$CDir/$id"
    $res = Join-Path $RepoDir "results\$id"
    $commit = (git -C $RepoDir rev-parse HEAD)
    $start = Ts
    Write-Host "[$(Ts)] ===== job $id (commit $($commit.Substring(0,8))) ====="

    # 1. send this version of the code to C
    # scp gets relative local paths only, so a drive letter like C: is never mistaken for a host
    Push-Location $RepoDir
    git -c core.autocrlf=false archive --format=tar -o "../exp-$id.tar" HEAD
    scp -q "../exp-$id.tar" ($CHost + ":exp-" + $id + ".tar")
    $up = $LASTEXITCODE
    Remove-Item "../exp-$id.tar" -ErrorAction SilentlyContinue
    Pop-Location
    if ($up -ne 0) {
        Write-Host "[$(Ts)] upload to C failed, will retry next round"
        return
    }

    # 2. unpack and run on C; output shows here and is saved to log.txt on C
    $remote = 'mkdir -p @DIR@ && cd @DIR@ && rm -rf code out exit_code && mkdir code out && tar -xf ~/exp-@ID@.tar -C code && rm -f ~/exp-@ID@.tar || exit 254; cd code; export JOB_ID=@ID@ OUT_DIR=$(cd ../out && pwd); { bash jobs/@ID@/run.sh; echo $? > ../exit_code; } 2>&1 | tee ../log.txt; exit $(cat ../exit_code)'
    $remote = $remote.Replace("@DIR@", $dir).Replace("@ID@", $id)
    ssh $CHost $remote
    $code = $LASTEXITCODE
    if ($code -eq 255 -or $code -eq 254) {
        Write-Host "[$(Ts)] could not reach C or set up the job there (code $code), will retry next round"
        return
    }

    # 3. bring back log.txt and out/, record status, push
    New-Item -ItemType Directory -Force -Path $res | Out-Null
    Push-Location $RepoDir
    scp -q -r ($CHost + ":" + $dir + "/log.txt") ($CHost + ":" + $dir + "/out") "results/$id" 2>$null
    Pop-Location
    $status = "exit_code=$code`nstart=$start`nend=$(Ts)`ncommit=$commit`n"
    [System.IO.File]::WriteAllText((Join-Path $res "status.txt"), $status)
    git -C $RepoDir add "results/$id"
    git -C $RepoDir commit -q -m "result $id (exit $code)"
    git -C $RepoDir pull -q --rebase origin $Branch
    if ($LASTEXITCODE -eq 0) { git -C $RepoDir push -q origin "HEAD:$Branch" }
    if ($LASTEXITCODE -ne 0) { Write-Host "[$(Ts)] push failed, will retry next round" }
    Write-Host "[$(Ts)] job $id finished, exit code $code"
}

Write-Host "[$(Ts)] polling $RepoUrl every $Interval s; jobs run on ${CHost}:$CDir. Ctrl+C to stop."
while ($true) {
    git -C $RepoDir pull -q --rebase origin $Branch
    if ($LASTEXITCODE -eq 0) {
        # push results that failed to push earlier
        $ahead = (git -C $RepoDir rev-list --count "origin/$Branch..HEAD")
        if ($ahead -ne "0") { git -C $RepoDir push -q origin "HEAD:$Branch" }
        $jobsDir = Join-Path $RepoDir "jobs"
        if (Test-Path $jobsDir) {
            foreach ($job in (Get-ChildItem -Directory $jobsDir | Sort-Object Name)) {
                $id = $job.Name
                if (-not (Test-Path (Join-Path $job.FullName "run.sh"))) { continue }
                if (Test-Path (Join-Path $RepoDir "results\$id")) { continue }
                if ($id -notmatch '^[A-Za-z0-9._-]+$') { Write-Host "skip bad job name: $id"; continue }
                Run-Job $id
            }
        }
    } else {
        Write-Host "[$(Ts)] git pull failed, will retry"
    }
    Start-Sleep -Seconds $Interval
}
