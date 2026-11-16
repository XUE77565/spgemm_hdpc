#!/usr/bin/env python3
"""FlexSpGEMM 142 测试集下载(UF 老镜像版,11-16 找到的活源):
www.cise.ufl.edu/research/sparse/MM/<group>/<name>.tar.gz —— Flex AE 脚本的第一备选,
tamu 主源 S3 AccessDenied 期间仍 200。tar 内含 <name>.mtx 与 <name>_b.mtx(取主件)。
断点续传式(有 .mtx 跳过);失败阵重试 3 轮;输出 external_sota/FlexSpGEMM/data/test142/。
"""
import csv, os, subprocess, sys, tarfile, tempfile, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(REPO, "external_sota/FlexSpGEMM/data/test142")
os.makedirs(DEST, exist_ok=True)
rows = list(csv.reader(open(os.path.join(REPO, "external_sota/FlexSpGEMM/data/mat-142.csv"))))
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0"

def fetch(name, group):
    out = os.path.join(DEST, name + ".mtx")
    if os.path.exists(out) and os.path.getsize(out) > 100:
        return "skip"
    url = f"https://www.cise.ufl.edu/research/sparse/MM/{group}/{name}.tar.gz"
    with tempfile.TemporaryDirectory(dir=DEST) as tmp:
        arc = os.path.join(tmp, "m.tar.gz")
        try:
            r = subprocess.run(["curl", "-s", "-L", "--max-time", "300", "-A", UA, "-o", arc, url],
                               capture_output=True, timeout=320)
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        if not os.path.exists(arc) or os.path.getsize(arc) < 100:
            return f"FAIL rc={r.returncode}"
        if open(arc, "rb").read(2) != b"\x1f\x8b":
            return f"NOTGZ {os.path.getsize(arc)}B"
        with tarfile.open(arc) as t:
            m = [e for e in t.getmembers() if e.name.endswith(f"/{name}.mtx")]
            if not m: return "NOMTX"
            with t.extractfile(m[0]) as s, open(out, "wb") as d:
                while True:
                    b = s.read(1 << 20)
                    if not b: break
                    d.write(b)
    return "ok"

pending = [(r[1], r[0]) for r in rows if len(r) >= 2]
print(f"清单 {len(pending)} 阵", flush=True)
for round_ in range(3):
    todo = [(n, g) for n, g in pending
            if not (os.path.exists(os.path.join(DEST, n + ".mtx")) and os.path.getsize(os.path.join(DEST, n + ".mtx")) > 100)]
    print(f"[{time.strftime('%H:%M')}] round {round_+1}: 待取 {len(todo)}", flush=True)
    if not todo: break
    fails = []
    for i, (n, g) in enumerate(todo):
        st = fetch(n, g)
        if i % 20 == 0 or st not in ("ok", "skip"):
            print(f"  [{i+1}/{len(todo)}] {n}: {st}", flush=True)
        if st != "ok": fails.append((n, st))
    if not fails: break
have = sorted(f for f in os.listdir(DEST) if f.endswith(".mtx"))
print(f"完成: {len(have)}/142", flush=True)
if len(have) < 142:
    print("缺:", ", ".join(f[:-4] for f in [r[1]+'.mtx' for r in rows] if f not in have)[:500], flush=True)
