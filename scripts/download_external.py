#!/usr/bin/env python3
"""下载外部72阵(tar.gz→解压→.mtx)"""
import os, subprocess, sys, tarfile
REPO = "/home/xueyizhou/spgemm-challenge"
OUT = os.path.join(REPO, "data/external_72")
os.makedirs(OUT, exist_ok=True)
for line in open(f"{REPO}/compare/batch437/external_72.txt"):
    parts = line.strip().split("\t")
    if len(parts) < 4: continue
    name, group, nnz, url = parts[0], parts[1], parts[2], parts[3]
    mtx_path = os.path.join(OUT, f"{name}.mtx")
    if os.path.exists(mtx_path):
        continue
    tar_path = os.path.join(OUT, f"{name}.tar.gz")
    if not os.path.exists(tar_path):
        print(f"下载 {name}...", flush=True)
        r = subprocess.run(["wget", "-q", "--timeout=120", url, "-O", tar_path],
                          capture_output=True, timeout=180)
        if r.returncode != 0 or not os.path.exists(tar_path) or os.path.getsize(tar_path) < 100:
            print(f"  FAIL {name}", flush=True)
            if os.path.exists(tar_path): os.remove(tar_path)
            continue
    # 解压找 .mtx
    try:
        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                if member.name.endswith(".mtx"):
                    f = tf.extractfile(member)
                    with open(mtx_path, "wb") as out:
                        out.write(f.read())
                    print(f"  OK {name} → {mtx_path}", flush=True)
                    break
    except Exception as e:
        print(f"  EXTRACT_FAIL {name}: {e}", flush=True)
    finally:
        if os.path.exists(tar_path):
            os.remove(tar_path)
print("DOWNLOAD_DONE", flush=True)
