#!/usr/bin/env python3
"""D1 oracle 采集(study/34 D 线第一步,v1.1:标定集含 first100 抽样)。

四武器逐行成本对齐 → 逐行"最优路"真值表原料(D2 成本律拟合用)。

臂(每臂=一个进程,进程内自带 5 轮 warmup;裸 env,大阵 mempool 红线规避):
  H×3 = allhash 强制(DITER_MIN_FLOP=999999999)→ HASH_ROWTIME
        9 列:r a_len flop est span ovf cycles d(=C 行 nnz 真值) bkid(hash 梯 bin)
  D×3 = dense 开门(DITER_MIN_FLOP=1 DITER_EST_MIN=1 SPANF=999999 OCCGATE=0)
        → DD_ROWTIME(r cyc mode;档外老内核 cursor/search)+ DSN_RT(r cyc;档位梯 numeric)
  M×3 = merge3(METHOD=merge3)→ MRG3_ROWTIME(r bucket cyc;first100 侧 C_merge)
  R×1 = 默认路由 → DUMP_ROUTING 二进制(est/span/flop/bkid/a_len;现役人口画像)
  K×1 = tools/k7_bench <mtx> 3 <rt>(独立台;10-08 memcheck pre2 0 err+值门 PASS)

纪律(内置):timeout --signal=TERM 自杀(harness 截杀=拥堵源,wedge#8 教训);
输出/日志直落盘不过 grep 管道;串行单卡降 churn。
用法:.venv/bin/python scripts/collect_d1_oracle.py [--reps 3] [--only H,D,K,...] [mat ...]
"""
import argparse, os, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_test")
KBIN = os.path.join(REPO, "tools", "k7_bench")
OUT = os.path.join(REPO, "compare", "d1_oracle")
LOGS = os.path.join(OUT, "logs")

BIG6 = ["bloweybq", "pre2", "c-43", "Si10H16", "in-2004", "nd24k"]
F100 = ["bcsstk32", "bcsstk23", "bcsstm23", "bcspwr08", "bcspwr05", "bcsstm01"]  # 规模分层抽样
BIG_DIR = os.path.join("data", "ocean", "square")
F100_DIR = os.path.join("data", "first100")

TIMEOUT_S = 1500  # TERM 自杀线(3× 预期;in-2004 读盘为长杆)


def sh(mat, arm, rep, mtx, env_extra, instruments):
    """一臂一发:instruments = {ENV: 出文件}。成功=全部出文件非空。"""
    tag = f"{mat}.{arm}{rep}"
    env = dict(os.environ, METHOD="hash")  # 裸 env:不设 mempool(大阵 host arena 红线)
    env.update(env_extra)
    outs = {}
    for k, ext in instruments.items():
        p = os.path.join(OUT, f"{mat}.{arm}{rep}.{ext}")
        env[k] = p
        outs[p] = False
    log = os.path.join(LOGS, tag + ".log")
    cmd = ["timeout", "--signal=TERM", str(TIMEOUT_S), BIN, mtx]
    with open(log, "w") as lf:
        r = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
    ok = r.returncode == 0
    for p in outs:
        outs[p] = os.path.exists(p) and os.path.getsize(p) > 0
    return ok and all(outs.values()), [p for p, g in outs.items() if not g]


def k7_arm(mat, mtx):
    """k7_bench 独立台:rep=3 内置热身+计时,rowtime 单发热态输出。"""
    rt = os.path.join(OUT, f"{mat}.K.tsv")
    log = os.path.join(LOGS, f"{mat}.K.log")
    cmd = ["timeout", "--signal=TERM", str(TIMEOUT_S), KBIN, mtx, "3", rt]
    with open(log, "w") as lf:
        r = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
    return r.returncode == 0 and os.path.exists(rt) and os.path.getsize(rt) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat", nargs="*", help="空=全标定集(big6×{H,D,R,K}+first100×{H,M,K})")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="", help="臂过滤,如 H,D,K(逗号分隔)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    want = set(x for x in a.only.split(",") if x)

    jobs = []  # (mat, arm, rep, mtx)
    mats = a.mat or (BIG6 + F100)
    for m in mats:
        d = F100_DIR if m in F100 else BIG_DIR
        mtx = os.path.join(d, m + ".mtx")
        arms = ["H", "M", "K"] if m in F100 else ["H", "D", "R", "K"]
        for arm in arms:
            if want and arm not in want:
                continue
            reps = 1 if arm in ("R", "K") else a.reps
            for rep in range(1, reps + 1):
                jobs.append((m, arm, rep, mtx))

    fails = []
    t0 = time.time()
    for i, (m, arm, rep, mtx) in enumerate(jobs, 1):
        if arm == "H":
            ok, missing = sh(m, arm, rep, mtx, {"DITER_MIN_FLOP": "999999999"},
                             {"HASH_ROWTIME": "tsv"})
        elif arm == "D":
            ok, missing = sh(m, arm, rep, mtx,
                             {"DITER_MIN_FLOP": "1", "DITER_EST_MIN": "1",
                              "SPANF": "999999", "OCCGATE": "0"},
                             {"DD_ROWTIME": "dd.tsv", "DSN_RT": "dsn.tsv"})
        elif arm == "M":
            ok, missing = sh(m, arm, rep, mtx, {"METHOD": "merge3"},
                             {"MRG3_ROWTIME": "tsv"})
        elif arm == "R":
            ok, missing = sh(m, arm, rep, mtx, {}, {"DUMP_ROUTING": "routing.bin"})
        elif arm == "K":
            ok = k7_arm(m, mtx)
            missing = [os.path.join(OUT, m + ".K.tsv")] if not ok else []
        else:
            continue
        st = "OK " if ok else "FAIL"
        print(f"[{i}/{len(jobs)}] {st} {m}.{arm}{rep} (+{time.time()-t0:.0f}s)"
              + (f" missing={missing}" if not ok else ""), flush=True)
        if not ok:
            fails.append(f"{m}.{arm}{rep}")
    print(f"\nD1 采集完成:{len(jobs)-len(fails)}/{len(jobs)} 成功,总耗时 {time.time()-t0:.0f}s")
    if fails:
        print("失败臂:", " ".join(fails))
        sys.exit(1)


if __name__ == "__main__":
    main()
