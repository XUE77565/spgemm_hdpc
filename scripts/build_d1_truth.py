#!/usr/bin/env python3
"""D1 真值表构建:四臂逐行成本 join → 逐行"最优路"真值表(D2 成本律拟合原料)。

输入(compare/d1_oracle/):
  {mat}.H{1..3}.tsv   9 列:r a_len flop est span ovf cycles d bkid(hash 梯)
  {mat}.D{1..3}.dd.tsv 3 列:r cyc mode(档外老内核 cursor/search)
  {mat}.D{1..3}.dsn.tsv 2 列:r cyc(档位梯 numeric)
  {mat}.K.tsv          2 列:r cyc(k7_bench,谓词域内)
  {mat}.M{1..3}.tsv    3 列:r bucket cyc(first100;行成本=桶 max=块墙钟)
  {mat}.R1.routing.bin 头 2 int + 5×rows int(est/span/flop/bkid/a_len;默认路由现役)

输出(compare/d1_oracle/truth/{mat}.truth.tsv):
  r a_len flop est span d dup bkid_def cyc_hash cyc_dense cyc_k7 cyc_mrg3 best
  · cyc_*=3 rep 中位(仅同臂全 rep 有值的行);ovf>0 的 hash 行剔除(rettry 重做,cyc 失真)
  · best=已测武器 argmin;k7 域外/稠密阵 L0 域界行如实留空
摘要:各行数 + 各武器赢面人口 + 采样行数合计(计划判据 ≥10 万)。
用法:.venv/bin/python scripts/build_d1_truth.py [mat ...]
"""
import argparse, os, statistics, struct, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "compare", "d1_oracle")
OUT = os.path.join(SRC, "truth")
BIG6 = ["bloweybq", "pre2", "c-43", "Si10H16", "in-2004", "nd24k"]
F100 = ["bcsstk32", "bcsstk23", "bcsstm23", "bcspwr08", "bcspwr05", "bcsstm01",
        "1138_bus", "ash219", "bp_0", "arc130"]


def median_maps(paths, ncols):
    """多 rep 文件 → {r: [各列中位]}(仅全 rep 出现的行)。"""
    acc = {}
    for p in paths:
        if not os.path.exists(p):
            return {}
        for ln in open(p):
            f = ln.split()
            if len(f) != ncols:
                continue
            r = int(f[0])
            acc.setdefault(r, []).append([int(x) for x in f[1:]])
    return {r: [statistics.median(v[i] for v in vs) for i in range(len(vs[0]))]
            for r, vs in acc.items() if len(vs) == len(paths)}


def load_routing(p):
    """DUMP_ROUTING 二进制 → {r: bkid}(第 4 数组)。"""
    if not os.path.exists(p):
        return {}
    with open(p, "rb") as f:
        rows, _ = struct.unpack("2i", f.read(8))
        f.seek(8 + 3 * rows * 4)
        bk = struct.unpack(f"{rows}i", f.read(rows * 4))
    return {r: bk[r] for r in range(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat", nargs="*", default=BIG6 + F100)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tot = 0
    print(f"{'mat':<10}{'rows':>9}{'hash':>8}{'dense':>8}{'k7':>8}{'mrg3':>8}{'best行':>8}")
    for m in a.mat:
        H = median_maps([os.path.join(SRC, f"{m}.H{i}.tsv") for i in range(1, a.reps + 1)], 9)
        DD = median_maps([os.path.join(SRC, f"{m}.D{i}.dd.tsv") for i in range(1, a.reps + 1)], 3)
        DSN = median_maps([os.path.join(SRC, f"{m}.D{i}.dsn.tsv") for i in range(1, a.reps + 1)], 2)
        # dense 子律类:dsn(档位梯 numeric)=2 / dd cursor=1 / dd search=0(docs/66 §8 mode 列)
        DCLASS = {r: 2 for r in DSN}
        for r, v in DD.items():
            DCLASS[r] = int(v[1])
        K = {}
        kp = os.path.join(SRC, f"{m}.K.tsv")
        if os.path.exists(kp):
            for ln in open(kp):
                r, c = ln.split()
                K[int(r)] = int(c)
        Mraw = [os.path.join(SRC, f"{m}.M{i}.tsv") for i in range(1, a.reps + 1)]
        M = {}
        if all(os.path.exists(p) for p in Mraw):
            acc = {}
            for p in Mraw:
                for ln in open(p):
                    r, b, c = ln.split()
                    acc.setdefault(int(r), {}).setdefault(int(b), []).append(int(c))
            M = {r: max(statistics.median(v) for v in bs.values()) for r, bs in acc.items()
                 if all(len(v) == a.reps for v in bs.values())}
        BDEF = load_routing(os.path.join(SRC, f"{m}.R1.routing.bin"))
        rows = sorted(set(H) | set(K))
        if not rows:
            print(f"{m:<10} 无数据(臂全失败=域界或未采集)")
            continue
        n_win = {"hash": 0, "dense": 0, "k7": 0, "mrg3": 0}
        op = os.path.join(OUT, f"{m}.truth.tsv")
        with open(op, "w") as fo:
            fo.write("r a_len flop est span d dup bkid_def cyc_hash cyc_dense cyc_k7 cyc_mrg3 best dclass\n")
            for r in rows:
                h = H.get(r)
                if not h:
                    continue
                a_len, flop, est, span, ovf, cyc_h, d, bkh = h
                if ovf > 0 or (cyc_h == 0 and r not in DD and r not in DSN and r not in K and r not in M):
                    continue   # ovf 行剔除;全武器无测度行不进表
                dd = DD.get(r)
                cyc_d = min(dd[0], DSN[r][0]) if (dd and r in DSN) else (dd[0] if dd else (DSN[r][0] if r in DSN else 0))
                cyc_k = K.get(r, 0)
                cyc_m = M.get(r, 0)
                cand = {"hash": cyc_h, "dense": cyc_d, "k7": cyc_k, "mrg3": cyc_m}
                cand = {k: v for k, v in cand.items() if v > 0}
                # dense 合法域(D2 v3):档内=dsn 候选;档外需 avgB=flop/a_len≥64(cursor 态)。
                # search 态(档外∧avgB<64)=幽灵人口(自然路由≈0,DN 臂终审)——强制臂实测=坏假设决策成本,不作 oracle 依据
                if "dense" in cand and span > 7277 and (a_len <= 0 or flop / a_len < 64):
                    del cand["dense"]
                best = min(cand, key=cand.get) if cand else "-"
                if best != "-":
                    n_win[best] += 1
                dup = flop / d if d else 0.0
                fo.write(f"{r} {int(a_len)} {int(flop)} {int(est)} {int(span)} {int(d)} "
                         f"{dup:.3f} {BDEF.get(r,-1)} {int(cyc_h)} {int(cyc_d)} {int(cyc_k)} "
                         f"{int(cyc_m)} {best} {DCLASS.get(r,-1)}\n")
        n = sum(n_win.values())
        tot += n
        print(f"{m:<10}{len(rows):>9}{len(H):>8}{len(DD)+len(DSN):>8}{len(K):>8}{len(M):>8}{n:>8}")
    print(f"\n合计 best 行数:{tot}(计划判据 ≥100,000)")
    sys.exit(0 if tot >= 100000 else 2)


if __name__ == "__main__":
    main()
