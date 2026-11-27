#!/usr/bin/env python3
"""合成训练阵生成器(两遍式行门 HH_BM2_FX 的标定用;dialog/qa-013 批准的方案 A)。

为什么要合成:行门的判别量是"每行乘积数 flop ÷ 该 bin 的哈希表长 HT"。data/train_disp 的真实训练阵量程只到 flop/HT≈8
(到头时两遍式对哈希的周期比 R 已贴近 1),而会输的行在 10 以上——训练池里没有。这里用受控结构把 flop/HT 铺到 1…192。
生成的阵不属于任何测试账(428 / flex142 / external_72),出处就是本脚本 + 下面的参数表 + 固定种子;阵只在作业目录里现生成,不进 git。

结构(一个阵由若干"组"拼成,组与组互不相交):
  一个组 = W 个行号,也是这 W 行全部非零所在的列号集合 I = {base + off + s·t, t = 0..W-1}(s = 步长;s>1 时一个长 s·W 的超块里交错着 s 个组)。
  组内每行在 I 里均匀随机取 k 个不同的列。于是 C = A·A 的这一行:
      乘积数 flop = k·k;不同列数 D ≈ W·(1 − e^{−k²/W})(k² ≥ 3.6W 时 ≥ 0.97W);列跨度 = s·W。
  同一个阵里 W、s 固定(行落在同一个 bin 附近),k 按一组目标比值 r 取:k = round(sqrt(r · 3.6 · W)),上限 W。
  3.6 = 对"表长 ≈ 2 × est、est ≈ 1.8 × D"的粗估,只用来把梯级铺到目标量程;每行实际落在哪个 bin、flop/HT 是多少,以标定时的逐行数据为准。
  超块里没被组占用的行号放一个对角元(保证没有空行)。
参数表(固定,改了就不是同一批训练阵):
  W ∈ {112, 224, 448, 900, 1800, 3600, 7200} × s ∈ {1, 16};
  目标比值 r:W < 3600 用 {1,1.5,2,3,4,6,8,12,16,24,32,48,64,96,128,192};W = 3600 用 {2,4,8,16,32,64,128};W = 7200 用 {2,8,32,128};
  每个 k 重复 reps = max(1, ⌈1500/W⌉) 组,且总行数不足 6000 时再加(生产入口的 dispatcher 对小阵可能走 merge3);
  种子 = 2026112700000 + 10·W + s;数值在 [0.5, 1.5) 均匀(全正,不制造消去)。
用法:gen_synth.py --out DIR [--only syn_w224_s1,syn_w448_s16] [--list]
输出:DIR/<名字>.mtx(MatrixMarket coordinate real general,1 起下标)与 DIR/MANIFEST.tsv(名字、n、nnz、W、s、k 列表、reps、种子)。
"""
import argparse, math, os, sys, time
import numpy as np

RATIOS = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192]
WS = [112, 224, 448, 900, 1800, 3600, 7200]
STRIDES = [1, 16]
HT_PER_W = 3.6
SEED0 = 2026112700000


def spec(W):
    if W >= 7200:
        ratios = [2, 8, 32, 128]
    elif W >= 3600:
        ratios = [2, 4, 8, 16, 32, 64, 128]
    else:
        ratios = RATIOS
    ks = []
    for r in ratios:
        k = max(8, min(W, int(round(math.sqrt(r * HT_PER_W * W)))))
        if k not in ks:
            ks.append(k)
    reps = max(1, int(math.ceil(1500.0 / W)))
    while W * len(ks) * reps < 6000:
        reps += 1
    return ks, reps


def build(W, s):
    ks, reps = spec(W)
    seed = SEED0 + 10 * W + s
    groups = [k for _ in range(reps) for k in ks]
    nblk = int(math.ceil(len(groups) / float(s)))
    n = nblk * s * W
    rng = np.random.default_rng(seed)
    used = np.zeros(n, dtype=bool)
    ri, ci = [], []
    t_local = np.arange(W, dtype=np.int64)
    for gi, k in enumerate(groups):
        base = (gi // s) * s * W + (gi % s)
        idx = base + s * t_local
        used[idx] = True
        sel = np.empty((W, k), dtype=np.int64)
        for t in range(W):
            sel[t] = rng.choice(W, size=k, replace=False)
        sel.sort(axis=1)
        ri.append(np.repeat(idx, k))
        ci.append(idx[sel].ravel())
    free = np.nonzero(~used)[0]
    if free.size:
        ri.append(free)
        ci.append(free)
    ri = np.concatenate(ri)
    ci = np.concatenate(ci)
    va = rng.uniform(0.5, 1.5, size=ri.size)
    return n, ri, ci, va, ks, reps, seed


def write_mtx(path, n, ri, ci, va, note):
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write("% " + note + "\n")
        f.write("%d %d %d\n" % (n, n, ri.size))
        step = 2000000
        for a in range(0, ri.size, step):
            b = min(ri.size, a + step)
            blk = np.column_stack([ri[a:b] + 1, ci[a:b] + 1]).astype(np.int64)
            lines = ["%d %d %.6g" % (i, j, v) for (i, j), v in zip(blk.tolist(), va[a:b].tolist())]
            f.write("\n".join(lines))
            f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    only = set(x for x in a.only.split(",") if x)
    specs = [(W, s) for W in WS for s in STRIDES]
    if a.list or not a.out:
        print("name\tn\tnnz(约)\tW\ts\tks\treps")
        for W, s in specs:
            ks, reps = spec(W)
            ng = len(ks) * reps
            n = int(math.ceil(ng / float(s))) * s * W
            print("syn_w%d_s%d\t%d\t%d\t%d\t%d\t%s\t%d" % (W, s, n, W * sum(ks) * reps, W, s, ",".join(map(str, ks)), reps))
        return 0
    os.makedirs(a.out, exist_ok=True)
    man = open(os.path.join(a.out, "MANIFEST.tsv"), "w")
    man.write("name\tn\tnnz\tW\ts\tks\treps\tseed\n")
    for W, s in specs:
        name = "syn_w%d_s%d" % (W, s)
        if only and name not in only:
            continue
        t0 = time.time()
        n, ri, ci, va, ks, reps, seed = build(W, s)
        note = "synthetic training matrix (tools/train_synth/gen_synth.py): W=%d s=%d ks=%s reps=%d seed=%d" % (
            W, s, ",".join(map(str, ks)), reps, seed)
        write_mtx(os.path.join(a.out, name + ".mtx"), n, ri, ci, va, note)
        man.write("%s\t%d\t%d\t%d\t%d\t%s\t%d\t%d\n" % (name, n, ri.size, W, s, ",".join(map(str, ks)), reps, seed))
        man.flush()
        print("%-16s n=%d nnz=%d ks=%s reps=%d  %.0fs" % (name, n, ri.size, ",".join(map(str, ks)), reps, time.time() - t0), flush=True)
    man.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
