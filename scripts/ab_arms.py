#!/usr/bin/env python3
"""多臂 env 电池:同一个二进制,臂 = 一组环境变量;逐阵逐轮把各臂交错发射。

用来判"运行时开关"类改动(开关关=基线臂,开=实验臂):两臂共用同一份机器码布局,
免掉"每次增删代码重洗布局、微阵 ±15%"的编译混沌(study/57 S1 定谳)。

口径与 ab_pair.py 同源(矩阵表/正则/环境直接 import 它的):METHOD 按阵(hash/adaptive)、
WARMUP_N=3、USE_MEMPOOL=1、MP_HOST_MB=8192、D6EXP=0;co = TOTAL(GPU) − h2d − d2h
(merge3 路径取 mrg3 TOTAL)。另记计时轮(最后一次 hash_product)的相位毫秒:
accumulate / compact+sort / retry / dense_direct。

判读:每个实验臂对首臂(基线)的逐阵中位比 + 测量分级(微 <1ms 不作单阵裁决 / 中 1-10ms /
稳 ≥10ms)+ GM + 最差阵 + nnz 逐位;并给 accumulate 相位的逐阵中位比(机制读数)。
绿 = GM ≤ 1.02 且无(非微级)阵 > 1.10 且 nnz 零异常(study/70 §七)。

用法:
  ab_arms.py --bin ./spgemm_x --out PREFIX --arm off: --arm m1:HH_BM2=1 --arm m2:HH_BM2=2
             [--rounds 3] [--gpu 0] [--only bcsstk17,Bump] [--extra name=path:method ...]
             [--no-battery] [--stat]
  --stat:每阵对每个含 HH_BM2= 的臂加发一次诊断跑(HH_BM2_STAT=1、WARMUP_N=0,不计时),
          把 [bm2-stat] 行(每 bin 走/未走两遍式的行数与 Σflop)写进 PREFIX_stat.csv。
输出:PREFIX.csv(逐发)、PREFIX.txt(判读)、PREFIX_stat.csv、PREFIX_fail.log(失败发的输出尾)。
"""
import argparse, csv, math, os, re, statistics, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_pair import MATS, RE_TOT, RE_H2D, RE_D2H, RE_M3, RE_NNZ   # noqa: E402

RE_PROF = re.compile(r"\[hash-prof\]\s+(\S+)\s+([\d.]+)\s*ms")
RE_STAT = re.compile(r"\[bm2-stat\] bin=(\d+) mode=(\d+) leg=(\d+) rows=(\d+) taken_rows=(\d+) "
                     r"taken_flop=(\d+) skip_rows=(\d+) skip_flop=(\d+) ovf_rows=(\d+)")
PHASES = ["accumulate", "compact+sort", "retry", "dense_direct"]
RE_TFLOP = re.compile(r"total_flop=(\d+)")
RE_AROWS = re.compile(r"Input A: (\d+) x (\d+), nnz = (\d+)")


def parse_arm(s):
    name, _, envs = s.partition(":")
    env = {}
    for kv in envs.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k.strip()] = v.strip()
    return name.strip(), env


def run_one(binary, path, method, gpu, arm_env, warm="3", extra=None, timeout=900):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, METHOD=method, WARMUP_N=warm,
               USE_MEMPOOL="1", MP_HOST_MB="8192", D6EXP="0")
    env.update(arm_env)
    if extra:
        env.update(extra)
    try:
        r = subprocess.run([binary, path], env=env, capture_output=True, text=True, timeout=timeout)
    except Exception as ex:   # 超时等
        return None, "EXC %r" % (ex,)
    out = r.stdout + r.stderr
    tot = RE_TOT.findall(out); h = RE_H2D.findall(out); d = RE_D2H.findall(out)
    m3 = RE_M3.findall(out); nnz = RE_NNZ.findall(out)
    if not nnz:
        return None, "rc=%s (无 Result C 行)\n%s" % (r.returncode, out[-3000:])
    if m3:
        co = float(m3[-1])                                  # merge3 路径:mrg3 口径(同 ab_pair)
    elif tot and h and d:
        co = float(tot[-1]) - float(h[-1]) - float(d[-1])
    else:
        co = float(tot[-1]) if tot else None
    if co is None or co <= 0:
        return None, "rc=%s (无计时行)\n%s" % (r.returncode, out[-3000:])
    prof = RE_PROF.findall(r.stderr)                        # 预热轮+计时轮顺序排列;取最后一个 h2d 之后
    last = 0
    for i, (k, _) in enumerate(prof):
        if k == "h2d":
            last = i
    ph = {}
    for k, v in prof[last:]:
        ph[k] = ph.get(k, 0.0) + float(v)
    res = {"co": co, "nnz": int(nnz[-1]), "rc": r.returncode}
    for p in PHASES:
        res[p] = ph.get(p, 0.0)
    res["stat"] = [tuple(int(x) for x in m) for m in RE_STAT.findall(r.stderr)]
    tf = RE_TFLOP.findall(r.stderr)                         # dbg 行 "[hash] avg_product=… (total_flop=…)"
    res["total_flop"] = int(tf[-1]) if tf else 0
    ar = RE_AROWS.findall(out)
    res["a_rows"] = int(ar[-1][0]) if ar else 0
    return res, None


def med(v):
    return statistics.median(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", required=True, help="name:K=V,K=V(首臂=基线)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--only", default="")
    ap.add_argument("--extra", action="append", default=[], help="name=path:method")
    ap.add_argument("--no-battery", action="store_true")
    ap.add_argument("--stat", action="store_true")
    a = ap.parse_args()

    arms = [parse_arm(s) for s in a.arm]
    mats = [] if a.no_battery else list(MATS)
    for e in a.extra:
        name, _, rest = e.partition("=")
        path, _, method = rest.rpartition(":")
        mats.append((name, path, method or "hash"))
    if a.only:
        keep = set(x for x in a.only.split(",") if x)
        mats = [m for m in mats if m[0] in keep]
    present, missing = [], []
    for m in mats:
        (present if os.path.isfile(m[1]) else missing).append(m)

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("# ab_arms: bin=%s gpu=%s rounds=%d" % (a.bin, a.gpu, a.rounds))
    for n, e in arms:
        say("#   arm %-10s %s" % (n, " ".join("%s=%s" % kv for kv in sorted(e.items())) or "(无额外 env)"))
    if missing:
        say("# 缺文件跳过: " + ", ".join("%s(%s)" % (m[0], m[1]) for m in missing))

    rows, stats, share = [], [], {}
    fail = open(a.out + "_fail.log", "w")
    for name, path, method in present:
        if a.stat:
            for an, ae in arms:
                if ae.get("HH_BM2", "0") == "0" and ae.get("HH_BM2_L1", "0") == "0":
                    continue
                res, err = run_one(a.bin, path, method, a.gpu, ae, warm="0", extra={"HH_BM2_STAT": "1"})
                if res is None:
                    fail.write("== STAT %s %s\n%s\n" % (name, an, err)); fail.flush()
                    print("%14s %-10s STAT FAIL" % (name, an), flush=True)
                    continue
                for st in res["stat"]:
                    stats.append((name, an) + st + (res["nnz"], res["total_flop"], res["a_rows"]))
                tr = sum(s[4] for s in res["stat"]); sr = sum(s[6] for s in res["stat"])
                tf = sum(s[5] for s in res["stat"]); sf = sum(s[7] for s in res["stat"])
                share[(name, an)] = (tr, sr, tf, sf, res["total_flop"], res["a_rows"])
                print("%14s %-10s STAT taken_rows=%d skip_rows=%d taken_flop=%d skip_flop=%d total_flop=%d a_rows=%d nnz=%d"
                      % (name, an, tr, sr, tf, sf, res["total_flop"], res["a_rows"], res["nnz"]), flush=True)
        for r in range(a.rounds):
            for an, ae in arms:
                res, err = run_one(a.bin, path, method, a.gpu, ae)
                row = {"mat": name, "arm": an, "round": r, "co": -1, "nnz": -1}
                for p in PHASES:
                    row[p] = -1
                if res is None:
                    fail.write("== %s %s r%d\n%s\n" % (name, an, r, err)); fail.flush()
                    print("%14s %-10s r%d FAIL" % (name, an, r), flush=True)
                else:
                    row.update({k: res[k] for k in ["co", "nnz"] + PHASES})
                    print("%14s %-10s r%d co=%.3f acc=%.3f cs=%.3f retry=%.3f nnz=%d"
                          % (name, an, r, res["co"], res["accumulate"], res["compact+sort"], res["retry"], res["nnz"]),
                          flush=True)
                rows.append(row)
    fail.close()

    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mat", "arm", "round", "co", "nnz"] + PHASES)
        w.writeheader(); w.writerows(rows)
    if a.stat:
        with open(a.out + "_stat.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mat", "arm", "bin", "mode", "leg", "rows", "taken_rows", "taken_flop",
                        "skip_rows", "skip_flop", "ovf_rows", "nnz", "total_flop", "a_rows"])
            w.writerows(stats)

    # ── 判读 ──
    by, nz = {}, {}
    for r in rows:
        if r["co"] > 0:
            d = by.setdefault((r["mat"], r["arm"]), {"co": [], "acc": [], "cs": [], "retry": []})
            d["co"].append(r["co"]); d["acc"].append(r["accumulate"])
            d["cs"].append(r["compact+sort"]); d["retry"].append(r["retry"])
        if r["nnz"] > 0:
            nz.setdefault(r["mat"], {}).setdefault(r["arm"], set()).add(r["nnz"])
    base = arms[0][0]
    verdicts = {}
    for an, _ in arms[1:]:
        say("")
        say("===== 判读:%s / %s(逐阵中位比;测量分级)=====" % (an, base))
        say("%14s %10s %10s %8s %6s  %9s %9s %8s %9s %9s  %s" % ("mat", base + "_ms", an + "_ms", "ratio", "级",
                                                                  "acc_" + base, "acc_" + an, "acc比",
                                                                  "flop占比", "行占比", "备注"))
        ratios, worst, bad, micro, nfail = [], 0.0, [], [], 0
        for name, _, _ in present:
            A, B = by.get((name, base)), by.get((name, an))
            if not A:                                       # 基线臂自己就跑不出来 = 这阵在本口径下不可测,不计入裁定
                nfail += 1
                say("%14s  不可测(基线臂无数据,不计入裁定;见 _fail.log)" % name)
                continue
            if not B:
                nfail += 1; bad.append(name)
                say("%14s  FAIL(实验臂缺数据,基线臂正常)" % name)
                continue
            na, nb = nz.get(name, {}).get(base), nz.get(name, {}).get(an)
            nnz_bad = (na != nb) or (na is None) or len(na) != 1
            ma, mb = med(A["co"]), med(B["co"])
            ratio = mb / ma
            cls = "微" if ma < 1.0 else ("中" if ma < 10.0 else "稳")
            aa, ab_ = med(A["acc"]), med(B["acc"])
            accr = ("%8.4f" % (ab_ / aa)) if aa and aa > 0.05 else "       -"
            note = []
            if nnz_bad:
                note.append("NNZ-MISMATCH!"); bad.append(name)
            if cls == "微":
                note.append("微级不作单阵裁决"); micro.append(name)
            else:
                if ratio > 1.10:
                    note.append("回归>10%")
                elif abs(ratio - 1) > 0.05:
                    note.append("Δ>5%")
                worst = max(worst, ratio)
            ratios.append(ratio)
            sh = share.get((name, an))                      # 两遍式覆盖面:走两遍式的 Σflop / 全阵 Σflop;行数 / 全阵行数
            fsh = ("%8.1f%%" % (100.0 * sh[2] / sh[4])) if sh and sh[4] > 0 else "        -"
            rsh = ("%8.1f%%" % (100.0 * sh[0] / sh[5])) if sh and sh[5] > 0 else "        -"
            say("%14s %10.3f %10.3f %8.4f %6s  %9.3f %9.3f %s %s %s  %s"
                % (name, ma, mb, ratio, cls, aa, ab_, accr, fsh, rsh, " ".join(note)))
        if ratios:
            gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
            green = (not bad) and gm <= 1.02 and worst <= 1.10
            say("GM(%s/%s) = %.4f | 非微级最差 = %.4f | nnz 异常/实验臂缺数据阵 = %d | 微级 %d 阵 | 有效 %d 阵"
                % (an, base, gm, worst, len(bad), len(micro), len(ratios)))
            say("裁定: " + ("绿(GM≤1.02 且无非微级阵>1.10 且 nnz 全同)" if green else "红"))
            verdicts[an] = (gm, worst, len(bad), green)
        else:
            say("裁定: 红(无有效数据)")
            verdicts[an] = (None, None, len(bad), False)
    with open(a.out + ".txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
