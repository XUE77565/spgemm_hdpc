#!/usr/bin/env python3
"""全集影子 / 相位地图采数:同一个二进制,臂 = 一组环境变量,逐阵逐轮把各臂交错发射;边跑边落盘。

与 ab_arms.py 同口径(co = TOTAL(GPU) − h2d − d2h,merge3 路径取 mrg3 TOTAL;USE_MEMPOOL=1、MP_HOST_MB=8192、D6EXP=0),
为跑全 428 做了三点不同:
  ① 逐发即时追加写 CSV(作业半途被杀也留得下已跑的部分);按文件大小升序跑(小阵先出信号);
  ② 轮数按文件大小分档(小阵噪声大、装载快 → 多轮;大阵稳、装载慢 → 少轮);
  ③ 记下计时轮的全部 [hash-prof] 相位(长表),并可对每发算一个"签名"(--sig-regex 匹配到的行的 md5),
     用来在全集上核对"各臂的门普查结果逐行相同"这类行为保持。
输出:PREFIX.csv          mat,tier,size_mb,arm,round,co,nnz,sig,dur_s(dur_s = 这一发从起进程到退出的墙钟秒数)
      PREFIX_phases.csv   mat,arm,round,phase,ms(只含计时轮)
      PREFIX_fail.log     失败发的输出尾
      PREFIX.hb           心跳:当前这一发的 阵 / 臂 / 轮 / 开始时刻 / 子进程 pid / 累计发数(每发覆写一次;
                          旁人要判断"是不是挂了",看这个文件的开始时刻与 --timeout,别看驱动进程的年龄)
超时:每一发都包在 coreutils 的 `timeout -k 30 <秒>` 里(ps 里看得见),并放进自己的进程组;驱动另有一层
      Python 侧的兜底超时(多 60 秒),到点连进程组一起杀。--budget-min 给整段设时间预算,用完就在阵与阵之间干净收尾。
判读用 scripts/shadow_sum.py。
用法:
  shadow428.py --bin ./spgemm_x --list reference/mats_428.csv --roots data/ocean/square,data/first100 --out PREFIX
               --arm off: --arm aa: --arm cen:HH_CENSUS=1 [--env DBG=1 --env HASH_LHT23=0] [--method adaptive]
               [--rounds 5,3,2] [--tier-mb 10,200] [--sig-regex '^\\[census-stat\\].*$'] [--limit N] [--gpu 0]
               [--timeout 900] [--budget-min 0]
"""
import argparse, csv, hashlib, os, re, signal, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_pair import RE_TOT, RE_H2D, RE_D2H, RE_M3, RE_NNZ   # noqa: E402

RE_PROF = re.compile(r"\[hash-prof\]\s+(\S+)\s+([\d.]+)\s*ms")


def parse_arm(s):
    name, _, envs = s.partition(":")
    env = {}
    for kv in envs.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k.strip()] = v.strip()
    return name.strip(), env


def run_one(binary, path, method, gpu, common, arm_env, sig_re, timeout, hb=None, tag=""):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, METHOD=method, WARMUP_N="3",
               USE_MEMPOOL="1", MP_HOST_MB="8192", D6EXP="0")
    env.update(common)
    env.update(arm_env)
    cmd = ["timeout", "-k", "30", str(int(timeout)), binary, path]      # 外层:coreutils timeout(到点 TERM,30 秒后 KILL)
    t0 = time.time()
    try:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                             text=True, errors="replace", start_new_session=True)
    except Exception as ex:
        return None, "EXC %r" % (ex,), 0.0
    if hb:
        try:
            with open(hb, "w", encoding="utf-8") as f:
                f.write("%s start=%s pid=%d timeout=%ds\n" % (tag, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), p.pid, int(timeout)))
        except OSError:
            pass
    try:
        so, se = p.communicate(timeout=timeout + 60)                    # 内层兜底:外层没起作用时,连进程组一起杀
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            so, se = p.communicate(timeout=30)
        except Exception:
            so, se = "", ""
        return None, "TIMEOUT(Python 兜底,%ds)\n%s" % (int(timeout) + 60, (so + se)[-2000:]), time.time() - t0
    except Exception as ex:
        return None, "EXC %r" % (ex,), time.time() - t0
    dur = time.time() - t0
    out = so + se
    tot = RE_TOT.findall(out); h = RE_H2D.findall(out); d = RE_D2H.findall(out)
    m3 = RE_M3.findall(out); nnz = RE_NNZ.findall(out)
    if not nnz:
        return None, "rc=%s%s (无 Result C 行)\n%s" % (p.returncode, "(=124:被 timeout 结束)" if p.returncode == 124 else "", out[-2000:]), dur
    if m3:
        co = float(m3[-1])
    elif tot and h and d:
        co = float(tot[-1]) - float(h[-1]) - float(d[-1])
    else:
        co = float(tot[-1]) if tot else None
    if co is None or co <= 0:
        return None, "rc=%s (无计时行)\n%s" % (p.returncode, out[-2000:]), dur
    prof = RE_PROF.findall(se)                # 预热轮 + 计时轮顺序排列;取最后一个 h2d 起的那一段
    last = 0
    for i, (k, _) in enumerate(prof):
        if k == "h2d":
            last = i
    ph = {}
    for k, v in prof[last:]:
        ph[k] = ph.get(k, 0.0) + float(v)
    sig = ""
    if sig_re is not None:
        lines = sig_re.findall(out)
        sig = hashlib.md5("\n".join(lines).encode("utf-8", "replace")).hexdigest()[:12] + ":%d" % len(lines)
    return {"co": co, "nnz": int(nnz[-1]), "ph": ph, "sig": sig}, None, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--list", required=True, help="阵名清单(首行表头 mat;容忍 CRLF)")
    ap.add_argument("--roots", default="data/ocean/square,data/first100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", required=True, help="name:K=V,K=V(首臂=基线)")
    ap.add_argument("--env", action="append", default=[], help="所有臂共用的 K=V")
    ap.add_argument("--method", default="adaptive")
    ap.add_argument("--rounds", default="5,3,2", help="小/中/大 三档各自的轮数")
    ap.add_argument("--tier-mb", default="10,200", help="小|中、中|大 的文件大小分界(MB)")
    ap.add_argument("--sig-regex", default="")
    ap.add_argument("--limit", type=int, default=0, help="只跑按大小排序后的前 N 个(冒烟用)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--timeout", type=int, default=900, help="单发上限(秒)")
    ap.add_argument("--budget-min", type=float, default=0, help="整段时间预算(分钟);>0 时用完就在阵与阵之间收尾")
    a = ap.parse_args()

    arms = [parse_arm(s) for s in a.arm]
    common = {}
    for kv in a.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            common[k.strip()] = v.strip()
    rounds = [int(x) for x in a.rounds.split(",")]
    t1, t2 = [float(x) for x in a.tier_mb.split(",")]
    sig_re = re.compile(a.sig_regex, re.M) if a.sig_regex else None
    roots = [x for x in a.roots.split(",") if x]

    names = []
    with open(a.list, encoding="utf-8", errors="replace") as f:
        for i, ln in enumerate(f):
            n = ln.strip().strip("﻿")
            if not n or (i == 0 and n.lower() == "mat"):
                continue
            if n not in names:
                names.append(n)
    mats, missing = [], []
    for n in names:
        p = None
        for rt in roots:
            c = os.path.join(rt, n + ".mtx")
            if os.path.isfile(c):
                p = c
                break
        if p is None:
            missing.append(n)
            continue
        mb = os.path.getsize(p) / 1e6
        mats.append((mb, n, p))
    mats.sort()
    if a.limit > 0:
        mats = mats[:a.limit]
    print("# shadow428: bin=%s gpu=%s method=%s 清单 %d 阵,找到 %d,缺 %d;轮数(小/中/大)=%s 分界=%s MB"
          % (a.bin, a.gpu, a.method, len(names), len(mats), len(missing), a.rounds, a.tier_mb), flush=True)
    for n, e in arms:
        print("#   arm %-8s %s" % (n, " ".join("%s=%s" % kv for kv in sorted(e.items())) or "(无额外 env)"), flush=True)
    if common:
        print("#   共用 env: " + " ".join("%s=%s" % kv for kv in sorted(common.items())), flush=True)
    if missing:
        print("# 缺文件: " + " ".join(missing[:40]) + (" …" if len(missing) > 40 else ""), flush=True)

    fcsv = open(a.out + ".csv", "w", newline="", encoding="utf-8")
    w = csv.writer(fcsv)
    w.writerow(["mat", "tier", "size_mb", "arm", "round", "co", "nnz", "sig", "dur_s"])
    fph = open(a.out + "_phases.csv", "w", newline="", encoding="utf-8")
    wp = csv.writer(fph)
    wp.writerow(["mat", "arm", "round", "phase", "ms"])
    fail = open(a.out + "_fail.log", "w", encoding="utf-8")
    t0 = time.time()
    nrun = nfail = 0
    stopped = ""
    for k, (mb, name, path) in enumerate(mats):
        if a.budget_min > 0 and (time.time() - t0) / 60 > a.budget_min:
            stopped = "时间预算 %.0f 分钟用完,停在第 %d 个阵之前(%s 及其后未跑)" % (a.budget_min, k + 1, name)
            print("# " + stopped, flush=True)
            break
        tier = 0 if mb < t1 else (1 if mb < t2 else 2)
        durs = []
        for r in range(rounds[tier]):
            for an, ae in arms:
                res, err, dur = run_one(a.bin, path, a.method, a.gpu, common, ae, sig_re, a.timeout,
                                        hb=a.out + ".hb", tag="%s %s r%d shot#%d" % (name, an, r, nrun + 1))
                nrun += 1
                durs.append(dur)
                if res is None:
                    nfail += 1
                    w.writerow([name, tier, "%.2f" % mb, an, r, -1, -1, "", "%.1f" % dur])
                    fail.write("== %s %s r%d (%.0fs)\n%s\n" % (name, an, r, dur, err)); fail.flush()
                else:
                    w.writerow([name, tier, "%.2f" % mb, an, r, "%.4f" % res["co"], res["nnz"], res["sig"], "%.1f" % dur])
                    for ph, ms in res["ph"].items():
                        wp.writerow([name, an, r, ph, "%.4f" % ms])
            fcsv.flush(); fph.flush()
        sd = sorted(durs)
        print("[%4d/%d] %-28s %8.1fMB 档%d 轮数%d 累计 %d 发(失败 %d) %.0f 分钟;单发秒数 中位 %.0f / 最长 %.0f"
              % (k + 1, len(mats), name, mb, tier, rounds[tier], nrun, nfail, (time.time() - t0) / 60,
                 sd[len(sd) // 2] if sd else 0, sd[-1] if sd else 0), flush=True)
    fcsv.close(); fph.close(); fail.close()
    print("# 完成:%d 阵,%d 发,失败 %d 发,用时 %.0f 分钟%s" % (len(mats), nrun, nfail, (time.time() - t0) / 60, (";" + stopped) if stopped else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
