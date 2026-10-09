#!/usr/bin/env python3
"""D1 测量模型 v2:同臂自洽相位摊账 + DN 自然语境覆盖(study/36 §五 修复,数据严谨版)。

v1 盲区:逐行成本=numeric 相 only,管线相位不对称未入模(pre2 2.606/c-64 14.854 定谳)。
v2 三规则(零 GPU,全从既有日志/采数提取):
  ① 同臂自洽摊账:
     dense 行 += D 臂 dense_count(ms)×K ÷ D 臂 dense 测度行数
       (D 臂强制全人口走 diter → count 相服务的人群=测度人群,分母分子同源)
     hash  行 += H 臂(compact+sort+retry)(ms)×K ÷ H 臂 hash_spa 测度行数
       (H 臂 allhash → compact/retry 服务 hash 全人口;分母用 hash_spa 测度行=轻微
        过摊,方向保守[抬高 hash 价],注记在案)
     k7 免税(count 即精确计数,k7_count 已在 K 臂计时内)
  ② DN 自然语境覆盖:DN 臂(默认路由)测到的 dense 行,cyc_dense 以 DN 为准
     (强制臂语境膨胀 10-20×于大阵的教训,DN 才是生产行为)
  ③ ctx 全行口径:不动本脚本 —— fit_d1_context.matrix_ctx 改从 R 臂 DUMP_ROUTING
     (全行 est/flop)计算,修 train/serve skew(另行补丁)。
K=2.4e6 cyc/ms(H100 时钟域近似,与 truth 采集同源)。
输出:compare/d1_oracle/truth_v2/<mat>.truth.tsv(列同 v1)。
"""
import argparse, os, re, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "compare", "d1_oracle")
OUT = os.path.join(SRC, "truth_v2")
LOGS = os.path.join(SRC, "logs")
CYC_PER_MS = 2.4e6
PH = re.compile(r"\[hash-prof\]\s+(\S+)\s+([\d.]+)\s*ms")

OLD_BIG = ["bloweybq", "pre2", "c-43", "Si10H16", "in-2004", "nd24k", "crankseg_2", "F2", "c-64",
           "bcsstk32", "bcsstk23", "bcsstm23", "bcspwr08", "bcspwr05", "bcsstm01", "1138_bus", "ash219",
           "pf2177", "HTC_336_9129", "333SP", "opt1", "af_shell4", "bp_0", "arc130", "bcsstk30", "bcsstk31"]
NEW_BIG = ["af_2_k101", "crankseg_1", "rajat20", "wb-edu", "3Dspectralwave2", "Ga41As41H72",
           "TSOPF_FS_b39_c7", "mult_dcop_03", "Ge99H100", "c-58"]


def last_phases(p):
    if not os.path.exists(p):
        return {}
    last = {}
    for k, v in PH.findall(open(p, errors="ignore").read()):
        last[k] = float(v)
    return last


def med3(paths, ncols):
    acc = {}
    for p in paths:
        if not os.path.exists(p):
            return {}
        for ln in open(p):
            f = ln.split()
            if len(f) == ncols:
                acc.setdefault(int(f[0]), []).append(float(f[1]))
    return {r: statistics.median(v) for r, v in acc.items() if len(v) == len(paths)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat", nargs="*", default=None)
    a = ap.parse_args()
    mats = a.mat or (OLD_BIG + NEW_BIG)
    os.makedirs(OUT, exist_ok=True)
    print(f"{'mat':<18}{'dc_ms(D)':>9}{'dN':>7}{'dense+':>9}{'cp_ms(H)':>9}{'hN':>8}{'hash+':>9}{'DN覆':>6}")
    for m in mats:
        tp = os.path.join(SRC, "truth", f"{m}.truth.tsv")
        if not os.path.exists(tp):
            continue
        rows = [ln.split() for ln in open(tp).readlines()[1:]]
        # ① 同臂摊账
        dph = last_phases(os.path.join(LOGS, f"{m}.D1.log"))
        hph = last_phases(os.path.join(LOGS, f"{m}.H1.log"))
        dc = dph.get("dense_count", 0.0)
        cp = hph.get("compact+sort", 0.0) + hph.get("retry", 0.0)
        ndense = max(1, sum(1 for r in rows if float(r[9]) > 0))
        nhash = max(1, sum(1 for r in rows if float(r[8]) > 0))
        # dense 税按工作量(flop)摊:pre2 三点验证 count=0.47 cyc/flop(预测 52.3 vs 实测 54.8ms,
        # 4.6% 误差)——按行摊差 900×(count 相位 ∝ Σflop 非行数)
        flop_sum = max(1, sum(int(r[2]) for r in rows))
        k_count = dc * CYC_PER_MS / flop_sum
        add_h = cp * CYC_PER_MS / nhash     # compact/retry 按行(输出规模近似;量级小)
        # ② DN 覆盖(默认路由真语境的 dense 行价)
        dn = med3([os.path.join(SRC, f"{m}.DN{i}.dd.tsv") for i in (1, 2, 3)], 3)
        dn_cov = 0
        with open(os.path.join(OUT, f"{m}.truth.tsv"), "w") as fo:
            fo.write("r a_len flop est span d dup bkid_def cyc_hash cyc_dense cyc_k7 cyc_mrg3 best dclass\n")
            for r in rows:
                ri = int(r[0])
                ch, cd = float(r[8]), float(r[9])
                if ch > 0:
                    ch += add_h
                if cd > 0:
                    if ri in dn:
                        cd = dn[ri]          # DN 自然语境为准(免强制臂膨胀)
                        dn_cov += 1
                    cd += k_count * int(r[2])
                fo.write(" ".join([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
                                   str(int(ch)), str(int(cd)), r[10], r[11], r[12], r[13]]) + "\n")
        print(f"{m:<18}{dc:>9.1f}{ndense:>7}{k_count:>9.3f}{cp:>9.1f}{nhash:>8}{add_h:>9.0f}{dn_cov:>6}")


if __name__ == "__main__":
    main()
