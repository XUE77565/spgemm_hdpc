#!/usr/bin/env python3
"""L0 系统级采数(study/37 第一步):48 阵 × 5 配置 × ×1 发,TOTAL(含 h2d/d2h)。

配置:
  A: 全 hash(DITER_MIN_FLOP=999999999,K7=0,D6EXP=0)
  B: 全 hash + k7 静态谓词(DITER_MIN_FLOP=999999999,K7=1,D6EXP=0)
  C: 默认路由(D6EXP=0,K7=0)= 当前生产
  D: 默认 + k7(K7=1,D6EXP=0)
  E: 全 hash + S99(= A + SPEC99=1;S99 已默认开,等效 A 但确认)
输出:compare/l0_calibration/l0_data.csv(mat,config,total_ms,net_ms)
用法:CUDA_VISIBLE_DEVICES=? .venv/bin/python scripts/l0_collect.py [--mats ...]
"""
import argparse, csv, os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_test")
DIR = os.path.join(REPO, "data", "ocean", "square")
OUT = os.path.join(REPO, "compare", "l0_calibration")
TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
H2D = re.compile(r"\[hash-prof\] h2d\s+([\d.]+)")
D2H = re.compile(r"\[hash-prof\] d2h\s+([\d.]+)")

MATS = ["bloweybq", "pre2", "c-43", "Si10H16", "in-2004", "nd24k", "crankseg_2", "F2", "c-64",
        "bcsstk32", "bcsstk23", "bcsstm23", "bcspwr08", "bcspwr05", "bcsstm01", "1138_bus", "ash219",
        "pf2177", "HTC_336_9129", "333SP", "opt1", "af_shell4", "bp_0", "arc130", "bcsstk30", "bcsstk31",
        "af_2_k101", "crankseg_1", "rajat20", "3Dspectralwave2", "Ga41As41H72",
        "TSOPF_FS_b39_c7", "mult_dcop_03", "Ge99H100", "c-58",
        "c-71", "msc23052", "exdata_1", "rajat25", "msc10848", "hugetrace-00020",
        "web-BerkStan", "cage13", "M6", "para-6", "offshore", "std1_Jac3"]

CONFIGS = {
    "A_allhash":  {"DITER_MIN_FLOP": "999999999", "K7": "0", "D6EXP": "0"},
    "B_allhash_k7": {"DITER_MIN_FLOP": "999999999", "K7": "1", "D6EXP": "0"},
    "C_default":  {"D6EXP": "0", "K7": "0"},
    "D_default_k7": {"D6EXP": "0", "K7": "1"},
}


def run(m, cfg, timeout=1200):
    env = dict(os.environ, METHOD="hash", WARMUP_N="3", USE_MEMPOOL="1", MP_HOST_MB="8192")
    env.update(cfg)
    try:
        r = subprocess.run([BIN, os.path.join(DIR, m + ".mtx")],
                           capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, None
    t = TOT.findall(r.stderr); h = H2D.findall(r.stderr); d = D2H.findall(r.stderr)
    if not t:
        return None, None
    total = float(t[-1])
    net = total - (float(h[-1]) if h else 0) - (float(d[-1]) if d else 0)
    return total, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mats", nargs="*", default=MATS)
    ap.add_argument("--configs", default=",".join(CONFIGS.keys()))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, "l0_data.csv")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for row in csv.DictReader(f):
                done.add((row["mat"], row["config"]))
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["mat", "config", "total_ms", "net_ms"])
        for m in a.mats:
            for cn in a.configs.split(","):
                if (m, cn) in done:
                    continue
                cfg = CONFIGS[cn]
                total, net = run(m, cfg)
                if total is None:
                    print(f"  {m:<18} {cn:<14} DNF", flush=True)
                    w.writerow([m, cn, "DNF", "DNF"]); f.flush()
                else:
                    print(f"  {m:<18} {cn:<14} total={total:.1f} net={net:.1f}", flush=True)
                    w.writerow([m, cn, f"{total:.1f}", f"{net:.1f}"]); f.flush()
    print("L0 采数完成 →", out_path)


if __name__ == "__main__":
    main()
