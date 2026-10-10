#!/usr/bin/env python3
"""L0 v2 五配置采数:A/A'/B/C/D × 337 阵(或 51 阵子集先行验证)。

配置:
  A  = 全 hash(DITER_MIN_FLOP=999999999,K7=0,FASTPATH=0)—— 基线全 hash
  A' = 全 hash + FASTPATH(DITER_MIN_FLOP=999999999,FASTPATH=1)—— 管线精简
  B  = 全 hash + k7(DITER_MIN_FLOP=999999999,K7=1)—— hash+k7
  C  = 默认路由(D6EXP=0,K7=0)—— 当前生产
  D  = 默认 + k7(K7=1)—— 默认+k7
输出:compare/l0_v2/l0_v2_data.csv
用法:CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/l0_v2_configs.py [--mats ...]
"""
import os, re, subprocess, sys, csv, argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_test")
DIR = os.path.join(REPO, "data", "ocean", "square")
OUT = os.path.join(REPO, "compare", "l0_v2")
TOT = re.compile(r"TOTAL\(GPU\)\s+([\d.]+)")
H2D = re.compile(r"\[hash-prof\] h2d\s+([\d.]+)")
D2H = re.compile(r"\[hash-prof\] d2h\s+([\d.]+)")

CONFIGS = {
    "A":  {"DITER_MIN_FLOP": "999999999", "K7": "0", "FASTPATH": "0"},
    "Ap": {"DITER_MIN_FLOP": "999999999", "K7": "0", "FASTPATH": "1"},
    "B":  {"DITER_MIN_FLOP": "999999999", "K7": "1", "FASTPATH": "0"},
    "C":  {"FASTPATH": "0", "K7": "0"},
    "D":  {"FASTPATH": "0", "K7": "1"},
}

DEFAULT_MATS = ["bloweybq", "pre2", "c-43", "Si10H16", "in-2004", "nd24k",
                "crankseg_2", "F2", "c-64", "Ge99H100", "SiO2", "msc23052",
                "333SP", "exdata_1", "opt1", "af_shell4"]


def run(m, cfg, timeout=1200):
    env = dict(os.environ, METHOD="hash", WARMUP_N="3",
               USE_MEMPOOL="1", MP_HOST_MB="8192", D6EXP="0")
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mats", nargs="*", default=DEFAULT_MATS)
    ap.add_argument("--configs", default=",".join(CONFIGS.keys()))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, "l0_v2_data.csv")
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
                if net is None:
                    print(f"  {m:<16} {cn:<4} DNF", flush=True)
                    w.writerow([m, cn, "DNF", "DNF"]); f.flush()
                else:
                    print(f"  {m:<16} {cn:<4} net={net:.1f}", flush=True)
                    w.writerow([m, cn, f"{total:.1f}", f"{net:.1f}"]); f.flush()
    print(f"L0 v2 采数完成 → {out_path}")
