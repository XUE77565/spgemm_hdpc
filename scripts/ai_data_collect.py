#!/usr/bin/env python3
"""
AI Router 全量消融数据收集(study/60 终极版)
覆盖 15+ 路由点 × ~100 阵 → ai_calibration_full.csv
每个路由点的消融通过 env 变量控制(不需要改代码)

路由点清单:
  1. method: hash vs dense vs merge3 (从已有 dispatcher 日志提取)
  2. bin45_kernel: HASH_H45/H55 on/off
  3. bin9_kernel: HASH_H69 on/off
  4. compact_sorted: HH_SORTED on/off
  5. est_expand: HASH_EXPAND=1.15 vs 1.4
  6. batched_variant: BATCHMR=0 vs 2
  7. block_ladder: HASH_BLK_LADDER legacy vs default
  8. csort_strategy: CSORT_CNT=0 vs 1
  9. flat: COMPACT_FLAT=0 vs 1
  10. lht23: HASH_LHT23=0 vs 1
"""
import csv, os, re, subprocess, sys, json, math
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "ai_calibration_full.csv")
COLS = ["mat", "routing_point", "option", "co_ms", "nnz", "n",
        "bins_summary", "route", "config_env"]

# ===== 配置矩阵: 每个路由点的选项及对应 env =====
ROUTING_POINTS = [
    {
        "name": "bin45_kernel",
        "options": [
            {"label": "hash_heavy", "env": {"HASH_H45": "1", "HASH_H55": "1"}},
            {"label": "lht", "env": {"HASH_H45": "0", "HASH_H55": "0"}},
        ],
        "relevant": lambda bins: bins.get(4,0)+bins.get(5,0) > 100,
    },
    {
        "name": "bin9_kernel",
        "options": [
            {"label": "heavy_16253", "env": {"HASH_H69": "1"}},
            {"label": "lht14", "env": {"HASH_H69": "0"}},
        ],
        "relevant": lambda bins: bins.get(9,0) > 10,
    },
    {
        "name": "compact_sorted",
        "options": [
            {"label": "sorted_copy", "env": {"HH_SORTED": "1"}},
            {"label": "unsorted_csort", "env": {"HH_SORTED": "0", "CSORT_CNT": "1"}},
        ],
        "relevant": lambda bins: bins.get(4,0)+bins.get(5,0)+bins.get(7,0)+bins.get(8,0) > 100,
    },
    {
        "name": "est_expand",
        "options": [
            {"label": "tight_115", "env": {"HASH_EXPAND": "1.15"}},
            {"label": "loose_14", "env": {"HASH_EXPAND": "1.4"}},
        ],
        "relevant": lambda bins: sum(bins.values()) > 1000,
    },
    {
        "name": "block_ladder",
        "options": [
            {"label": "default", "env": {}},
            {"label": "legacy", "env": {"HASH_BLK_LADDER": "legacy"}},
        ],
        "relevant": lambda bins: bins.get(6,0)+bins.get(7,0) > 50,
    },
    {
        "name": "csort_strategy",
        "options": [
            {"label": "counting", "env": {"CSORT_CNT": "1"}},
            {"label": "radix", "env": {"CSORT_CNT": "0"}},
        ],
        "relevant": lambda bins: bins.get(6,0)+bins.get(7,0) > 50,
    },
    {
        "name": "flat_compact",
        "options": [
            {"label": "flat_on", "env": {"COMPACT_FLAT": "1"}},
            {"label": "row_copy", "env": {"COMPACT_FLAT": "0"}},
        ],
        "relevant": lambda bins: bins.get(2,0)+bins.get(3,0)+bins.get(4,0)+bins.get(5,0) > 100,
    },
    {
        "name": "lht23",
        "options": [
            {"label": "on", "env": {"HASH_LHT23": "1"}},
            {"label": "off", "env": {"HASH_LHT23": "0"}},
        ],
        "relevant": lambda bins: bins.get(2,0)+bins.get(3,0) > 100,
    },
]

def run_one(mat_path, env_extra, gpu="1", timeout=600):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, METHOD="adaptive",
               WARMUP_N="3", USE_MEMPOOL="1", MP_HOST_MB="8192", D6EXP="0")
    env.update(env_extra)
    try:
        r = subprocess.run(["./spgemm_test", mat_path], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
        tot = re.findall(r"TOTAL\(GPU\)\s+([\d.]+)", out)
        h = re.findall(r"\[hash-prof\] h2d\s+([\d.]+)", out)
        d = re.findall(r"\[hash-prof\] d2h\s+([\d.]+)", out)
        m3h = re.findall(r"\[mrg3-prof\] h2d\s+([\d.]+)", out)
        m3d = re.findall(r"\[mrg3-prof\] d2h\s+([\d.]+)", out)
        nnz = re.findall(r"Result C:.*nnz = (\d+)", out)
        bins = re.findall(r"bins: ([0-9: ]+)", out)
        route = re.findall(r"\[adapt\][^\n]*→ (\w+)", out)

        if tot and h and d:
            co = float(tot[-1]) - float(h[-1]) - float(d[-1])
        elif tot and m3h and m3d:
            co = float(tot[-1]) - float(m3h[-1]) - float(m3d[-1])
        else:
            co = -1

        bins_dict = {}
        bins_str = bins[-1] if bins else ""
        for tok in bins_str.split():
            if ':' in tok:
                p = tok.split(':')
                try: bins_dict[int(p[0])] = int(p[1])
                except: pass

        return {
            'co_ms': co,
            'nnz': int(nnz[-1]) if nnz else -1,
            'bins': bins_str[:80],
            'bins_dict': bins_dict,
            'route': route[-1] if route else '?',
        }
    except Exception as e:
        return {'co_ms': -1, 'nnz': -1, 'bins': '', 'bins_dict': {}, 'route': f'ERR:{type(e).__name__}'}

def select_matrices(all_mats, target=80):
    """选择多样化的矩阵(覆盖不同 bin 人口)"""
    print("选阵(需要先跑一遍获取 bins)...", flush=True)

    # 先用已有数据筛选
    bins_cache = {}
    cal_file = os.path.join(REPO, "results/calibration_data.csv")
    if os.path.exists(cal_file):
        for r in csv.DictReader(open(cal_file)):
            if r['mat'] != 'mat' and r.get('bins_summary'):
                if r['mat'] not in bins_cache:
                    bins_cache[r['mat']] = r['bins_summary']

    if len(bins_cache) < 20:
        print("bins 缓存不足,跳过筛选用全量", flush=True)
        return all_mats[:target]

    # 解析 bins 并按多样性选择
    candidates = []
    for mat, bs in bins_cache.items():
        bins = {}
        for tok in bs.split():
            if ':' in tok:
                p = tok.split(':')
                try: bins[int(p[0])] = int(p[1])
                except: pass
        candidates.append((mat, bins))

    # 分层:有 bin9 的、有 bin4/5 的、有 bin0 的、其他的
    has_bin9 = [c for c in candidates if c[1].get(9,0) > 10]
    has_bin45 = [c for c in candidates if c[1].get(4,0)+c[1].get(5,0) > 100]
    has_bin0 = [c for c in candidates if c[1].get(0,0) > 1000]
    others = [c for c in candidates if c not in has_bin9 and c not in has_bin45 and c not in has_bin0]

    selected = []
    # 每类选一些(确保覆盖)
    for group, quota in [(has_bin9, 15), (has_bin45, 25), (has_bin0, 15), (others, 25)]:
        group.sort(key=lambda c: -sum(c[1].values()))
        selected.extend([c[0] for c in group[:quota]])

    return selected[:target]

def main():
    gpu = sys.argv[1] if len(sys.argv) > 1 else "1"

    # 获取全部阵名
    all_mats = [r["mat"] for r in csv.DictReader(
        open(os.path.join(REPO, "compare/batch437/confirm16_428.csv"))
    ) if r["mat"] != "mat"]

    # 选阵
    cal_mats = select_matrices(all_mats)
    print(f"标定阵: {len(cal_mats)} 个 × {len(ROUTING_POINTS)} 路由点 × 2 选项 = "
          f"{len(cal_mats)*len(ROUTING_POINTS)*2} 配置", flush=True)

    # 断点续跑
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            if r.get('mat') and r['mat'] != 'mat':
                done.add((r['mat'], r['routing_point'], r['option']))
    else:
        with open(OUT, "w", newline="") as f:
            csv.writer(f).writerow(COLS)

    fout = open(OUT, "a", newline="")
    w = csv.writer(fout)

    total_configs = 0; skip = 0
    for mat in cal_mats:
        mat_path = None
        for d in ("data/ocean/square", "data/first100"):
            p = os.path.join(REPO, d, mat + ".mtx")
            if os.path.exists(p):
                mat_path = p; break
        if not mat_path: continue

        # 先跑一次默认获取 bins
        default_result = run_one(mat_path, {}, gpu)
        bins_dict = default_result['bins_dict']

        for rp in ROUTING_POINTS:
            # 只在相关阵上消融(省时间)
            if not rp["relevant"](bins_dict):
                skip += len(rp["options"])
                continue

            for opt in rp["options"]:
                key = (mat, rp["name"], opt["label"])
                if key in done: continue

                result = run_one(mat_path, opt["env"], gpu)
                if result['co_ms'] > 0:
                    env_str = json.dumps(opt["env"])
                    w.writerow([mat, rp["name"], opt["label"], f"{result['co_ms']:.3f}",
                               result['nnz'], "", result['bins'][:60], result['route'], env_str])
                    fout.flush()
                    total_configs += 1

        print(f"  [{cal_mats.index(mat)+1}/{len(cal_mats)}] {mat} 完成", flush=True)

    print(f"\nAI 消融数据收集完成: {total_configs} 新配置 + {skip} 跳过(不相关)", flush=True)
    print("AI_DATA_DONE", flush=True)

if __name__ == "__main__":
    main()
