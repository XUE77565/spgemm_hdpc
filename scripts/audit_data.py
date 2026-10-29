#!/usr/bin/env python3
"""标定阵选择:从 428 阵中选 ~40 阵,覆盖 bin9/bin45/bin0/大阵/微阵各人口带"""
import csv, os, subprocess, re

def get_bins(mat, repo):
    """快速跑一次取 bins(只读 bins 行,不用计时)"""
    p = None
    for d in ("data/ocean/square", "data/first100"):
        c = os.path.join(repo, d, mat + ".mtx")
        if os.path.exists(c):
            p = c
            break
    if not p:
        return None
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", METHOD="adaptive",
               WARMUP_N="1", USE_MEMPOOL="1", MP_HOST_MB="8192", D6EXP="0")
    try:
        r = subprocess.run(["./spgemm_test", p], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=300)
        bins = re.findall(r"bins: ([0-9: ]+)", r.stdout + r.stderr)
        return bins[-1] if bins else ""
    except:
        return ""

def parse_bins(bins_str):
    d = {}
    for tok in bins_str.split():
        if ":" in tok:
            k, v = tok.split(":")
            d[int(k)] = int(v)
    return d

def select_calibration_set(all_mats, repo, target=40):
    """选标定阵:确保覆盖各人口带"""
    # 先取已有的 bins 数据(从 loss_audit 或直接跑)
    cache_path = os.path.join(repo, "results", "bins_cache.json")
    bins_cache = {}
    if os.path.exists(cache_path):
        import json
        bins_cache = json.load(open(cache_path))

    # 需要新跑的
    mats_info = []
    for m in all_mats:
        if m in bins_cache:
            info = bins_cache[m]
        else:
            bs = get_bins(m, repo)
            if not bs:
                continue
            info = parse_bins(bs)
            bins_cache[m] = info
        bin9 = info.get("9", 0)
        bin45 = info.get("4", 0) + info.get("5", 0)
        bin0 = info.get("0", 0)
        total = sum(info.values())
        mats_info.append((m, bin9, bin45, bin0, total))

    # 保存缓存
    import json
    json.dump(bins_cache, open(cache_path, "w"))

    # 分层选择
    selected = []
    # bin9 有人口的阵(至少 1 行): 前 10(按 bin9 人口降序)
    bin9_mats = sorted([x for x in mats_info if x[1] > 0], key=lambda x: -x[1])
    selected.extend(bin9_mats[:10])
    # bin4/5 有人口的阵: 前 10(按 bin45 人口降序,排除已选)
    already = set(x[0] for x in selected)
    bin45_mats = sorted([x for x in mats_info if x[2] > 100 and x[0] not in already], key=lambda x: -x[2])
    selected.extend(bin45_mats[:10])
    # 微阵(bin0 主导): 5 个
    already = set(x[0] for x in selected)
    bin0_mats = sorted([x for x in mats_info if x[3] > x[4]*0.5 and x[0] not in already], key=lambda x: -x[3])
    selected.extend(bin0_mats[:5])
    # 大阵(total > 1M): 5 个
    already = set(x[0] for x in selected)
    big_mats = sorted([x for x in mats_info if x[4] > 1000000 and x[0] not in already], key=lambda x: -x[4])
    selected.extend(big_mats[:5])
    # 随机补充到 target
    already = set(x[0] for x in selected)
    import random
    random.seed(42)
    rest = [x for x in mats_info if x[0] not in already]
    random.shuffle(rest)
    selected.extend(rest[:max(0, target - len(selected))])

    return [x[0] for x in selected[:target]]
