#!/usr/bin/env python3
"""S4 决策表生成器(study/60):从标定数据生成 routing_table.h
读 calibration_data.csv + confirm16(训练子集的 baseline 数据)→
按签名分格 → 每格最优变体 → C 头文件(查表数组)。
用法:python3 scripts/gen_routing_table.py
"""
import csv, os, re, sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL = os.path.join(REPO, "results/calibration_data.csv")
CONF = os.path.join(REPO, "compare/batch437/confirm16_428.csv")
TRAIN = os.path.join(REPO, "results/train_356.txt")
OUT_H = os.path.join(REPO, "src/routing_table.h")

def load_data():
    """加载标定数据 + confirm16 baseline(训练子集)"""
    # 标定变体数据
    cal = defaultdict(dict)  # mat → variant → co_ms
    for r in csv.DictReader(open(CAL)):
        if r['mat'] == 'mat': continue
        try:
            co = float(r['co_ms'])
            if co > 0:
                cal[r['mat']][r['variant']] = co
        except:
            pass

    # confirm16 baseline(训练子集)
    train_mats = set(l.strip() for l in open(TRAIN) if l.strip())
    for r in csv.DictReader(open(CONF)):
        m = r['mat']
        if m in train_mats and m in cal:
            try:
                co = float(r['adpt13_ms'])
                if co > 0:
                    cal[m]['baseline'] = co
            except:
                pass

    # 提取 bins 信息(从标定数据的 bins_summary 列)
    bins_info = {}
    for r in csv.DictReader(open(CAL)):
        if r['mat'] == 'mat': continue
        if r['mat'] not in bins_info and r.get('bins_summary'):
            bins_info[r['mat']] = r['bins_summary']

    return cal, bins_info

def parse_bins(bins_str):
    d = {}
    for tok in bins_str.split():
        if ':' in tok:
            parts = tok.split(':')
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                try:
                    d[int(parts[0])] = int(parts[1])
                except ValueError:
                    pass
    return d

def compute_signature(mat, bins_str):
    """计算矩阵签名(用于分格)"""
    bins = parse_bins(bins_str)
    total = sum(bins.values())
    if total == 0:
        return None
    bin9_frac = bins.get(9, 0) / total
    bin45_frac = (bins.get(4, 0) + bins.get(5, 0)) / total
    bin0_frac = bins.get(0, 0) / total
    return (bin9_frac, bin45_frac, bin0_frac, total)

def find_optimal_variant(variants_co):
    """给定 mat 的各变体时间,返回最优配置"""
    # 配置映射:baseline = H69 on + H45/H55 on
    # no_h69 = H69 off + H45/H55 on
    # no_h4555 = H69 on + H45/H55 off
    # no_all = all off
    configs = {
        'baseline': {'H69': 1, 'H45': 1, 'H55': 1},
        'no_h69':   {'H69': 0, 'H45': 1, 'H55': 1},
        'no_h4555': {'H69': 1, 'H45': 0, 'H55': 0},
        'no_all':   {'H69': 0, 'H45': 0, 'H55': 0},
    }
    best_v = min(variants_co, key=variants_co.get)
    return configs.get(best_v, configs['baseline'])

def main():
    cal, bins_info = load_data()
    print(f"标定数据: {len(cal)} 阵", flush=True)

    # 有完整变体数据的阵
    complete = {m: v for m, v in cal.items() if len(v) >= 2}
    print(f"≥2 变体: {len(complete)} 阵", flush=True)

    # 按签名分格
    # 格定义:bin9_frac(0/低/高) × bin45_frac(0/低/高) × 规模(小/中/大)
    CELLS = {}
    for m, variants in complete.items():
        bs = bins_info.get(m, "")
        sig = compute_signature(m, bs)
        if sig is None:
            continue
        bin9f, bin45f, bin0f, total = sig
        # 分格
        b9_cell = 'none' if bin9f < 0.01 else ('low' if bin9f < 0.3 else 'high')
        b45_cell = 'none' if bin45f < 0.01 else ('low' if bin45f < 0.3 else 'high')
        size_cell = 'small' if total < 10000 else ('medium' if total < 1000000 else 'large')
        cell_key = f"{b9_cell}_{b45_cell}_{size_cell}"

        best = find_optimal_variant(variants)
        if cell_key not in CELLS:
            CELLS[cell_key] = defaultdict(int)
        cfg_str = f"H69={best['H69']},H45={best['H45']}"
        CELLS[cell_key][cfg_str] += 1

    # 每格取多数投票
    table = {}
    for cell, votes in sorted(CELLS.items()):
        winner = max(votes, key=votes.get)
        total_votes = sum(votes.values())
        table[cell] = (winner, total_votes, dict(votes))
        print(f"  {cell:<25} → {winner} (n={total_votes}, votes={dict(votes)})", flush=True)

    # 生成 C 头文件
    with open(OUT_H, 'w') as f:
        f.write("// S4 决策表(study/60):从标定数据自动生成\n")
        f.write("// 签名格:bin9_frac(0/low/high) × bin45_frac(0/low/high) × 规模(small/medium/large)\n")
        f.write("// 生成时间: 自动\n\n")
        f.write("#ifndef ROUTING_TABLE_H\n#define ROUTING_TABLE_H\n\n")
        f.write("// 每格的最优配置(H69/H45 开关)\n")
        f.write("typedef struct { int h69; int h45; int h55; } route_cfg_t;\n\n")
        f.write("// 查表:输入 bin9_frac/bin45_frac/total_rows 带的索引\n")
        f.write("// 返回该格的最优配置\n")
        f.write("static inline route_cfg_t routing_table_lookup(\n")
        f.write("    int bin9_level,  // 0=none 1=low 2=high\n")
        f.write("    int bin45_level, // 0=none 1=low 2=high\n")
        f.write("    int size_level)  // 0=small 1=medium 2=large\n")
        f.write("{\n")
        f.write("    // 默认:全开(基线)\n")
        f.write("    route_cfg_t cfg = {1, 1, 1};\n\n")

        for cell, (winner, n, votes) in table.items():
            # 解析 cell_key → 索引
            parts = cell.split('_')
            b9_idx = {'none': 0, 'low': 1, 'high': 2}.get(parts[0], 0)
            b45_idx = {'none': 0, 'low': 1, 'high': 2}.get(parts[1], 0)
            size_idx = {'small': 0, 'medium': 1, 'large': 2}.get(parts[2], 1)
            # 解析 winner
            h69 = 1 if 'H69=1' in winner else 0
            h45 = 1 if 'H45=1' in winner else 0
            f.write(f"    if (bin9_level=={b9_idx} && bin45_level=={b45_idx} && size_level=={size_idx})\n")
            f.write(f"        cfg = (route_cfg_t){{{h69}, {h45}, {h45}}};  // {cell} n={n}\n")

        f.write("\n    return cfg;\n}\n\n")
        f.write("#endif // ROUTING_TABLE_H\n")

    print(f"\n决策表已生成 → {OUT_H}", flush=True)
    print(f"格数: {len(table)}", flush=True)

if __name__ == '__main__':
    main()
