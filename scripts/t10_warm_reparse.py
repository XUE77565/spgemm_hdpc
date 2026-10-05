#!/usr/bin/env python3
"""T10 广域电池暖轮重解析器(10-06 口径第四层之后的钦定口径)。

背景(study/32 10-06 追加节):旧解析跨轮取 max,把第 1 轮冷 JIT 的相位
(小 dense 阵 dense_direct 暖轮的 3-6×)吸进相位和,而 ocean 侧取迭代
2-10 中位免冷——系统性虚高我方小阵。本脚本按【轮】分组,只取暖轮。

口径(双声明):
  our  = TOTAL(GPU) 按轮切分;净窗 = TOTAL−h2d−d2h;暖轮 = 第 2 轮起的中位
  ocean = bench "Iteration N done. Time: X ms" 迭代 2-10 中位(原样)

用法:
  python3 scripts/t10_warm_reparse.py /tmp/t10_our_*.err   # 自动配对 /tmp/t10_oc_<m>_<r>.log
输出:逐阵比值 + 几何均值(--phase 时改吐暖轮相位和中位,口径偏乐观档)
"""
import re, sys, glob, statistics, collections, math

PH = ['count_flop', 'mh_construct', 'mh_merge', 'est_scan', 'binning', 'accumulate', 'retry',
      'cnnz_scan', 'compact+sort', 'dense_count', 'dense_direct',
      'hash_count', 'hash_direct', 'oc_rescale', 'oc_samp']


def our_rounds(err_path):
    """返回 (净窗列表, 相位和列表),按 TOTAL(GPU) 行切轮。"""
    tots, h2s, d2s = [], [], []
    rounds, cur = [], {}
    for l in open(err_path):
        for k in PH:
            mm = re.match(r'\[dbg[\d. ]*ms\] \[hash-prof\] ' + k + r'\s+([\d.]+)', l)
            if mm:
                cur[k] = float(mm.group(1))
                break
        t = re.search(r'TOTAL\(GPU\)\s+([\d.]+)', l)
        if t:
            tots.append(float(t.group(1)))
            rounds.append(cur)
            cur = {}
        h = re.search(r'\[hash-prof\] h2d\s+([\d.]+)', l)
        if h:
            h2s.append(float(h.group(1)))
        d = re.search(r'\[hash-prof\] d2h\s+([\d.]+)', l)
        if d:
            d2s.append(float(d.group(1)))
    n = min(len(tots), len(h2s), len(d2s))
    nets = [tots[i] - h2s[i] - d2s[i] for i in range(n)]
    psums = [sum(r.values()) for r in rounds if r]
    return nets, psums


def oc_med(log_path):
    its = [float(x) for x in re.findall(r'Iteration \d+ done\. Time: ([\d.]+) ms', open(log_path).read())]
    return statistics.median(its[1:]) if len(its) > 1 else None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    phase_mode = '--phase' in sys.argv
    res = collections.defaultdict(list)
    for f in sorted(glob.glob(args[0]) if args else []):
        key = f.split('t10_our_')[1]
        m = key.rsplit('_', 1)[0]
        try:
            nets, psums = our_rounds(f)
            vals = psums if phase_mode else nets
            c = oc_med('/tmp/t10_oc_' + key.replace('.err', '.log'))
            if len(vals) > 1 and c:
                res[m].append(statistics.median(vals[1:]) / c)
        except Exception:
            pass
    alls = []
    for m, rr in sorted(res.items(), key=lambda x: statistics.median(x[1]) if x[1] else 9):
        if not rr:
            continue
        med = statistics.median(rr)
        alls.append(med)
        print('%-22s %7.3fx (n=%d)' % (m, med, len(rr)))
    if alls:
        gm = math.exp(sum(map(math.log, alls)) / len(alls))
        w = sum(1 for x in alls if x < 0.98)
        l = sum(1 for x in alls if x >= 1.02)
        tag = '暖轮相位和(偏乐观档)' if phase_mode else '暖轮净窗(T2 认证兼容)'
        print('\n%s 几何均值(%d 阵): %.4f | 赢 %d 平 %d 输 %d' % (tag, len(alls), gm, w, len(alls) - w - l, l))


if __name__ == '__main__':
    main()
