#!/usr/bin/env python3
# fig_paper_structure.py —— 论文结构流程图(Ocean ICS'26 图风对标)
#
# 图 1 fig/paper_structure_overview.{png,pdf}
#   系统总览管线:CSR → host 特征扫描 → DISPATCH_MODEL 分流 →
#   hash 六相位管线(H2D∥count 重叠 → 估计 → binning → bin 内核阵列 → compact → D2H)
#   与 merge3 支路;overflow retry / 整阵回退环。
# 图 2 fig/paper_structure_routing.{png,pdf}
#   行级路由判据链:est/span/flop/kc 过字节模型门链 → bin 梯(内核阵列);
#   每门标注结构化判据(零裸数字哲学:阈值全部挂推导出处)。
#
# 用法:.venv/bin/python scripts/fig_paper_structure.py
# 风格靶标:Ocean Fig2 —— 圆角矩形、sans-serif、灰(host/搬运)/蓝(symbolic)/橙(numeric)、
#   细连接线、A→C 端点;一张图一个论点,箭头带标签。

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── 风格层(对标 Ocean 印刷素雅)────────────────────────────────────────────
C_GRAY_F, C_GRAY_E = "#EFEFEF", "#8C8C8C"   # host / 数据搬运相位
C_BLUE_F, C_BLUE_E = "#C9DCEF", "#5F87B5"   # symbolic(估计/计数)+ hash 族内核
C_ORNG_F, C_ORNG_E = "#FBE2C8", "#D08A4F"   # numeric dense 族内核
C_GREN_F, C_GREN_E = "#D9EAD3", "#7AA36B"   # merge3 支路(独立算法族)
C_TXT, C_SUB, C_ARROW = "#1A1A1A", "#555555", "#3A3A3A"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,   # TrueType 嵌入(期刊要求)
})

DPI = 200
_UPP = 0.0   # units-per-pt(每张画布在 new_canvas 里设定;rbox 行排版用)


def new_canvas(w_in, h_in):
    """100×100 数据坐标画布;记录 pt→unit 换算率。"""
    global _UPP
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _UPP = 100.0 / h_in / 72.0
    return fig, ax


def rbox(ax, x, y, w, h, lines, fc=C_GRAY_F, ec=C_GRAY_E, lw=1.0, fs=8.0,
         weight="normal", tc=C_TXT, rounding=1.2, zorder=3):
    """圆角框;lines = [(text, fontsize, color, weight), ...] 竖排居中(行距 1.45×字号)。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={rounding}",
                 fc=fc, ec=ec, lw=lw, zorder=zorder))
    if isinstance(lines, str):
        lines = [(lines, fs, tc, weight)]
    lh = [f * 1.45 * _UPP for _, f, _, _ in lines]
    cy = y + h / 2 + sum(lh) / 2
    for (text, fsz, col, wgt), dy in zip(lines, lh):
        cy -= dy / 2
        ax.text(x + w / 2, cy, text, ha="center", va="center",
                fontsize=fsz, color=col, fontweight=wgt, zorder=zorder + 1)
        cy -= dy / 2


def arr(ax, x0, y0, x1, y1, lw=0.9, color=C_ARROW, ls="-", head="-|>",
        conn="arc3,rad=0", zorder=2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=head, color=color,
                 lw=lw, linestyle=ls, mutation_scale=9, connectionstyle=conn,
                 shrinkA=0, shrinkB=0, zorder=zorder))


def polyarrow(ax, pts, lw=0.9, color=C_ARROW, ls="-", zorder=2):
    """折线箭头:pts = [(x,y),...],仅末段带箭头。"""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls,
                solid_capstyle="round", zorder=zorder)
    arr(ax, *pts[-2], *pts[-1], lw=lw, color=color, ls=ls, zorder=zorder)


def label(ax, x, y, text, fs=6.3, color=C_SUB, ha="left", weight="normal",
          zorder=6):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va="center",
            fontweight=weight, zorder=zorder)


def _pdf_meta(path, title):
    """期刊工具链基础元数据(Title/Creator);pypdf 缺席时静默跳过。"""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print(f"  (pypdf absent, metadata skipped: {path})")
        return
    r = PdfReader(path)
    w = PdfWriter()
    w.append(r)
    w.add_metadata({"/Title": title, "/Creator": "scripts/fig_paper_structure.py"})
    with open(path, "wb") as fh:
        w.write(fh)


# ════════════════════════════════════════════════════════════════════════════
# 图 1:系统总览管线
# ════════════════════════════════════════════════════════════════════════════
def fig_overview():
    fig, ax = new_canvas(7.16, 7.9)

    # ── 端点与 host 段 ──
    rbox(ax, 36, 95.5, 28, 4.0, [("A  (CSR, host)", 8.5, C_TXT, "bold")])
    rbox(ax, 24, 88.0, 52, 6.0, [
        ("host row-scan  O(nnz)", 8.0, C_TXT, "bold"),
        ("features:  log flop · log n · log max-row · log skew", 6.6, C_SUB, "normal"),
    ])
    rbox(ax, 24, 79.5, 52, 6.5, [
        ("DISPATCH_MODEL  (learned logistic)", 8.0, C_TXT, "bold"),
        ("score = w · features + b    (synthetic-calibrated, disjoint training)", 6.4, C_SUB, "normal"),
    ])
    arr(ax, 50, 95.5, 50, 94.0)
    arr(ax, 50, 88.0, 50, 86.0)

    # ── 分流 ──
    arr(ax, 40, 79.5, 40, 77.0)
    arr(ax, 76, 79.5, 76, 76.8)
    label(ax, 41.0, 78.5, "score < 0", 6.5)
    label(ax, 77.0, 78.5, "score ≥ 0", 6.5)

    # ── hash 管线容器(左主支)──
    ax.add_patch(FancyBboxPatch((2, 9.0), 62, 68.0,
                 boxstyle="round,pad=0,rounding_size=1.6",
                 fc="#FBFCFD", ec="#B9C4CE", lw=1.1, zorder=1))
    ax.text(4.0, 74.8, "hash pipeline", fontsize=8.5, color="#40566B",
            fontweight="bold", va="center", zorder=2)

    # 相位 1:H2D ∥ count 重叠
    rbox(ax, 5, 65.0, 27.5, 8.0, [
        ("copy stream", 7.0, C_TXT, "bold"),
        ("row-ptr first, then", 6.2, C_SUB, "normal"),
        ("ci / val chunks (L2/4)", 6.2, C_SUB, "normal"),
    ])
    rbox(ax, 34.5, 65.0, 27.5, 8.0, [
        ("kernel stream", 7.0, C_TXT, "bold"),
        ("count-flop per chunk", 6.2, C_SUB, "normal"),
        ("(flop, max-B-len)", 6.2, C_SUB, "normal"),
    ], fc=C_BLUE_F, ec=C_BLUE_E)
    arr(ax, 32.8, 71.5, 34.2, 71.5, head="<|-|>", lw=0.8, color="#8A7CC4")
    arr(ax, 32.8, 67.0, 34.2, 67.0, head="<|-|>", lw=0.8, color="#8A7CC4")
    label(ax, 33.5, 62.8, "overlap", 6.0, "#8A7CC4", ha="center")
    label(ax, 5.0, 63.4, "gate: avg B-row ≤ 8 nonzeros  (span re-pass defers)", 6.0)
    arr(ax, 36, 77.0, 36, 73.0)

    # 相位 2:估计
    rbox(ax, 5, 53.5, 57, 8.5, [
        ("row estimation  (symbolic)", 7.2, C_TXT, "bold"),
        ("avg product ≤ 64 (= warp × 2) → exact flop upper bound", 6.2, C_SUB, "normal"),
        ("else MinHash: construct bottom-k sketch → merge partitions", 6.2, C_SUB, "normal"),
        ("est = min(2·E[sketch], flop) × expand 1.15   (one-sided 95% CI)", 6.2, C_SUB, "normal"),
    ], fc=C_BLUE_F, ec=C_BLUE_E)
    arr(ax, 33.6, 65.0, 33.6, 62.0)

    # 相位 3:binning
    rbox(ax, 5, 46.0, 57, 6.0, [
        ("row binning", 7.2, C_TXT, "bold"),
        ("byte-model cascade per row  →  bucket-ordered row list", 6.2, C_SUB, "normal"),
    ], fc=C_BLUE_F, ec=C_BLUE_E)
    arr(ax, 33.6, 53.5, 33.6, 52.0)

    # 相位 4:numeric 内核阵列(两行)
    label(ax, 5.0, 44.8, "numeric kernel array   (multi-stream, one kernel per bin)",
          7.0, C_TXT, weight="bold")
    krow1 = [
        ("linear scan", "est ≤ 16 = warp/2\nno table", C_GRAY_F, C_GRAY_E),
        ("warp-batched", "est ≤ 32 ∧ kc ≤ warp\n64-slot table", C_BLUE_F, C_BLUE_E),
        ("SMEM ladder", "open addressing\n2·est slots, α = 0.5", C_BLUE_F, C_BLUE_E),
        ("dense-window", "wide rows, low dup\ncolumn windows", C_ORNG_F, C_ORNG_E),
    ]
    krow2 = [
        ("small-span", "span ≤ 256 cols\ndense sub-bins", C_ORNG_F, C_ORNG_E),
        ("global table", "est > SMEM cap\n+ segmented sort", C_BLUE_F, C_BLUE_E),
        ("largest bitmap", "straggler rows\n0.25 B/col window", C_ORNG_F, C_ORNG_E),
        ("dense direct", "output ≥ 15% dense\nwrite final CSR", C_ORNG_F, C_ORNG_E),
    ]
    xw, gap, y4a, y4b = 13.0, 1.6, 36.0, 26.4
    for row, y in ((krow1, y4a), (krow2, y4b)):
        for i, (t, s, fc, ec) in enumerate(row):
            rbox(ax, 5 + i * (xw + gap), y, xw, 8.4,
                 [(t, 6.4, C_TXT, "bold"), (s, 5.7, C_SUB, "normal")],
                 fc=fc, ec=ec, lw=0.9, rounding=0.9)
    arr(ax, 33.6, 46.0, 33.6, 44.4)
    # overflow → retry 侧环(欠估行重做)
    rbox(ax, 45.0, 21.0, 17.0, 5.0, [
        ("overflow retry", 6.2, C_TXT, "bold"),
        ("flop-sized table", 5.6, C_SUB, "normal"),
    ], fc="#FFFFFF", ec=C_ORNG_E, lw=0.9)
    arr(ax, 55.3, 26.4, 55.3, 26.0, ls="--", color="#B0713A", lw=0.8, head="-")
    polyarrow(ax, [(55.3, 26.0), (55.3, 21.0)], ls="--", color="#B0713A", lw=0.8)
    polyarrow(ax, [(53.5, 21.0), (53.5, 20.0)], ls="--", color="#B0713A", lw=0.8)
    label(ax, 63.0, 24.5, "est under-count\nrows", 5.6, "#B0713A")

    # 相位 5:compact(全宽,承接内核阵列)
    rbox(ax, 5, 14.5, 57, 5.5, [
        ("compact / count-sort", 7.0, C_TXT, "bold"),
        ("sorted bins copy free   ·   others in-block csort", 6.0, C_SUB, "normal"),
    ])
    for xx in (11.5, 26.1, 40.7, 55.3):
        arr(ax, xx, 26.4, xx, 20.0, lw=0.8)

    # 相位 6:D2H
    rbox(ax, 5, 9.5, 38, 4.2, [("D2H  packed CSR", 7.0, C_TXT, "bold")])
    arr(ax, 24, 14.5, 24, 13.7)

    # ── merge3 支路(右)──
    ax.add_patch(FancyBboxPatch((68, 42.0), 30, 34.0,
                 boxstyle="round,pad=0,rounding_size=1.6",
                 fc="#F7FBF5", ec="#9DB893", lw=1.1, zorder=1))
    ax.text(69.5, 73.8, "merge3", fontsize=8.5, color="#3F5E35",
            fontweight="bold", va="center", zorder=2)
    rbox(ax, 70.5, 63.0, 25, 7.0, [
        ("count upper bound", 6.8, C_TXT, "bold"),
        ("per-row intermediates", 6.0, C_SUB, "normal"),
    ], fc=C_BLUE_F, ec=C_BLUE_E, lw=0.9)
    rbox(ax, 70.5, 53.5, 25, 7.5, [
        ("expand to COO", 6.8, C_TXT, "bold"),
        ("col-ordered runs per k", 6.0, C_SUB, "normal"),
    ], lw=0.9)
    rbox(ax, 70.5, 43.5, 25, 8.0, [
        ("K-bucket warp merge", 6.8, C_TXT, "bold"),
        ("dedup + sum in registers", 6.0, C_SUB, "normal"),
        ("no hash, no sort", 6.0, C_SUB, "normal"),
    ], fc=C_GREN_F, ec=C_GREN_E, lw=0.9)
    arr(ax, 83, 63.0, 83, 61.0)
    arr(ax, 83, 53.5, 83, 51.5)
    # hash → merge3 整阵回退(arena 超限)
    arr(ax, 64.0, 57.0, 68.0, 57.0, ls=(0, (4, 2.5)), color="#A33", lw=0.9)
    label(ax, 66.0, 59.0, "arena exceeded\n→ full fallback", 5.6, "#A33", ha="center")

    # ── 汇流输出 ──
    rbox(ax, 36, 2.0, 28, 4.0, [("C  (CSR, host)", 8.5, C_TXT, "bold")])
    polyarrow(ax, [(24, 9.5), (24, 7.6), (44, 7.6), (44, 6.0)])
    polyarrow(ax, [(83, 42.0), (83, 7.6), (56, 7.6), (56, 6.0)])

    fig.savefig("fig/paper_structure_overview.png", dpi=DPI)
    fig.savefig("fig/paper_structure_overview.pdf")
    plt.close(fig)
    _pdf_meta("fig/paper_structure_overview.pdf",
              "MMSpGEMM hash pipeline overview: dispatch, overlapped "
              "estimation, binned numeric kernel array")
    print("fig/paper_structure_overview.{png,pdf} written")


# ════════════════════════════════════════════════════════════════════════════
# 图 2:行级路由判据链
# ════════════════════════════════════════════════════════════════════════════
def fig_routing():
    fig, ax = new_canvas(7.16, 5.9)

    # ── 门链(自上而下 = 代码优先级)──
    gates = [
        ("G1  largest bitmap window", C_ORNG_F, C_ORNG_E,
         ["est > SMEM table cap  ∧  span ≥ 16K cols  ∧  dup ≤ 2",
          "est/span ≤ 1/8  ∧  flop ≥ 4096 (= 48 KB / 12 B)",
          "bytes:  0.25 · span + 16 · est ≤ 12 · ht + 24 · est"],
         ("straggler rows: 1 CTA/row", "bitmap + prefix in SMEM, vals streamed")),
        ("G2  small-span dense", C_ORNG_F, C_ORNG_E,
         ["est > 64  ∧  span ≤ 256 cols  ∧  2 · est ≥ span",
          "(array smaller than any hash table)"],
         ("dense sub-window accumulate", "span-binned block shapes")),
        ("G3  linear scan, no table", C_GRAY_F, C_GRAY_E,
         ["est ≤ 16 = warp_size / 2",
          "(scan cost ≈ probe cost at α = 0.5)"],
         ("register linear scan", "")),
        ("G4  dense-window iteration", C_ORNG_F, C_ORNG_E,
         ["dup < 8   (contention ≪ compact tax)",
          "flop ≥ 4096 ∧ est ≥ 2048   (SMEM tier / 12 B)",
          "16 · flop ≥ span  ∧  16 · est ≥ span   (occupancy)"],
         ("column-window accumulate", "direct addressing, no probing")),
        ("G5  global hash table", C_BLUE_F, C_BLUE_E,
         ["est > 16K slots   (SMEM opt-in cap)",
          "(table must live in DRAM)"],
         ("global table + segmented", "radix sort (cub)")),
        ("G6  SMEM hash ladder", C_BLUE_F, C_BLUE_E,
         ["kc ≤ warp ∧ est ≤ 32 → 64-slot batch",
          "kc ≤ warp ∧ est ≤ 64 → 128-slot batch",
          "else  ht = 2 · est slots  (α = 0.5)"],
         ("open-addressing SMEM tables", "32 → 16K slots, one bin each")),
    ]
    n = len(gates)
    top, bot = 95.0, 12.5
    gh = (top - bot) / n
    boxh = gh - 2.6
    for i, (name, fc, ec, conds, target) in enumerate(gates):
        y = top - (i + 1) * gh
        rbox(ax, 21, y, 40, boxh,
             [(name, 6.8, C_TXT, "bold")] + [(c, 6.0, C_SUB, "normal") for c in conds],
             fc=fc, ec=ec, lw=0.9, rounding=0.9)
        t_title, t_sub = target
        tlines = [(t_title, 6.2, C_TXT, "bold")]
        if t_sub:
            tlines.append((t_sub, 5.7, C_SUB, "normal"))
        rbox(ax, 66, y, 32.5, boxh, tlines, fc=fc, ec=ec, lw=0.8, rounding=0.9)
        mid = y + boxh / 2
        arr(ax, 61, mid, 66, mid, lw=0.8)
        label(ax, 63.5, mid + 2.2, "hit", 5.6, C_SUB, ha="center")
        if i:
            arr(ax, 41, y + gh - 2.6, 41, y + boxh, lw=0.8)
            label(ax, 42.2, (y + gh - 2.6 + y + boxh) / 2, "else", 5.6)

    # ── 输入特征(与 G1 同高,水平进入)──
    rbox(ax, 1.5, 81.0, 15.5, 14.0, [
        ("per-row", 6.8, C_TXT, "bold"),
        ("features", 6.8, C_TXT, "bold"),
        ("", 3.0, C_SUB, "normal"),
        ("est  distinct est.", 6.2, C_TXT, "normal"),
        ("span  column width", 6.2, C_TXT, "normal"),
        ("flop  products", 6.2, C_TXT, "normal"),
        ("kc  input chains", 6.2, C_TXT, "normal"),
    ], fc=C_BLUE_F, ec=C_BLUE_E)
    arr(ax, 17.0, 88.0, 21.0, 88.0, lw=0.9)

    # ── 左下图例(色编码 = 内核族)──
    label(ax, 2.0, 66.0, "kernel family", 6.6, C_TXT, weight="bold")
    for j, (cfill, cedge, txt) in enumerate([
        (C_GRAY_F, C_GRAY_E, "table-free linear"),
        (C_BLUE_F, C_BLUE_E, "hash (SMEM / global)"),
        (C_ORNG_F, C_ORNG_E, "dense / bitmap"),
    ]):
        yy = 61.5 - j * 6.5
        ax.add_patch(FancyBboxPatch((2.0, yy - 1.8), 3.2, 3.6,
                     boxstyle="round,pad=0,rounding_size=0.6",
                     fc=cfill, ec=cedge, lw=0.8, zorder=4))
        label(ax, 6.4, yy, txt, 6.0)

    # ── 底部推导注释带 ──
    rbox(ax, 1.5, 1.5, 97, 9.0, [
        ("every threshold is derived, none fitted", 6.6, C_TXT, "bold"),
        ("α = 0.5 ⇒ E[probes] = 1.5 ⇒ table = 2·distinct (Knuth)    ·    4096 = 48 KB SMEM tier ÷ 12 B/slot", 5.9, C_SUB, "normal"),
        ("16K slots = opt-in cap ÷ 12 B    ·    bitmap 0.25 B/col = 1-bit key + 1-bit prefix    ·    contention 12 B/dup = 3 cycles × 4 B", 5.9, C_SUB, "normal"),
    ], fc="#FFFFFF", ec="#B9C4CE", lw=0.9)

    fig.savefig("fig/paper_structure_routing.png", dpi=DPI)
    fig.savefig("fig/paper_structure_routing.pdf")
    plt.close(fig)
    _pdf_meta("fig/paper_structure_routing.pdf",
              "Per-row routing cascade: byte-model gates to kernel bins")
    print("fig/paper_structure_routing.{png,pdf} written")


# ════════════════════════════════════════════════════════════════════════════
# 图 3:merge3 列域并行归并(对标 FIANLSPGEMMPPT.pdf 第 6 页,论文化精修)
# ════════════════════════════════════════════════════════════════════════════
def fig_merge3():
    fig, ax = new_canvas(7.16, 4.8)

    CHAIN_Y = [78.0, 68.0, 58.0]          # 三条 k 链的行中心(两面板共用 → 对照对齐)
    CHAINS  = ["K₀", "K₁", "K₂"]          # 行 i 的三条输入链 A[k,:]
    DATA    = [[1, 9], [0, 3, 8], [2, 6, 9]]

    def cell(cx, cy, num, fc="#FFFFFF", ec="#8C8C8C", w=4.4, h=4.6, fs=6.4):
        rbox(ax, cx - w / 2, cy - h / 2, w, h, [(str(num), fs, C_TXT, "bold")],
             fc=fc, ec=ec, lw=0.8, rounding=0.5)

    # ══ 左面板:prior row merge(串行选择)══════════════════════════════════
    ax.add_patch(FancyBboxPatch((2, 8), 44, 88,
                 boxstyle="round,pad=0,rounding_size=1.6",
                 fc="#FCFCFC", ec="#C4C4C4", lw=1.0, zorder=1))
    label(ax, 4.0, 92.5, "prior row merge — one warp per row, serial select",
          7.2, "#555555", weight="bold")
    label(ax, 4.0, 88.8, "row i  =  merge of k-chains  A[k₀,:]  A[k₁,:]  A[k₂,:]",
          5.9, C_SUB)
    # 三条链:浅灰长条 + 数字格(右移,左侧留给 merge 框 → 水平箭头零交叉)
    for cy, nums in zip(CHAIN_Y, DATA):
        ax.add_patch(FancyBboxPatch((19, cy - 3.0), 25, 6.0,
                     boxstyle="round,pad=0,rounding_size=0.8",
                     fc="#EFEFEF", ec="none", zorder=2))
        lo, hi = 23.0, 41.0
        n = len(nums)
        for j, v in enumerate(nums):
            cell(lo + j * ((hi - lo) / max(n - 1, 1)) if n > 1 else (lo + hi) / 2,
                 cy, v)
    # 串行归并循环框(左侧竖框,高度贴合三条链)
    rbox(ax, 4, 52, 11.5, 32.0, [
        ("warp merge", 6.2, C_TXT, "bold"),
        ("loop", 6.2, C_TXT, "bold"),
        ("", 2.5, C_SUB, "normal"),
        ("shuffle-min", 5.8, C_SUB, "normal"),
        ("emit winner", 5.8, C_SUB, "normal"),
        ("advance chain", 5.8, C_SUB, "normal"),
        ("", 2.5, C_SUB, "normal"),
        ("loop × flop", 5.8, "#888888", "normal"),
    ], fc="#EFEFEF", ec=C_GRAY_E, lw=0.9)
    for cy in CHAIN_Y:
        arr(ax, 19.0, cy, 15.5, cy, lw=0.7)               # 三链水平汇入
    # 串行发射 → 结果
    arr(ax, 9.75, 52, 9.75, 25.0, lw=0.9)
    label(ax, 11.2, 40.0, "serial emit", 5.8, C_SUB)
    for j, v in enumerate([0, 1, 2, 3, 6, 8, 9]):
        cell(10.5 + j * 5.35, 22.5, v, fc="#EFEFEF", ec=C_GRAY_E)
    label(ax, 8.3, 18.0, "C[i,:]  column-sorted, one element at a time",
          5.9, C_SUB)
    ax.text(4.0, 13.0, "serial dependency — heavy rows are stragglers",
            fontsize=6.0, color="#A33333", style="italic", va="center", zorder=6)
    ax.text(4.0, 10.0, "(why fast GPU SpGEMM abandoned merge for hash)",
            fontsize=5.8, color="#A33333", va="center", zorder=6)

    # ══ 面板间大箭头 ═════════════════════════════════════════════════════
    arr(ax, 46.6, 52, 53.4, 52, lw=2.6, color="#B9C4CE", head="-|>",
        conn="arc3,rad=0")
    label(ax, 50, 55.5, "same row,\nsame chains", 5.8, "#7A8791", ha="center")
    label(ax, 50, 47.5, "add a value-\ndomain axis", 5.8, "#3F5E35", ha="center")

    # ══ 右面板:merge3(列值域不交桶)════════════════════════════════════
    ax.add_patch(FancyBboxPatch((54, 8), 44, 88,
                 boxstyle="round,pad=0,rounding_size=1.6",
                 fc="#FAFDF9", ec="#9DB893", lw=1.0, zorder=1))
    label(ax, 56.0, 92.5, "merge3 — disjoint column-domain buckets",
          7.2, "#3F5E35", weight="bold")
    label(ax, 56.0, 88.6, "bounds split row span [min col, max col] into K equal ranges",
          5.8, C_SUB)
    # 三个竖桶(青色渐深)
    BX0, BW, BG = 56.0, 12.4, 1.9
    BFILL = ["#EAF5F0", "#D2EADD", "#B6DCCA"]
    BIN    = [[1, 0, 2], [None, 3, 6], [9, 8, 9]]        # 桶 × 链(演示 K=3)
    for b in range(3):
        bx = BX0 + b * (BW + BG)
        ax.add_patch(FancyBboxPatch((bx, 54), BW, 32.5,
                     boxstyle="round,pad=0,rounding_size=1.0",
                     fc=BFILL[b], ec="#7AA36B", lw=0.9, zorder=2))
        label(ax, bx + BW / 2, 83.6, f"bucket {b}", 6.0, "#3F5E35", ha="center",
              weight="bold")
        for r in range(3):                               # 数字落位:行=链,x=桶内居中
            v = BIN[b][r]
            if v is not None:
                cell(bx + BW / 2, CHAIN_Y[r], v)
    # 桶 → warp-shfl 药丸 → 桶内有序短条
    OUT = [[0, 1, 2], [3, 6], [8, 9]]
    for b in range(3):
        bx = BX0 + b * (BW + BG)
        cx = bx + BW / 2
        arr(ax, cx, 54, cx, 48.2, lw=0.8, color="#7AA36B")
        rbox(ax, cx - 5.6, 42.4, 11.2, 5.6, [("warp-shfl merge", 6.0, C_TXT, "bold")],
             fc="#FFFFFF", ec="#7AA36B", lw=0.9, rounding=2.8)
        arr(ax, cx, 42.4, cx, 37.0, lw=0.8, color="#7AA36B")
        vals = OUT[b]
        cw = 3.6 if len(vals) == 3 else 4.2
        for j, v in enumerate(vals):
            cell(cx + (j - (len(vals) - 1) / 2) * (cw + 0.5), 34.5, v,
                 fc=C_GREN_F, ec=C_GREN_E, w=cw)
    # 桶 2 的重复列去重说明并入底部注释带(桶内两个 9 由 warp-shfl 内 shuffle-add 合并)
    # 拼接 → 结果
    for b in range(3):
        cx = BX0 + b * (BW + BG) + BW / 2
        arr(ax, cx, 32.2, cx, 29.0, lw=0.7, color="#7AA36B")
    ax.plot([BX0 + BW / 2, BX0 + 2 * (BW + BG) + BW / 2], [29.0, 29.0],
            color="#7AA36B", lw=0.8, zorder=2)
    arr(ax, 77, 29.0, 77, 22.0, lw=0.9, color="#7AA36B")
    label(ax, 79.0, 26.0, "concatenate\n(offsets = in-row scan)", 5.8, "#3F5E35")
    for j, v in enumerate([0, 1, 2, 3, 6, 8, 9]):
        cell(60.5 + j * 5.35, 19.5, v, fc=C_GREN_F, ec=C_GREN_E)
    label(ax, 60.5, 15.2, "C[i,:]  sorted directly", 5.9, "#3F5E35", ha="center")
    ax.text(56.0, 11.2, "parallel axes:  rows  ×  k-chains  ×  column buckets",
            fontsize=6.0, color="#3F5E35", style="italic", va="center", zorder=6)

    # ══ 底部要点注释带 ══
    rbox(ax, 2, 1.0, 96, 5.6, [
        ("disjoint value ranges ⇒ bucket outputs concatenate in sorted order", 6.4, C_TXT, "bold"),
        ("parallel and order-preserving · no global sort · duplicate columns shuffle-added within bucket · one warp per (row, bucket)", 5.8, C_SUB, "normal"),
    ], fc="#FFFFFF", ec="#B9C4CE", lw=0.9)

    fig.savefig("fig/paper_structure_merge3.png", dpi=DPI)
    fig.savefig("fig/paper_structure_merge3.pdf")
    plt.close(fig)
    _pdf_meta("fig/paper_structure_merge3.pdf",
              "merge3: column-domain parallel k-way merge "
              "(prior serial row merge vs disjoint column buckets)")
    print("fig/paper_structure_merge3.{png,pdf} written")


# ════════════════════════════════════════════════════════════════════════════
# 图 4/5:hash 与 merge 分支机制图 —— v2 顶刊视觉层
#   规范(ref/ 顶会样板 OpSparse SC22 / MatRaptor / HSMU Fig.1 归纳):
#   · 低饱和语义色,无边框或极浅边;同一颜色语义跨图复用
#   · header-bar 标题栏框(深色带白字标题 + 浅色主体)
#   · 箭头细(≤1pt)小锐头,主流程实线 / 次要虚线,直线与直角折线(不用弧线)
#   · 数据值用等宽字体;格子直角细边
#   · 泳道近无色(#FBFBFA)+ 极浅边;阶段轴 = 细线 + 小三角
#   · 图内无大标题(论文 caption 承担)
# ════════════════════════════════════════════════════════════════════════════
V_BLUE = ("#D6E7F5", "#4A7BB0")   # hash 族 (body fill, header/edge)
V_ORNG = ("#FBE6CC", "#B97A2A")   # dense 族
V_GREN = ("#D9F0DE", "#3F8A5A")   # merge 族
V_GRAY = ("#F0F0F0", "#6B6B6B")   # host / 数据搬运
V_TXT, V_SUB, V_ARR = "#1A1A1A", "#6A6A6A", "#3B3B3B"
V_RED = "#B03A2E"
MONO = "DejaVu Sans Mono"

_CHAIN_Y = [78.0, 70.0, 62.0]
_CHAINS  = [[1, 9], [0, 3, 8], [2, 6, 9]]
_C_OUT   = [0, 1, 2, 3, 6, 8, 9]
_LANES_V = [(60.0, 84.5, "L2 · HBM  (global memory)"),
            (35.0, 57.5, "shared memory  (per block)"),
            (10.5, 32.5, "registers  (per warp)")]


def v_lanes(ax):
    for y0, y1, name in _LANES_V:
        ax.add_patch(FancyBboxPatch((6.5, y0), 91.5, y1 - y0,
                     boxstyle="round,pad=0,rounding_size=0.6",
                     fc="#FBFBFA", ec="#E3E3E3", lw=0.6, zorder=1))
        ax.text(4.2, (y0 + y1) / 2, name, fontsize=5.6, color="#8F8F8F",
                ha="center", va="center", rotation=90, zorder=2)


def v_phases(ax, names):
    ax.plot([8, 97], [88.5, 88.5], color="#CFCFCF", lw=0.6, zorder=1)
    for x, n in zip((19.5, 50.5, 82.5), names):
        ax.text(x, 91.2, n, fontsize=6.3, color=V_TXT, fontweight="bold",
                ha="center", va="center", zorder=3)
    for xm in (35.0, 67.0):
        ax.plot(xm, 88.5, marker=">", markersize=3.0, color=V_ARR, zorder=3)


def v_hdrbox(ax, x, y, w, h, title, body, pal, tfs=6.2, bfs=5.5):
    """header-bar 框:顶部深色标题带(白字)+ 浅色主体。body = [line, ...]"""
    fill, head = pal
    hh = tfs * 1.45 * _UPP + 1.1
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=0.6",
                 fc=fill, ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((x, y + h - hh), w, hh,
                 boxstyle="round,pad=0,rounding_size=0.6",
                 fc=head, ec="none", zorder=4))
    ax.text(x + w / 2, y + h - hh / 2, title, fontsize=tfs, color="#FFFFFF",
            fontweight="bold", ha="center", va="center", zorder=5)
    lh = bfs * 1.55 * _UPP
    cy = y + (h - hh) / 2 + lh * len(body) / 2
    for line in body:
        cy -= lh / 2
        ax.text(x + w / 2, cy, line, fontsize=bfs, color=V_TXT,
                ha="center", va="center", zorder=5)
        cy -= lh / 2


def v_cell(ax, cx, cy, txt, w=3.2, h=4.4, fill="#FFFFFF", edge="#8F8F8F",
           fs=5.9, lw=0.7):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0,rounding_size=0.3",
                 fc=fill, ec=edge, lw=lw, zorder=3))
    ax.text(cx, cy, str(txt), fontsize=fs, color=V_TXT, fontweight="bold",
            fontfamily=MONO, ha="center", va="center", zorder=4)


def v_arr(ax, pts, lw=0.9, color=V_ARR, ls="-"):
    """直线/直角折线箭头(顶刊规范:不用弧线)。pts = [(x,y), ...]"""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls,
                solid_capstyle="round", zorder=2)
    arr(ax, *pts[-2], *pts[-1], lw=lw, color=color, ls=ls)


def v_note(ax, x, y, text, fs=5.5, color=V_SUB, ha="left", bold=False):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va="center",
            fontweight="bold" if bold else "normal", zorder=6)


def v_load(ax):
    """load 列(两图平行):三条 k 链。"""
    for cy, nums in zip(_CHAIN_Y, _CHAINS):
        for j, v in enumerate(nums):
            v_cell(ax, 18.0 + j * 3.6, cy, v, fill="#F2F2F2", edge="#9A9A9A")
    v_note(ax, 8.6, _CHAIN_Y[1], "k-chains of row i\nA[k₀,:]  A[k₁,:]  A[k₂,:]", 5.4)


def v_emit(ax, tag):
    for j, v in enumerate(_C_OUT):
        v_cell(ax, 70.0 + j * 4.0, 73.0, v, w=3.6, fill=V_GREN[0], edge=V_GREN[1])
    v_note(ax, 70.0, 79.4, f"C[i,:]  ·  {tag}", 5.7, "#2F6B46", bold=True)


def v_foot(ax, line1, line2):
    ax.plot([6.5, 97.5], [8.8, 8.8], color="#DDDDDD", lw=0.6, zorder=1)
    ax.text(52, 5.7, line1, fontsize=6.0, color=V_TXT, fontweight="bold",
            ha="center", va="center", zorder=3)
    ax.text(52, 2.7, line2, fontsize=5.5, color=V_SUB,
            ha="center", va="center", zorder=3)


def fig_hash_kernel():
    fig, ax = new_canvas(7.16, 4.7)
    v_lanes(ax)
    v_phases(ax, ("load k-chains", "hash accumulate", "extract & emit"))
    v_load(ax)

    # 链 → warp(直线收敛)
    for cy, ty in zip(_CHAIN_Y, (26.0, 23.0, 20.0)):
        v_arr(ax, [(26.6, cy), (36.0, ty)], lw=0.8)
    v_hdrbox(ax, 36, 14, 29, 16.5, "warp · 32 lanes",
             ["lane t_j owns cursor of chain k_j",
              "products  a_ik × B[k,:]  streamed"], V_GRAY)

    # 插入箭头 → SMEM 表
    for x in (46.0, 50.0, 54.0):
        v_arr(ax, [(x, 30.5), (x, 42.6)], lw=0.8)

    # 开放寻址表:8 槽样例,每槽 [key|val],槽下地址 s0..s7
    keys = ["0", "1", "9", "3", "2", "6", "8", ""]
    for j, k in enumerate(keys):
        cx = 38.0 + j * 3.35
        v_cell(ax, cx - 0.8, 45.2, k, w=1.6, h=3.8, fs=5.4,
               fill="#FFFFFF" if k else "#F4F4F4",
               edge="#4A7BB0" if k else "#BBBBBB")
        ax.add_patch(FancyBboxPatch((cx, 43.3), 1.5, 3.8,
                     boxstyle="round,pad=0,rounding_size=0.3",
                     fc="#EDF3FA", ec="#BBBBBB", lw=0.6, zorder=3))
        ax.text(cx, 40.6, f"s{j}", fontsize=4.6, color="#9A9A9A",
                fontfamily=MONO, ha="center", va="center", zorder=4)
    v_arr(ax, [(41.35, 47.6), (44.7, 47.6)], lw=0.7, color=V_RED, ls="--")
    v_note(ax, 43.0, 49.9, "h(9) → s₁ full → +1 probe", 5.3, V_RED, ha="center")
    v_note(ax, 38.0, 54.4, "open-addressing table · 16 slots = 2 × 7 distinct (α = 0.5)", 5.5)
    v_note(ax, 38.0, 37.4, "12 B/slot (key 4 + val 8) · E[probes] = 1.5", 5.5)
    v_note(ax, 38.0, 35.0, "pass 1: count (1-bit/col bitmap) · pass 2: insert + add (CAS probe)", 5.5)

    # extract → 有序条 → C 行
    v_arr(ax, [(62.4, 45.2), (68.8, 45.2)], lw=0.9)
    for j, v in enumerate(_C_OUT):
        v_cell(ax, 70.0 + j * 3.35, 45.2, v, w=3.0, fill=V_BLUE[0], edge=V_BLUE[1])
    v_note(ax, 70.0, 49.6, "numeric direct-write → compact (sorted bins copy-free)", 5.5)
    v_arr(ax, [(83.4, 43.0), (83.4, 71.0)], lw=0.9)
    v_emit(ax, "sorted after compact")

    v_foot(ax,
           "one kernel per ladder bin — table sized by row est: SMEM ladder 32 → 16K slots"
           " · overflow → flop-sized retry · est > cap → global table",
           "cf. merge branch: register merge — no table, no sort")

    fig.savefig("fig/paper_structure_hash.png", dpi=DPI)
    fig.savefig("fig/paper_structure_hash.pdf")
    plt.close(fig)
    _pdf_meta("fig/paper_structure_hash.pdf",
              "hash branch: per-row open-addressing accumulate "
              "(warp lanes, SMEM ladder table, extract and emit)")
    print("fig/paper_structure_hash.{png,pdf} written")


def fig_merge_kernel():
    fig, ax = new_canvas(7.16, 4.7)
    v_lanes(ax)
    v_phases(ax, ("load k-chains", "bucket + warp merge", "concat & emit"))
    v_load(ax)

    # 链 → 桶边界(直线收敛)
    for cy, ty in zip(_CHAIN_Y, (47.0, 45.0, 43.0)):
        v_arr(ax, [(26.6, cy), (36.4, ty)], lw=0.8)
    for j, v in enumerate([0, 3, 7, 10]):
        v_cell(ax, 38.0 + j * 3.6, 45.0, v, w=3.2, fill="#FFFFFF", edge=V_GREN[1])
    ax.text(44.4, 40.6, "bnd[0..K]", fontsize=4.6, color="#9A9A9A",
            fontfamily=MONO, ha="center", va="center", zorder=4)
    v_note(ax, 38.0, 54.4, "bucket bounds = row span [min col, max col] ÷ K", 5.5)
    v_note(ax, 38.0, 37.4, "count pass: 1 B flag / column within bucket", 5.5)
    v_note(ax, 38.0, 35.0, "bounds tight to product span (load-balanced)", 5.5)

    # 边界 → warp 归并(Register 泳道)+ 桶序条
    v_arr(ax, [(49.0, 42.8), (49.0, 32.2)], lw=0.8)
    v_hdrbox(ax, 36, 16.5, 29, 15.5, "(row, bucket) = one warp",
             ["shuffle-min → smallest column head",
              "shuffle-add → duplicates (9 + 9 → 9)"], V_GREN)
    for xs, vals in (([38.0, 40.6, 43.2], [0, 1, 2]),
                     ([48.0, 50.8], [3, 6]),
                     ([56.0, 58.8], [8, 9])):
        for x, v in zip(xs, vals):
            v_cell(ax, x, 13.0, v, w=2.4, h=3.6, fs=5.2,
                   fill=V_GREN[0], edge=V_GREN[1])

    # concatenate → C 行
    v_arr(ax, [(60.2, 13.0), (68.0, 16.5)], lw=0.8)
    v_hdrbox(ax, 68, 12.8, 28, 8.0, "concatenate",
             ["bucket offsets = in-row scan"], V_GREN)
    v_arr(ax, [(82.0, 20.8), (82.0, 71.0)], lw=0.9)
    v_emit(ax, "concatenated sorted")

    v_foot(ax,
           "disjoint value ranges ⇒ concatenation is already sorted"
           " · duplicate columns shuffle-added · no table, no sort",
           "cf. hash branch: SMEM / global open-addressing tables")

    fig.savefig("fig/paper_structure_merge.png", dpi=DPI)
    fig.savefig("fig/paper_structure_merge.pdf")
    plt.close(fig)
    _pdf_meta("fig/paper_structure_merge.pdf",
              "merge branch: column-bucket warp merge "
              "(disjoint value ranges, shuffle merge, concatenation)")
    print("fig/paper_structure_merge.{png,pdf} written")


if __name__ == "__main__":
    fig_overview()
    fig_routing()
    fig_merge3()
    fig_hash_kernel()
    fig_merge_kernel()
