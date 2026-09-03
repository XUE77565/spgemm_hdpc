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


if __name__ == "__main__":
    fig_overview()
    fig_routing()
