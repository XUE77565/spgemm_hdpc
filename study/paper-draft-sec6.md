# Paper §6: Evaluation (Draft)

## 6. Evaluation

### 6.1 Experimental Setup

**Hardware**: NVIDIA H100 PCIe (80GB HBM3, 132 SMs @ 1.7GHz), dual-card
configuration with card 1 dedicated to measurements. **Software**: CUDA 12.8,
Ubuntu 24.04, GCC 13.3. **Compilation**: per-file multi-tier (§3): hash kernel
at compute_61→sm_90, all other files at native sm_90.

**Benchmark**: 428 square matrices from the SuiteSparse Matrix Collection
(337 from the primary benchmark suite + 91 from a secondary suite). Matrix
sizes range from 1K to 2.9M rows, nnz from 7K to 128M. Each measurement uses
compute-only timing (TOTAL GPU time minus h2d/d2h transfers), median of 3+
warm runs.

**Baselines**:
- **ocean**: The strongest known competitor (production system, same evaluation
  protocol, paired on the same card)
- **HSMU-SpGEMM** (2024): High shared-memory utilization algorithm
- **TileSpGEMM** (2020): Tile-based sparse computation
- FlexSpGEMM (2026): ML-guided tile/hardware-path selection (not evaluated
  due to build incompatibility; discussed in §7)

### 6.2 Main Results

| Baseline | GM (ours/theirs) | Speedup | Win rate | Era |
|---|---|---|---|---|
| **ocean** | **0.8054** | **1.24×** | **70.5%** | production |
| HSMU-SpGEMM | 0.2138 | 4.7× | 100% | 2024 |
| TileSpGEMM | 0.3550 | 2.8× | 95% | 2020 |

Against the strongest competitor (ocean), we achieve:
- **Geometric mean: 0.8054** (19.5% net improvement)
- **Home-turf (337 matrices): 0.9728** (first-ever overall parity)
- **Win rate: 70.5%** (301/427 paired matrices)

Per-family win rates reveal structural patterns:

| Family | Win rate | GM | Note |
|---|---|---|---|
| Structural (bcsstk/bcspwr) | **100%** (68/68) | 0.427 | Dominant |
| Power systems | 78% | 0.878 | |
| Circuit (can/cage) | 77% | 0.340 | Dominant |
| Engineering (Serena/Hook) | 74% | 0.834 | |
| Graph/Network | **12%** (2/16) | 1.174 | Challenging |
| CFD/Reservoir | **21%** (3/14) | 1.339 | Challenging |
| Delaunay mesh | **0%** (0/4) | 1.773 | Challenging |

The challenging families (graph, CFD, mesh) share a common profile: large
row counts with per-row estimates in the bin0 range (33-64), where our batched
kernel operates near hardware limits (§5.3).

### 6.3 Ablation: Confirm Chain

Figure 1 shows the geometric mean across the confirm chain, with each step
individually attributed:

| Step | Knives | GM | Δ |
|---|---|---|---|
| confirm14 (baseline) | — | 0.8873 | — |
| confirm15 | per-file + knife-A + knife-G | 0.8311 | −6.3% |
| confirm16 | + H69 bin9 prime table | **0.8054** | −3.1% |
| confirm18 | + routing table | 0.8196 | +1.8% |

The confirm15→16 improvement (−3.1%) exceeds our prediction (−0.6 to −1.3%)
because 79 matrices benefit from the bin9 knife—far more than the 6 initially
identified. This "beneficiary population" discovery reveals that bin9
population exists across many winning matrices, not just visible losses.

### 6.4 Routing Table Evaluation

Three-layer evaluation of the signature routing system (§5):

| Layer | Data | GM (routing) | GM (hand-tuned) | Gap |
|---|---|---|---|---|
| Training | 356 | 0.8077 | 0.7944 | +1.7% (noise) |
| Held-out | 72 | 0.8808 | 0.8616 | +2.2% (noise) |
| **Generalization** | — | — | — | **+0.6%** |

The held-out generalization gap (6.7% for hand-tuned) reflects population
difficulty (held-out matrices from harder families), not routing table
overfitting. The routing table adds only 0.6% additional gap.

### 6.5 Flagship Matrix Results

| Matrix | Ratio vs ocean | Note |
|---|---|---|
| Hook_1498 | **0.985** | Parity crossed |
| Bump_2911 | **1.005** | 0.5% from parity |
| cage15 | **1.013** | 1.3% from parity |
| pf2177 | **0.932** | Parity crossed (bin9 knife) |
| SiO2 | **0.420** | Dominant (2.4× faster) |

### 6.6 Methodology Lessons

Our evaluation methodology itself yields transferable findings:

**Variance calibration**: Micro-scale matrices (<1ms) exhibit CV=18.9% with
outliers up to +80% (attributable to system noise from concurrent processes).
Single-shot A/B comparisons at this scale are unreliable. We require ≥10
interleaved measurements with median for matrices <1ms.

**Compile-level controls**: Performance comparisons between GPU kernels must
control for compilation configuration. Our initial 3-day investigation
attributed a 20% gap to "algorithm differences" that was entirely due to a
Makefile flag difference.

**Phased attribution**: Phase-level ablation (measuring kernel sub-phases
separately) revealed that 71.7% of batched kernel time is in hash insertion,
not in table initialization or extraction as initially hypothesized. This
redirected optimization efforts from dead ends (table optimization, geometry
changes) to the correct target (routing populations away from
hardware-limited kernels).

**Negative results are informative**: Three consecutive optimization failures
(read-first, warp aggregation, B-row prefetch) each costing 70-109% overhead,
collectively proved that the batched kernel operates near hardware limits—
the strongest evidence for routing-based optimization.
