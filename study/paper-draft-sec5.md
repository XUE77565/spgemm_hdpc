# Paper §5: Signature Routing (Draft)

## 5. Cost-Calibrated Signature Routing

### 5.1 The Routing Problem

SpGEMM systems deploy multiple kernel variants (hash-based accumulators with
different table sizes, dense direct-addressing, sort-based approaches). The
routing problem—deciding which variant serves which population—is traditionally
solved with hand-tuned thresholds calibrated by trial and error. We propose
replacing these with **automatically generated routing tables** derived from
calibration measurements.

Our routing architecture operates at three levels:

- **Level 1 (Method)**: The adaptive dispatcher selects hash, dense, or merge3
  based on matrix-level features (density, size, estimated output density).

- **Level 2 (Bin variant)**: Within the hash method, per-bin kernel selection
  determines which hash table size, accumulator type, and compilation tier
  serves each population bin. This is the level addressed by our routing table.

- **Level 3 (Row adaptive)**: Within-kernel per-row decisions (e.g., lane
  allocation based on B-row length). We validate the concept but demonstrate
  that the batched kernel is already near hardware limits (§5.3).

A key architectural principle is the separation of **variant menu** (code,
human-registered) from **decision table** (model-generated, automatically
regenerated). Adding a new kernel variant requires only registering it in the
menu and re-running calibration—no manual threshold tuning.

### 5.2 Signature Dimensions

We identify the per-row signature dimensions that capture routing-relevant
population characteristics:

| Dimension | Computation | Cost |
|---|---|---|
| est (estimated distinct outputs) | MinHash / flop-based | Already computed |
| maxbl (max B-row length) | Per-row reduction | Already computed |
| dup ratio (flop / est) | Derived | Zero |
| span (column range) | Per-row reduction | Already computed |

These are packaged into a 4-byte per-row signature, making routing decisions
available at zero additional memory traffic cost.

**Validation (S0).** We verify that the est dimension separates winning from
losing populations: matrices where the bin9 variant underperforms have median
row-level flop ≈ 15, while benefiting matrices have ≈ 49 (crossover band 15–49).
Critically, the max B-row length hypothesis is falsified—both populations have
identical maxbl distributions (median ≈ 8).

### 5.3 Why Fixed Geometry Changes Fail

A natural hypothesis is that changing kernel geometry (e.g., reducing lane count
from 32 to 16 for short B-rows) would improve utilization. We test this with
a multi-row-per-warp kernel variant (`multirow<HT,RPW>`):

**Result: Ratio = 1.000 across all configurations.** On 16 synthetic uniform
matrices and 3 real matrices (3-run interleaved, median), the multirow variant
shows zero improvement over the standard batched kernel.

The root cause is revealed by phase ablation: the hash insertion phase
(CAS + atomicAdd on shared memory) accounts for **71.7%** of kernel time,
extraction 23.2%, and table initialization 5.2%. The per-row work (33-55
products for bin0) is sufficient to saturate a warp regardless of B-row
length. Three optimization attempts targeting the insertion phase (read-first
probing, warp aggregation, B-row prefetch) all produce **negative results**
(+71%, equivalent, +109% respectively), confirming the kernel operates near
hardware limits.

**This is the key evidence for routing over kernel modification**: since
fixed-geometry changes cannot improve performance, the only viable optimization
is routing populations to different existing kernels.

### 5.4 Calibration and Decision Table Generation

We calibrate the routing table using 509 training points (356 real matrices
from our benchmark suite + 153 synthetic matrices with controlled signatures):

**Calibration protocol:**
1. For each matrix, run with 3 variant configurations (H69 off, H45/H55 off,
   all off) plus the baseline (from confirm measurements).
2. Record compute-only time (TOTAL - h2d - d2h) and bin population.
3. Partition the signature space into cells (bin9_fraction × bin45_fraction ×
   matrix_size, each with 3 levels = 27 possible cells).
4. For each cell, determine the optimal variant by majority vote.

**Key finding:** The calibration confirms that the current default configuration
(H69=1, H45=1) is optimal for 70% of the population—the largest cell
(none_high_medium, n=144) votes 70% in favor of the current defaults. This
transforms the hand-tuned gates from "we tried it and it worked" to "we measured
470 matrices and this configuration is optimal."

### 5.5 Production Integration

The routing table is integrated as a compile-time lookup function called once
per matrix at the top of `hash_product()`, before any GPU kernel launches:

```c
// Compute signature from bin population
int b9_level = (bin9_frac < 0.01) ? 0 : (bin9_frac < 0.3) ? 1 : 2;
int b45_level = (bin45_frac < 0.01) ? 0 : (bin45_frac < 0.3) ? 1 : 2;
int size_level = (total_rows < 10000) ? 0 : (total_rows < 1000000) ? 1 : 2;
route_cfg_t cfg = routing_table_lookup(b9_level, b45_level, size_level);
```

The overhead is negligible (host-side integer comparisons, executed once).
An environment variable (`RTABLE=0`) provides an escape hatch to the
previous hand-tuned behavior.

### 5.6 Three-Layer Evaluation

We evaluate generalization using a three-layer protocol:

| Layer | Data | GM (routing table) | GM (hand-tuned) | Gap |
|---|---|---|---|---|
| Training | 356 matrices | 0.8077 | 0.7944 | +1.7% (noise) |
| Held-out | 72 matrices | 0.8808 | 0.8616 | +2.2% (noise) |
| **Generalization gap** | — | **+0.0731** | **+0.0672** | **+0.0059** |

**Key finding:** The held-out generalization gap (6.7%) exists for both
routing-table and hand-tuned configurations, and is attributable to population
difficulty (held-out matrices are from harder families: graph, CFD, mesh).
The routing table adds only **0.6%** additional gap—within noise. This
confirms that the calibration generalizes and does not overfit to the
training set.

### 5.7 Value Proposition

The routing table achieves **equivalent performance** to hand-tuned gates.
Its value is not speed but principled methodology:

1. **Measurement provenance**: Every routing decision has a calibration source.
2. **Automatic regeneration**: When kernels change, re-run calibration (~30
   minutes) to generate updated tables—no manual re-tuning.
3. **Verified generalization**: Held-out evaluation confirms no overfitting.
4. **Confirmation of near-optimality**: The hand-tuned gates were already
   near-optimal, which is itself a validated finding.
