# Paper §5: Physics-Informed Adaptive Routing (Draft v2, 11-02)

> 重写说明:原版=S4 查表时代;本版整合 11-02 全部实验
> (分层测绘/AI Router v3/漂移四臂/C 部署/held-out 40)。
> 主张纪律:不 claim "AI 比 if-else 准";不 claim 首次行级路由;
> 不 claim regret 界(无探索项);分数贡献归 §3-§4 的刀,本节主张可维护性。

## 5. Physics-Informed Adaptive Routing

### 5.1 The Routing Problem

SpGEMM systems deploy multiple kernel families (hash accumulators with
ladder-scaled tables, dense direct addressing, merge-based). The routing
problem—deciding which variant serves which matrix and which row
population—is traditionally solved with hand-tuned thresholds. On a fixed
benchmark, hand-tuning can always reach near-oracle performance; the cost is
weeks of expert labor, and the result is **brittle**: when kernels evolve,
calibrated answers silently expire (§5.5 measures this).

Our routing architecture operates at three levels:

- **Level 1 (Method)**: the adaptive dispatcher (a trained logit) selects
  hash, dense, or merge3 from matrix-level features.
- **Level 2 (Population)**: within the hash method, per-bin selection
  determines table size, accumulator kernel, and extraction strategy per
  estimated-output-length bin.
- **Level 3 (Build)**: per-file compilation tier selection (compute_61 vs
  native sm_90, §4)—a routing dimension we discovered.

A key architectural principle is the separation of the **variant menu**
(code, human-registered) from the **decision logic** (model-generated,
automatically regenerated).

### 5.2 Where Routing Matters: A Layered Survey

Before building a learned router, we measure **where routing has headroom**.
We ablate 8 sub-switch routing points (kernel-variant toggles within the
hash method) on 66 matrices, two options each (658 paired measurements),
and compare per-matrix oracle routing against the best *fixed* choice:

| Decision layer | Best fixed option achieves | Routing headroom |
|---|---|---|
| Method/variant combos (4 options) | 51% of oracle decisions | **real headroom** |
| bin4/5 kernel (hash_heavy vs LHT) | 100% (total-time 1.000×) | none |
| bin9 kernel (prime table vs LHT14) | 60% (small sample) | marginal |
| compact strategy (sorted copy vs csort) | 90% | ~none |
| est expansion (1.15 vs 1.4) | 89% | cliff-only |
| block ladder / csort / flat / lht23 | ~100% | none |

Two findings:

1. **Sub-switch defaults are near-oracle** (0.7% total headroom): any
   model with <100% accuracy necessarily loses to a fixed rule here.
   The correct engineering response is to *freeze* these as mechanisms,
   not to route them.
2. **Mis-routing stakes are extremely uneven**: the est-expansion cliff
   (overflow → retry storm) reaches 30× on affected matrices, while most
   other points cost ≤1.3×. To our knowledge this stakes map of the
   SpGEMM routing space has not been published.

### 5.3 Three Generations of Decision Logic

**Generation 1 — hand-tuned gates** (weeks of trial-and-error; the
deployed defaults). **Generation 2 — calibration tables** (470-matrix
calibration → 27-cell signature grid → lookup; held-out gap only +0.6%,
proving the hand gates were near-optimal and generalizable). Both are
**static**: they encode the kernel library as it existed at calibration
time.

**Generation 3 — physics-informed cost model with online updates.** For
each decision point and option we predict

```
cost(option | x) = phys(x) + (w·x̄ + b)
```

where `phys(x)` is a closed-form roofline model (table clear × occupancy,
insert atomics, extraction scan, HBM traffic—computed from hardware
constants) and `w·x̄+b` is a linear residual on standardized features
(14 dimensions: log-size, density, bin population fractions, est
quantiles, interactions), trained supervised on paired ablation data.

The design is matched to four constraints that rule out alternatives:
(i) **sample efficiency**—80 matrices suffice (each option has ~15
parameters; tree ensembles starve at this size); (ii) **extrapolation**—
physics holds outside the training range, trees clamp to leaf values;
(iii) **drift adaptation**—online SGD after every run (§5.5), where
offline-trained models expire; (iv) **zero deployment tax**—host-side,
~50 ns per decision, no GPU changes. Cold-start (no data) degrades
gracefully to the physics prior.

### 5.4 Kernel Evolution Invalidates Static Routing (Measured)

Our own kernel history provides a controlled drift experiment. We keep
two binaries from a compile-layer campaign (§4) plus current production:

| Era | Binary | bin4/5 optimal (14 matrices, paired) |
|---|---|---|
| A (Oct 27) | native sm_90 build | LHT wins 12/14 |
| B (Oct 27) | compute_61 build | LHT wins 12/14 |
| C (Oct 30) | production (per-file + knives) | **hash_heavy wins 13/14** |

Three days of kernel evolution (ordered direct-write, sorted copy,
variant promotion) **flipped the optimal routing on 13/14 matrices**.
A static table calibrated in era A collapses when era C arrives:

| Arm | Era-A accuracy | Era-C accuracy | Era-C time regret |
|---|---|---|---|
| Static table (calibrated era A) | 100% | **14%** | 1.128× |
| always hash_heavy | 14% | 100% | 1.000× |
| always LHT | 86% | 0% | 1.130× |
| **Online model** (era-A prior, era-C stream) | — | **86%** | **1.009×** |

The online model—initialized with the era-A prior and updated by SGD on
the era-C observation stream—recovers within two passes (57% → 86% →
86%). **Stale routing costs 12.8% fleet-wide; the adapting model costs
0.9%.** Static approaches in prior work (IA-SpGEMM, Wang et al., offline
ML in FlexSpGEMM) structurally cannot run this experiment: their kernels
do not change.

*Engineering note.* The first execution of this experiment produced a
false negative (29% recovery): unstandardized interaction features
(magnitude ~170) saturated the clipped error on both options, freezing
the relative update. The failure mode—instrument saturation masking the
mechanism—recurs across our campaign and is worth documenting.

### 5.5 Deployment

The router is integrated at the top of `hash_product()` as an opt-in
path (`AIROUTER=1`; default off): features are assembled from bin
populations already computed by the pipeline (zero extraction cost),
costs predicted per decision point, and defaults overridden only when
the predicted win exceeds 5% (**confidence gating**: worst case is
exactly the hand-tuned default). On a three-matrix sanity A/B the
overhead is indistinguishable from noise (ratios 1.012 / 1.008 / 1.000),
and the model's prediction matches measured ground truth where checked
(e.g., af_shell1: predicted hash_heavy 2.31 < LHT 2.51 in log₂-ms;
measured 7.80 vs 8.55 ms).

### 5.6 Generalization on Unseen Matrices

We pre-register a held-out protocol before running: 213 on-disk matrices
never touched by any router data pool, filtered by the sealed size rule
(nnz ≥ 1e5 or dim ≥ 1e4), evenly sampled to 40 (selection rule fixed
before measurement). Result: 38/40 exercise the hash path; the gated
router **flips zero defaults**; paired ratio geomean 1.013 (t = 0.94 vs
1.0, not significant—the two configurations execute identical kernels).

**Interpretation.** The *lower bound* is proven: on unseen matrices the
router falls back to defaults with zero regression and zero overhead—
the conservative design holds. The *upper bound* (router beats
hand gates on unseen matrices) is **not tested by this set**: all 40
are small (0.3–2.5 ms) with negligible bin4/5/9 populations—the
decision points barely exist. Testing the upper bound requires large
unseen matrices; the external download remains blocked, and we report
this as a limitation rather than extrapolating.

Combined with §5.4, the complete claim is: **no evidence → safe
fallback to defaults; evidence + kernel drift → online adaptation
recovers near-optimal routing with no human re-calibration.**

### 5.7 What We Do and Do Not Claim

We claim: (1) the layered survey of routing headroom and stakes;
(2) measured decision drift under kernel evolution and its automation
cost (12.8% vs 0.9%); (3) a deployable, zero-tax, confidence-gated
router with proven safe fallback; (4) calibration methodology that
replaced weeks of hand-tuning with one measurement pass.
We do not claim: that learned routing beats hand-tuned gates on the
tuning benchmark (sub-switch defaults are near-oracle there—§5.2 shows
this is structural); first row-level routing (hybrid per-row formats
date to SpMV work in 2009); or formal regret bounds (the update is
greedy; ε-exploration is future work). The performance score of the
system comes from the kernel and compilation contributions of §3–§4;
this section contributes maintainability and adaptivity.
