# Decidability-Graded Dispatch for Sparse Matrix Multiplication on GPUs

**Draft v0.1 (2026-10-20).** Authors: TBD. Target: SC/PPoPP.
Numbers stamped; figure slots marked [FIG]. Chinese notes in《》are working annotations.

---

## Abstract

Sparse general matrix multiplication (SpGEMM) on GPUs is notoriously input-sensitive:
no single accumulation strategy wins across the space of matrices, and modern systems
therefore face a *dispatch problem* — deciding, per input, which pipeline configuration
to run. The 2026 state of the art, Ocean, solves this with a static threshold between
an estimation-based and a symbolic workflow; earlier work (IA-SpGEMM, ICS'19) trains a
single learned model to predict the best variant. We argue that dispatch decisions
should be **graded by decidability**: decisions that analytic features resolve must not
be learned; decisions with structural signatures belong to compact formulas; only the
irreducible residue justifies machine learning — and every learned decision must be
backed by a **runtime certificate** that can invalidate it and fall back to a safe form.

We instantiate this principle in a three-layer dispatcher over a heterogeneous SpGEMM
arsenal (seven accumulator families, a 3-phase merge-based form, an estimation front-end
with MinHash sampling, and certificate-gated phase elimination). On the union of both
camps' benchmarks (428 matrices: the original challenge set and Ocean's own selection,
which overlap in a single matrix), our deployed system attains a geometric-mean
paired ratio of **0.95 vs Ocean over three same-night measurement campaigns**
(i.e., 5% faster), wins 2.2-2.4x on the challenge set, and generalizes to a held-out
population of 70 never-seen matrices with a 22.3% dispatch gain and **zero regressions**.
We further present an empirical *frontier curve*: without certificates, no pre-run
learned selector can jointly maximize aggregate speed and bound per-matrix regression —
a result that elevates certificates from engineering hygiene to an architectural
necessity. We contribute the forensic methodologies (phase-share and routing-trace
analysis, bandwidth accounting, SASS-level auditing) that located each of our wins.

---

## 1 Introduction

SpGEMM C = A·A underpins algebraic multigrid, graph analytics, and graph neural
networks. On GPUs its cost is dominated by the *accumulator* — the per-row
structure that deduplicates and sums intermediate products — and no accumulator
form wins everywhere: hash tables, dense bitmaps, expand-sort-compress (ESC),
k-way merge, and cursor scans each own a population of inputs. The field's
trajectory has been to multiply strategies (six accumulator families coexist in
current systems) and then to add a dispatcher on top.

**The dispatch problem is subtle.** Our starting observation is a negative result:
per-row kernel-level wins do not compose into system-level wins. In our own campaign,
a phase in which every kernel family beat its Ocean counterpart at the kernel level
coincided with *no movement* in end-to-end score: launch overheads, occupancy
interactions, and fixed "ceremony" costs exist only at system scope
(§6.1). Conversely, fixing the pipeline *changes* the optimal dispatch — we document
a configuration flip (SiO2) in which the best arm for a matrix inverts after a
pipeline repair. Dispatch quality is therefore not a static property to be tuned once
but a moving target coupled to the pipeline beneath it.

**Prior art dispatches flat.** IA-SpGEMM trains MatNet, a deep model over 2,700
matrices, to predict the best format-algorithm pair — a single learned layer with no
safety net. Ocean selects between an estimation-based and a symbolic workflow with a
static analysis threshold, and repairs estimation failures by retry. Neither grades
decisions by how *decidable* they are, and neither bounds the cost of a wrong
dispatch.

**This paper.** We propose **decidability-graded dispatch** with three layers and a
safety net:

1. *Analytic layer.* Decisions resolved by O(n) header features are never learned.
   Our pre-gate sends n ≥ 5000 matrices directly to a lean hash configuration —
   a rule discovered by forensic analysis showing the legacy dispatcher had never
   misrouted such matrices but paid a 4% decision-machinery tax on them. A
   band-structure rule (skew > 1700 ∧ avg-row < 10) routes an extreme-skew
   population to the dense-ladder configuration, worth −30% on its (previously
   5x-losing) population.
2. *Formula layer.* A four-variable logistic formula (flop proxy, dimension,
   max-row, skew), retrained on 91 measured merge3 labels (LOO accuracy 94.5%),
   chooses between the hash and merge forms for small matrices.
3. *Certificate layer.* Overflow certificates invalidate aggressive decisions and
   fall back to the merge form, which is always safe; a degeneracy veto handles
   pathological inputs. An empirical *frontier curve* (§6.4) shows that without
   this layer, no operating point of the learned selector jointly achieves
   aggregate speed and bounded per-matrix regression.

**Results.** On the union benchmark (428 matrices — the union of the original
challenge set and Ocean's own 337-matrix selection; we show the field's benchmark
suites are mutually disjoint to within one matrix), our deployed system achieves
**0.95x** paired geometric mean versus Ocean over three same-night campaigns,
**2.2-2.4x** on the challenge set, and beats the per-matrix oracle of any single arm.
On 70 held-out matrices never touched by any fitting, dispatch gains fully replicate
(+22.3% vs the best single arm) with zero regressions — evidence that the rules
encode structure, not benchmark idiosyncrasy. A leakage audit bounds the worst case:
even if the entire retraining delta were in-sample illusion, the union ratio remains
below 1.0.

We also contribute kernel-level artifacts that the dispatcher selects among — most
notably an **entry-parallel compact-copy kernel** that lifts the output-compaction
phase from 8-16% to near-peak memory bandwidth on small-row populations (+9.1%
same-session on its affected population), and a certificate-gated elimination of the
dense-count phase (DCEST) whose trade-off we characterize as a paradox: the phase is
most expensive exactly where eliminating it hurts most.

**Contributions.**
- A deployed, measured, decidability-graded dispatch architecture for SpGEMM (§4);
- The frontier curve: an empirical proof that pre-run learned dispatch without
  certificates cannot bound tail regressions (§6.4);
- A concrete arsenal of accumulator kernels and runtime mechanisms with per-population
  wins (§5, inventory in Appendix A-E of the companion technical report);
- An evaluation methodology: source-line-level calibration auditing, same-session A/B
  protocols, leakage worst-case bounds, held-out adjudication, and full-phase sidecar
  dumps that make every number recomputable (§6).

---

## 2 Background and the Shape of the Problem

### 2.1 SpGEMM on GPUs

Gustavson's row-wise formulation parallelizes over output rows; each row's products
must be accumulated (deduplicated by column) into the output. The canonical two-pass
workflow computes a symbolic pass (exact per-row output sizes) before numerics; the
symbolic pass averages ~28% of runtime in Ocean's own accounting, motivating the
2026 estimation paradigm (HyperLogLog-based size estimation with runtime retry),
which itself sends 48% of Ocean's evaluation matrices back to the exact workflow —
evidence that estimation is a *gated niche*, a structure our certificates generalize.

### 2.2 Accumulator families

Six families with distinct populations (§5): dense SPA/bitmaps, shared-memory hash,
ESC, k-way merge, cursor scans, and hybrid designs. No single family dominates;
our system maintains seven routed weapons plus a 3-phase merge form.

### 2.3 Dispatch in prior systems

IA-SpGEMM (ICS'19): single learned layer, format+variant granularity, no safety net.
Ocean (ICS'26): static analytic threshold between two workflows; retry on estimation
failure. Wang et al. (TACO'25): ML selects the estimation method within one form.
Du et al. (2022): compression-ratio prediction for output structure. None dispatches
at pipeline-configuration granularity; none grades decidability; none carries
invalidation certificates. [Related-work section to cite: spECK, nsparse, Winter'19,
ASpT, TileSpGEMM, opSparse, HSMU (HPCA'25), MMSpGEMM (PACT'25), gMAGNUS, CombBLAS,
Trident; estimation lineage Cohen'96/97, Amossen'14, HLL.]

---

## 3 Overview: the Decision Chain

[FIG: decision-flow diagram — reproduced from slides P8]

```
header scan (O(n)):
  n >= 5000 ─┬ skew>1700 and avg<10 → hash,C-config   (band rule)
             └ else → hash,A-config (pre-gate) ─ overflow cert → merge3 fallback
  n < 5000 ──┬ avg<2 and n>=10k → hash                (degeneracy veto)
             └ 4-var logit → hash | merge3
```

One command (`METHOD=adaptive`) exercises the full chain. Each layer's scope was
fixed by data: the pre-gate by routing-trace forensics (the legacy dispatcher never
misrouted large matrices; its 4% loss was self-tax), the band rule by loss-population
profiling (the antidote for a 5x-losing population was present in the arsenal but
never routed to), the logit by label retraining, the vetoes by held-out failure
analysis.

---

## 4 The Three Layers (details to expand)

### 4.1 Analytic layer
Pre-gate; band rule; degeneracy veto. Design rule: a decision enters this layer only
when O(n) features separate winners from losers with zero observed collateral.

### 4.2 Formula layer
Logistic formula on (log flop-proxy, log n, log max-row, log skew); 91 measured
labels; LOO 94.5%; −5.6% end-to-end on the small-matrix population.

### 4.3 Certificate layer
DCEST overflow certificates (dense-count elimination with invalidation);
hash-overflow → merge3 fallback; the frontier-curve argument for why this layer
must exist (§6.4). S3 adjudication: after a min-of-5 same-day protocol, the
deployed system's true regression tail is 2 matrices totaling 171 µs across 428.

---

## 5 The Arsenal (kernel contributions; expand from A-E inventory)

- Estimation front-end: MinHash dual estimator (mh_construct/mh_merge), −43% kernel
  optimization via instruction-ledger forensics; estimation-free gate for tiny-product
  inputs.
- Accumulator weapons with per-population ratios (table from A-E inventory).
- merge3: column-domain bucketed warp-merge, 3-phase; dense direct-write branch.
- Runtime mechanisms: DCEST V3 (phase elimination + certificates); entry-parallel
  compact copy (COMPACT_FLAT) — bandwidth forensics (8-16% of peak) → binary-search
  row location → coalesced global stream; +9.1% same-session on the affected
  population, −24-40% on mesh-family matrices; population gate (entry coverage ≥ 50%).
- csort ladder v2 / two-level counting sort.

---

## 6 Evaluation

### 6.1 Setup and methodology
Union benchmark (428; suites disjoint to within one matrix [FIG: Venn]).
Same-night paired measurement; calibration audited to source lines (opponent's
epilogue proven inside its timed region, SpGEMM.cuh:1020/1365); full-phase sidecars;
opponent cross-night stability 0.9966. Single disclosed asymmetry: the opponent's
decision D2H transfers are inside its wall clock; ours are device-side.

### 6.2 Headline results
Three same-night campaigns: **0.9486 / 0.9588 / 0.9560** (bracket 0.95 ± 0.005).
Battlefields: challenge-set 0.41-0.45 (2.2-2.4x), opponent's set 1.174 (17.4%
behind, concentrated in kernel-level residual populations), union 0.95.
Per-layer attribution: pre-gate 1.0331→0.9486; logit −5.6% (small population);
band rule −30% (treated population); FLAT −1.1% on the full opponent set.
Deployment exceeds the four-arm oracle (0.9486 vs 0.9684) because the merge arm was
outside the legacy choice set.

### 6.3 Generalization and leakage audit
Held-out 70 matrices (disjoint from all fitting): +22.3% dispatch gain, 0/70
regressions >1.10. Worst-case leakage bound: 0.9979 < 1.0. Size-band structure
(<1k: 0.372, 48/48 wins; >100k: 1.398).

### 6.4 The frontier curve
Guard-parameter sweep of the learned selector: GM 1.0012 with 14 matrices >1.10
regression vs GM 1.0141 with 1 — no operating point achieves both. Certificates
break the frontier. [FIG: frontier scatter]

### 6.5 Microbenchmarks and case studies
Mesh family (delaunay/road/rajat): copy-bandwidth forensics, FLAT effect,
per-matrix trajectories 2.2-2.4x → 1.3-1.8x. Band family (blowey): routing rescue.
dense_count trade-off paradox (bucketed CD/C ratios). Measurement-protocol findings:
sub-millisecond single-shot cross-day comparisons are unusable (6/9 noise
acquittals); same-session A/B is the only admissible knife-level protocol.

---

## 7 Related Work (expand §2.3 with full citations)

## 8 Conclusion

Decidability-graded dispatch with certificate fallback turns heterogeneous SpGEMM
arsenals into a deployable system: 5% faster than the state of the art on the union
of both camps' benchmarks, with provably bounded dispatch risk and verified
generalization. The frontier curve argues the certificate layer is not optional
hygiene but the enabling mechanism for learned dispatch at all.

---

## TODO / figure list
- [FIG1] decision chain; [FIG2] benchmark Venn; [FIG3] frontier curve;
  [FIG4] size-band bar; [FIG5] phase-share stacked bars (ours vs Ocean);
  [FIG6] mesh-family per-matrix trajectories; [FIG7] held-out scatter.
- Expand §4/§5 from study/37,40,41,45,48,49 + A-E inventory; §6 tables from
  compare/batch437 CSVs (script: analyze_batch437.py outputs).
- Venue formatting; author list; artifact appendix (tags timed-confirm428+).
