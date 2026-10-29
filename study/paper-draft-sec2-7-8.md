# Paper §2, §7, §8 (Draft)

## 2. Background and Related Work

**SpGEMM algorithms.** GPU SpGEMM methods fall into three families:
hash-based accumulators (per-row hash tables in shared memory), sort-based
approaches (sort all products then aggregate), and dense accumulation
(direct addressing for rows with high output density). Hybrid approaches
select among these per matrix or per row.

**Algorithm selection.** IA-SpGEMM [Winter et al., 2019] adaptively
re-purposes GPU threads and selects algorithm per matrix. Wang et al. [2025]
formulate selection as a per-matrix choice problem. FlexSpGEMM [2026] uses
machine learning (LightGBM, LLM, SVM) to predict tile sizes and hardware
paths (CUDA cores vs Tensor cores). All prior work operates at the matrix
level or tile level; our system operates at the row level and additionally
includes compilation tier as a routable dimension.

**GPU compilation.** NVIDIA's compilation pipeline (source → PTX → SASS)
allows specifying different PTX levels for the same target architecture.
This mechanism is typically used for forward compatibility (running old
binaries on new GPUs), not as a performance optimization. Our work is the
first to systematically characterize its performance impact on SpGEMM kernels
and to exploit it as a per-file routable dimension.

**Cost models for sparse computations.** Learned cost models have been applied
to database query optimization and sparse matrix computations. Our approach
differs in using physical cost models (closed-form expressions derived from
hardware invariants like lane utilization and occupancy) rather than black-box
machine learning, making the routing decisions interpretable and cheap to
recalibrate.

## 7. Discussion

**Limitations.** Our compile-layer findings are specific to CUDA 12.8 and
the H100 architecture; different ptxas versions may produce different
optimization patterns. The routing table was calibrated on our 428-matrix
benchmark; while held-out evaluation shows no overfitting, external
validation on independently-collected matrices remains future work (our
download attempt was blocked by the SuiteSparse server).

**FlexSpGEMM comparison.** FlexSpGEMM [2026] represents a complementary
approach: ML-predicted tile/hardware-path routing with Tensor Core hybrid
execution. We were unable to build it in our environment (requires CUDA 11.8
and specific GCC versions). The two systems optimize along orthogonal axes:
FlexSpGEMM targets tile-level decisions with ML on Tensor Core vs CUDA core
selection; we target row-level population routing with physical cost models
and compilation-tier selection. A combined system could leverage both.

**Generalization beyond SpGEMM.** The routing architecture (menu/decision-table
separation, signature-based routing, calibration-driven table generation)
is domain-agnostic. Any GPU kernel family with multiple variants (SpMM,
graph traversals, stencil computations) could adopt this framework: register
variants, measure calibration population, generate tables, deploy.

**Compile-layer sensitivity as a general phenomenon.** Our finding that PTX
level affects performance by up to 52% and correctness (through shared memory
layout changes) likely applies to other compute-intensive GPU kernels beyond
SpGEMM. We recommend that GPU performance studies routinely report and control
compilation configuration, similar to how CPU studies control compiler flags.

## 8. Conclusion

We presented a multi-dimensional routing system for SpGEMM on modern GPUs that
addresses three orthogonal decision axes: compilation tier, algorithm variant,
and population routing. Our key discoveries include: (1) PTX-level compilation
sensitivity that can account for 52% performance differences and affect
correctness, (2) the revival of a previously-falsified kernel through
compilation changes, (3) an in-kernel ordered direct-write mechanism that
eliminates sorting carrier overhead, and (4) a calibrated routing table system
that replaces hand-tuned thresholds with measurement-based decisions.

Evaluated on 428 matrices against three external baselines and the strongest
known competitor, our system achieves 19.5% net improvement (GM 0.8054) with
100% numerical correctness. The routing table achieves equivalent performance
to hand-tuned gates while providing measurement provenance and automatic
regeneration capability.

Our methodology findings—variance calibration requirements for micro-scale
measurements, compile-level controls for fair comparison, and the diagnostic
value of negative results—contribute to the broader GPU performance engineering
community beyond SpGEMM.
