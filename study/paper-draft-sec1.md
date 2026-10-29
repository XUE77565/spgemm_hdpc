# Paper §1: Introduction (Draft)

## 1. Introduction

Sparse matrix-matrix multiplication (SpGEMM) is a fundamental kernel in graph
analytics, scientific computing, and machine learning. On modern GPUs, SpGEMM
performance depends on a complex interplay of algorithm choice, kernel geometry,
compiler behavior, and input matrix characteristics. Existing approaches
optimize along a single dimension: algorithm selection methods pick the best
algorithm per matrix, auto-tuning frameworks search kernel parameters, and
hand-tuned systems adjust thresholds by trial and error.

We demonstrate that this single-dimensional view is fundamentally incomplete.
Through a systematic investigation motivated by a three-day performance mystery,
we discover that **compilation-level decisions**—specifically, the PTX
(intermediate representation) level used in NVIDIA's compilation pipeline—can
account for performance differences of up to 52%, dwarfing many algorithmic
improvements. More critically, these compilation effects are **conditional**:
the same kernel compiled with different PTX levels not only performs differently
but can exhibit correctness failures (out-of-bounds shared memory access) on
specific input populations.

This discovery motivates a **multi-dimensional routing system** that treats
compilation tier, algorithm variant, and population routing as orthogonal
decision axes within a unified framework. Our system makes three key
contributions:

**Contribution 1: Compile-Layer Characterization.** We provide the first
systematic study of PTX-level sensitivity in SpGEMM kernels. We identify
compute_61 as a lone performance peak (with compute_70 and compute_80 falling
back to native performance), demonstrate asymmetric benefits across kernel
families, and document three novel link-layer hazards including template
instantiation weak merge—a silent wrong-codegen mechanism. Our per-file
multi-tier compilation architecture with automated verification (symbol
disjointness, coverage audit, baseline drift check) makes these benefits
safely deployable.

**Contribution 2: Algorithmic Knife Chain.** Leveraging the compile-layer
discovery, we revive a previously-falsified kernel (hash_heavy) that
underperformed by 77% under native compilation but becomes the strongest
self-developed kernel under compute_61 (reaching 85% of the theoretical
ceiling). We introduce an in-kernel ordered direct-write mechanism that
eliminates the sorting carrier overhead, and a prime-number hash table tier
(16,253 slots) that replaces a power-of-two table suffering from occupancy
disasters. Together, these "knives" reduce the performance gap to the strongest
competitor from 28% to 5%.

**Contribution 3: Signature Routing System.** We replace all hand-tuned
routing thresholds with automatically generated decision tables derived from
calibration measurements. Our system packages per-row features (estimated
distinct outputs, maximum B-row length, duplication ratio) into compact
signatures, calibrates variant performance across 509 training points (356
real + 153 synthetic matrices with controlled characteristics), and generates
routing tables through majority-vote cell assignment. A three-layer evaluation
(training / held-out / cross-family) confirms that the routing table achieves
equivalent performance to hand-tuned gates with only 0.6% additional
generalization gap.

**Contribution 4: Comprehensive Evaluation.** We evaluate on 428 matrices
against three external baselines and one real competitor. Our system achieves
a geometric mean ratio of 0.8054 (19.5% faster) against the strongest
competitor, while maintaining 100% numerical correctness across all 428
matrices. Per-family analysis reveals 100% win rate on structural matrices
and identifies remaining challenges in graph/network populations.

### 1.1 Motivation: The Decision Drift Problem

A key insight from our investigation is what we term **decision drift**: as
kernels are modified (algorithm improvements, compilation changes, new variants),
previously-optimal routing decisions become stale. We observe this phenomenon
repeatedly: adding a kernel variant changes the optimal routing for populations
that the new kernel does not even serve (through compilation-level perturbation
effects); modifying a hash table size shifts the crossover point between hash
and dense paths; and reviving a kernel (via compilation changes) invalidates
the population assignments that had been tuned for the previous version.

Traditional hand-tuning cannot address decision drift because there is no
systematic mechanism to detect when decisions become stale or to update them.
Our calibration-based routing system solves this by making the update process
automatic: re-run calibration (~30 minutes), regenerate tables, redeploy.

### 1.2 Paper Organization

§2 reviews related work. §3 presents the compile-layer discovery and per-file
architecture. §4 describes the algorithmic improvements. §5 details the
signature routing system. §6 presents evaluation results. §7 discusses
limitations and future work. §8 concludes.
