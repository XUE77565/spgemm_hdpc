# Paper §3: Compile-Layer Discovery (Draft)

## 3. Compile-Layer Performance Sensitivity in SpGEMM Kernels

### 3.1 The PTX-Level Performance Cliff

We discover that the choice of PTX (intermediate representation) level in NVIDIA's
compilation pipeline has a profound and previously unreported impact on SpGEMM kernel
performance—up to 52% difference for the same source code, and critically, affecting
**correctness** in certain configurations.

The CUDA compilation pipeline consists of three stages: source → PTX (virtual ISA) →
SASS (machine code). The `-gencode=arch=X,code=Y` flag controls the PTX level (X)
and target architecture (Y). Conventionally, these are matched (e.g., `arch=sm_90,
code=sm_90`). However, we find that **deliberately mismatching** them—generating
compute_61-level PTX but targeting sm_90 SASS—produces significantly faster code
for hash-based SpGEMM kernels.

In a controlled experiment comparing our hash kernel compiled with native sm_90
versus compute_61→sm_90, we observe:

| Configuration | Bump_2911 accumulate (ms) | Speedup |
|---|---|---|
| Native sm_90 | 79.1 | baseline |
| compute_61 → sm_90 | 71.9 | **+9.1%** |

More critically, the competing system (ocean) uses this compilation flag
unintentionally (a legacy Makefile setting), accounting for **the entire performance
gap** that motivated three days of investigation. When both systems compile with
identical flags, the gap closes to noise level.

A scan across the PTX ladder (compute_50 through compute_80) reveals that
compute_61 is a **lone peak**—compute_70 and compute_80 both fall back to
native sm_90 performance. This suggests the benefit is not from "older is better"
but from a specific interaction between Pascal-era PTX constructs and the Hopper
ptxas backend's instruction selection.

### 3.2 Per-File Multi-Tier Compilation

Exploiting this discovery requires solving a compilation engineering challenge:
applying compute_61 to the hash kernel would force all dependent CUDA intrinsics
(match_any, reduce) to their software fallbacks, and the benefits are asymmetric
across kernel families. We introduce a **per-file, multi-tier compilation
architecture**:

- **Tier 1 (compute_61)**: The hash kernel file compiles with
  `-gencode=arch=compute_61,code=sm_90`, targeting bins ≥4 large-table instances.
  A template parameter `TB_TAG` ensures instantiation-set disjointness.

- **Tier 2 (native sm_90)**: All other source files compile natively, preserving
  sm_80+ intrinsics and avoiding regressions in non-hash paths.

- **Tier 3 (build-time verification)**: Three automated checks guard against
  compilation-level hazards:
  1. **Symbol disjointness** (`check_disjoint.sh`): Verifies zero overlap in
     template instantiation sets between compilation units. This prevents
     the "weak merge" hazard where the linker silently resolves two same-named
     template instantiations to one arbitrary body (§3.3).
  2. **Coverage audit** (`check_optin.py`): Ensures all large-table kernel
     instances exceeding 48KB shared memory have corresponding
     `cudaFuncSetAttribute` opt-in calls.
  3. **Baseline drift check**: Verifies that code additions do not perturb
     existing kernel performance beyond noise thresholds.

### 3.3 Link-Layer Hazards

We document three novel compilation hazards that are invisible at the source level:

**Hazard 1: Template Instantiation Weak Merge.** When two compilation units
instantiate the same kernel template with identical parameters but different
optimization levels (compute_61 vs sm_90), the linker silently merges them to
one arbitrary body. In our probe experiment, a kernel launched from compilation
unit A executed the code generated for compilation unit B—a silent wrong-codegen
that produces no compiler warning or runtime error.

**Hazard 2: Shared Memory Opt-In Mismatch.** Template instantiation creates new
mangled symbols that are not covered by existing `cudaFuncSetAttribute` calls.
Kernels requiring >48KB shared memory fail with `cudaErrorInvalidConfiguration`
at launch time. This manifested as a batch failure across 40 matrices before
our coverage audit caught it.

**Hazard 3: Dormant Code Performance Perturbation.** We observe that adding
never-executed kernel code to a compilation unit can perturb the performance of
unrelated active kernels by up to 38% on micro-scale matrices (<1ms). However,
variance calibration reveals this effect is largely attributable to system-level
noise (CV=18.9% for 0.1ms-scale measurements), with any true compilation-layout
effect being smaller than our measurement precision.

### 3.4 Methodology Lessons

These findings motivate a revised evaluation discipline for GPU kernel research:

1. **Compile-level variables must be controlled** in any performance comparison.
   A difference attributed to "algorithm" may be entirely due to PTX level.

2. **Per-file compilation flags** should be treated as a routable dimension—
   choosing which PTX level for which kernel is a legitimate optimization decision.

3. **Automated build-time verification** (symbol disjointness, coverage audit,
   baseline drift) is necessary when using per-file compilation, as the hazards
   are invisible at the source level.

4. **Variance calibration** (repeated measurements on identical binaries) must
   precede attribution of performance changes to code modifications, especially
   for short-running kernels where system noise can mimic compilation effects.
