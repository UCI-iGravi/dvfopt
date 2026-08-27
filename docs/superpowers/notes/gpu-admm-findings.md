# Batched GPU ADMM for window QPs — findings

**Verdict: NO-GO** for the two regimes the engine actually runs in (cold and
warm-from-neighbour). A batched GPU ADMM is 3x *slower* per QP than the CPU
process pool's measured ceiling there. It is 6.7x faster than the pool only in
the degenerate near-solution warm regime, which the engine does not spend its
time in.

Prototype: [`benchmarks/gpu_admm_proto.py`](../../../benchmarks/gpu_admm_proto.py)
(research only — imports nothing from `dvfopt`, ships nothing to the library).
Box: i7-13700 + RTX 3050 8 GB, cupy 14.2 / CUDA 12.9, 40 captured real
giant-tile window QPs (`benchmarks/output/qp_capture/`).

## Environment note

The isolated venv at `%TEMP%\qpb` had `cupy-cuda12x` but no CUDA 12 libraries —
only a system CUDA **v13.0** toolkit, so `import cupy` succeeded (cudart) while
cuBLAS/cuSPARSE failed with `DLL load failed while importing cublas`. Fixed by
installing the pip wheels:

```
uv pip install --python %TEMP%\qpb\Scripts\python.exe \
  nvidia-cublas-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-curand-cu12 \
  nvidia-cufft-cu12 nvidia-nvjitlink-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
```

## The QPs

n = 16018, m ∈ {21632, 26834}. **P is diagonal** (5202 entries, all 2.0) — the
elastic step term; `q` is 0/1000 on 10816 slack variables. A has 89258 nnz with
rows of exactly 1/3/5/7 nnz, and **QPs with the same m share an identical
sparsity pattern** (only the values differ). No equality rows.

So the reduced KKT `M = P + σI + ρA'A` has 226620 nnz, and P being diagonal
means the whole linear-solve difficulty lives in `A'A`.

## What was built

OSQP-faithful batched ADMM in cupy: Ruiz equilibration (10 sweeps, cost
scaling), σ=1e-6, ρ₀=0.1 with per-row equality bump, α=1.6 over-relaxation,
adaptive ρ, unscaled-residual termination at eps_abs = eps_rel = 1e-3 checked
every 25 iterations, max_iter 4000. K QPs are padded to a common m and stacked
as one **block-diagonal CSR**, so each operator application is a single cusparse
SpMV over the whole batch.

Two linear-solve strategies:

- **indirect** — Jacobi-preconditioned CG, warm-started across ADMM iterations,
  cuOSQP's tolerance rule (`max(1e-7, min(0.15·√(r_prim·r_dual), prev/5))`).
- **direct-hybrid** — per-QP `scipy.sparse.linalg.splu` of M on the CPU, then
  the per-iteration triangular solves batched on the GPU via
  `cupyx.scipy.sparse.linalg.spsolve_triangular`.

**Batch compaction** turned out to be mandatory and is implemented: without
retiring converged QPs the batch runs until the *slowest* member finishes, and
per-QP cost then *grows* with K (K=64 cold went 4862 → 1501 ms/QP once
compaction was added). Converged QPs are frozen and the block-diagonal operator
is rebuilt from the survivors.

`--self-check` runs a GPU-free correctness test against a closed-form QP.

## Numbers — per-QP wall clock (ms), 40 real QPs

CPU is single-core OSQP measured in the same session (eps 1e-3, polishing off,
max_iter 4000). "pool" is that divided by the measured 2.6x process-pool ceiling.

| regime | CPU 1 core (mean / median) | CPU pool ceiling | GPU K=8 | K=32 | K=64 | K=128 |
|---|---|---|---|---|---|---|
| **cold** | 1367 / 1310 | 526 | 1756 | 2027 | **1501** | 1593 |
| **warm_prev** (engine-realistic) | 757 / 562 | 291 | 1060 | 1406 | **969** | 1048 |
| warm_self (from own solution) | 252 / 98 | 97 | 23.5 | 38.1 | 21.2 | **14.4** |

GPU/pool throughput ratio at the best K: **cold 0.35x, warm_prev 0.30x,
warm_self 6.7x**.

Median ADMM iterations: CPU OSQP 2162 (cold) / 888 (warm_prev); GPU 512 (cold) /
375 (warm_prev) — our ADMM needs ~4x *fewer* iterations, and still loses, because
each iteration costs ~25 CG iterations.

Without compaction, for reference: K=32 cold 4993, K=64 cold 4862, K=64
warm_self 1361 ms/QP.

### Direct-hybrid — cost of ONE ADMM iteration (not a whole solve)

| K | GPU batched tri-solve | CPU splu setup | LU nnz/QP | GPU MiB/QP |
|---|---|---|---|---|
| 8 | 20.5 ms/QP | 72 ms | 1.89M | 22 |
| 16 | 11.0 ms/QP | 71 ms | 1.88M | 22 |
| 32 | 7.1 ms/QP | 75 ms | 1.90M | 22 |
| 64 | **4.4 ms/QP** | 74 ms | 1.90M | 22 |

CPU OSQP does a *complete* ADMM iteration (factorized KKT solve included) in
0.63 ms. `spsolve_triangular` is 7x worse at K=64 for the linear solve alone,
because triangular solves are level-sequential and batching only widens each
level rather than shortening the chain. **Direct-hybrid is not viable** — the
factorization also costs 8.9x fill (226620 → 2.0M nnz) at 22 MiB/QP.

### Memory

Indirect: **7.1–9.2 MiB/QP resident** (2.2 MiB/QP of that is the block-diagonal
A/Aᵀ operators; the rest is ADMM + PCG vectors). 8 GB fits roughly **700 QPs**,
call it 600 with fragmentation headroom. Direct-hybrid: 22 MiB/QP → ~330.

### Break-even batch size

- **cold / warm_prev: there is none.** The GPU never beats even one CPU core on
  the full 40-QP set; it saturates at K≈64 and gets slightly worse at K=128.
- **warm_self: K=1.** 25.4 ms/QP at K=1 already beats CPU's 98 ms median,
  because that regime terminates in ≤25 ADMM iterations with ~0 CG iterations —
  it measures fixed overhead, not solve work.

## Why it loses

The SpMV throughput advantage is real and is *not* the problem:

| quantity | value |
|---|---|
| A + Aᵀ SpMV pair, K=64 | 0.0222 ms/QP |
| measured PCG iteration, K=64 | 0.0549 ms/QP (2.5x the SpMV pair — vector ops, kernel launches, per-iteration sync) |
| CG iterations per ADMM iteration | **25.3** |
| ⇒ GPU cost per ADMM iteration | 1.39 ms/QP |
| CPU OSQP cost per ADMM iteration (direct KKT) | **0.63 ms/QP** |

One factorized KKT solve on the CPU costs less than 25 Jacobi-preconditioned CG
iterations on the GPU, even at 30x the SpMV throughput. Batching amortizes the
*launch* overhead (which is why K=1 → K=64 helps a lot) but cannot change the
25:1 work ratio.

Two secondary taxes, both measured:

- **Straggler tail.** convmax/convmed is 6.2x (cold) and 5.9x (warm_prev): the
  hardest QP needs 3200 iterations where the median needs 512. Compaction
  recovers most of this, but shrinking the batch also gives back the
  amortization that made the GPU fast.
- **Fused-kernel headroom.** A perfectly fused PCG (no launch/sync overhead,
  0.0222 ms/QP/iteration) with *zero* straggler waste would give 710 ms/QP cold
  and 540 ms/QP warm_prev. Still **below** the pool's 526 / 291 ms/QP. The
  idealized ceiling loses too.

## Correctness

All 40 QPs converged to eps_abs = eps_rel = 1e-3 in every batch and regime
(`conv N/N` at K = 1…128).

| | max bound violation | median | objective gap vs captured OSQP (median / max) |
|---|---|---|---|
| GPU indirect (all K, cold) | 6.8e-3 … 1.5e-2 | ~5e-4 | +2e-4 … +6e-3 / 2.1e-1 |
| GPU indirect (warm_self) | 1.8e-2 | | 0.0 / 2.8e-2 |
| **control:** CPU OSQP no-polish vs captured polished | 7.7e-3 | 4.1e-4 | +2.9e-4 / **6.4e-2** |

The control row is the point: OSQP itself, run at the same 1e-3 tolerance
without polishing, differs from the captured (polished) solutions by up to 6.4%
in objective on the small-objective QPs (objectives span 50 → 1.4e5, so an
absolute gap of ~10 reads as 20% on a QP whose optimum is 50). Our ADMM's
2.1e-1 max is looser than that but the same phenomenon and the same order;
medians agree to ~1e-3. No polish step was implemented.

## Recommendation

**No-go.** Do not wire a batched GPU ADMM into the windowed engine. Reasons, in
order of how hard they are to remove:

1. The 25:1 CG-iterations-per-KKT-solve ratio is the whole story, and even a
   perfectly fused, straggler-free implementation stays below the CPU pool.
2. Getting to "go" requires a preconditioner strong enough to cut CG to ~5
   iterations (IC(0), or an algebraic multigrid on `A'A`) — batched on the GPU,
   for a matrix whose pattern is shared across QPs. That is a research project,
   not an integration.
3. Integrating it would require turning the facade/CLI per-slice loop into a
   **batch producer** (accumulate windows across many slices, solve K at once,
   scatter results back), which breaks the engine's per-window
   grow-on-failure/no-damage control flow. Not worth paying for a 0.3x.

The one genuinely promising fact for a future attempt: **QPs with the same m
share an identical sparsity pattern**, so a shared-pattern batched SpMV kernel
(values `(K, nnz)`, indices read once) is available, as is a shared *symbolic*
factorization — a batched numeric Cholesky with one symbolic analysis for the
whole batch is the only direct route that could plausibly beat CPU OSQP. That,
not PCG, is where a second attempt should go.

Reproduce:

```
python benchmarks/gpu_admm_proto.py --self-check
python benchmarks/gpu_admm_proto.py --qp-dir benchmarks/output/qp_capture --batches 8,32,64,128
python benchmarks/gpu_admm_proto.py --qp-dir benchmarks/output/qp_capture --batches 8,16,32,64 \
    --strategy direct --skip-cpu
```
