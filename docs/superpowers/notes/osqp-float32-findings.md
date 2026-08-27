# float32 OSQP for the windowed isqp engine — feasibility study

**Verdict: NO-GO.** A single-precision OSQP builds cleanly and is genuinely
2.4–3.0x faster *per ADMM iteration*, but it returns a coarser solution at the
same nominal tolerance. The windowed engine pays that back with 43–160% more
SQP iterations, and on one of the three hard crops it never converges at all.
Net wall over the four in-engine cases: **332.5 s (float64) -> 401.0 s
(float32)** — slower, and worse.

Box: i7-13700, Windows 11. `dvfopt` @ `origin/main` (7cd79a1), osqp 1.1.3.

---

## 1. Build recipe (worked, exactly as written)

The osqp-python 1.x wrapper is scikit-build-core over a `FetchContent`'d
`osqp/osqp` v1.0.0 C core. Its own `CMakeLists.txt` pins `OSQP_USE_LONG=OFF`
but says nothing about `OSQP_USE_FLOAT`, so the C library's own
`option(OSQP_USE_FLOAT ...)` is reachable from the command line, and
`src/osqp/interface.py` already keys its numpy dtype off it
(`self._dtype = np.float32 if self.ext.OSQP_USE_FLOAT == 1 else np.float64`).
No patching needed.

```bash
# 1. isolated venv (short path — Windows MAX_PATH bites during the CMake build)
uv venv --python 3.12 C:\Users\Andy\AppData\Local\Temp\osqp32
uv pip install --python C:\Users\Andy\AppData\Local\Temp\osqp32\Scripts\python.exe numpy scipy

# 2. sdist (the PyPI wheel is float64; --no-binary is not enough on its own,
#    installing the tarball by path forces the source build)
curl -sL -o osqp-1.1.3.tar.gz \
  https://files.pythonhosted.org/packages/source/o/osqp/osqp-1.1.3.tar.gz
```

Then, from a **cmd.exe** shell (MSVC is not on the default PATH):

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set CMAKE_ARGS=-DOSQP_USE_FLOAT=ON
set SKBUILD_CMAKE_DEFINE=OSQP_USE_FLOAT=ON
set OSQP_ALGEBRA_BACKEND=builtin
uv pip install --python "C:\Users\Andy\AppData\Local\Temp\osqp32\Scripts\python.exe" ^
    --no-cache "C:\...\osqp-1.1.3.tar.gz"
```

Notes:
- No cmake/ninja needed on PATH — scikit-build-core pulls both from PyPI into
  the build isolation env. VS 18 Community ships MSVC 14.50 but no bundled
  `cmake.exe` at the usual `Common7\IDE\CommonExtensions\...` location.
- `vcvars64.bat` prints `'vswhere.exe' is not recognized` and still works.
- Build time ~60 s (includes the `git clone` of osqp v1.0.0 by `FetchContent`).

**Verification**

```
OSQP_USE_FLOAT = 1
OSQP_USE_LONG  = 0
_dtype = numpy.float32   _itype = numpy.int32
capabilities = 29
r.x.dtype = float32      (2-var reference QP solves correctly, 25 iters)
```

`osqp.capabilities()` is not exposed; the authoritative flag is
`osqp.OSQP().ext.OSQP_USE_FLOAT` (mirrored by `._dtype`).

---

## 2. Landmine found: polish failure silently returns a stale solution

In single precision, OSQP v1.0.0's **polish** step fails on these KKT systems:

```
ERROR in LDL_factor: ... There are zeros in the diagonal matrix
ERROR in osqp_solve: Failed polishing
```

and `osqp_solve` handles it as (`src/osqp_api.c`):

```c
exitflag = polish(solver);
if (exitflag > 0) { c_eprint("Failed polishing"); goto exit; }   /* skips store_solution() */
```

so the caller gets the **previous contents of the solution buffer** while
`info.status` still reads `"solved"` and `info.obj_val` still reports the ADMM
objective. On a fresh solver that is all zeros; inside the engine (which reuses
one `osqp.OSQP()` across SQP iterations via `prob.update(...)`) it would be the
*previous* iteration's step — silently stale, and `isqp_solve`'s
`np.all(np.isfinite(z))` guard does not catch it.

**45 of the 80 captured solves** (23 of the 40 QPs) are hit with
`polishing=True` in float32. `delta` (1e-6 … 1e-3) does not help. Every float32
number below therefore uses `polishing=False`.

This is not a float32-only bug in principle — it is an upstream failure path
that returns garbage as "solved". float64 just never trips the factorisation
here.

Control: float64 with `polishing=False` is indistinguishable from float64 with
polish on, in-engine (192.5 s / L2 280.31 / 780 SQP iters with polish vs
193.1 s / L2 280.61 / 693 without). **Polish is not load-bearing for this
engine**, so the float32 numbers are a fair precision-only comparison.

---

## 3. Captured-QP accuracy and speed

40 real subproblems from `benchmarks/output/qp_capture/` (n = 16018,
m = 21632–26834), `eps_abs = eps_rel = 1e-3`, `max_iter = 8000`,
`polishing=False`, cold and warm-started from the captured `x`/`y`.
Reproduce with `benchmarks/osqp_float32_bench.py`.

Caveat: this whole section ran while another timing job held 5–6 cores, so the
absolute wall figures are inflated for both builds (the two runs were
back-to-back under comparable load). Iteration counts, violations and
objectives are unaffected; the matched-work table at the end of the section was
run twice to bound the contention noise.

| | float64 | float32 | ratio |
|---|---|---|---|
| cold, total wall | 353.8 s | 76.5 s | **4.62x** |
| cold, total ADMM iters | 101 150 | 44 225 | 2.29x fewer |
| cold, µs / ADMM iter | 3497 | 1731 | **2.02x** |
| warm, total wall | 59.7 s | 6.3 s | **9.50x** |
| warm, total ADMM iters | 14 850 | 2 625 | 5.66x fewer |
| warm, µs / ADMM iter | 4023 | 2396 | 1.68x |
| median &#124;Δobj&#124;/&#124;obj&#124; cold | — | — | 1.1e-2 (max 2.6e-1) |
| median &#124;Δobj&#124;/&#124;obj&#124; warm | — | — | 4.1e-3 (max 5.7e-2) |
| median max bound violation | 3.8e-4 / 5.6e-4 | 9.8e-4 / 9.2e-4 | ~2x worse |
| worst max bound violation | 7.7e-3 | 1.5e-2 | 2x worse |

**The headline 4.6x / 9.5x is not a speedup — it is an early exit.** float32
terminates at far fewer iterations because its ADMM trajectory (scaling and rho
are computed in the working precision) reaches the *same* tolerance sooner at a
*worse* point. Cross-checked by recomputing OSQP's own residuals in float64
from the returned `x`/`y`: OSQP's reported `prim_res`/`dual_res` match exactly
in both builds, and both satisfy their thresholds. The termination test is
honest; the tolerance is simply loose enough (`eps_dual ≈ 1.0`, because
`‖q‖∞ = 1000` — the elastic penalty `rho` — dominates the relative term) that
float32 can stop 10x earlier.

Example (qp0, cold): float64 4025 iters, obj 8246.19, violation 4.9e-4;
float32 425 iters, obj 7718.55, violation 7.0e-3.

Note both builds routinely exceed the 1e-3 target margin on raw bound
violation at `eps=1e-3` (float64 up to 7.7e-3). The engine tolerates that
because the elastic slack absorbs it and feasibility is re-checked
geometrically per window — but it means "well below 1e-3" is not a bar the
*current* float64 configuration clears either.

**Matched-work throughput** (`eps≈0`, `adaptive_rho=0`, `check_termination=0`,
exactly 500 iterations, min of 3 reps, two rounds) — this is the honest
arithmetic/bandwidth number:

| QP | float64 µs/iter | float32 µs/iter | ratio |
|---|---|---|---|
| 0 | 2534 / 2630 | 926 / 867 | 2.7–3.0x |
| 8 | 4111 / 4388 | 1078 / 1120 | 3.8–3.9x |
| 20 | 3592 / 3799 | 1267 / 1098 | 2.8–3.5x |
| 24 | 4038 / 4449 | 1368 / 1532 | 2.9x |
| 39 | 3741 / 2438 | 1498 / 1772 | 1.4–2.5x |

**2.4–3.0x per iteration**, above the ~2x width-halving ceiling — consistent
with the KKT factor crossing a cache threshold once its float payload halves,
which is exactly the memory-bandwidth story that motivated this study. The
lever is real; the problem is what the SQP loop does with it.

---

## 4. In-engine measurement

`windowed_correct(raw, 'isqp', constraint=SimplexConstraint2DBilinear(...),
objective=NoneObjective(), threshold=0.01)` at engine defaults
(`qp_backend='hybrid'`, clarabel installed in both venvs), threads pinned
(`OMP/OPENBLAS/MKL/RAYON=1`). float32 runs force `polishing=False`; float64
runs are stock.

All rows below were measured under **light background load** (one leftover
worker, ~19% CPU). Quality columns are deterministic — they reproduced exactly
between the heavy-load and light-load passes — so only wall moved.

| case | build | wall | simplex folds | folds_after | damage | L2 move | SQP iters | windows |
|---|---|---|---|---|---|---|---|---|
| B0039 z16 (320×456) | float64 | **192.5 s** | 0 | 0 | 0 | 280.31 | 780 | 28 |
| | float32 | **157.6 s** (1.22x) | 0 | 0 | 0 | **331.44** (+18%) | 1114 (+43%) | 29 |
| z0_cluster (35×42) | float64 | 44.0 s | 0 | 1 | 0 | 542.73 | 387 | 9 |
| | float32 | 35.4 s (1.24x) | **1** | 1 | 0 | 551.22 (+1.6%) | 619 (+60%) | 9 |
| z0_sliver (53×52) | float64 | 65.7 s | 0 | **0** | 0 | 25.32 | 540 | 15 |
| | float32 | **175.2 s (0.37x)** | 0 | **9** | 0 | 16.83 | 1395 (+158%) | 27 |
| z16_twist (50×50) | float64 | 30.3 s | 0 | 0 | 0 | 125.72 | 128 | 2 |
| | float32 | 32.8 s (0.92x) | 0 | 0 | 0 | 123.21 | 285 (+123%) | 2 |
| **total** | float64 | **332.5 s** | 0 | 1 | 0 | | 1835 | |
| | float32 | **401.0 s** | 1 | 10 | 0 | | 3413 | |

Heavy-load pass, for the record (another timing job was running: 5–6 python
workers, 30–55% CPU): z16 f64 287.4 s / f32 180.6 s; cluster f64 56.5 / f32
45.4; sliver f64 91.6 / f32 257.0; twist f64 40.7 / f32 50.2. Same quality
numbers throughout. Only the light-load pass is quoted above.

Reading:

- The **no-damage invariant holds** in float32 (`damage = 0` everywhere) — the
  frozen-ring construction is precision-independent, as designed.
- z16 is the best case and still only **1.22x**, bought with **+18% L2 move**.
  For a method whose whole selling point is minimal displacement, an 18% larger
  move is not a free speedup.
- **z0_sliver is the disqualifier.** float64 clears it (18 -> 0 folds, 65.7 s).
  float32 leaves **9 of 18**, after 2.7x the wall, 2.6x the SQP iterations and
  1.8x the windows — the coarse QP steps fail window targets, which triggers
  grow-on-failure, extra rounds and extra windows. Its smaller L2 (16.8) is not
  a win: it is the move it did not make on the folds it did not fix.
- Every case pays 43–158% more SQP iterations. The 2.4–3.0x per-iteration gain
  buys 1.2x at best and -2.7x at worst.

---

## 5. Go / no-go

**No-go**, on three independent grounds:

1. **Net slower end to end** (332.5 s -> 401.0 s over the four cases). The
   per-iteration win is real but the SQP loop spends it on extra iterations.
2. **Quality regression**: +18% L2 move on z16, one residual simplex fold on
   z0_cluster, and 9 unfixed folds on z0_sliver. Feasibility is the engine's
   product; float32 stops delivering it.
3. **Upstream polish landmine** (§2): OSQP v1.0.0 reports "solved" while
   returning a stale buffer. Any float32 deployment would have to ship
   `polishing=False` *and* a guard, or carry a patched C core.

### If it were a go, what shipping it would cost on Windows

Non-trivial, and worth stating even though the answer is no:

- **No float32 wheel exists.** PyPI ships one `osqp` distribution, float64.
  A `[solvers]` extra cannot express "the same package, different CMake
  define" — it would need a separately-named wheel (`osqp-float32`) built and
  published by us for every cp3.10–3.13 × win/linux/mac target, or a source
  build at install time.
- **A source build at install time means MSVC on the user's box** (VS Build
  Tools ≥ 14.4x), plus `FetchContent`'s `git clone` of the osqp C repo at
  build time — i.e. network access from inside the build. Neither is
  acceptable for `pip install dvfopt[solvers]`.
- **Dual-precision at runtime** would mean two importable extension modules
  (the wrapper hardcodes `OSQP_EXT_MODULE_NAME = ext_builtin` per build), so
  they cannot coexist in one environment without renaming the module in a fork.
- `isqp.py` would need a precision probe (`osqp.OSQP()._dtype`) to force
  `polishing=False`, tighten `osqp_eps` to compensate for the early exit, and
  document that `qp_max_iter` no longer means the same thing.

### The one thing worth keeping from this

The matched-work measurement (§3, 2.4–3.0x per ADMM iteration) confirms the
memory-bandwidth diagnosis. The way to collect that win is **not** narrower
floats: it is fewer/smaller KKT solves. Cheaper levers already in the codebase
point the same direction — `coarse_to_fine`, `max_window_area`, and the
`hybrid` interior-point backend all cut ADMM iteration *count*, which is the
term float32 accidentally proved dominates.

---

## Reproducing

```bash
# captured-QP suite (run once per interpreter, then diff)
<f64-python> benchmarks/osqp_float32_bench.py --capture <repo>/benchmarks/output/qp_capture \
    --no-polish --out f64.json
<f32-python> benchmarks/osqp_float32_bench.py --capture <repo>/benchmarks/output/qp_capture \
    --no-polish --out f32.json
python benchmarks/osqp_float32_bench.py --compare f64.json f32.json

# in-engine
<f32-python> benchmarks/osqp_float32_bench.py --no-polish \
    --engine <repo>/data/dvfs/b0039/b0039_laplacian_deformation_field.npy --z 16 --out e.json
```
