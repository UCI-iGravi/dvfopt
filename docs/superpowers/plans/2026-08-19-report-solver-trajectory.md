# Interactive-Report Solver Trajectory Animation

> **For agentic workers:** executed inline this session (no subagent). Steps use `- [ ]`.

**Goal:** Let the interactive cohort report animate how a field's Jacobian-determinant map deforms **across the solver's iterations**, not just before→after.

**Approach:** Capture K intermediate DVF states from `correct_dvf_25d`'s `progress_callback` (the cohort's 3D corrector — it streams full intermediate volumes), embed the displayed slice's Jdet for K evenly-sampled frames as float16 base64, and add a play/scrub timeline to the existing vanilla-JS canvas viewer. Graceful: no captured frames → today's before/after viewer, unchanged.

**Scope / limits (honest):**
- **3D cohort report only** (`run_cohort_benchmark`, serial) in v1. The 2D-sections path is parallel (ProcessPoolExecutor) and its default strategies (`slp`/`slsqp_windowed`) don't stream intermediate fields, so it keeps before/after — noted as follow-up.
- Frames embedded (K≤8, jdet-only) keep the single-portable-file design; size bounded by the K cap.

**Tech:** existing `benchmarks/interactive_report.py` viewer, `benchmarks/cohort_benchmark.py`, `correct_dvf_25d(progress_callback=)`, `jacobian_det3D`.

## Global constraints
- Ruff clean, tests via `pytest tests/`. Don't touch dvfopt core solvers.
- Report must still be self-contained (no external assets).
- Never hold K full volumes in memory: capture only a 3-slice stub `phi[:, zc-1:zc+2]` per fire (zc is known from the initial field before solving).

---

### Task 1: trajectory capture + payload (cohort_benchmark.py)

**Files:** `benchmarks/cohort_benchmark.py`, `tests/test_cohort_benchmark.py`

**Interfaces produced:**
- `make_25d_corrector(**kw)` closure gains optional `frames` sink: `corrector(phi, frames=None)`. When `frames` is a list, it installs a `progress_callback` that appends `np.asarray(state['phi']).copy()` (full volume — the pipeline hands one per fire; caller slices).
- `_sample_trajectory(volumes, phi_init, phi_out, zc, threshold, k=8) -> list[dict]`: prepend `phi_init`, append `phi_out`, evenly sample to ≤k; each frame → `{"jdet": ir.b64_floats(jacobian_det3D(f)[zc]), "n_neg": int, "label": str}`. Returns `[]` if `volumes` empty.
- `_build_3d_payload(..., frames=None)` gains `frames`; when non-empty adds `"traj": [f["jdet"] ...]`, `"traj_labels": [...]` to the payload.

- [ ] Step 1: test `_sample_trajectory` (planted 3-slice volume list → K frames, first=input jdet, last=output jdet, len≤k). Run → fail.
- [ ] Step 2: implement `_sample_trajectory` + `frames` sink in `make_25d_corrector` + `frames` arg on `_build_3d_payload`. In `run_cohort_benchmark` serial loop, when `interactive`, capture: `frames_vol=[]; phi=corrector(phi_init.copy(), frames=frames_vol)` and pass `_sample_trajectory(frames_vol, ...)`. Guard the `frames=` call with a try/TypeError so a non-capturing corrector still works.
- [ ] Step 3: test `_build_3d_payload(frames=[...])` includes `traj`; `frames=None`/`[]` omits it. Run → pass. Ruff. Commit.

### Task 2: viewer timeline (interactive_report.py)

**Files:** `benchmarks/interactive_report.py`, `tests/test_interactive_report.py`

- [ ] Step 1: `_field_block` passes `data.traj` / `data.traj_labels` through the JSON when present; adds (only then) a `<input type=range class=traj>` + `<button data-act=play>▶</button>` + `<span class=trajlbl>` to the controls.
- [ ] Step 2: `_JS` Viewer: if `data.traj`, `const trajF32=data.traj.map(b64ToF32), imgTraj=trajF32.map(f=>buildImage(f,W,H,thr,vmax))`; state `trajIdx=-1`. `curImg()` returns `trajIdx>=0?imgTraj[trajIdx]:(showAfter?imgA:imgB)`; `draw()` uses `curImg()`. Slider sets `trajIdx`, updates label, `draw()`; play toggles a `setInterval` stepping `trajIdx` 0→last then stops. Hover in traj mode reads `trajF32[trajIdx][i]` (jdet only; skip dy/dx line). Toggling before/after or reset sets `trajIdx=-1`. Disable quiver while `trajIdx>=0`.
- [ ] Step 3: test `build_interactive_report` with a `traj` payload contains the slider + `data-act=play` and the base64 frames; a payload without `traj` has neither (graceful). `_JS` is a static string — assert the frame-decode + `curImg` tokens exist. Run → pass. Ruff. Commit.

### Task 3: end-to-end + docs

- [ ] Step 1: extend an existing `run_cohort_benchmark(..., interactive=True)` synthetic test (fields= bypass, dz=0) to assert the produced report.html contains `data-act=play` and a non-empty `traj` for a folded 3D field; feasible/flat field → no `traj`.
- [ ] Step 2: note the feature in `benchmarks/cohort_benchmark.py` / `interactive_report.py` docstrings + CHANGELOG [Unreleased]. Full `pytest tests/`, ruff. Commit. PR → verify CI → squash-merge.

## Self-review
- Data availability verified empirically (25d progress_callback yields `{'phi': full volume}`). ✔
- Memory bound: capture stores volumes only during one serial solve, sampled to ≤8 slices before embedding; the payload holds only K slice-jdets. (If a 528-slice volume's callback fires hundreds of times, `frames_vol` transiently holds those volume refs — acceptable for one field at a time; a stub-slice optimization is a noted follow-up if it bites.)
- Graceful degradation: TypeError guard + empty-frames → unchanged before/after. ✔
- Types: `traj` is `list[str]` (base64); `traj_labels` `list[str]`. JS `data.traj` optional. ✔
