# Pin-chain measurement scripts (frozen)

These are the throwaway scripts that produced the pin-chain measurements (findings note,
section 14; `table.md` here is the 7-brain cohort table). They are kept verbatim as the record
of how the numbers were made — **not maintained, not imported anywhere, excluded from ruff**.
They read the gitignored cohort data and write under `benchmarks/output/probe_source_space/`.

- `full_chain_v5.py` — the full-volume chain (pin read → pairwise drop → AMG re-fill → per-slice
  2D engine → 2.5D → census + landmark residual), `python full_chain_v5.py <brain> [variant]`,
  env knobs `PIN_TAU` (`auto` = the tau rule), `PIN_C`, `PIN_DETREND`.
- `cg_bench.py` — the re-fill CG benchmark (Jacobi-PCG vs pyamg AMG-PCG).
- `table.md` — the cohort results table.

Superseded by the library: `dvfopt.pipeline_pins.correct_dvf_pins` (CLI
`dvfopt correct --pipeline pins`), `dvfopt.dvf.pins`, `dvfopt.dvf.refill`. The landmark-residual
column is benchmark-side only — the library never reads correspondence files.
