## Findings

Fill in after the benchmark completes. Expected pattern:

- **Sparse synthetic / multi-cluster real data:** big speedup
  (5–20×) because each cluster is small relative to the full field.
  L1 should be ≤ global m14 (or close, since each cluster sees only
  its own anchor neighborhood).
- **Single-large-cluster cases (`z12_30x30_*` and `z12_full`):**
  the `fallback_size_ratio` triggers and we fall back to global m14.
  Wall-clock is then `global m14 wall + cluster-detection overhead`
  — essentially unchanged.

If sparse cases show the expected speedup, the prototype is worth
promoting into the public surface (as `iterative_2d_tri_refine_repair`
with an optional `schwarz=True` flag or a new
`iterative_2d_tri_schwarz_refine` entry point).
