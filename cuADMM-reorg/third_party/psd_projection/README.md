# psd_projection (vendored subset)

Copied verbatim from the `psd_projection` submodule at commit b566590 (the frozen version, plan decision Q9), as
required by the developer prompt ("Copy code into the new tree when you need it"): the composite polynomial filters,
the Lanczos norm estimate, LOBPCG and the helpers that `SDPSolver` calls.

| File | Origin |
|---|---|
| `include/psd_projection/{check,composite_FP16,lanczos,lobpcg,utils}.h` | `psd_projection/include/psd_projection/` |
| `src/{composite_FP16,lanczos,lobpcg,utils}.cu` | `psd_projection/src/` |

Not copied: `eig_FP32_psd`, `eig_FP64_psd` (the solver does not call them), the MATLAB bindings and the tests. The FP32
and FP32-emulated composite filters were copied in Phase 0 and removed in Phase 1 (plan decision Q3: not ported; they
were opt-in paths that no default used). Local edits: `lobpcg.h` documents `D` as a vector of `m` eigenvalues (B12).

Plan Phase 6a moves the FP16 filter, the Lanczos estimate, LOBPCG and the precision/symmetrization helpers to
`src/cone/psd/kernels/` under cuADMM's error macros (B15) with one copy of each helper (B10); the FP32 and
FP32-emulated filters are not ported (Q3). Until then this directory keeps the submodule's include guard and
`CHECK_*` macros (which only print), exactly as the parent tree builds them.

License: neither repository has a license yet (plan O4).
