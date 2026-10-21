# Decisions, questions and deviations

Settled directions (plan D1-D8, Q1-Q9) are followed as written. This file records the choices the plan leaves to the
implementation, deviations from its letter, and open questions, each with its reason. Entries are dated.

## 2026-10-21

1. **Phase 0 copies the current solver into `cuADMM-reorg/src/legacy/`.** The plan's Phase 0 edits `main` in
   place; the developer prompt makes `cuADMM-reorg/` a self-contained project that must build and test without the
   parent tree. The copied tests exercise `SDPSolver`, so the project needs the solver from day one. The copy is
   verbatim (c23e67e), kept in `src/legacy/` with the headers under `src/legacy/include/cuadmm/`, and is decomposed
   phase by phase into the layout of plan Section 6.1; what remains in `src/legacy/` at Milestone 1 belongs to
   Phases 5-7 (cone tiers, PSD projection, CHOLMOD y-step, scaling).

2. **The submodule subset is a shared library.** `cuadmm_psd_kernels` (the copied `psd_projection` sources) is
   built `SHARED`, as the submodule builds `psd_lib`: the solver and the submodule both define
   `max_dense_vector_zero_kernel` (B10), and a static library gives a duplicate symbol at link time. The duplicate
   goes away in Phase 6a with the kernel move.

3. **The FP32 and FP32-emulated composite filters are copied in Phase 0 and removed in Phase 1.** Q3 says they are
   not ported; the Phase 0 copy is verbatim so that the copied tests (`large_projection_test.hpp` uses
   `composite_FP32.h`) build unchanged. Phase 1 (cleanup) removes the two kernels, their `ProjectionMethod` values
   (which become errors, principle 4) and the tests of those values, recorded in `CHANGES.md`.

4. **Reference for `residual_balancing` is 2d8a755 (factor 2), not c23e67e (factor 1.2).** As the developer
   prompt prescribes. The starting commit c23e67e differs from 2d8a755 only in `src/solver.cu` and `solver.h`
   (`admm_sigma_factor = 1.2` replacing `sigscale`), so its `legacy_adaptive` reproduces neither reference policy:
   the Phase 0 copy is compared with 2d8a755 on `fixed` cases only. In Pre-0 the factor becomes `penalty.factor`
   (default 2; 1.2 selectable), as the plan says.

5. **Machine.** The reference workstation of the plan (2x RTX 6000 Ada, CUDA 12.9, SuiteSparse 7.6.1) is not
   available; the pinned configuration is one A100-SXM4-80GB on FASRC with CUDA 12.9.1, GCC 12.2.0, SuiteSparse
   7.10.1 (`tools/reference/pinned_config.md`). Bitwise statements hold for this configuration. B9 (CHOLMOD 7.6.1
   failure on the rank-deficient test) cannot be reproduced here (7.10.1 passes, as the plan notes for 7.10.1).

6. **GPU budget.** Only the `gpu` partition (A100) is used, in allocations under one hour, without H200 jobs. The
   Phase 4b GPU gates (up to 12.6 H200-hours) are prepared as scripts with projected hours and are not submitted
   without approval.

7. **MATLAB.** No MATLAB on the build machines: the MEX gateway source is kept buildable through CMake
   (`-DCUADMM_BUILD_MATLAB=ON`, needs `find_package(Matlab)`) but is neither compiled nor tested here; this is
   reported as "not run", not as passed.

8. **Reference printing patch scope.** `print17.patch` raises all three `setprecision(10)` of the reference
   `main.cu` to 17 (history, JSON summary, `json_number`), not only the history line, so that the summary values
   can be compared too. It changes printing only.

9. **History column names of the new CLI.** The new history CSV uses the plan's names (`eta_p`, `eta_d`, `eta_g`,
   `sigma`); the comparison tool maps the legacy names (`errRp`, `errRd`, `relgap`, `sig`) onto them.

10. **Baselines are stored outside the repository.** The reference run directories (histories, solutions of the
    large problems are tens of MB) live in `$CUADMM_BASELINES/<config>/`; the repository keeps the case lists, the
    job scripts, the configuration record and the comparison reports (`docs/baselines/`), so the baselines can be
    regenerated from the recorded commits and patches.

11. **The hybrid-only policy `fixed_sgs_then_legacy_admm` is dropped in Pre-0.** The plan's policy table lists
    `fixed`, `legacy_adaptive`, `residual_balancing` and `balance_freeze` only, the hybrid mode is kept "as a pure
    recovery choice until Q5", and this policy's rule (the c23e67e plain-ADMM rule with factor 1.2, counted from the
    switch) is neither reference rule, so nothing could verify it bit for bit. Removing a name is an error, never
    an alias (plan principle 4); recorded in `CHANGES.md`. The hybrid mode remains available with every policy
    (the adaptive rules then count the global iteration, as they did at both references).

12. **`sigma_0` is not clamped for `residual_balancing`.** The option table says `penalty.sigma0` is "clamped into
    the bounds except in `legacy_adaptive`", but the policy is defined as the 2d8a755 rule and must be bitwise with
    it, and 2d8a755 never clamped `sigma_0` (its comment says so). The clamp is applied by `balance_freeze` (whose
    pseudocode in plan Section 3.5 starts with `clamp(sigma0, ...)`) and not by `residual_balancing`.
