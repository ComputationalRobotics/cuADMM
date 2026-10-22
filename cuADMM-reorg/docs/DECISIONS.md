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

13. **One build directory per verified commit.** The Phase 1 GPU run used a build directory that I had rebuilt
    with later changes while its tests were running, so its CLI tests exercised the next commit's binary (the
    comparison runs were identical either way, since the new CLI accepts the old flags). From then on every commit
    that produces GPU evidence is built from its own worktree into `$HOME/cuadmm-builds/reorg-<commit>`
    (`tools/slurm/build_and_compare.sh`), and `docs/PROGRESS.md` names the commit of each run.

14. **`algorithm.tau` is restricted until Phase 4.** The ported engine hard-codes 1.618 (ADMM) and 1.95 (sGS phase);
    the facade rejects any other value with a message rather than silently ignoring the option. Phase 4's
    `multiplier_step` takes `tau` as a parameter.

15. **The reference policies ignore the hysteresis options.** `penalty.win_ratio`, `.guard`, `.residual_weight` and
    `.schedule` belong to `balance_freeze`; `legacy_adaptive` and `residual_balancing` keep their fixed values
    (bitwise with 9bb4ba5 and 2d8a755). An explicit value that contradicts a reference rule (a win ratio other than
    1.35, a changed schedule, `guard` off for `residual_balancing`, a factor other than 2 for `legacy_adaptive`) is
    rejected, so that no option silently changes meaning.

## 2026-10-21 (Phase 2b)

16. **The SDPA sense is `max`.** The plan maps SDPA's dual (D) `max <F_0, Y>` onto cuADMM's primal `min <-F_0, Y>`
    and asks for objective values "in the file's sense". `ProblemData::sense = Maximize` for `.dat-s` input makes
    the reported `pobj` the published SDPA value (e.g. 607.6055 for buck3), while the TXT conversions of the PLATO
    experiments stay minimizations (their `pobj` is the negative). `tools/baselines/cases_formats.json` records the
    sign per case so that the GPU acceptance compares the right number.
17. **The SeDuMi and SDPT3 readers symmetrize with `value * sqrt2/2` per mirrored half.** The two halves are merged
    by summation, which gives exactly `value * sqrt2` for a symmetric matrix (scaling by 2 is exact), so SeDuMi
    files converted from SDPA data reproduce the SDPA reader bit for bit (buck3: 0 difference, better than the
    2e-16 of the plan).
18. **Untouched PSD blocks are simplified too.** A PSD block that neither A nor C touches satisfies "data on the
    diagonal only" vacuously and becomes an `l` block (reported as "untouched by A and C"); this is exact and
    avoids an eigendecomposition per iteration. The PLATO converter kept such blocks as `s` (reported); the
    reference TXT files of trto1, mater-1 and buck3 have none, so the bitwise comparisons are unaffected.
19. **Duplicate `(i, j, k, l)` entries of MOSEK's `bara` / `barc` are summed** (as `sparse()` does in
    `convert_mosek2sedumi.m` and as MOSEK's block-triplet interface documents); the count is reported. SDPA
    duplicates are rejected (as the converter does); SeDuMi/SDPT3 duplicates cannot occur in a MATLAB sparse
    matrix except through the mirroring, which is the intended merge.
20. **MOSEK layout of the problem**: one `u` block (free scalars), one `l` block (nonnegative, shifted, negated
    and ranged scalars in variable order, then the bound slacks, then the row slacks in row order), then the bar
    variables; the extra constraints (second rows of ranged rows, bound rows of ranged variables) come after the
    file's rows. Fixed variables (`blx = bux`) are eliminated. Free rows are dropped. The mapping is kept in
    `SourceInfo::mosek_*` so that the writer restores `xx` and `y` of the file; `xc`, `skc`, `skx` are not
    written (not used by SPOT's `recover_mosek_sol_blk`).
21. **MOSEK duals of a maximization.** MOSEK's dual variables of a maximization problem have the opposite sign and
    its slacks are non-positive (`bars = -S`, `y = -y_cuADMM`, `slx = min(t, 0)`, `sux = min(-t, 0)`); the
    writer follows that convention. Open question for the maintainers: SPOT's SOS pipeline uses `barx`, `xx`
    and the objective values only, so the dual sign convention has not been checked against a MOSEK run.
22. **`io.sense` on formats whose sense is fixed** (SDPA, SeDuMi, SDPT3, TXT, and `.h5` with a stored sense) is
    accepted when it agrees and rejected when it contradicts, instead of silently changing the reporting.
23. **The solution file of a non-TXT convention**: `--save-dir` names the file when it carries the convention's
    extension (`.h5`, `.mat`, `.ini-s`), otherwise a directory that receives `solution.<ext>` (and `solution.json`
    for SDPA). For `.h5` input the group `/solution` is added to (or replaced in) the problem file itself when it
    is named; `certificate.txt` exists only in the TXT convention (as today).
24. **The `.mat` solution files keep the version of the input** (v7.3 for a v7.3 input, v5 otherwise), compressed
    with zlib; `make_sdpt3_testfiles.py` writes v7.3 files with h5py to test matio's v7.3 path independently.
25. **HDF5 and matio are found by small modules of our own** (`cmake/FindHDF5Lib.cmake`, `cmake/FindMatio.cmake`,
    `HDF5_ROOT` / `MATIO_ROOT`) instead of CMake's `FindHDF5`, which runs the `h5cc` wrapper; the build options are
    tri-state (`AUTO` = on when found, `ON` = required, `OFF`). zlib (`.dat-s.gz`) is a third such option.
26. **The conversion report is printed by the front-ends**, not by the library (plan: "each reader prints a
    conversion report"): `read_problem` is silent and fills `source.report`; the CLI and `cuadmm-convert` print the
    lines and the summary JSON stores them (`conversion_report`). The legacy TXT reader keeps its own printing.
27. **"Relative difference" of the cross-format checks** is `max |a - b| / max |b|` over the entries of At, b and
    C (the 16-decimal TXT files of SPOT limit the agreement to about 5.6e-17 in this measure, matching the plan's
    6e-17; the elementwise relative difference of tiny entries is larger, up to 3e-15, because of the TXT rounding).
