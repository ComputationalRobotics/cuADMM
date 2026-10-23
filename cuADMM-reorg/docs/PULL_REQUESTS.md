# Pull requests of Milestone 1

`main` is protected and the GitHub CLI is not available on the build machines, so the pull requests are opened on
the web from the pushed branches (`reorg`, `reorg-io`). The plan asks for "one or a few per phase"; the branch is
linear apart from the two merges of `reorg-io`, so the simplest sequence is one PR per range below, each opened
against the previous one's head (or a single PR of `reorg` onto `main` with these sections as its description).
Every commit builds and passes its tests; the GPU evidence per commit is in docs/PROGRESS.md.

## PR 1: Phase 0 (0d91b67, 7411625)

Skeleton of `cuADMM-reorg/` as a self-contained CMake project next to the original tree (untouched); the vendored
subset of the PSD-projection submodule; the comparison tool `tools/compare_runs.py` (full-precision histories and
returned X, y, S; first differing iteration and field); the reference worktrees and builds of 9bb4ba5, 2d8a755 and
2d8a755 without `CUADMM_PLAIN_ADMM_ONLY` (printing-only precision patches in `tools/reference/`); the pinned
configurations; 53 fixed-sigma, legacy_adaptive, residual_balancing and sGS baselines; the copied generator tests.
Evidence: self-checks 6 of 6, baselines 53 of 53 identical on both pinned configurations.

## PR 2: Pre-0 and Phase 1 (75eb6ed, f8de01f)

Penalty policy names restored to the references (`legacy_adaptive` = 9bb4ba5 with Monitor1, `residual_balancing` =
2d8a755); the one-line fixes of the plan (B2, B5, B6, B12, B16, B17, B21, B22, Q3, P4) with results unchanged on the
exact paths. Evidence: 49 of 49 identical, tests 99 of 101 (two formatting expectations fixed later in d6720b6).

## PR 3: Phase 2 (b44596b)

`SolverOptions` as the single source of defaults, the option table (CLI flags and aliases, documentation),
`Solver` facade, `ProblemData` with 64-bit CSC, the CLI on the option table, the MEX gateway (source only: no
MATLAB on the build machines). Evidence: 49 of 49 identical, tests as above.

## PR 4: Phase 2b (7b28aac, eb3ee06, 9907cda, df2a112, merges b64cc18 and f10e149)

Readers for HDF5 (native format), SDPA (`.dat-s`, `.dat-s.gz`), SeDuMi, SDPT3 and MOSEK `prob` `.mat` files;
solution writers in each convention; `cuadmm-convert`; exact simplifications; three build configurations
(HDF5/matio/zlib AUTO/ON/OFF). An independent review against the plan's mappings found three correctness bugs and
one build break (HDF5 1.10), fixed with tests. Evidence: 51 CPU tests; on the GPU trto1 and mater-1 converge from
TXT, SDPA and HDF5 with objectives within 1.4e-4 of validated references, and the SDPA, HDF5 and SeDuMi histories
are bit for bit the TXT histories; the summary JSON of the new CLI parses (4640e8a).

## PR 5: Phases 3 and 4 (59e943a, d6720b6, 36e8552)

Thin GPU core (one set of checks, device buffers, `GpuContext`); the algorithm layer (`AdmmState`,
`IterationScheme`, the penalty policies `fixed`, `legacy_adaptive`, `residual_balancing`, `balance_freeze`,
stopping rules, best-iterate tracking, observers) with the legacy engine's `solve()` as a wrapper over the new
loop; B23 (best iterate by the maximum DIMACS error when validations exist). Evidence: 53 of 53 identical on both
configurations (sGS, hybrid and the sGS-phase adaptive rule included), tests 101 of 101.

## PR 6: Phase 4b (92dca50, 353b96d, 2625dfb, 4f23c27)

The validated CPU replica of the sigma study extended with `balance_freeze` (cross-checked bit for bit against the
C++ policy on 15 parameter cases), the screen driver and analysis, the screen's results (513 runs, 28.8 core-hours):
the freeze iteration never acts; factor 1.2 + halve_on_reversal + freeze bias 4 loses no problem at sigma_0 0.5
and 1 against fixed sigma = 1 and the old rule, keeps the variation bounded, has no spike, ties the time gate,
and loses vibra1 at sigma_0 = 2 (DECISIONS 36). The default policy is unchanged; the GPU gates are generated and
await approval.

---
Each description ends with: 🤖 Generated with [Claude Code](https://claude.com/claude-code)

# Pull requests of Milestone 2

The same arrangement: the branch `reorg` is linear apart from the merges of `reorg-bench6a` (4dbb110) and
`reorg-linsys` (364c87e); the release branch `cuADMM-2.0` points at its tip with the tag `v2.0.0-rc1` (pushed with
the SSH remote; no push to `main`). One PR per range below, or one PR of `cuADMM-2.0` onto `main` with these
sections as its description. Every commit builds and passes its tests; the GPU evidence per commit is in
docs/PROGRESS.md.

## PR 7: Phase 5, the cone layer (1cca25e, a9521e2)

`src/cone/`: Cone, ConeRegistry, ConeProduct, FreeCone, NonnegativeCone, SliceTable, PsdCone on the engine's tiers,
`termination/Certificate` (per-block bounds, B24); AdmmState and the facade on the product; the free-variable
tests of plan Section 9.3 and the new generator cases. Evidence: 53 of 53 identical, 118 tests.

## PR 8: Phase 6a, the PSD strategies (14178fa, 684b99f, 72347eb, merge 4dbb110)

The composite / Lanczos / LOBPCG kernels moved in with the deterministic Lanczos start (B8, B10, B11, B15), the
honoured LOBPCG options (DECISIONS 45); the projector classes (JacobiBatchedEvd, BatchedEvd, StreamedEvd, FullEvd,
Lobpcg, CompositeFilter), RankAnalyzer and LargeBlockPolicy with the per-block warm-up threshold and the honoured
`psd.*` options (DECISIONS 46-48); the projector benchmark of `reorg-bench6a` and the warm-up default
`composite_fp16` for blocks of at least 5000 (DECISIONS 54). Evidence: 53 of 53 identical at 14178fa and 684b99f,
130 tests; no baseline case has a block of 5000.

## PR 9: Phase 6b, the PSD memory model (02991f1, 4cd46a5, 974e373, 8998801)

WorkspaceArena and StreamPool; LOBPCG and Lanczos on caller-provided workspaces (B7); LargeTier (one arena region,
the SYRK products, the warm starts), BatchedTier (B19, B20, the chunked batched EVD), PsdCone owning the tiers, the
memory plan at setup with the budget check and the reported peak; the compat products of main as options and in
the C++ API (DECISIONS 49-53); the comparison tooling for extra flags and tolerances. Evidence (at 364c87e): 53 of
53 identical with the compat products; within rounding with the defaults (47 of 53 at every iteration, the LOBPCG
cases at the level of its tolerance); the peak within 0.2% of the plan; A100 time per iteration 0.85-0.99 of 6a;
138 tests.

## PR 10: Phase 7, the linear systems (merge 364c87e of `reorg-linsys`, ed9d598)

The backends of `reorg-linsys` (NormalSystem, DenseCusolverSolver, CudssSolver, presolve, DECISIONS 40-44) and
their integration: CholmodSolver wrapping the engine's y-step bit for bit, the backend selection (`linsys.backend
auto / cholmod / dense / cudss`), the presolve wiring, the benchmark `cuadmm_linsys_bench` and its report
(DECISIONS 55-57). Evidence: 53 of 53 identical with the compat products at ed9d598; dense and cuDSS end to end
with the same iteration counts as CHOLMOD; 142 tests.

## PR 11: Phase 10, the documents and the release (0a29158, 8e483f8 and the final docs commit)

README for 2.0, `docs/MIGRATION.md`, `docs/OPTIONS.md` regenerated, `examples/`, `CMakePresets.json`,
`docs/Doxyfile` with the `cuadmm_docs` target, PROGRESS / CHANGES / DECISIONS complete with the Milestone 2
status, the open items (the freeze redesign of DECISIONS 37, `linsys.autotune`, the warm-up threshold) and the
deferred Phases 8 and 9; the tag `v2.0.0-rc1` on `cuADMM-2.0`.
