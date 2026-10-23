# Implementation plan for Milestones 2 and 3 (Phases 5, 6a, 6b, 7, 10), updated after the Phase-4b gates

Authorized on 2026-10-22 ("proceed beyond the Phase 5 boundary ... complete Phases 5, 6a, 6b, and 7 in order,
followed by Phase 10 documentation and release preparation; Phases 8 and 9 deferred"). Sources: the plan
(`reorg-plan/cuadmm_reorg_plan.tex`, revision 2, sha256 `56e1366c...`), the developer prompt (`e0589bd5...`),
docs/PROGRESS.md, docs/DECISIONS.md (1-37) and docs/phase4b_gates_report.md. Branch `reorg` from aa5e4e2;
everything inside `cuADMM-reorg/`; the original tree, the submodule and `main` untouched; the final version is
pushed to the branch `cuADMM-2.0`.

## What the gate findings change

- **Penalty policy (DECISIONS 37).** `residual_balancing` stays the default through Milestones 2 and 3;
  `balance_freeze` keeps the screen's parameters (factor 1.2, halve_on_reversal, bias 4) as the documented
  candidate; no sigma tuning, no repetition of G0-G3. The freeze redesign (a freeze after the practical stage, or a
  freeze time and bias that scale with the problem; the gap-aware vote O8) is recorded as unresolved in the Phase
  10 documentation and is not implemented here.
- **Reference timings.** The gates measured the sm_90 RelWithDebInfo build 5-10% slower than the campaign's Release
  build on the same H200 runs (trto1 24.5 vs 23.3 s, buck1 70.1 vs 68.9 s, pendulum 686 vs 622 s). Bitwise
  acceptance keeps the pinned RelWithDebInfo configurations; Phase 10 adds a Release preset and the benchmarks of
  Phases 6a/7 run with it where only times are compared (recorded per run).
- **Benchmark protocol.** The comparator policy of every Milestone 2/3 benchmark is `residual_balancing` with its
  parameters explicit (as in the gates); fixed sigma = 1 where the plan asks for it. No H200: the GPU work of
  Milestones 2 and 3 fits the standing allowance (partition `gpu`, one A100 under one hour per job; `gpu_test` MIG
  under three hours), about 3 hours of A100 and 3 hours of MIG in total (per-commit bitwise comparisons, the
  Phase-6a projector benchmark, the Phase-6b memory and time checks, the Phase-7 backend benchmark).
- **Validation.** The independent CPU validator stays as it is (host reference measures per cone, kept separate
  from the GPU cones of Phase 5); B23's ranking is kept.

## Phase 5: cone layer (bitwise)

Scope (plan Sections 8-9): `src/cone/cone.h` (`Cone`, `ConeKind`, `Side`, `ConeMeasures`, `ScalingCompatibility`,
`HostReferenceMeasure`), `ConeProduct` (owns one object per cone kind, maps the svec slices, runs the projections,
joins them), `ConeRegistry` (input code -> factory; used by the readers, the MEX and the validator), `FreeCone`
(`project_dual`: 0; `project_primal`: identity), `NonnegativeCone` (max(z, 0)), `PsdCone` wrapping the engine's
three tiers unchanged (the code of `AdmmState::project_psd_tiers` moves into it; the tiers' buffers stay in the
engine until Phase 6). The cone step becomes `ConeProduct::project_dual(Xinput, S)`: the PSD cone unpacks its
blocks, projects, packs; the nonnegative and free cones project their slices with exact elementwise kernels
(identical results to today's fused scatter kernel, which handled them inside `matrices_to_vector`).
`Cone::measure` provides the per-block measures of X, S and C - A^T y (today's `SDPSolver::cone_measures`), and
`termination::Certificate` computes the certified lower bound with explicit per-block bounds (B24): a missing
bound for a block makes the bound "unavailable" in the result and the summary. Tests: cone property tests
(idempotence, Moreau decomposition, orthogonality, nonexpansiveness, membership) on random data for the three
cones and the product; the free-variable tests of plan Section 9.3 (free and nonnegative only, free only, rank
deficient free columns, free with a large PSD block through LOBPCG, warm start with S_u != 0, sGS-ADMM with free
blocks, certificate with per-block bounds and with a missing bound); the generator cases added to
`test/regression_problems.hpp` and `apps/cli/gen_problems.cpp`. Acceptance: 53 of 53 baseline cases bitwise, the
GPU and CPU test suites, on the MIG configuration (per-commit build). "PushBox_N=30_SOS against MOSEK" needs MOSEK
reference solutions that are not on the machines: recorded as not run.

## Phase 6a: PSD strategies, bit for bit

Scope (plan Section 10): `src/cone/psd/` with `batched/` (`JacobiBatchedEvd`, `BatchedEvd`, `StreamedEvd` on
today's resident buffers), `large/` (`LargeBlockState`, `RankAnalyzer`, `LargeBlockPolicy`, projectors `FullEvd`,
`Lobpcg`, `CompositeFilter`), `kernels/` (the FP16 composite filter with its coefficient table as data, the
Lanczos estimate, LOBPCG, precision and symmetrization helpers moved in from `third_party/psd_projection` under
the CUADMM checks, one copy of each helper: B10, B11, B15); the vendored copy removed once the parity tests pass.
B8: the Lanczos start vector from a counter-based generator seeded by (options.seed, block index), so the
composite path is deterministic. B13: the LOBPCG tolerance proportional to the stopping tolerance (constant
calibrated on the generator's large-block cases). Option `psd.large.final_product = gemm_compat` keeps today's
products in this phase. The benchmark `bench/psd_projector_bench` compares `CompositeFilter` with `FullEvd` at
n in {2000, 3000, 5000, 10000, 20000} (time and accuracy on synthetic spectra; A100, under one hour); then the
default warm-up `composite_fp16` for n >= 5000 is switched on (option default change, documented). Acceptance:
exact and LOBPCG paths bitwise (the baselines and the generator cases with large blocks); the composite path by
accuracy, fallback and end-to-end tests with and without the warm-up (its results change: deterministic start).

## Phase 6b: PSD memory model (within rounding)

Scope (plan Sections 10.4-10.6): `core/StreamPool`, `core/WorkspaceArena`; the large tier's arena (M of n_max^2
and W of the EVD workspace size) with the per-block pipeline (unpack, strategy, pack; SYRK on the smaller
eigen-side; LOBPCG and Lanczos buffers inside W, B7); small and medium tiers: the third buffer dropped (in-place
scaling and strided-batched GEMM, B19), chunked batched EVD from the memory budget and one workspace per stream
(B20); `Solver::setup` prints the memory plan and fails early above `psd.large.memory_budget`. Acceptance: the
histories within rounding (relative differences at the level of the projection error, `compare_runs.py --rtol`),
measured peak memory within 5% of the plan on the generator's large-block cases and the PLATO problems with the
largest blocks available locally, time per iteration not worse than Phase 6a (A100, Release).

## Phase 7: linear systems

Scope (plan Section 11): `src/linsys/` with `NormalSystem` (A A^T + eps I, lower triangle, assembled on the host
with CHOLMOD or on the device with cuSPARSE), `NormalOperator` (matrix-free), `NormalEquationSolver` (the
interface with the inexact hooks), `CholmodSolver` (today's code moved out of `SDPSolver`, dense mode included:
bitwise), `CudssSolver` (optional, `CUADMM_WITH_CUDSS`; the cuDSS 1.0 wheel from PyPI provides the library and
header on the build machines, so it is built and tested here), `DenseCusolverSolver` (Xpotrf/Xpotrs, Dsytrf/
Xsytrs), `Presolve` (dependent rows from the factorization pivots, consistency check of b, y mapped back;
off by default and in every bitwise comparison), the selection rule of Section 11.5 (`auto` = CHOLMOD with
today's dense rule until the benchmark reports), `linsys.autotune`, and `bench/linsys_bench` with the report
(analysis, factorization, per-solve time, peak memory, relative residual, end-to-end time to eta < 1e-4) on the
problems available locally (SPOT samples including the SOS form, the pendulum, PLATO; `neosfrbr25` and the larger
SPOT instances are not on the machines). Acceptance: CHOLMOD bitwise; cuDSS within 1e-10 relative residual per
solve and the same iteration counts within a few percent end to end.

## Phase 10: documentation and release preparation

README rewrite for the new project (build, options, formats, policies, validation, the solver's guarantees and
limits), Doxygen configuration, examples on the new options, migration notes (the old MEX signature, the API
defaults that changed, the policy names), CHANGES/DECISIONS/PROGRESS complete, the unresolved freeze redesign
(DECISIONS 37) and the deferred Phases 8 and 9 stated; a Release CMake preset; tag `v2.0.0-rc1` on the branch
`cuADMM-2.0`. Switching the original tree over stays the maintainers' decision (out of scope).

## Order, workflow and delegation

Phases run in order 5, 6a, 6b, 7, 10, each with small commits, the CPU tests on the CPU allocation, a clean
per-commit build and the bitwise (or tolerance) comparison on the MIG, and PROGRESS/DECISIONS/CHANGES updated
before the next phase. Independent pieces are delegated to subagents on their own branches and merged after
review: the cuDSS and dense cuSOLVER backends with the presolve (against the `NormalEquationSolver` interface,
unit tests on synthetic systems) during Phases 5-6, and the Phase-6a projector benchmark harness.
