# Task: build the reorganized cuADMM in `cuADMM-reorg/`

You are reorganizing cuADMM, a GPU solver for large multi-block conic programs: ADMM with PSD, nonnegative
and free blocks, PSD projection in three size tiers, and CHOLMOD normal-equation solves. Follow the plan in
`reorg-plan/cuadmm_reorg_plan.tex` (compiled: `reorg-plan/cuadmm_reorg_plan.pdf`).

Build the reorganized solver as a new, self-contained project in the folder `cuADMM-reorg/` at the repository
root. The existing code stays untouched: it is the reference you test against, and it keeps working while you
build the new one.

## Read first

1. The whole plan. The key sections are:
   - 3, the mathematics;
   - 6, the architecture, directory layout, naming and public API;
   - 7 to 11, the algorithm layer, cones, free variables, PSD projection and linear systems;
   - 12, problem input and output (the data formats);
   - 15, the options and their defaults;
   - 16, testing;
   - 17, the migration phases;
   - 18, the decisions;
   - Appendix B, which maps each current file to its new location.

   The decisions D1–D8 and Q1–Q9 are settled; do not reopen them without asking. O1–O9 (Section 18.2) are
   still open.
2. The current sources:
   - `include/cuadmm/solver.h` and `src/solver.cu` (the class `SDPSolver` does almost everything);
   - `src/main.cu` (the CLI) and `MATLAB/cuadmm_MATLAB.cu`;
   - `src/problem.cu` and `src/utils/io.cu` (the text reader);
   - `src/kernels/`, `include/cuadmm/`;
   - the submodule `psd_projection/src/` (composite filter, Lanczos estimate, LOBPCG);
   - `test/`.
3. The existing format converters, which define today's data conventions:
   - `experiments/2026-09-29_plato_cold_plain_admm_sigma/scripts/sdpa_to_cuadmm.py` (SDPA to TXT), including
     the checks your SDPA reader must keep;
   - `examples/sdpt3_to_txt.m`, `sedumi_to_txt.m`, `mosek_to_txt.m` and `examples/utils/` (MATLAB). Defects
     B28 and B29 describe where they are wrong.
   - In SPOT (`https://github.com/ComputationalRobotics/SPOT`):
     - `SPOT/MATLAB/mosek_standard_sdp_test_1.m` and `_2.m`, which build the MOSEK `prob` struct;
     - `recover_mosek_sol_blk.m`, which reads a MOSEK solution back.
4. Wherever the plan says "today", "as today" or "as coded", the current source is the specification. Read
   it; do not guess.

## Ground rules

- **Write only inside `cuADMM-reorg/`.** Do not edit:
  - the original `src/`, `include/`, `MATLAB/`, `examples/` or `test/`;
  - the `psd_projection` submodule;
  - the plan;
  - SPOT or any other repository. The SPOT changes of O9 are for SPOT's maintainers.

  Copy code into the new tree when you need it, for example the FP16 composite kernel, the Lanczos estimate
  and LOBPCG.
- **Self-contained build.** `cuADMM-reorg/` is its own CMake project. It must configure, build and test
  without including or linking anything from the parent tree.
- **Git.** Work on a feature branch (for example `reorg`) in small commits, each of which builds and passes
  its tests. `main` is protected: merge through pull requests, one or a few per phase, and never push to
  `main` directly.
- **No silent behavior changes.** Any change in results gets a new option value or policy name (plan
  principle 4) and an entry in `cuADMM-reorg/docs/CHANGES.md`.
- **Follow the plan's conventions.** Use the naming and coding rules of plan Section 6.2. Allocate nothing
  inside the iteration loop. Make `determinism = strict` the default.
- **Ask before you:**
  - change a settled decision;
  - run any H200 job, or any GPU job longer than one hour;
  - add a dependency the plan does not list;
  - touch anything outside `cuADMM-reorg/`.
- **Keep two logs.**
  - `docs/PROGRESS.md`: per phase, its status and the acceptance commands with their output.
  - `docs/DECISIONS.md`: questions, deviations from the plan, and the reason for each.

## Reference builds

Build these from `git worktree` checkouts. Use the same machine, compiler, CUDA, SuiteSparse and BLAS as for
your own build, because bitwise results hold only for one configuration.

| Reference | Used for |
|---|---|
| `9bb4ba5` | `legacy_adaptive`, the historical σ rule including `Monitor1` |
| `2d8a755` | `fixed` σ, and `residual_balancing` (guard `max(1,·)`, factor 2, σ ∈ [1e-3, 1e3]) |
| `2d8a755` without `CUADMM_PLAIN_ADMM_ONLY` | the sGS-ADMM path (plan Phase 4) |

**Bitwise** means the reference and your build agree exactly on two things:
- the per-iteration history: pobj, dobj, η_p, η_d, η_g and σ;
- the returned X, y and S.

This must hold on the generator test problems and on two real PLATO problems, `trto1` and `mater-1`. The
reference solvers read only the TXT format, so all bitwise comparisons use TXT input.

**Comparison script.** Write it first, for example `cuADMM-reorg/tools/compare_runs.py`. It runs a reference
and the new build with equivalent options and reports the first iteration and field that differ.

**Output precision.**
- The reference CLIs print histories with 10 digits. In the reference worktrees only, apply a patch that
  changes printing only, to 17 digits, and record it in the script.
- `--save-solution` already writes 17 digits.

## References for the readers (Phase 2b)

- **TXT.** Your reader must produce exactly the arrays that today's `Problem::from_txt` produces. That includes
  the legacy conventions: 0-based indices, and a bare size in `blk.txt` meaning `s` (the SPOT moment files
  use it). It never simplifies.
- **MOSEK `prob`.** `examples/SPOT/data/MOSEK/*.mat` and `examples/SPOT/data/TXT/*` are the same seven
  instances. The `.mat` files are MATLAB v5 and contain only `prob`.
  - Three of them have been checked against their TXT versions: PushBox N=30 moment and SOS, and PushT N=10
    moment. Read with the mapping of plan Section 12, they reproduce the TXT data to within rounding
    (relative differences ≤ 6·10⁻¹⁷). The mapping:
    - the scalar variables come first, as one `u` block, then the bar variables in order;
    - the lower-triangle entries of `bara` and `barc` go to svec, multiplied by √2 off the diagonal;
    - for a maximization, C = −c.
  - The moment files are minimizations and the SOS file is a maximization. The `.mat` files do not record
    this, so the tests pass the sense explicitly.
- **SDPA and SeDuMi.** PLATO publishes both formats at `https://plato.asu.edu/ftp/sdp/`:
  - `.dat-s.gz` files, which your reader reads directly;
  - `.mat.gz` files in SeDuMi format (`buck3.mat` stores `At`, `b`, `c`, `K`); gunzip them first.

  `sdpa_to_cuadmm.py` is the reference conversion for `.dat-s`. For `buck3`, the SeDuMi file was checked
  against it: read with the plan's mapping and the diagonal-block simplification, it gives the same blocks
  (`s 320`, `s 321`, `l 544`) and the same data to within rounding (relative differences ≤ 2·10⁻¹⁶).
- **SDPT3.** The repository has no SDPT3 files. Write test files yourself (for example with
  `scipy.io.savemat`) from instances you already have. Include an `s` entry that lists several block sizes.
- **Objectives.** Published objectives for the PLATO and Kocvara instances are in
  `experiments/2026-09-29_plato_cold_plain_admm_sigma/data/` (`published_objectives.json`,
  `reference_objectives.csv`).

## Order of work

The plan describes in-place edits of `main`. Here every phase happens inside `cuADMM-reorg/` and is checked
against the reference builds.

### Phase 0: skeleton and safety net

Set up:
- the CMake project and a test runner;
- the reference builds and the comparison script;
- fixed-σ baselines;
- the known-optimum problem generator and the tests copied from `test/`.

**Acceptance:** the comparison script reports zero differences when it compares each reference build with
itself, and the copied tests pass.

### Port without changing results

This covers plan Pre-0, Phases 1 to 5 (including 2b), 6a and the CHOLMOD part of 7. Bring the solver into the
target layout of plan Section 6.1 piece by piece, in the plan's order:

1. **Options and front-ends** (Phase 2): `SolverOptions`, the `Solver` facade, the CLI and MATLAB.
2. **Input and output formats** (Phase 2b, plan Section 12; fixes B27–B29). This step depends only on step 1
   and can run in parallel with steps 3 and 4.
   - **Readers** for:
     - the native HDF5 format (schema version 1 of Section 12);
     - SDPA `.dat-s` and `.dat-s.gz`;
     - SeDuMi, SDPT3 and MOSEK `prob` `.mat` files, in both v5 and v7.3;
     - the legacy TXT directory.

     Each reader produces one `ProblemData` with 64-bit sizes, fills the CSC arrays without a COO copy, and
     prints a conversion report.
   - **Format detection** from the extension and, for `.mat`, from the variables present. `io.format`
     overrides it.
   - **Objective sense.** MOSEK files take it from a `sense` variable saved next to `prob`, or from
     `io.sense`. The reader stops if neither is given.
   - **Exact simplifications** (`io.simplify`, on by default, each one reported): 1×1 and diagonal-only PSD
     blocks become `l` blocks. Never for TXT input.
   - **Writers:** solutions in each input's convention (the MOSEK output must work with SPOT's
     `recover_mosek_sol_blk`), the HDF5 problem writer, and `cuadmm-convert`.
   - **Optional dependencies.** HDF5 and matio are optional at build time (`CUADMM_WITH_HDF5`,
     `CUADMM_WITH_MATIO`). The build and the tests must pass with them on and off; format tests skip what is
     not compiled in.

   **Acceptance** (plan Phase 2b):
   - the TXT reader gives the same `ProblemData` as today;
   - the format tests of plan Section 16 pass;
   - PLATO instances read from either format reach their published objectives;
   - reading 10⁸ nonzeros from `.h5` is limited by disk bandwidth, not by parsing.
3. **Core** (Phase 3): `GpuContext`, `ExecContext`, buffer types, one set of error checks.
4. **Algorithm layer** (Phase 4):
   - `IterationScheme`;
   - the penalty policies `FixedPenalty` and `ResidualBalancingPenalty`;
   - `LegacyPenalty`, which reproduces 9bb4ba5 bit for bit, `Monitor1` included;
   - `balance_freeze`, implemented but not yet the default;
   - `KktEvaluator`, `StoppingRule`, `BestIterateTracker`;
   - observers with 17-digit histories.
5. **Cone layer** (Phase 5): `FreeCone`, `NonnegativeCone` and the PSD cone, with the free-variable
   requirements of plan Section 9.
6. **PSD strategies** (Phase 6a), on today's memory layout, with the FP16 composite kernel copied in. The
   FP32 and FP32-emulated filters are not ported.
7. **`CholmodSolver`** behind `NormalEquationSolver` (the bitwise part of Phase 7).

Include the Phase 1 fixes that leave results unchanged: B2, B5, B6, B12, B15–B17 and B21. B22 changes only
the composite path, which is accepted on accuracy.

**Acceptance** for every step except step 2, which has its own: bitwise against the reference for each
penalty policy, and for sGS-ADMM as plan Phase 4 specifies. The composite path is the exception: its Lanczos
start is random today (B8), so it is checked for accuracy, fallback and end-to-end results instead.

### Changes to results, one labelled step each, with tolerance-based acceptance

- **Phase 4b.** Tune `balance_freeze` and make it the default. This can start once Phase 4 is done. Run the
  CPU screen on the σ-study replica first (ask the maintainers for it). The GPU gates come after, and need
  approval.
- **Phase 6a.**
  1. Make the Lanczos start deterministic (B8).
  2. Run the benchmark of the FP16 filter against `FullEvd` at n ∈ {2000, 3000, 5000, 10000, 20000}.
  3. Switch on the default warm-up: `composite_fp16` for blocks with n ≥ 5000, none for smaller blocks.
- **Phase 6b.** Add the large-block memory arena, the SYRK final product, and the workspace fixes for small
  and medium blocks.

  **Acceptance:** histories agree within rounding, and the measured peak memory is within 5% of the
  setup-time memory plan.
- **Phase 7.** Add `CudssSolver` (an optional dependency), `DenseCusolverSolver`, presolve and the benchmark
  report.

Phases 8 and 9 are deferred. CBF input and the second-order cones of SeDuMi, SDPT3 and MOSEK data come with
Phase 9; until then the readers reject them with a clear message.

**Out of scope:** switching over, that is, replacing the original tree with `cuADMM-reorg/` and removing the
submodule. The maintainers decide that once everything passes.

### Milestone 1

Phase 0, the port through Phase 4 (including Phase 2b), and the Phase-4b CPU screen. **Stop at this point and
report** before you start Phase 5.

## Environment

- **CUDA:** 12.8 or later (12.9 on the reference workstation). `cusolverDnXsyevBatched` needs 12.8.
- **Libraries:**
  - SuiteSparse/CHOLMOD is required.
  - HDF5, matio and zlib are used by the readers from Phase 2b; HDF5 and matio are optional at build time.
  - cuDSS is optional and only needed from Phase 7.
- **Language:** C++17. Python 3 with NumPy, SciPy and h5py for the tools and for generating test data.
- **Hardware:** the reference workstation has 2× RTX 6000 Ada (48 GiB each). The solver uses one GPU,
  chosen by `gpu.device`.
- **CPU-only tests:** tests that do not need a GPU must run on the CPU. These include the penalty policies on
  scripted residual sequences, option parsing, and the readers and writers.
- **Data:**
  - The SPOT samples are in `examples/SPOT/data/`, in MOSEK and TXT form.
  - The PLATO and Kocvara instances are public: `https://plato.asu.edu/ftp/sdp/` and
    `https://web.mat.bham.ac.uk/kocvara/pennon/problems.html`.
  - Ask the maintainers for the larger SPOT and Mittelmann problems, the MOSEK reference solutions and the
    σ-study replica.

## Stop and ask when

- an acceptance check fails and you cannot explain the first iteration that differs;
- the plan and the current code disagree about current behavior (the code wins for "today", but report it);
- an open decision blocks you (plan Section 18.2, O1–O9);
- a step needs more time, GPU hours or memory than the plan budgets.

## At the end of each phase, report

- what changed, and in which files;
- the acceptance commands and their output;
- time per iteration and peak memory, compared with the reference build;
- open questions.
