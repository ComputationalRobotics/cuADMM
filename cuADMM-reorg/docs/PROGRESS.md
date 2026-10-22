# Progress log of the cuADMM reorganization (Milestone 1)

Per phase: status, what was done, the acceptance commands with their output, and the checks still pending.
Conventions: "pending" means not yet run; a check that cannot be run here (no MATLAB, no H200, no second
configuration) is marked "not run: <reason>", never "passed".

Documents followed (sha256 of the versions read, both from commit c23e67ef58146e553e8c688fdbfbb675533677a3 of
ComputationalRobotics/cuADMM, which was origin/main on 2026-10-21):

- `reorg-plan/cuadmm_reorg_plan.tex` (revision 2, October 20, 2026): `56e1366cf3c3c48a9f87c1f64bf04c452cdbebf38466f54c25b83b82ddf0688c`
- `reorg-plan/developer_prompt.md`: `e0589bd54cc400d2bf1fb7839130493af2908e41304393b54346f653ab06ba86`

Repository state at the start: `origin/main` = c23e67e (no newer commit); submodule `psd_projection` at b566590.
Work happens on branch `reorg`, started from c23e67e, inside `cuADMM-reorg/` only.

Machine: FASRC Cannon. Builds on CPU nodes (`sapphire`/`shared`, 16 cores); GPU runs on one A100-SXM4-80GB of
partition `gpu` in allocations under one hour (no H200 used; approval would be needed). Full pinned configuration:
`tools/reference/pinned_config.md`.

## Phase 0: skeleton and safety net

Status: skeleton built; GPU acceptance pending (see below).

Done (2026-10-21):

- Reference worktrees `$HOME/cuadmm-worktrees/{ref-9bb4ba5,ref-2d8a755,ref-2d8a755-sgs}` (submodule b566590), with
  the printing-only patch `tools/reference/print17.patch` and, for the sGS build, `tools/reference/sgs_no_plain_only.patch`;
  built with `$HOME/cuadmm-tools/build_ref.sh` into `$HOME/cuadmm-builds/<name>/` (sm_80, RelWithDebInfo):

      === ref-9bb4ba5 Wed Oct 21 15:42:23 EDT 2026   build exit code: 0
      === ref-2d8a755 Wed Oct 21 15:43:27 EDT 2026   build exit code: 0
      === ref-2d8a755-sgs Wed Oct 21 15:44:24 EDT 2026   build exit code: 0
      (cuobjdump --list-elf libcuadmm_lib.so: sm_80 cubins; CMakeCache: -gencode arch=compute_80,code=sm_80, RelWithDebInfo)

- `tools/compare_runs.py` (written first, as the developer prompt asks): runs two builds with equivalent options on
  one TXT problem (`run`), one build twice (`selfcheck`), a JSON list of cases (`batch`), or compares two run
  directories (`compare`); reports the first differing iteration and field of the history (pobj, dobj, eta_p, eta_d,
  eta_g, sigma), the first differing entry of X, y, S, and per-field counts. Checked on hand-made directories
  (identical -> exit 0; a one-ulp difference in dobj at iteration 2 and in S[2] -> reported, exit 1).
- `cuADMM-reorg/` CMake project (`CMakeLists.txt`): library `cuadmm` from the verbatim copy of `src/`, `include/`
  (c23e67e) in `src/legacy/`, the submodule subset in `third_party/psd_projection/` (shared library
  `cuadmm_psd_kernels`), `cuadmm_exe` (copied CLI), `cuadmm_gen_problems` (writes the generator problems as TXT),
  `cuadmm_tests` (copied `test/tests.cpp` and the 14 test headers it includes, `regression_problems.hpp`,
  `solver_setup.hpp`, `test/data/`), optional MATLAB MEX target (`matlab/`, not built here: no MATLAB).
  Build on the CPU node (`tools/build.sh $HOME/cuadmm-builds/reorg-phase0`):

      [ 23%] Built target cuadmm_gen_problems ... [ 39%] Built target cuadmm_psd_kernels
      [ 90%] Built target cuadmm  [ 97%] Built target cuadmm_exe  [100%] Built target cuadmm_tests
      build exit code: 0

- Baseline case lists `tools/baselines/cases_{fixed,legacy_adaptive,residual_balancing,sgs}.json` (21 + 10 + 10 + 8
  cases: the 13 generator problems, `trto1`, `mater-1`) and the GPU job scripts `tools/slurm/phase0_baselines.sh`
  (configuration record, problem generation, self-checks, baselines) and `tools/slurm/run_tests.sh`.

Acceptance (developer prompt): the comparison script reports zero differences when it compares each reference
build with itself, and the copied tests pass.

GPU configuration used: `a100-40gb-mig3g20gb_cuda12.9_gcc12.2_ss7.10.1_t8` (one MIG 3g.20gb slice of an
A100-SXM4-40GB on partition `gpu_test`, 8 threads; the full-A100 allocation on `gpu` stayed queued behind ~1000
jobs). Record: `$HOME/cuadmm-baselines/a100-40gb-mig3g20gb_cuda12.9_gcc12.2_ss7.10.1_t8/config.txt`.

- Self-checks (each reference against itself; `tools/slurm/phase0_baselines.sh`, 2026-10-21 16:21-16:33, Phase 0
  build of commit 7411625): 6 of 6 identical (9bb4ba5 fixed and legacy_adaptive on PSD_Free, 2d8a755 fixed and
  residual_balancing on PSD_Free and fixed on trto1, 2d8a755-sgs sGS on PSD_Free).
- Baselines (`$CUADMM_BASELINES/<config>/{fixed,legacy_adaptive,residual_balancing,sgs}/<case>/a`): 55 of 55
  comparisons identical, 0 different, 0 failed runs:

      fixed (21 cases, 2d8a755 against the Phase 0 copy of c23e67e): all identical, 800-20000 iterations
      legacy_adaptive (10 cases, 9bb4ba5 run twice): all identical (6000-iteration runs reach the Monitor1 region)
      residual_balancing (10 cases, 2d8a755 run twice): all identical
      sgs (8 cases, 2d8a755 without CUADMM_PLAIN_ADMM_ONLY run twice): all identical

  A first attempt with `--tol 0` failed because the reference CLI rejects a zero tolerance; the fixed-iteration cases
  use `--tol 1e-300` (never reached; the hidden sGS tau reduction never fires).
- Copied tests (`tools/slurm/run_tests.sh`, build of 7411625): 87 tests, 86 passed, 1 failed:
  `FixedSigma.LegacyAdaptiveAndPlainAdmmUnchanged`, the golden test that fails by construction on `main` (plan B14:
  the guard of the October 13 rule blocks the first change of the 9bb4ba5 golden). Pre-0 restores the rule and the
  test passes again (see Phase 1 below). Runtime 87 s.

Phase 0 is accepted on this configuration. The same baselines are to be regenerated on the full A100 when the
`gpu` allocation is granted (one more configuration, same scripts).

## Pre-0 and Phase 1 (inside `cuADMM-reorg/`)

Status: implemented (commits 75eb6ed Pre-0, f8de01f Phase 1); GPU evidence below.

- Pre-0: `legacy_adaptive` = 9bb4ba5, `residual_balancing` = 2d8a755 with `penalty.factor`, policy parameters and
  build commit in the summary, hybrid-only policy removed (`docs/CHANGES.md`, `docs/DECISIONS.md` 11-12).
- Phase 1: B2, B5, B6, B12, B16, B17, B21, B22, Q3/B1, P4 (`docs/CHANGES.md`).
- Comparison of the build of f8de01f (first run from `$HOME/cuadmm-builds/reorg-dev`, 16:26-16:34) with the
  references on every baseline case: 49 of 49 identical (fixed 21 against 2d8a755, legacy_adaptive 10 against
  9bb4ba5, residual_balancing 10 against 2d8a755, sGS 8 against 2d8a755-sgs), 0 different.
- Tests of that run: 101 tests, 98 passed; the golden test passes (B14 fixed). The 3 failures were CLI tests that
  ran against the Phase 2 binary, because the build directory had been rebuilt during the run (DECISIONS 13); the
  clean rerun of f8de01f from its own worktree and build directory (`tools/slurm/build_and_compare.sh f8de01f`) is
  recorded below when it completes.

## Phase 2: options and front-ends

Status: implemented; GPU evidence recorded below (b44596b: 49 of 49 identical on the MIG configuration).

- `include/cuadmm/{options,problem,result,solver,io,cuadmm}.h`, `src/options/options.cc` (option table, setter,
  getter, validation, documentation), `src/io/{problem_data,txt,io}.cc`, `src/report/writers.{h,cc}`,
  `src/solver/solver_facade.cu` (the facade over the ported engine), `apps/cli/cuadmm.cu` (the CLI on the table,
  with today's flags as aliases), `matlab/cuadmm_mex.cu` and `matlab/+cuadmm/solve.m` (not built: no MATLAB),
  `docs/OPTIONS.md` (generated by `cuadmm_exe --print-options markdown`).
- CPU tests (`cuadmm_cpu_tests`, run on the CPU node): `Options.*` 4 tests pass (table consistency and round trip of
  every option, parsing, validation, documentation coverage).
- Acceptance (plan): bitwise for equivalent options; the option table generated into the documentation. The GPU run
  of the Phase 2 commit against the references (`tools/slurm/build_and_compare.sh <commit> phase2 <config> reorg`)
  is recorded below when it completes.

## Phase 2b: input and output formats

Status: implemented on branch `reorg-io` (commits listed below); CPU acceptance run; the GPU acceptance (published
objectives from every format, `tools/baselines/cases_formats.json`) is for the parent run.

Done (2026-10-21), plan Section 12, developer prompt "References for the readers (Phase 2b)":

- Readers: native HDF5 schema 1 (`src/io/h5.cc`, also the writer, `/solution` and `/start`), SDPA sparse
  `.dat-s` / `.dat-s.gz` with the checks of `sdpa_to_cuadmm.py` (`src/io/sdpa.cc`, `text_source.cc`), SeDuMi,
  SDPT3 and MOSEK `prob` `.mat` through matio, v5 and v7.3 (`sedumi.cc`, `sdpt3.cc`, `mosek.cc`, `mat_common.cc`);
  the legacy TXT reader untouched. Every reader fills the CSC of At directly through `CscBuilder` (in-place bucket
  sort, duplicates merged or rejected per format, zeros dropped; `csc_builder.cc`), reports what it did
  (`source.report`), keeps the DIMACS kinds and the mapping to the file's blocks (`original_block_*`), and applies
  the exact simplifications of `io.simplify` (`simplify.cc`).
- Detection (`io.cc`): extension, file signature, and the variables of a `.mat` file; `io.format` overrides.
- Writers: `write_solution` in the input's convention or `io.save_format` (`solution_writers.cc`), the TXT problem
  writer (`txt_writer.cc`), `write_problem` (HDF5, chunked, optional deflate, `source_sha256` by `sha256.cc`).
- Tools: `cuadmm-convert` (`apps/cli/cuadmm_convert.cc`), `cuadmm_io_bench` (`bench/io_read_bench.cc`),
  `tools/io/make_sdpt3_testfiles.py` (the SDPT3 test files of `test/data`, v5 by scipy and v7.3 by h5py).
- CMake: `CUADMM_WITH_HDF5`, `CUADMM_WITH_MATIO`, `CUADMM_WITH_ZLIB` (AUTO/ON/OFF; `cmake/FindHDF5Lib.cmake`,
  `cmake/FindMatio.cmake`); `--formats` of the CLI lists what is compiled in; absent readers fail with a message.
- Tests (CPU, `test/io_*_test.hpp` included from `test/cpu_tests.cpp`): builder/svec/sha256, SDPA edge cases and the
  PLATO instances against the reference TXT conversions, simplifications and the mapping back, TXT and SDPA
  writers, HDF5 round trips (problem, solution, warm start, malformed files, detection), hand-made SeDuMi / SDPT3 /
  MOSEK files in v5 and v7.3 (A vs At and N = m, asymmetric PSD storage, K.q rejected, an `s` entry with several
  sizes, `l`/`u` blocks, MOSEK bounds and ranges, missing sense, maximization, cones rejected), the Python-written
  SDPT3 files, buck3 SeDuMi vs SDPA, the three SPOT MOSEK files vs TXT, and the MOSEK/SeDuMi/SDPT3 solution writers
  read back through matio. Tests of a format that is not compiled in skip (GTEST_SKIP).

Data prepared (read-only inputs untouched): `$HOME/cuadmm-data/plato_sdpa/` (`buck1`, `buck3`, `trto1`, `trto2`,
`mater-1`, `mater-2` `.dat-s` extracted from the Kocvara zips, plus `trto1.dat-s.gz`), `$HOME/cuadmm-data/plato_sedumi/buck3.mat`
(from `https://plato.asu.edu/ftp/sdp/buck3.mat.gz`; `trto1.mat.gz` and `mater-1.mat.gz` do not exist there: 404),
`$HOME/cuadmm-data/plato_h5/` (conversions for the GPU cases).

Builds (CPU node holy8a26208, job 54419762):

    BUILD_JOBS=8 tools/build.sh $HOME/cuadmm-builds/reorg-io -DCUADMM_WITH_HDF5=ON -DCUADMM_WITH_MATIO=ON -DCUADMM_WITH_ZLIB=ON \
        -DHDF5_ROOT=$HOME/cuadmm-io-env -DMATIO_ROOT=$HOME/cuadmm-io-env -DZLIB_ROOT=$HOME/cuadmm-io-env
    -- cuADMM-reorg: HDF5 TRUE (1.14.6), matio TRUE (1.5.30), zlib TRUE (1.3.2)   ...   build exit code: 0
    BUILD_JOBS=6 tools/build.sh $HOME/cuadmm-builds/reorg-io-nodeps -DCUADMM_WITH_HDF5=OFF -DCUADMM_WITH_MATIO=OFF -DCUADMM_WITH_ZLIB=OFF
    -- cuADMM-reorg: HDF5 FALSE (), matio FALSE (), zlib FALSE ()   ...   build exit code: 0

Acceptance (CPU node holy8a26208, 2026-10-21 21:19, build directories above):

- `./cuadmm_cpu_tests` with HDF5 + matio + zlib: `33 tests from 8 test suites ran. [ PASSED ] 33 tests.`
  Without the libraries (`reorg-io-nodeps`): `33 tests ... [ PASSED ] 22 tests. [ SKIPPED ] 11 tests` (the gz, HDF5
  and .mat tests skip; the "reader not compiled in" messages are checked by hand below).
- The TXT reader gives the same ProblemData as today: `src/io/txt.cc` is unchanged (the reader body is the Phase 2
  one over the verbatim `Problem::from_txt`); `IoSimplify.TxtWriterRoundTrip` and the PLATO comparisons read it.
- Cross-format checks printed by the tests (`max|diff| / max|ref|` over the entries of At, b and C; 0 = bitwise):

      trto1 / mater-1 / buck3 .dat-s vs the reference TXT conversion (sdpa_to_cuadmm.py):  0 / 0 / 0  (bitwise, same blocks and DIMACS kinds)
      trto1 .dat-s.gz vs .dat-s:                                                            0 (bitwise)
      buck3 SeDuMi .mat vs .dat-s (plan: <= 2e-16; blocks s 320, s 321, l 544):             0 (bitwise; the 544 block is reported as simplified)
      PushBox_N=30_MOMENT MOSEK vs TXT (plan: <= 6e-17):   At 5.551e-17, b 0, C 5.551e-17  (374258 entries; most entries bitwise, the rest limited by the 16-decimal TXT)
      PushBox_N=30_SOS (maximization) MOSEK vs TXT:        At 3.925e-17, b 4.163e-17, C 0
      PushT_N=10_MOMENT MOSEK vs TXT:                      At 5.551e-17, b 0, C 4.930e-17
      HDF5 round trips (plain, deflate 4, after /solution was added, trto1 .dat-s -> .h5): 0 (bitwise, metadata equal)
      SDPT3 v5 (matio) vs v7.3 (matio), and the scipy-written v5 / h5py-written v7.3 files of test/data: same arrays

- Formats listed by the CLI: `txt sdpa sdpa.gz h5 sedumi sdpt3 mosek` (with the libraries) and `txt sdpa`
  (without); a `.mat`, `.h5` or `.dat-s.gz` input of the second build fails with
  "reading .mat files (SeDuMi, SDPT3, MOSEK) needs matio, which is not compiled into this build (CUADMM_WITH_MATIO)",
  "the native HDF5 format is not compiled into this build (CUADMM_WITH_HDF5)", "... is gzip-compressed, and this
  build has no zlib (CUADMM_WITH_ZLIB): decompress the file first".
- HDF5 read rate (`cuadmm_io_bench`, node-local /tmp of holy8a26208, 1e8 nonzeros: 100 PSD blocks of side 1000,
  m = 100000, 1000 entries per column, 1539 MB uncompressed; page cache warm, which is the best a disk can do):

      generate (write)                    0.85 s   1801 MB/s
      read .h5                            0.71 s   2158 MB/s   140 Mnnz/s   (three runs: 0.711-0.775 s)
      read .h5, deflate level 1 (846 MB)  6.9-7.3 s  115-123 MB/s  14 Mnnz/s  (decompression-bound)
      read the TXT conversion (2470 MB)   33.8 s   73 MB/s     2.96 Mnnz/s  (the legacy reader, as the plan's 3.5e6/s)
      dd of the .h5 file from the page cache: 5.2 GB/s

  Reading is 23 x faster per nonzero than the TXT parser and reaches 2.2 GB/s, above the bandwidth of the node's
  local disk and of the network file systems here; the remaining cost is the copy out of the page cache plus the
  structure checks (sorted columns, ranges, finiteness), i.e. no parsing. Before the early exit of the
  simplification scan (commit below) the same read took 1.5 s (1 GB/s).
- `docs/OPTIONS.md` regenerated (`cuadmm_exe --print-options markdown`); the io options are "now".

GPU acceptance (for the parent): `tools/baselines/cases_formats.json` lists trto1, mater-1 and buck3 from `.dat-s`,
from `.h5` (`$HOME/cuadmm-data/plato_h5/`, converted with `cuadmm-convert --compress 1`), from the TXT reference,
and buck3 from the SeDuMi `.mat` and its `.h5` conversion (policy fixed, sigma0 1, max_iter 20000, tol 1e-4, with
the published objective and the sign that relates the summary's pobj to it); `tools/io/check_published_objectives.py
<cuadmm_exe> $HOME/cuadmm-data <out dir>` runs them and reports the relative deviation. Not run here (no GPU in
this allocation).

Not done / not verified:

- The solution output read back by SPOT's `recover_mosek_sol_blk.m`: no MATLAB here; the writer's `barx` / `bars`
  layout (lower triangles column by column, in `bardim` order, `xx` / `slx` for the scalar block) is checked by
  `IoMat.MosekSolutionWriterReadBack` and on PushT_N=10_MOMENT through matio instead.
- MOSEK dual signs of a maximization (DECISIONS 25) are the documented MOSEK convention, not compared with a
  MOSEK run.
- Second-order cones (`K.q`, `blk 'q'`, `prob.cones`/`accs`), CBF, complex data: rejected with a message (Phase 9).
- The MEX gateway is not built (no MATLAB); it still takes the in-memory arrays of Phase 2.

## Phase 3: thin core

Status: implemented (with Phase 4 in one commit); GPU evidence pending.

- `src/core/check.h` (one set of macros, `cuadmm::Error`; the legacy and vendored headers map onto it, B15),
  `src/core/device_buffer.h`, `src/core/gpu_context.{h,cu}`. The ported engine keeps its streams and handles until
  Phase 6b (stream pool, arena).
- Acceptance: bitwise (checked with Phase 4 below).

## Phase 4: algorithm layer

Status: implemented; GPU evidence on the MIG configuration recorded below (59e943a: 53 of 53 identical); the
second configuration (full A100) is in progress.

- `src/admm/{admm_state,iteration_scheme,penalty_policy,admm_solver}`, `src/termination/{stopping_rule,
  best_iterate_tracker}`, `src/report/observers`; the engine's `solve()` is a wrapper over `AdmmSolver::run`
  (DECISIONS 16); the facade runs the loop with observers (`Solver::add_observer`).
- CPU tests (`cuadmm_cpu_tests`, 13 tests, pass on the CPU node): `PenaltyPolicy.*` replay the reference histories of
  `test/data/golden/` through the policies and reproduce every sigma exactly: `PSD_Free` legacy_adaptive 6000
  iterations (136 changes), `mater-1` legacy_adaptive to tolerance 1e-4 (8117 iterations, 141 changes, 2 Monitor1
  corrections), `PSD_Free` residual_balancing 6000 iterations (135 changes); `balance_freeze` on scripted sequences
  (freeze at the iteration, ping-pong by net movement with the +,+,-,- pattern, band centre, bias, halve-on-reversal,
  sigma_0 clamp, state round trip, equality with residual_balancing before a freeze).
- Acceptance (plan): bitwise for ADMM with each existing policy; sGS-ADMM with FixedPenalty and tau = 1.95 against
  the 2d8a755 build without `CUADMM_PLAIN_ADMM_ONLY` at fixed sigma and tolerance 1e-300. The case lists gained the
  sGS-phase adaptive rule, two hybrid cases and one sGS case to tolerance with the reduction rule on. The GPU run is
  recorded below when it completes.
- B23 (done after the merge of Phase 2b): the externally evaluated snapshots are ranked by the maximum DIMACS error,
  validated ones first, the practical merit only between snapshots without a finite DIMACS error
  (`ValidationResult::better_than`; CHANGES.md "Phase 4, B23"); the GPU test
  `ValidationStopping.NoStopWhenConeFailsAndBestIterateReturned` checks the returned iterate against that ranking.

## Phase 4b: CPU screen of `balance_freeze`

Status: the CPU screen is running (Stage 1 and the comparators submitted on 2026-10-21 at 21:16); results are
recorded below as they complete. The GPU gates are prepared as commands only (they need approval: H200, or jobs
over one hour).

- The validated sigma-study replica (plan: "ask the maintainers for it") is in this account:
  `$HOME/cuadmm-experiments/2026-10-19_sigma_rule_check/emu-run/cuadmm_emu.py`, v1.1 (2026-10-19), sha256
  `f213e9c6875900491af0eddce14e1c818a20416ab32ec534585a6e63cb40db18`, with its cross-check against the GPU
  (`emu-run/results/crosscheck.md`), its rule comparison (`emu-run/results/rule_comparison.md`) and the report
  (`report/`). `tools/sigma_screen/cuadmm_emu.py` is that file plus the `balance_freeze` rule (v1.2, sha256
  `b0a5814e684d1a4b01366742838ce0cb6439bb2b91c236353635f44258f3af6b`): `BalanceFreezeState`, a transcription of
  `ResidualBalancingPenalty` with the freeze enabled (`src/admm/penalty_policy.cc`), selected as
  `--rule balance_freeze[:key=value,...]` with the option names of `penalty.*`; checkpoints carry the freeze state;
  `summary.json` gains `freeze_iteration` and `freeze_reason` (DECISIONS 19).
- The replica is unchanged for the other rules: the sigma study's v1.1 suite (`emu-run/test_emu_rules.py`, run
  against `tools/sigma_screen/` on the CPU node) passes, so v1.2 = v1.1 (= v1.0) on its cases.
  `tools/sigma_screen/test_balance_freeze.py` (7 tests: the rule parser, the freeze at K_f, ping-pong with the
  +,+,-,- pattern and the band centre, the last value and the bias, halve-on-reversal for three factors, the sigma_0
  clamp, the emulator against the state object on trto1, the checkpoint round trip on vibra1) passes on the CPU node.
- Cross-check of the C++ policy against the replica: `tools/sigma_screen/replay_policy.py` drives
  `BalanceFreezeState` with the residual sequences of the golden histories (PSD_Free residual_balancing, 6000
  iterations; mater-1 legacy_adaptive, 8117 iterations) for 15 parameter cases and writes
  `test/data/golden/balance_freeze_cases_{PSD_Free,mater-1}.csv` (sigma after every iteration, shortest round-trip
  decimals, and the names of the log entries); `PenaltyPolicy.BalanceFreezeAgreesWithTheReplicaOn{PsdFree,Mater1}`
  replay them through `make_penalty_policy` and compare every sigma and every log entry exactly: pass
  (`cuadmm_cpu_tests`: 15 tests pass on the CPU node). Cases on PSD_Free at sigma_0 = 100: the defaults (ping-pong
  freeze at 1000); `freeze_iteration=3000,pingpong_min_iter=100000` (K_f); `factor=1.2,step_schedule=halve_on_reversal`
  (step freeze at 121); `step_schedule=halve_on_reversal,pingpong_min_iter=100000,freeze_iteration=100000` (step
  freeze at 141); `freeze_value=last,freeze_bias=2`; `freeze_bias=4,pingpong_window=12`;
  `pingpong_window=6,pingpong_net=0`; sigma_0 = 5000 (clamped to 1000); sigma_0 = 1e-4 with
  `sigma_min=1e-2,sigma_max=1e2,factor=1.5`; `freeze_iteration=5000,pingpong_min_iter=100000,freeze_bias=0.5,
  freeze_value=band_centre`; `factor=1.2,freeze_iteration=20000,pingpong_window=12`. On mater-1 at sigma_0 = 1: the
  defaults; `factor=1.2,step_schedule=halve_on_reversal`; `freeze_iteration=20000,pingpong_window=12`;
  `freeze_iteration=50000,pingpong_window=8,pingpong_net=2,freeze_bias=2,freeze_value=last`.
- Screen design (plan Table 2, fractional): Stage 1 = factor {2, 1.2} x step_schedule {constant, halve_on_reversal}
  x K_f {1e4, 2e4, 5e4} (12 settings) at sigma_0 in {0.5, 1, 2} on the nine small PLATO problems (trto1, trto2,
  buck1, buck2, vibra1, vibra2, mater-1, mater-2, shmup1): 324 runs. Comparators: old (legacy_adaptive) and new
  (residual_balancing) at the three sigma_0 values, fixed sigma = 1: 63 runs. Stage 2 = freeze_value {band centre,
  last} x freeze_bias {1, 2, 4} for the two best Stage-1 settings at sigma_0 = 1 (90 runs), then the best two of
  Stage 2 at sigma_0 in {0.5, 2}. Protocol of the sigma study: tolerance 1e-4, validation every 50 iterations,
  strict stage max|DIMACS| <= 1e-6, iteration cap 200,000 (DECISIONS 18). Driver `tools/sigma_screen/screen.sh`:
  one 1-core Slurm array task per run (seas_compute, cascadelake nodes as in the sigma study, 15 minutes, checkpoints
  every 120 s, resumes on a second call); outputs outside the repository in
  `$HOME/cuadmm-experiments/2026-10-21_phase4b_screen/` (the pinned replica copy and its sha256, `runs/`, `logs/`,
  `results/`, `specs/*.pending.*`). Gates and tables: `tools/sigma_screen/analyze.py` (G1 no problem lost against
  fixed sigma = 1 and the old rule at the same sigma_0; G2 the changes stop; G3 no change followed by a >= 10x rise
  of the validated error; G4 a better shifted geometric mean (shift 10 s) of the estimated GPU time than both
  residual_balancing and fixed sigma = 1; G5 robust across sigma_0).
- Commands (from `cuADMM-reorg/tools/sigma_screen`, Python = `~/cuadmm-analysis-env/bin/python -B`):
  `make_specs.py stage1 specs/stage1.txt`; `make_specs.py comparators specs/comparators.txt`;
  `SCREEN_ROOT=$HOME/cuadmm-experiments/2026-10-21_phase4b_screen ./screen.sh specs/comparators.txt 24 00:15:00`;
  `... ./screen.sh specs/stage1.txt 40 00:15:00`; `srun -c 1 ... analyze.py specs/stage1.txt specs/comparators.txt`.
- Submitted: smoke test (array 54488454, two short runs, both finished in 13 s); comparators array 54489077 (63 tasks,
  24 at a time) and Stage 1 array 54489148 (324 tasks, 40 at a time), 2026-10-21 21:16. Projected cost: at most
  about 20 core-hours for Stage 1 if every run hit the cap, less in practice.
- GPU gates (plan: G0-G3, at most 12.6 H200-hours): prepared, not submitted. `tools/sigma_screen/gpu_gates.sh
  <cuadmm_exe> "<flags of the chosen setting>"` writes one sbatch file per gate (partition `gpu_h200`,
  `--gres=gpu:nvidia_h200:1`, 16 cores, 96-128 GB; the campaign protocol: tolerance 1e-4, validation every 50
  iterations, strict stage max|DIMACS| <= 1e-6, time caps) and prints the projection. Caps: G0 trto1, buck1,
  pendulum N=80 at fixed sigma = 1 (600, 600, 900 s: 0.58 h); G1 trto1, buck1 (600 s) and vibra1, mater-1, mater-2,
  shmup1 (1800 s) x {balance_freeze, residual_balancing, fixed sigma = 1} (7.0 h); G2 pendulum N=80 at sigma_0 in
  {1, 100} x {balance_freeze, residual_balancing} (900 s: 1.0 h; the comparators 124 s old rule and 625 s fixed
  sigma = 1 are the paper-replication runs); G3 mater-3..6 x {balance_freeze, residual_balancing} (1800 s: 4.0 h).
  Sum of the caps 12.58 H200-hours; runs that converge stop earlier (the expected total from the CPU screen's
  iteration counts is recorded with the screen results). The setting's flags are filled in once the CPU screen has
  chosen it; the jobs need the user's approval before submission (H200, and G1/G3 exceed one hour).

### GPU evidence of the Pre-0/Phase 1 and Phase 2 commits (clean per-commit builds, MIG configuration)

- f8de01f (Phase 1), `tools/slurm/build_and_compare.sh f8de01f phase1 ... legacy@pre0` (16:39-16:59): comparison with
  the references on every baseline case 49 of 49 identical, 0 different, 0 failed runs (fixed 21 against 2d8a755,
  legacy_adaptive 10 against 9bb4ba5, residual_balancing 10 against 2d8a755, sGS 8 against 2d8a755-sgs). Tests:
  101 ran, 99 passed, 2 failed: `FixedSigmaCli.BackwardCompatibleAndNewFlags` and `HybridSwitchCli.SgsIterationsOption`,
  both expectation mismatches of that commit's own CLI output format (the 17-digit printing writes `1.3500000000000001`
  and `1.6180000000000001` where the tests expect `1.35` and `1.618`); the golden test `FixedSigma.
  LegacyAdaptiveAndPlainAdmmUnchanged` passes (B14 fixed by Pre-0). The formatting is fixed from Phase 2 on by the
  shortest round-trip printing (`std::to_chars`).
- b44596b (Phase 2), `build_and_compare.sh b44596b phase2 ... reorg` (the new CLI, 16:43-17:03): 49 of 49 identical,
  0 different, 0 failed runs; tests 101 ran, 99 passed, the same 2 formatting expectations failed (the commit's tests
  still expected the 17-digit form in one place and the shortest form in the other; both fixed in the working tree
  that the Phase 4 commit carries, where `cuadmm_cpu_tests` and the CLI tests use the shortest round-trip form).
- 59e943a (Phase 3 + 4), `build_and_compare.sh 59e943a phase4 ... reorg` (20:50-21:00): 53 of 53 identical, 0
  different, 0 failed runs (fixed 21 against 2d8a755, legacy_adaptive 10 against 9bb4ba5, residual_balancing 10
  against 2d8a755, sGS 12 against 2d8a755-sgs: 8 fixed-sigma sGS cases, the sGS-phase adaptive rule on PSD_Free,
  two hybrid cases with the switch at 101 and 301, and one sGS run to tolerance 1e-5 with the reduction rule on).
  Tests: 101 ran, 99 passed, 2 failed, the same two CLI formatting expectations as before
  (`FixedSigmaCli.BackwardCompatibleAndNewFlags` expected `"sigma_max": 1000000` where `std::to_chars` wrote `1e+06`;
  `HybridSwitchCli.SgsIterationsOption` expected `1.618` in the history CSV, which prints 17 significant digits).
  Both fixed in d6720b6 (`json_number` prints integral values in fixed notation; the hybrid test parses tau;
  CHANGES.md Phase 4b): the two tests pass on the MIG with the rebuilt working tree (21:17).
- 92dca50 (d6720b6 + the Phase 4b preparation), `build_and_compare.sh 92dca50 phase4b ... reorg` (21:22-21:35,
  MIG configuration): 53 of 53 identical, 0 different, 0 failed runs; tests 101 ran, 101 passed.
- Second configuration `a100-sxm4-80gb_cuda12.9_gcc12.2_ss7.10.1_t16` (partition `gpu`, job 54480621, from 21:08,
  `tools/slurm/full_a100_session.sh 54480621 59e943a phase4 reorg`): Phase 0 baselines 59 of 59 identical
  (6 self-checks and the 53 reference cases, 16 threads); the comparison of 59e943a is in progress and is recorded
  here when it completes.
