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

Status: implemented; GPU evidence pending.

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
- MOSEK dual signs of a maximization (DECISIONS 21) are the documented MOSEK convention, not compared with a
  MOSEK run.
- Second-order cones (`K.q`, `blk 'q'`, `prob.cones`/`accs`), CBF, complex data: rejected with a message (Phase 9).
- The MEX gateway is not built (no MATLAB); it still takes the in-memory arrays of Phase 2.

### Phase 2b review fixes (2026-10-21, commits on `reorg-io` after eb3ee06)

Independent review of the merged tree (findings against plan Section 12): fixed 1-5, 7-9, 11, 13; 6, 10, 12 are
documented (DECISIONS 32-35, CHANGES "Phase 2b review fixes"). Each fix has a test:

- A. clean build without `HDF5_ROOT` found the system HDF5 1.10.5, which lacks `H5Treclaim`: `h5.cc` now uses
  `H5Treclaim` from 1.12 and `H5Dvlen_reclaim` before (`H5_VERSION_GE`).
- 1. string attributes / `/meta/report` read with the stored character set (h5py's variable-length UTF-8, fixed
  ASCII, variable ASCII, fixed UTF-8; int32 datasets): `tools/io/make_h5_testfiles.py` -> `test/data/h5_*.h5`,
  `IoH5.FilesWrittenByH5pyAndMatlabStyle`.
- 2. MOSEK `slx`/`sux` of ranged and fixed variables (`SourceInfo::mosek_var_slack/var_cost/fixed_*`, also in the
  `.h5` metadata): `IoMat.MosekWrittenDualsSatisfyTheDualConstraint` (c - a'y = slx - sux for every variable
  kind, slc - suc = y, sign conventions, min and max; the reviewer's minimal case gives y = 0, slx = 0, sux = 1).
- 3. `/At/rowind`, `/b/index`, `/C/index` range-checked before the simplification scan:
  `IoH5.MalformedIndicesAreRejectedBeforeTheSimplification` (2^40, 5, -1, C index 2^40; simplify on and off).
- 4. `/solution` and `/start` in the file's block layout with `/solution/blocks` (`file_block_layout`,
  `to_file_layout`, `from_file_layout` in `simplify.cc`); `solution_blocks.txt` with the TXT convention:
  `IoH5.SolutionAndWarmStartInTheFileBlockLayout`, `IoSimplify.TxtSolutionWriterAndFormatDispatch`.
- 5. missing `sense` attribute rejected unless `io.sense`: `IoH5.MissingSenseAttributeNeedsIoSense`.
- 7. `.cbf` / `.cbf.gz` -> "CBF comes with Phase 9": `IoSimplify.TxtSolutionWriterAndFormatDispatch`.
- 8. MOSEK quadratic terms rejected: `IoMat.MosekBoundsRangesSenseAndRejections`.
- 9. SDPA text source read errors are errors: `IoSdpa.ExplicitZerosDoNotTouchABlockAndReadErrorsAreReported`
  (a directory opened as a file), which also covers 13 (a block touched by an explicit zero only is untouched).
- 11. HDF5 files closed explicitly with the status checked; the SDPA JSON sidecar, `certificate.txt` and
  `solution_blocks.txt` streams checked.

Builds and tests after the fixes (CPU node holy8a26208):

    tools/build.sh ~/cuadmm-builds/reorg-io -DCUADMM_WITH_HDF5=ON -DCUADMM_WITH_MATIO=ON -DCUADMM_WITH_ZLIB=ON -DHDF5_ROOT=... -DMATIO_ROOT=... -DZLIB_ROOT=...
      -- cuADMM-reorg: HDF5 TRUE (1.14.6), matio TRUE (1.5.30), zlib TRUE (1.3.2);  build exit code: 0
      ./cuadmm_cpu_tests: 39 tests from 8 test suites ran. [ PASSED ] 39 tests.   --formats: txt sdpa sdpa.gz h5 sedumi sdpt3 mosek
    tools/build.sh ~/cuadmm-builds/reorg-io-nodeps -DCUADMM_WITH_HDF5=OFF -DCUADMM_WITH_MATIO=OFF -DCUADMM_WITH_ZLIB=OFF
      -- cuADMM-reorg: HDF5 FALSE (), matio FALSE (), zlib FALSE ();  build exit code: 0
      ./cuadmm_cpu_tests: 39 tests ran. [ PASSED ] 23 tests. [ SKIPPED ] 16 tests.   --formats: txt sdpa
    tools/build.sh ~/cuadmm-builds/reorg-io-auto        (clean configure, no HDF5_ROOT, the environment of ~/cuadmm-tools/env.sh)
      -- cuADMM-reorg: HDF5 TRUE (1.10.5), matio FALSE (), zlib TRUE (1.2.11)   (/usr/lib64/libhdf5.so, /usr/lib64/libz.so);  build exit code: 0
      ./cuadmm_cpu_tests: 39 tests ran. [ PASSED ] 32 tests. [ SKIPPED ] 7 tests.   --formats: txt sdpa sdpa.gz h5

The reviewer's probe files (`~/cuadmm-builds/reorg-io/io_test_tmp/review_probe/`) with the io-env build: the four
string variants and the int32 file convert; `huge_idx.h5`, `huge_C_idx.h5`, `vlen_utf8_oob.h5`, `vlen_utf8_neg.h5`
are rejected with "/At/rowind[2] = 1099511627776 outside [0, 5)" and the like (exit 1, no crash).

## Phase 3: thin core

Status: not started.

## Phase 4: algorithm layer

Status: not started.

## Phase 4b: CPU screen of `balance_freeze`

Status: not started. The validated sigma-study replica is available locally
(`$HOME/cuadmm-experiments/2026-10-19_sigma_rule_check/emu-run/cuadmm_emu.py`, v1.1, with its results and
cross-check reports); it will be copied into `tools/sigma_screen/` for the screen.
