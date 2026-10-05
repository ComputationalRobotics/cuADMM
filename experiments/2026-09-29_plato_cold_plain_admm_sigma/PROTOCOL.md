# Protocol: cold-start plain ADMM on the PLATO / Kocvara sparse SDPs, adaptive versus fixed σ

Frozen on 2026-09-29, before any GPU run of this campaign. Changes after that date are listed under "Deviations" at the end, with their reasons.

Scope:

- **No MOSEK.** MOSEK is not run, and no MOSEK solution is used for initialization.
- **Cold start only.** No warm start and no sGS-ADMM.
- **One H200 per timed solve.**

Accuracy criteria: `accuracy_protocol.md`. Data provenance and conversion checks: `dataset_inventory.csv`, `conversion_validation.md` and `logs/conversion/`.

## Solver build

- **Branch.** `experiment/plato-cold-plain-admm-sigma`, from the hybrid branch at 09687d2.
- **Commits.**
  - b89e509: validator, with the DIMACS measures and the S cone.
  - d477f1b: plain-ADMM-only build, strict stage, σ log.
  - 2898f5c: tests.
- **The plain-ADMM-only build.** It defines `CUADMM_PLAIN_ADMM_ONLY`:
  - `solve()` and the CLI reject every sGS option;
  - the sGS second y-update throws if it is ever reached;
  - every summary records `y_solves` and `y_solves_per_iteration`, which must be exactly 1.
- **Official build.** `~/cuadmm-builds/plato_official`, built on the H200 node by `scripts/build_version.sh` from a clean tree at the commit above. It uses the flags of the hybrid campaign's `hybrid_official` build: `-DCMAKE_BUILD_TYPE=Release`, `-gencode=arch=compute_90,code=sm_90` and the same SuiteSparse and BLAS, so the two builds are comparable bit for bit. Its manifest is in `builds/official/`, and it is frozen for the whole campaign.
- **Checks before any timed run:**
  - the full test suite;
  - compute-sanitizer memcheck, initcheck and synccheck;
  - bit identity with the 09687d2 build (`~/cuadmm-builds/hybrid_official`) run as pure plain ADMM. X, y and S must be byte-identical after a fixed number of iterations, for fixed and adaptive σ, on four PLATO instances (`scripts/compare_builds.sh`).

## Configurations

Every run starts cold: X = 0, y = 0, S = 0, with the solver's own scaling. It uses plain ADMM (`--algorithm admm`) and the same data. Iterations are bitwise deterministic (CSR_ALG2 SpMV), so there is no seed.

| label | options | meaning |
|---|---|---|
| adaptive | `--sigma-policy legacy_adaptive --sig 1` | the historical adaptive σ rules (schedule factor 2; Monitor1 after 5,000 iterations), from σ₀ = 1 |
| fixed | `--sigma-policy fixed --sig 1` | the same σ₀, never changed |
| tuned | `--sigma-policy fixed --sig σ*` | σ* chosen on the pilot (Stage 2) by the rule below |

**Initial σ = 1.** This is the value of the upstream example for this collection: `MATLAB/example_sdpt3.m` solves vibra4 with `sigma = 1.0`. The CLI default is 100, which is in the pilot grid.

**Common options.**

- `--tol 1e-4`: the internal residual only triggers validations.
- `--validate-tol 1e-4`: the practical criteria.
- `--validate-threads 16`.
- `--max-iter 1000000000`: no iteration cap, so the time limit binds.
- `--time-limit`: the stage limit, in seconds of solve time; validation time is included.
- `--strict-dimacs-tol 1e-6`: in Stages 1 and 3.
- `--save-solution`, which saves the returned iterate, and `--save-final-dir`, which saves the final iterate at a limit.
- `--history`, `--validation-history` and `--sigma-log`.
- LOBPCG and every other solver option stay at their defaults. LOBPCG is used only for blocks larger than 1,000 whose rank is below 3% of n; its calls and fallbacks are recorded.

**Required checks.**

- Every fixed-σ run must report `sigma_changes == 0`, `sigma_log_entries == 0` and `final_sig == sigma0`.
- Every adaptive run logs each σ change: the iteration, the old and new σ, the rule, and the residuals and win counters the rule saw.

## Stopping and accuracy

The practical target is the first validation snapshot with all of the following:

- eta_p, eta_d, eta_g ≤ 1e-4;
- the X, Z and S normalized cone violations ≤ 1e-4;
- all values finite.

This is recorded as the practical milestone. The run then continues until a validated snapshot has max_abs_DIMACS ≤ 1e-6, or until the time limit. It returns that snapshot, or else the validated snapshot with the smallest max_abs_DIMACS.

The Stage 2 pilot runs stop at the practical target, with no strict stage.

**Revalidation.** After each run, `scripts/dimacs_validate.py` recomputes every measure on the CPU from the saved returned and final iterates. It is independent of the solver's validator (see `logs/validator_crosscheck.json`). The report uses these values, and any disagreement with the in-solver values is reported.

**Statuses** (definitions in `accuracy_protocol.md`):

- **STRICT_VALIDATED**: the returned iterate passes the strict target. For shmup4 and shmup5 the objective part is undecidable at 1e-6, and the table says so.
- **PRACTICAL_VALIDATED**: the practical target was reached, but the strict target was not reached within the limit.
- **TIMEOUT**: no practical target within the limit. Time-to-target is reported as "> limit", and both the best and the final iterate are validated.
- **NOT_VALIDATED**: a crash, non-finite values, or disagreement between the two validators.
- **OUT_OF_MEMORY**.
- **CONVERSION_FAILED**: none; all 26 conversions pass.

**Threshold times.** For each run, the first validation snapshot with eta ≤ 1e-3, 1e-4, 1e-5, 1e-6 and with max_abs_DIMACS ≤ 1e-3, 1e-4, 1e-5, 1e-6, with its iteration and solve-clock time. These are exact to the validation interval.

**End-to-end time.** Load + CUDA initialization + solver initialization + solve, measured by the solver. The driver also records the process wall time.

## Stages and limits

**Stage 0: checks.** The build, tests, sanitizers and bit identity, as listed above.

**Stage 1: smoke** (`plans/stage1.json`, 52 runs).

- **Runs.** All 26 datasets × {adaptive, fixed}, with a limit of 120 s each and the strict stage on.
- **Validation interval.** 100 when vec_len < 1e5, 200 when < 1e6, otherwise 500.
- **Purpose.** Finite values, the factorization report (the rank of mater-5 and mater-6), memory, and the validation cost.
- **Validation-interval rule, applied to both configurations of a dataset in Stages 2 and 3.**
  - v = the larger per-snapshot validation time of the dataset's two smoke runs.
  - t = the smaller per-iteration time without validation.
  - The interval is the smallest value in {50, 100, 200, 500, 1,000, 2,000, 5,000, 10,000} with v/(interval · t) ≤ 0.10, which keeps the validation overhead at or below about 10%.
  - Written to `plans/validation_intervals.json`.
- **Budget projection.** A projection of H200-hours from Stage 1, stated before Stage 2.

**Stage 2: σ pilot** (`plans/stage2.json`, 40 runs).

- **Datasets.** The five predeclared pilot datasets, the third member of every family: buck3, mater-3, shmup3, trto3 and vibra3.
- **Configurations.** The adaptive reference, plus fixed σ ∈ {1e-3, 1e-2, 1e-1, 1, 10, 100, 1000}.
- **Limit.** 900 s each, stopping at the practical target.
- **Frozen selection rule** for σ*:
  1. the largest number of pilot datasets reaching the practical target within 900 s;
  2. then the smallest geometric mean over the 5 datasets of the solve time to the practical target, with a timeout counted as 1,800 s (twice the limit);
  3. geometric means within 5% of the best count as tied, and the tie goes to the σ closest to 1 in log scale.
- **In-sample and held out.** σ* is one value for all datasets. In Stage 3, results on the 5 pilot datasets are marked in-sample and the other 21 held out.

**Stage 3: official** (`plans/stage3.json`).

- **Runs.** All 26 datasets × {adaptive, fixed, tuned}, with a 1 h limit each and the strict stage on. If σ* = 1, tuned is the fixed configuration and is not repeated.
- **Order.** Datasets by increasing vec_len. Configurations alternate within a dataset: A-T-F on even, F-T-A on odd.
- **Repetitions.** Every configuration whose rep 1 took under 10 minutes end to end gets reps 2 and 3, interleaved. Longer runs have one labelled repetition. Reported times are medians over the repetitions, with the range. Iterations are deterministic, so they must agree across repetitions; this is checked.

**Optional extension, from 1 h to 3 h, for a dataset.** Only if all of the following hold:

- neither configuration reached the target at 1 h;
- both are still improving, measured by max_abs_DIMACS over the last 20% of the run;
- at least one is plausibly close: within 10× of the threshold;
- both configurations get the same extension, run as new 3 h runs from a cold start, not continued;
- the budget allows it.

**Equal-limit comparison.** Adaptive and fixed always get the same limit. If neither reaches a target, they are compared by the external errors of their returned iterates after equal wall time. No winner is declared from internal residuals.

## Budget and hardware

**Budget.**

- H200-hours = allocated H200 GPUs × allocation wall time (sacct Elapsed), including build, tests, sanitizers and idle time.
- **The global limit is 48 H200-hours.** Before the projected or actual usage exceeds it, the campaign stops and asks the user.
- CPU-hours for revalidation, and storage, are also reported.

**Hardware.**

- Partition `gpu_h200`, one H200 per solve, and one solve at a time per GPU.
- `nvidia-smi` is recorded before and after every run.
- A cudaMalloc test runs on the device before the campaign.
- If the campaign uses two allocations at once, each dataset's configurations run on the same allocation, back to back.

## Deviations

All of these were made before any timed run.

1. **Stage 0 bit-identity check, first version.** The v1 check compared the returned iterate of a run without validation against the final iterate of a run with validation. Without validation, the solver returns its best-KKT iterate rather than its last one, so these are different iterates. The check was stopped after one case. That case was identical build to build (X, y, S and the history on trto3, fixed σ); its partial log is `logs/stage0/compare_builds_v1_flawed_partial.txt`. The v2 check replaces it, and compares like with like:
   - without validation: the returned iterates and histories of the two builds;
   - with validation: the checkpoint iterates at iterations 3,000 and 6,000;
   - for each build: the history with validation against the history without it.
2. **A test bug found on the H200.** `PlainAdmmOnlyCli` expected exactly 600 y solves, but the run converges at iteration 212. The test was fixed in 21464b1 and the build dir was rebuilt incrementally. The library, executable and psd_projection library are byte-identical before and after the fix (`logs/stage0/hashes_*`), and the suite then passes 87 of 87 tests.
3. **initcheck reports inside cuSOLVER.** The first sanitizer pass was clean for all 5 memcheck cases and for initcheck on the tests, trto2 and mater-2.
   - **The reports.** initcheck on buck4 (blocks 673 and 672) reported 530,046 uninitialized 8-byte global reads. All 100 printed reports come from cuSOLVER's internal tridiagonalization kernel `sytrd4_gpu`, called by `cusolverDnXsyevd` from `single_eig_cusolver`, the medium and large block EVD.
   - **Why it is not an input problem.** The input matrix is fully written: `vector_to_matrices_kernel` writes both triangles. memcheck is clean, and the iterates are bit-identical across processes and builds.
   - **Stopping the pass.** The error reporting slowed the run to 4.9 s per iteration, so the pass was stopped during initcheck on trto5; that partial log is `*.INTERRUPTED.log`.
   - **The second pass** (`scripts/sanitizers2.sh`) reruns initcheck on buck4 and trto5 with `--kernel-name-exclude kns=sytrd4_gpu`. This shows whether any other kernel reads uninitialized memory. It then runs synccheck on all 5 cases.
4. **Reports outside the excluded kernels, and Stage 1 held.**
   - **The reports.** On 2026-09-30 the filtered initcheck of buck4 (`kns=sytrd` excluded) reported 1,237,088 reads. All 100 printed ones are in cuBLAS's `sm90_xmma_syr2k_…_cublas` kernel, launched through `cublasLtDDDMatmul` ← `cublasDsyr2k_v2_64` ← `cusolverDnXsyevd` ← `single_eig_cusolver`. The filtered trto5 run shows the same kernel and stack, via `large_full_eig_project`. cuADMM and psd_projection never call syr2k.
   - **Why the filtered runs are not conclusive.** A kernel excluded from initcheck is not instrumented, so its writes are not recorded. Memory written by the excluded sytrd kernels then looks uninitialized to the kernels that read it next, such as the syr2k trailing update of the same tridiagonalization. This fits the rise from 530,046 to 1,237,088 reports. Excluding kernels is diagnostic isolation, not evidence of a clean unfiltered initcheck.
   - **Stage 1 held.** Following the instruction to stop the benchmark progression on any report outside the excluded kernels, the STOP file was created at 14:00:44, before Stage 1 started. Job 49451471 finishes its sanitizer cases, and its Stage 1 driver then exits without a run.
   - **The diagnosis job** (49453464, `scripts/diag/`) runs:
     - `cusolverDnXsyevd` as cuADMM calls it, with its workspace and outputs left unwritten, zeroed, NaN-filled or garbage-filled, comparing the outputs bitwise;
     - the same program under initcheck;
     - complete, unfiltered initcheck tallies of every report of the cuADMM cases, by kernel and named host frames.
   - **Coverage gap.** LOBPCG was not used in any sanitizer case (LOBPCG calls = 0), so psd_projection's LOBPCG path has no sanitizer coverage yet.
   - **Diagnosis result** (job 49453464, 14:04–14:23; logs in `logs/diag/`):
     1. **Workspace fills.** `cusolverDnXsyevd`, called exactly as cuADMM calls it at n = 673 and n = 1761, gives bitwise-identical eigenvalues and eigenvectors in all four cases: workspace, W and info unwritten, zeroed, NaN-filled (0xFF bytes) or garbage-filled. The output is also stable over 3 repetitions. The contents of the memory initcheck flags do not influence the result.
     2. **The same program under initcheck.** With the workspace unwritten, it reports 10,416 reads, all under `cusolverDnXsyevd`. With the caller's buffers zero-filled, it reports 0. So the reads are of the caller-provided workspace, which cuADMM allocates with `cudaMalloc` and never initializes.
     3. **Complete, unfiltered tallies of the cuADMM cases.** Every report was attributed, with none left unprinted:
        - buck4, the planned case (200 iterations): 530,046 reports;
        - trto5 (3 iterations, validation every iteration): 30,869 reports.
        All are in `sytrd4_gpu`, launched by `cusolverDnXsyevd` from `single_eig_cusolver` via `SDPSolver::solve`, `large_full_eig_project` or `cone_measures`. **There are 0 reports in cuADMM, psd_projection, cuSPARSE or any other kernel.** The filtered runs' syr2k reports came from the exclusion, as described above.
     - **Conclusion.** This is cuSOLVER-internal reading of caller workspace that it has not yet written, and it does not affect results. It is no correctness problem, and the benchmark resumes once synccheck is clean. The official build is unchanged: initializing the workspace would silence initcheck, but it would alter the frozen build and change nothing numerically.
5. **LOBPCG sanitizer coverage** (added after Stage 1, before any further benchmarking).
   - **Why.** Stage 1 used psd_projection's LOBPCG on shmup4 and shmup5, with 7,846 to 22,275 calls per run, but no earlier sanitizer case did (LOBPCG calls = 0 there).
   - **The case.** Job 49475917 runs memcheck with leak check, synccheck and a complete unfiltered initcheck tally on shmup4 with fixed σ = 1 for 120 iterations. That covers LOBPCG from the first rank analysis and the full-EVD re-evaluation at iteration 100.
   - **Stopping rule.** The same as before: any report in a cuADMM or psd_projection kernel stops the benchmark progression.
   - **Result of the LOBPCG case** (job 49475917, 9 min).
     - memcheck: 0 errors, 0 bytes leaked. synccheck: 0 errors. Both over 216 LOBPCG calls and 2 fallbacks to cuSOLVER.
     - The complete unfiltered initcheck tally has 1,076,979 reports, all under cuSOLVER frames:
       - 36,339 in `sytrd4_gpu` under `cusolverDnXsyevd`, as before;
       - **1,040,640 in `geqr2_smem_domino_fast` under `cusolverDnDgeqrf`**, called from psd_projection's `lobpcg`.
     - There are 0 reports in cuADMM or psd_projection kernels.
6. **The QR reports under psd_projection's LOBPCG.**
   - **The concern.** Unlike an eigensolver workspace, a QR factorization reads its whole input matrix. So the benchmark progression stays held until it is shown that the QR input is fully written and that the flagged reads do not influence the result.
   - **Code inspection** (`psd_projection/src/lobpcg.cu`). The input block XRD = [X_k, R_k, Delta_X_k] is written entirely by three `cudaMemcpy` calls before every `geqrf`; Delta_X_k is initialized from X_k before the loop. The caller-provided `tau_xrd`, `d_work_xrd` and devInfo come from bare `cudaMalloc`.
   - **Diagnosis** (job 49476960, `scripts/diag/geqrf_workspace_test.cu`). It reproduces the `geqrf` + `orgqr` step at shmup4's shapes, for the first-iteration block (Delta_X_k = X_k) and a generic block, with tau, workspace and devInfo left unwritten, zeroed, NaN-filled or garbage-filled. It compares R and Q bitwise, and reruns under initcheck.
   - **A script bug.** The verdict line printed by that job uses the wrong awk fields (10, 12 and 14 instead of 11, 13 and 15), which would always report a match. That printed verdict is not used; `scripts/diag/geqrf_verdict.sh` recomputes it from the raw RESULT lines.
   - **Result** (job 49476960, 1.5 min; `logs/diag/geqrf_*`).
     - In all 4 cases (n, m = 1681, 21 and 1680, 20; first-iteration and generic blocks), R and Q are bitwise identical across the four fills and stable across repetitions (`geqrf_workspace_fills_verdict_recomputed.txt`).
     - Under initcheck, the reproduction reports 243,456 reads in `geqr2_smem_domino_fast` with the caller's tau, workspace and devInfo unwritten, and 0 with them zero-filled.
     - **Conclusion.** This is the same benign pattern as `syevd`: cuSOLVER reads caller-provided workspace or outputs before writing them, and the result does not depend on their contents. It is no psd_projection bug and no correctness problem.
     - **All sanitizer work is complete.**
       - memcheck: 6 of 6 cases clean.
       - synccheck: 6 of 6 cases clean.
       - initcheck: clean on the tests, trto2 and mater-2. Every report on buck4, trto5 and shmup4/LOBPCG is attributed to cuSOLVER's `sytrd4_gpu` or `geqr2_smem_domino_fast` reading caller workspace, with 0 in cuADMM, psd_projection or any other kernel.
7. **Reduced official stage: plan A**, chosen by the user on 2026-09-30 after the Stage 1 projection.
   - **Why.** The declared Stages 2–3 were projected at 67.7 H200-hours expected and 82.1 at most, against 29.12 remaining.
   - **What runs.** Stage 3 becomes adaptive vs fixed σ = 1 on all 26 datasets, with the same **1,800 s** cap for both. Everything else is as declared: identical cold starts, the strict DIMACS stage (1e-6) after the practical target, the Stage 1 validation intervals, alternating configuration order, repetitions 2–3 for runs under 10 minutes end to end, and independent revalidation.
   - **What is dropped.** The Stage 2 σ pilot, the tuned fixed-σ configuration, and the 1 h → 3 h extension.
   - **Consequences for the evidence.**
     - The comparison is adaptive vs fixed at the same σ₀ = 1 only. There is no evidence on a tuned fixed σ.
     - Runs that would need 30–60 minutes are TIMEOUT. They are still compared by the external errors of their returned iterates after equal wall time.
   - **Execution.** `plans/stage3.json` (`make_plan.py stage3 --plan-a`) is split by `scripts/partition_plan.py` into two concurrent 1-H200 jobs, `plans/stage3_part{1,2}.json`. Each dataset's runs stay in one part, back to back on the same GPU. The parts are balanced on their maximum cost, 11.64 and 11.65 H200-hours, and each job's limit is 12.5 h, so at most 25.0 H200-hours can be used against 29.12 remaining.
