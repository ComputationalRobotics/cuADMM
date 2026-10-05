# Cold-start plain ADMM on the PLATO / Kocvara sparse SDPs: adaptive vs fixed σ

**Campaign:** 2026-09-29 to 2026-10-01.
**Branch:** `experiment/plato-cold-plain-admm-sigma`.
**Official solver build:** code commit 21464b1, library sha256 2787f184…, executable 621b459a…, frozen.

## Scope

- **Solver.** Plain ADMM only, from a cold start: X = 0, y = 0, S = 0. No sGS, no hybrid switching, no warm start.
- **No MOSEK.** MOSEK was never run and its license was never touched (`logs/job_inspection_and_mosek_cleanup.md`).
- **Instances.** All 26 structural-optimization SDPs of Kocvara's collection, the source of PLATO's sparse-SDP Kocvara instances.
- **Configurations.** Two, with identical data, cold starts and caps:
  - **adaptive:** `--sigma-policy legacy_adaptive --sig 1`, the historical rules;
  - **fixed:** `--sigma-policy fixed --sig 1`.
- **Plan A.** The official stage is the reduced plan A chosen after the Stage 1 projection (`PROTOCOL.md`, Deviation 7):
  - adaptive vs fixed σ = 1 on all 26 datasets, with the same **1,800 s** cap for both;
  - no σ pilot and no tuned fixed σ;
  - the strict DIMACS stage after the practical target;
  - repetitions 2–3 for runs under 10 minutes.
- **Statuses** come from an **independent revalidation** of every returned iterate (`scripts/dimacs_validate.py`, NumPy). It agrees with the solver's own validator to within 4.9e-10 in max_abs_DIMACS on all 72 official runs.

## Accuracy criteria (`accuracy_protocol.md`)

**PLATO itself.**

- PLATO reports the six DIMACS error measures with a 40,000 CPU-second limit.
- It classifies a run as "a" when an error exceeds 1e-4 and as "f" when one exceeds 1e-2.
- It mandates **no** stopping tolerance and publishes no reference objectives.

**Adopted here.**

- **Practical target:** eta = max(eta_p, eta_d, eta_g) ≤ 1e-4, every normalized cone violation (X, S, C − A*y) ≤ 1e-4, and all values finite.
- **Strict target:** max_abs_DIMACS ≤ 1e-6, and a relative objective error ≤ 1e-6 against Kocvara's published objective after the scale corrections of section 4 of `accuracy_protocol.md`.
- For shmup4 and shmup5 the reference itself is uncertain to about 2e-6, so their objective check cannot be decided at 1e-6.
- The solver stops on the strict target using DIMACS alone, since it has no reference objective. A run whose returned iterate then fails the objective check is reported **PRACTICAL_VALIDATED**: mater-1 and mater-2 with adaptive σ, and mater-1 with fixed σ.

## Before the timed runs (Stage 0)

- **Tests.** The official build passes 87 of 87 tests. One test bug was fixed, which left the library and executable byte-identical.
- **Bit identity.** The build is byte-identical to the 09687d2 build run as plain ADMM, on 4 instances × fixed and adaptive σ: returned iterates, checkpoints and histories (`logs/stage0/compare_builds.txt`).
- **compute-sanitizer.**
  - memcheck and synccheck are clean on 6 cases, including psd_projection's LOBPCG path, which Stage 1 showed shmup4 and shmup5 use.
  - initcheck reports occur only inside cuSOLVER's `sytrd4_gpu` (eigensolver) and `geqr2_smem_domino_fast` (QR) kernels.
  - Complete, unfiltered tallies (530,046, 30,869 and 1,076,979 reports) attribute every report to these kernels, with **0 in any cuADMM or psd_projection kernel**.
  - Standalone reproductions show cuSOLVER reading caller workspace it has not yet written: the eigen-decomposition and QR outputs are bitwise identical for unwritten, zeroed, NaN-filled and garbage-filled workspaces, and zero-filling removes every report (`logs/diag/`, Deviations 3–6).
  - **No correctness problem.**
- **Stage 1 smoke** (52 runs of 120 s): all ran, and validation intervals were derived with an overhead of at most 10% (`plans/validation_intervals.json`).

## Official results (Stage 3, plan A; `results/official_table.md`, `results/comparison.md`)

| | adaptive σ | fixed σ = 1 |
|---|---|---|
| practical target reached (of 26) | **8** | 6 |
| strict target reached | 4 (vibra1, mater-3, mater-4, mater-5) | 4 (trto1, buck1, vibra1, mater-2) |
| TIMEOUT at 1,800 s | 18 | 20 |
| PLATO-style class of the returned iterate: clean / "a" / "f" | 8 / 0 / 18 | 6 / 3 / 17 |
| cost per iteration (geometric mean, adaptive / fixed) | 1.04× | |

**Where both reach the practical target** (vibra1, mater-1, mater-2, mater-3):

- adaptive is faster on all 4;
- the geometric-mean speedup of adaptive over fixed is **41×** to the practical target and **12.6×** end to end;
- on the strict target, only vibra1 validates under both, with a 14× speedup.

**Where only one configuration reaches the target:**

- adaptive only: shmup1, mater-4, mater-5, mater-6;
- fixed only: trto1, buck1.

**Where neither reaches it** (16 datasets): after the same 30 minutes, the returned iterate has the lower external max DIMACS error with **fixed on 12** and with adaptive on 4 (shmup2–5).

**Over equal wall time across all 26:** adaptive has the lower external max DIMACS error on 13 vs 12 datasets at 60 s, 12 vs 14 at 300 s, and 12 vs 14 at the cap.

**The pattern is by family, not uniform:**

- **mater (6 instances).** Adaptive drives σ down to 0.001–0.08 and converges orders of magnitude faster:
  - it reaches the practical target in 3–189 s on all six;
  - fixed σ = 1 reaches it only on mater-1, mater-2 and mater-3, in 74–596 s.
- **trto, buck and vibra (15 instances).** Adaptive σ oscillates: 179 to 5,652 changes per run, with final values from 0.006 to 100. The external errors show periodic spikes up to about 1e2 (`results/plots/stage3_dimacs.png`), and adaptive validates only vibra1. Fixed σ = 1:
  - validates trto1, buck1 and vibra1;
  - on the other 12, ends with a lower external max DIMACS error, by 1.3× to about 1,900× (largest on buck2 and vibra2).
- **shmup (5 instances).** Adaptive validates shmup1 at 1,707 s. On shmup2–5 neither validates, and adaptive ends lower in error.
  - On shmup4 and shmup5, adaptive σ changes break LOBPCG's warm start. LOBPCG then falls back to the full eigensolver 7–12× more often (149 vs 22 and 206 vs 17 fallbacks).
  - As a result adaptive completes only 1.1× and 2.9× fewer iterations in the same time (96 vs 32 ms per iteration on shmup5).
- **The eight PLATO-benchmarked instances** (buck5, mater-6, shmup4, shmup5, trto4, trto5, vibra4, vibra5): within 30 minutes only mater-6 reaches any target, adaptive's practical target at 189 s. All others end with max DIMACS errors of 0.2–4.8 for both configurations, which is PLATO class "f".

**Tolerances reached, per dataset:** `results/tolerances_reached.md`. Every run has thresholds at 1e-3, 1e-4, 1e-5 and 1e-6 for eta and max_abs_DIMACS in `results/stage3_runs.csv`, along with equal-time snapshots at 60 s, 300 s and 1,800 s.

## Timeouts, failures, memory

- **Exit status.** All 52 official runs plus 20 repetitions exited with code 0. There were no crashes, no non-finite values and no OUT_OF_MEMORY.
- **Memory.** Peak GPU memory was 9,454 MiB (shmup5), and CHOLMOD reported no tiny or non-positive pivots.
- **Timeouts.** 38 runs are TIMEOUT: 18 adaptive and 20 fixed.
  - Their best and final iterates are both saved and revalidated.
  - Their time-to-target is reported as "> 1800 s".
  - The solve clock of a time-limited run also includes its final validation and the final-iterate save, up to 16 s for shmup5.
- **CONVERSION_FAILED.** None: all 26 conversions pass every check (`conversion_validation.md`).

## Deviations from the frozen protocol (`PROTOCOL.md`)

1. The bit-identity check was redone correctly. Its first version compared different iterates.
2. A test bug was fixed.
3. The initcheck reports in cuSOLVER were diagnosed.
4. Stage 1 was held during that diagnosis.
5. The LOBPCG sanitizer case was added.
6. The QR reports under LOBPCG were diagnosed.
7. **Plan A:** the σ pilot and the tuned σ were dropped, and the cap was cut from 1 h to 30 min for budget reasons, by the user's choice.

**Not done:** the tuned fixed-σ comparison, runs beyond 30 minutes, and the 1 h → 3 h extension.

## Resources

- **GPU.** 41.51 H200-hours of the 48 approved (`logs/budget.log`). This includes 15.09 h that an allocation sat idle on 2026-09-29 after the session attached to it ended; all later work used self-terminating batch jobs.
- **CPU.** 239 core-hours.
- **Storage.** 5.8 GB in `~/cuadmm-experiments/2026-09-29_plato_cold_plain_admm_sigma/`: iterates, per-iteration histories and diagnostics.

## Files

| File | Content |
|---|---|
| `plato_cold_plain_admm_sigma_results.xlsx` | Every table and every run; σ updates; references; inventory; budget |
| `results/official_table.md`, `results/stage3_table.csv` | The official table: one row per dataset and configuration, medians over repetitions |
| `results/comparison.md`, `results/stage3_comparison.csv`, `results/stage3_comparison_summary.json` | Adaptive vs fixed |
| `results/stage3_runs.csv`, `results/stage1_runs.csv` | Every run with all metrics, best and final validation, revalidation |
| `results/plots/` | External eta and max DIMACS against time; adaptive σ against iteration |
| `runs/<stage>/<run>/` | `run.json`, `summary.json`, `validation_history.csv.gz`, `sigma_log.csv.gz`, `stdout.log.gz`, `postval_{returned,final}.json` |
| `logs/` | Slurm actions, budget, campaign timeline, sanitizers, diagnosis, verification (`verification_stage3.txt`: all checks pass) |
| `results/iterate_manifest.sha256` | SHA-256 of every saved iterate, which stays on the cluster |
