# Phase-6a projector benchmark: the FP16 composite filter against the exact FP64 EVD

Plan Section "PSD cone projection", Scope item of "Integrating the composite polynomial filter": "Phase 6a benchmarks
the filter against `FullEvd` at n in {2000, 3000, 5000, 10000, 20000} on the pinned GPUs to confirm or move the
threshold" `psd.large.warmup.min_size` (default 5000). Program: `bench/psd_projector_bench.cu` (target
`cuadmm_psd_projector_bench`, option `CUADMM_BUILD_BENCH`); tables: `bench/psd_projector_table.py RESULTS_DIR`.

## What is measured

Per size n and spectrum family, one dense symmetric Z = Q diag(lambda) Q^T (FP64, column-major, symmetrized) with Q
the orthogonal factor of the QR of a Gaussian matrix from a counter-based generator seeded by (seed, family, n).
Both paths run as today's engine runs them (`SDPSolver::large_full_eig_project`, `SDPSolver::large_composite_project`):

- **FullEvd**: on a fresh copy of Z, `cusolverDnXsyevd` (FP64, vectors, lower triangle) in place, W <- max(W, 0),
  T = Q diag(W) (`cublasDdgmm`), P = T Q^T (one `cublasDgemm`). Reported: the time per projection, the device
  workspace of the `cusolverDnXsyevd_bufferSize` query as w_evd(n) / n^2 doubles, the host workspace, and the
  device memory of the path (`cudaMemGetInfo` before and after allocating the workspace, W, info, T and P).
- **CompositeFilter**: on a fresh copy of Z, ||Z||_F; `approximate_two_norm` (FP64 Lanczos, 20 steps on Z^2:
  up = sqrt(theta + ||Z^2 q - theta q||), lo = sqrt(theta)); scale s = 1.05 up (`COMPOSITE_FP16_SCALE_MARGIN`);
  `composite_FP16` on Z / s (7 steps a X + b X^3 + c X^5 with damping 1.01: 22 `cublasGemmEx` on FP16 inputs with
  FP32 accumulation, then P = (1/2) A (I + X_7), symmetrized), rescaled by s; ||P||_F; the finiteness check and the
  norm test ||P||_F <= (1 + 1e-2) ||Z||_F (`COMPOSITE_NORM_SLACK`). Reported: the time per projection and its split
  (Lanczos, filter with the two scalings, checks = the two norms); the accuracy of every run against the FP64
  projection of the same Z, ||P_fp16 - P_fp64||_F / ||P_fp64||_F (median and maximum over the runs); for the last
  run lambda_min(P_fp16) from one more `cusolverDnXsyevd` (FP64, eigenvalues only), negative when the result is not
  PSD; the fraction of runs rejected by the checks (warm-up included); the Lanczos estimate against the true
  ||Z||_2 = max |lambda_i(Z)| from the FP64 eigenvalues of Z (up / ||Z||_2 and s / ||Z||_2, min..max over the runs);
  the device memory of the path (the FP32 workspace of 3 n^2 floats and the FP16 workspace of 3 n^2 halves, n^2
  rounded up to a multiple of 4, and the result buffer). Not timed: the O(n^2) copy of the input that today's
  engine keeps to restore the block on rejection, and the copy of the result into the P buffer (the memory model
  of the plan removes the first; both are milliseconds).
- **speedup** = median FullEvd time / median CompositeFilter time.

Timing: `cudaEvent` pairs on the legacy default stream, which every call uses, `cudaDeviceSynchronize` before the
first event and `cudaEventSynchronize` on the last; one warm-up run, then the median of 5 runs (3 at n = 20000);
the host wall time of the same region is recorded next to it in `timings.csv` and agrees with the event time.
Math modes (plan B1): `cublasSetMathMode(CUBLAS_DEFAULT_MATH)` on the one cuBLAS handle, as the engine sets it;
the filter sets `CUBLAS_COMPUTE_32F_FAST_16F` with `CUBLAS_GEMM_DEFAULT_TENSOR_OP` in each `cublasGemmEx`; the FP64
GEMMs and GEMVs run under the default mode. The Lanczos start vector of the vendored kernel comes from
`std::random_device` (plan B8 makes it deterministic in Phase 6a proper), so the estimate, the scale and the FP16
result differ slightly from run to run; everything else is deterministic.

Spectrum families (eigenvalues of Z, n values each; r = ceil(0.03 n)):

| family | eigenvalues |
|---|---|
| full_rank | uniform in [-1, 1] |
| low_pos_rank | r values uniform in [0.01, 1], the rest uniform in [-1, -0.01] (the dual-side low-rank case: few positive eigenvalues) |
| low_neg_rank | r values uniform in [-1, -0.01], the rest uniform in [0.01, 1] (the primal low-rank case) |
| clustered | c + noise, c in {-1, 0, 1} in equal thirds, noise uniform in [-1e-3, 1e-3] (a third of the spectrum inside the filter's transition region) |
| spread | sign * 10^u, u uniform in [-6, 0]: magnitudes log-uniform in [1e-6, 1], random signs (shows the FP16 limits) |

## Results on the A100 (job 54783686, 2026-10-22)

One NVIDIA A100-SXM4-80GB, 5 runs per path after one warm-up (3 at n = 20000), every size and family completed in
7.2 minutes of program time (the 40-minute budget was never reached, nothing was skipped). Times are medians of the
`cudaEvent` time per projection in milliseconds; `rejected` counts the warm-up run too; the Lanczos columns are
ranges over the runs.

| n | family | FullEvd ms | Composite ms | of which Lanczos / filter / checks ms | speedup | rel. Frobenius error median / max | lambda_min(P_fp16) | rejected | Lanczos up / true (min..max) | scale / true (min..max) |
|---:|---|---:|---:|---|---:|---|---:|---:|---|---|
| 2000 | full_rank | 44.0 | 6.7 | 3.2 / 3.4 / 0.1 | 6.58 | 7.80e-04 / 7.84e-04 | -1.26e-03 | 0/6 | 1.00300..1.00518 | 1.0531..1.0554 |
| 2000 | low_pos_rank | 39.9 | 6.7 | 3.2 / 3.4 / 0.1 | 5.98 | 1.58e-03 / 1.59e-03 | -2.41e-04 | 0/6 | 1.00420..1.00583 | 1.0544..1.0561 |
| 2000 | low_neg_rank | 39.8 | 6.7 | 3.2 / 3.4 / 0.1 | 5.97 | 3.32e-04 / 3.36e-04 | -7.62e-05 | 0/6 | 1.00250..1.00336 | 1.0526..1.0535 |
| 2000 | clustered | 40.1 | 6.7 | 3.2 / 3.4 / 0.1 | 6.01 | 1.16e-03 / 1.16e-03 | -1.19e-03 | 0/6 | 1.00017..1.00023 | 1.0502..1.0502 |
| 2000 | spread | 38.9 | 6.7 | 3.2 / 3.4 / 0.1 | 5.83 | 1.81e-03 / 1.84e-03 | -1.77e-03 | 0/6 | 1.00323..1.00878 | 1.0534..1.0592 |
| 3000 | full_rank | 100.6 | 13.9 | 4.1 / 9.6 / 0.2 | 7.25 | 8.12e-04 / 8.13e-04 | -1.15e-03 | 0/6 | 1.00112..1.00222 | 1.0512..1.0523 |
| 3000 | low_pos_rank | 100.3 | 13.8 | 4.0 / 9.6 / 0.2 | 7.25 | 1.42e-03 / 1.44e-03 | -2.30e-04 | 0/6 | 1.00233..1.00317 | 1.0524..1.0533 |
| 3000 | low_neg_rank | 100.6 | 13.8 | 4.0 / 9.6 / 0.2 | 7.27 | 3.19e-04 / 3.20e-04 | -5.79e-05 | 0/6 | 1.00349..1.00391 | 1.0537..1.0541 |
| 3000 | clustered | 100.8 | 13.9 | 4.1 / 9.6 / 0.2 | 7.28 | 1.16e-03 / 1.16e-03 | -1.21e-03 | 0/6 | 1.00013..1.00015 | 1.0501..1.0502 |
| 3000 | spread | 97.7 | 13.9 | 4.0 / 9.6 / 0.2 | 7.05 | 1.53e-03 / 1.54e-03 | -1.74e-03 | 0/6 | 0.99972..1.00066 | 1.0497..1.0507 |
| 5000 | full_rank | 344.3 | 44.4 | 8.1 / 35.9 / 0.4 | 7.76 | 7.84e-04 / 7.86e-04 | -1.03e-03 | 0/6 | 1.00304..1.00363 | 1.0532..1.0538 |
| 5000 | low_pos_rank | 343.1 | 44.4 | 8.2 / 35.9 / 0.4 | 7.73 | 1.37e-03 / 1.38e-03 | -2.36e-04 | 0/6 | 1.00226..1.00450 | 1.0524..1.0547 |
| 5000 | low_neg_rank | 343.6 | 44.4 | 8.1 / 35.9 / 0.4 | 7.74 | 3.37e-04 / 3.37e-04 | -5.80e-05 | 0/6 | 1.00322..1.00437 | 1.0534..1.0546 |
| 5000 | clustered | 344.7 | 44.5 | 8.2 / 36.0 / 0.4 | 7.74 | 1.17e-03 / 1.17e-03 | -1.23e-03 | 0/6 | 1.00017..1.00019 | 1.0502..1.0502 |
| 5000 | spread | 334.9 | 44.5 | 8.1 / 36.0 / 0.4 | 7.53 | 1.70e-03 / 1.70e-03 | -1.94e-03 | 0/6 | 1.00134..1.00569 | 1.0514..1.0560 |
| 10000 | full_rank | 1827.5 | 228.2 | 22.7 / 204.3 / 1.1 | 8.01 | 7.77e-04 / 7.78e-04 | -1.06e-03 | 0/6 | 1.00260..1.00313 | 1.0527..1.0533 |
| 10000 | low_pos_rank | 1823.3 | 227.8 | 22.7 / 203.9 / 1.1 | 8.00 | 1.39e-03 / 1.40e-03 | -2.32e-04 | 0/6 | 1.00253..1.00292 | 1.0527..1.0531 |
| 10000 | low_neg_rank | 1827.0 | 227.8 | 22.7 / 203.9 / 1.1 | 8.02 | 3.29e-04 / 3.30e-04 | -5.76e-05 | 0/6 | 1.00242..1.00287 | 1.0525..1.0530 |
| 10000 | clustered | 1828.6 | 228.2 | 22.7 / 204.0 / 1.1 | 8.01 | 1.16e-03 / 1.16e-03 | -1.22e-03 | 0/6 | 1.00016..1.00019 | 1.0502..1.0502 |
| 10000 | spread | 1777.0 | 228.4 | 22.8 / 204.3 / 1.1 | 7.78 | 1.81e-03 / 1.81e-03 | -2.00e-03 | 0/6 | 1.00254..1.00597 | 1.0527..1.0563 |
| 20000 | full_rank | 12359.3 | 1616.6 | 78.4 / 1533.4 / 4.3 | 7.65 | 8.02e-04 / 8.02e-04 | -1.22e-03 | 0/4 | 1.00251..1.00289 | 1.0526..1.0530 |
| 20000 | low_pos_rank | 12331.0 | 1614.9 | 78.3 / 1532.4 / 4.3 | 7.64 | 1.49e-03 / 1.50e-03 | -2.40e-04 | 0/4 | 1.00232..1.00293 | 1.0524..1.0531 |
| 20000 | low_neg_rank | 12356.6 | 1610.5 | 78.4 / 1527.9 / 4.3 | 7.67 | 3.35e-04 / 3.35e-04 | -6.54e-05 | 0/4 | 1.00308..1.00358 | 1.0532..1.0538 |
| 20000 | clustered | 12375.1 | 1613.4 | 78.4 / 1530.8 / 4.3 | 7.67 | 1.16e-03 / 1.16e-03 | -1.23e-03 | 0/4 | 1.00017..1.00018 | 1.0502..1.0502 |
| 20000 | spread | 11983.8 | 1614.4 | 78.4 / 1531.7 / 4.3 | 7.42 | 1.71e-03 / 1.71e-03 | -2.00e-03 | 0/4 | 1.00155..1.00257 | 1.0516..1.0527 |

| n | w_evd(n) device bytes | w_evd(n) / n^2 (doubles) | Xsyevd host workspace bytes | FullEvd path GiB (workspace + W + info + T + P) | FullEvd path / n^2 | CompositeFilter path GiB (FP32 + FP16 workspaces + P) | CompositeFilter path / n^2 | runs per path | true ||Z||_2 (full_rank) | r+ / r- (full_rank) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2000 | 130295960 | 4.0717 | 0 | 0.186 | 6.226 | 0.100 | 3.342 | 5 | 0.999782 | 1010 / 990 |
| 3000 | 291311896 | 4.0460 | 0 | 0.408 | 6.088 | 0.221 | 3.291 | 5 | 0.999835 | 1566 / 1434 |
| 5000 | 805344024 | 4.0267 | 0 | 1.127 | 6.050 | 0.609 | 3.272 | 5 | 0.999964 | 2509 / 2491 |
| 10000 | 3210423960 | 4.0130 | 0 | 4.482 | 6.016 | 2.426 | 3.256 | 5 | 0.999811 | 5043 / 4957 |
| 20000 | 12820583832 | 4.0064 | 0 | 17.902 | 6.007 | 9.688 | 3.251 | 3 | 0.999992 | 9977 / 10023 |

Per-run data: `phase6a_projector_benchmark_data/a100_timings.csv`; summary rows: `a100_results.csv`; GPU and
options: `a100_run_info.txt`; the job log: `a100_job_54783686.out`. The host wall time of every timed region agrees
with the event time to within 0.1 ms (columns `wall_ms` and `ms`).

### Observations

- **Time.** FullEvd (full_rank): 44.0, 100.6, 344, 1827, 12359 ms at n = 2000, 3000, 5000, 10000, 20000;
  CompositeFilter: 6.7, 13.9, 44.4, 228, 1617 ms, of which the Lanczos estimate 3.2, 4.1, 8.1, 22.7, 78.4 ms and
  the filter 3.4, 9.6, 35.9, 204, 1533 ms. The speedup is 5.8-6.6x at n = 2000, 7.1-7.3x at 3000, 7.5-7.8x at 5000,
  7.8-8.0x at 10000 and 7.4-7.7x at 20000: flat from n = 3000 on. At n = 2000 the Lanczos loop (40 FP64 GEMVs with
  host-side dot products and norms, and its own allocations) is half of the composite time; from n = 5000 on the
  22 FP16 GEMMs dominate (at n = 20000 they run at about 230 TFLOPS: 22 x 2 n^3 flop in 1.53 s). Within a family
  the EVD runs agree to 0.3% or better, except in the very first case of the run (n = 2000 full_rank: 39.9-44.1 ms, median
  44.0 ms, the GPU's first seconds; the other families at n = 2000 give 39-40 ms, so 6.0x is the typical speedup at
  n = 2000 and the 6.6x of that row is inflated) and two cases with one slower run (1.2-1.3%); the `spread`
  family's EVD is 2-3% faster than the others. Other outliers exist only in the warm-ups (the first composite call
  at n = 2000 took 106 ms, library initialization) and in one of the 25 composite runs at n = 10000 (295 ms against
  a 228 ms median); the medians are unaffected.
- **Accuracy.** The relative Frobenius error of the FP16 projection does not depend on n: 3.2e-4 to 3.4e-4
  (low_neg_rank), 7.8e-4 to 8.1e-4 (full_rank), 1.16e-3 to 1.17e-3 (clustered), 1.37e-3 to 1.58e-3 (low_pos_rank),
  1.53e-3 to 1.84e-3 (spread). The absolute error is about 1e-3 ||Z||_2 per eigenvalue in every family; the relative
  error is larger when ||P_fp64||_F is small (low_pos_rank keeps only 3% of the spectrum) and smaller when P is
  almost Z (low_neg_rank). lambda_min(P_fp16) is between -5.8e-5 (low_neg_rank) and -2.0e-3 (spread) times
  ||Z||_2 ~ 1 at every size: the FP16 result is never PSD, as expected of a polynomial filter whose step is
  approximated to about 1e-3; the `spread` family (half of its eigenvalues below 1e-3 in magnitude, where the filter
  cannot resolve the sign) gives the largest violation but only a slightly larger Frobenius error, because those
  eigenvalues contribute little to the norm.
- **Rejections.** 0 of 140 composite runs failed the finiteness check or the norm test
  ||P||_F <= (1 + 1e-2) ||Z||_F (the measured ||P||_F / ||Z||_F is at most 0.985 in `timings.csv`: the projection
  shrinks the norm).
- **Lanczos estimate.** up / ||Z||_2 lies in 0.9997..1.0088 over the 140 runs; it fell below 1 once (n = 3000,
  spread, run 5: 0.99972), confirming that the 20-step estimate is not a certified bound. The certified lower bound
  lo / ||Z||_2 lies in 0.9952..0.99999. With the margin 1.05 the scale s satisfied s / ||Z||_2 >= 1.0497 in every
  run, so the scaled spectrum stayed within [-0.953, 0.953], well below the ~1.02 at which the filter overflows.
  The estimate is tightest on the clustered spectrum (three eigenvalues of Z^2, up / true = 1.0002) and loosest on
  the spread spectrum.
- **Memory.** w_evd(n) / n^2 = 4.072, 4.046, 4.027, 4.013, 4.006 doubles at n = 2000..20000 (12.82 GB at
  n = 20000), host workspace 0 bytes, consistent with the plan's 4.16 n^2 at n = 1000 and 4.004 n^2 at n = 30000.
  In today's layout the FullEvd path allocates 6.0-6.2 n^2 doubles (workspace + T + P: 17.9 GiB at n = 20000) and
  the composite path 3.25-3.34 n^2 (FP32 and FP16 workspaces + P: 9.7 GiB at n = 20000); the FP32/FP16 buffers
  alone are 2.25 n^2, which fit inside the EVD workspace region W of the plan's arena.

## Supplementary run on a gpu_test MIG slice (job 54791237)

The same program for n in {2000, 3000, 5000, 10000} on one A100-SXM4-40GB MIG 3g.20gb slice (42 of 108 SMs; n = 20000
needs about 36 GB and does not fit the 20 GB slice), 5 runs per path. The accuracy columns reproduce the A100 ones
(same seeds, same kernels; the FP16 results differ only through the random Lanczos start vector), the times are
1.5-2.4x longer and the speedup is 5.1-5.3x at n = 2000, 6.2-6.4x at 3000, 6.4-6.6x at 5000 and 6.8-7.1x at 10000.
Data: `phase6a_projector_benchmark_data/mig_*`.

| n | family | FullEvd ms | Composite ms | of which Lanczos / filter / checks ms | speedup | rel. Frobenius error median / max | lambda_min(P_fp16) | rejected | Lanczos up / true (min..max) | scale / true (min..max) |
|---:|---|---:|---:|---|---:|---|---:|---:|---|---|
| 2000 | full_rank | 65.3 | 12.4 | 4.3 / 7.9 / 0.2 | 5.27 | 7.80e-04 / 7.82e-04 | -1.21e-03 | 0/6 | 1.00248..1.00489 | 1.0526..1.0551 |
| 2000 | low_pos_rank | 65.2 | 12.4 | 4.3 / 7.9 / 0.2 | 5.26 | 1.57e-03 / 1.58e-03 | -2.36e-04 | 0/6 | 1.00050..1.00536 | 1.0505..1.0556 |
| 2000 | low_neg_rank | 65.1 | 12.4 | 4.3 / 7.9 / 0.2 | 5.25 | 3.29e-04 / 3.35e-04 | -7.40e-05 | 0/6 | 1.00246..1.00323 | 1.0526..1.0534 |
| 2000 | clustered | 65.6 | 12.4 | 4.3 / 7.9 / 0.2 | 5.30 | 1.16e-03 / 1.16e-03 | -1.20e-03 | 0/6 | 1.00017..1.00023 | 1.0502..1.0502 |
| 2000 | spread | 63.5 | 12.4 | 4.3 / 7.9 / 0.2 | 5.13 | 1.75e-03 / 1.84e-03 | -1.81e-03 | 0/6 | 1.00067..1.00748 | 1.0507..1.0579 |
| 3000 | full_rank | 186.6 | 29.4 | 7.1 / 21.9 / 0.3 | 6.35 | 8.11e-04 / 8.12e-04 | -1.32e-03 | 0/6 | 1.00093..1.00202 | 1.0510..1.0521 |
| 3000 | low_pos_rank | 186.2 | 29.3 | 7.1 / 21.9 / 0.3 | 6.34 | 1.41e-03 / 1.42e-03 | -2.27e-04 | 0/6 | 1.00260..1.00360 | 1.0527..1.0538 |
| 3000 | low_neg_rank | 186.6 | 29.4 | 7.1 / 22.0 / 0.3 | 6.35 | 3.18e-04 / 3.20e-04 | -6.39e-05 | 0/6 | 1.00289..1.00393 | 1.0530..1.0541 |
| 3000 | clustered | 187.5 | 29.4 | 7.1 / 21.9 / 0.3 | 6.39 | 1.16e-03 / 1.16e-03 | -1.21e-03 | 0/6 | 1.00012..1.00016 | 1.0501..1.0502 |
| 3000 | spread | 181.4 | 29.4 | 7.1 / 21.9 / 0.3 | 6.18 | 1.53e-03 / 1.53e-03 | -1.73e-03 | 0/6 | 1.00046..1.00070 | 1.0505..1.0507 |
| 5000 | full_rank | 662.1 | 100.7 | 14.8 / 85.2 / 0.8 | 6.57 | 7.86e-04 / 7.86e-04 | -9.42e-04 | 0/6 | 1.00290..1.00357 | 1.0530..1.0537 |
| 5000 | low_pos_rank | 659.8 | 100.6 | 14.7 / 85.2 / 0.7 | 6.56 | 1.37e-03 / 1.37e-03 | -2.45e-04 | 0/6 | 1.00271..1.00416 | 1.0528..1.0544 |
| 5000 | low_neg_rank | 659.3 | 100.7 | 14.8 / 85.2 / 0.7 | 6.55 | 3.36e-04 / 3.39e-04 | -6.68e-05 | 0/6 | 1.00323..1.00364 | 1.0534..1.0538 |
| 5000 | clustered | 662.8 | 100.6 | 14.8 / 85.1 / 0.7 | 6.59 | 1.17e-03 / 1.17e-03 | -1.23e-03 | 0/6 | 1.00019..1.00021 | 1.0502..1.0502 |
| 5000 | spread | 640.9 | 100.7 | 14.8 / 85.2 / 0.7 | 6.37 | 1.70e-03 / 1.70e-03 | -1.97e-03 | 0/6 | 1.00284..1.01050 | 1.0530..1.0610 |
| 10000 | full_rank | 3849.5 | 545.3 | 50.2 / 492.3 / 2.7 | 7.06 | 7.78e-04 / 7.78e-04 | -9.83e-04 | 0/6 | 1.00244..1.00350 | 1.0526..1.0537 |
| 10000 | low_pos_rank | 3841.6 | 545.0 | 50.1 / 492.3 / 2.7 | 7.05 | 1.40e-03 / 1.40e-03 | -2.31e-04 | 0/6 | 1.00262..1.00314 | 1.0528..1.0533 |
| 10000 | low_neg_rank | 3841.7 | 544.9 | 50.1 / 492.1 / 2.7 | 7.05 | 3.29e-04 / 3.31e-04 | -6.46e-05 | 0/6 | 1.00255..1.00297 | 1.0527..1.0531 |
| 10000 | clustered | 3848.8 | 544.7 | 50.0 / 492.0 / 2.7 | 7.07 | 1.16e-03 / 1.16e-03 | -1.23e-03 | 0/6 | 1.00017..1.00020 | 1.0502..1.0502 |
| 10000 | spread | 3723.2 | 544.8 | 50.1 / 492.0 / 2.7 | 6.83 | 1.81e-03 / 1.81e-03 | -1.99e-03 | 0/6 | 1.00359..1.00596 | 1.0538..1.0563 |

## GPU, build and jobs

| item | value |
|---|---|
| GPU (A100 run) | NVIDIA A100-SXM4-80GB, 108 SMs, 81175 MiB (node holygpu8a22203), `nvidia-smi` driver 610.57.04 (CUDA driver API 13.3) |
| GPU (MIG runs) | NVIDIA A100-SXM4-40GB MIG 3g.20gb, 42 SMs, 20096 MiB (nodes holygpu7c26201, holygpu7c26203), same driver |
| CUDA | runtime 12.9 (module `cuda/12.9.1-fasrc01`), cuBLAS 12.9.1 (120901), cuSOLVER 11.7.5 (11705) |
| Build | commit 89b4500 (clean), GCC 12.2.0, RelWithDebInfo (`-O2 -g -DNDEBUG`), `CMAKE_CUDA_ARCHITECTURES=80-real`, `tools/build.sh $HOME/cuadmm-builds/reorg-bench6a -DCUADMM_BUILD_BENCH=ON` on the CPU allocation 54779332 |
| Math modes | `CUBLAS_DEFAULT_MATH` on the handle; the filter's `cublasGemmEx` with `CUBLAS_COMPUTE_32F_FAST_16F` |
| Jobs | 54783059 (`gpu_test`, smoke: n = 2000 and 5000, 17 s), 54791237 (`gpu_test`, n <= 10000, 3 min 24 s), 54783686 (`gpu`, `--gres=gpu:nvidia_a100-sxm4-80gb:1 -c 16 --mem=96G`, elapsed 7 min 19 s, queued 1 h 22 min) |
| A100 time used | 7 min 19 s (`sacct` elapsed), 0.12 A100-hours; `gpu_test` 3 min 41 s in total |
| Seed | 20261022 (eigenvalues, Gaussian matrix); the Lanczos start vector is unseeded in the vendored kernel |

## Conclusion on the threshold n >= 5000

The plan sets `psd.large.warmup.min_size = 5000` because 5000 is the smallest size evaluated in the composite-filter
paper and "the GEMM advantage needs n in the thousands". The measurements do not single out n = 5000: on the A100
the composite projection is already about 6x faster than the exact EVD at n = 2000 (39-44 ms against 6.7 ms per
block and iteration), 7.1-7.3x at n = 3000, and the speedup is flat at 7.4-8.0x from n = 3000 to 20000; the accuracy
(relative error 3e-4 to 1.8e-3, lambda_min(P_fp16) >= -2.0e-3 ||Z||_2) is the same at every size, and none of the
140 runs was rejected. The per-projection numbers therefore support lowering the default to
`psd.large.warmup.min_size = 2000`, the smallest size measured; 5000 remains a valid conservative value and blocks
with 1000 < n < 2000 should keep exact projections until measured (the speedup at n = 2000 is already limited by the
Lanczos loop rather than by the GEMMs, so the crossover is below 2000, but it was not measured). Two things this
benchmark does not decide and the Phase-6a end-to-end tests with and without the warm-up must: whether the 1e-3
projection error costs ADMM iterations during the warm-up (it ends at eta < 1e-2, where that error is below the
residual), and whether the Lanczos overhead at small n matters next to the rest of an iteration. The switch
criteria (eta < 1e-2 or k > 5000) are not affected.

## Reproduction

```
# build (CPU node), run (one A100), tables
tools/build.sh $HOME/cuadmm-builds/reorg-bench6a -DCUADMM_BUILD_BENCH=ON
sbatch -p gpu --gres=gpu:nvidia_a100-sxm4-80gb:1 -c 16 --mem=96G -t 00:59:00 -J bench6a \
  --wrap "source ~/cuadmm-tools/env.sh; export OMP_NUM_THREADS=16; \
          $HOME/cuadmm-builds/reorg-bench6a/cuadmm_psd_projector_bench --out $HOME/cuadmm-scratch-bench6a/results"
python -B bench/psd_projector_table.py $HOME/cuadmm-scratch-bench6a/results [--timings]
# options: --smoke (n = 2000, one run), --sizes, --families, --reps, --reps-large, --budget-min, --seed
```
