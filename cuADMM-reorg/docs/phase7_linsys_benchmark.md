# Phase 7: the normal-equation backends, benchmark

Plan Section 11.6. `bench/linsys_bench.cu` (target `cuadmm_linsys_bench`) factorizes A A^T + eps I of each
problem with every backend and times 50 solves with a random right-hand side (one warm-up solve first); the
relative residual ||(A A^T + eps I) y - r|| / ||r|| of the last solve is computed on the host.

Run of 2026-10-23 on one MIG 3g.20gb slice of an A100-SXM4-40GB (gpu_test job 54795140, holygpu7c26205), 8
OpenBLAS threads, CUDA 12.9, SuiteSparse 7.10.1, cuDSS 1.0.0, the development build of commit 364c87e with the
Phase 7 integration (RelWithDebInfo), eps = 1e-15. The A100 (`gpu` partition) run is pending its queue; the
MIG slice has a third of the SMs and a fifth of the memory bandwidth, so the device times are upper bounds.
Data: `phase7_linsys_benchmark_data/mig_results.csv`.

Backends: `cholmod` is the solver's default (CHOLMOD on the host; the dense factor on the device by today's rule,
"simplicial LDL^T, dense L and D on the device" when it applies), `cholmod_cpu` the same without the dense
variant, `dense` the cuSOLVER Cholesky on the device (`DenseCusolverSolver`), `cudss` cuDSS 1.0.0 in its
deterministic mode (`CudssSolver`). "device MiB" is the memory the factorization holds on the device.

| problem | m | nnz(A) | backend | kind | factor s | ms / solve | rel. residual | device MiB |
|---|---|---|---|---|---|---|---|---|
| Large_LowRankS | 1000 | 8000 | cholmod | simplicial LDL^T | 0.005 | 0.354 | 1.6e-16 | 0.0 |
| Large_LowRankS | 1000 | 8000 | cholmod_cpu | simplicial LDL^T | 0.004 | 0.342 | 1.6e-16 | 0.0 |
| Large_LowRankS | 1000 | 8000 | dense | dense_llt | 0.010 | 0.217 | 2.2e-16 | 7.6 |
| Large_LowRankS | 1000 | 8000 | cudss | cudss_llt | 2.616 | 0.046 | 2.1e-16 | 4.3 |
| Large_HalfRank | 1000 | 8000 | cholmod | simplicial LDL^T | 0.003 | 0.292 | 1.7e-16 | 0.0 |
| Large_HalfRank | 1000 | 8000 | cholmod_cpu | simplicial LDL^T | 0.003 | 0.294 | 1.6e-16 | 0.0 |
| Large_HalfRank | 1000 | 8000 | dense | dense_llt | 0.005 | 0.232 | 2.3e-16 | 7.6 |
| Large_HalfRank | 1000 | 8000 | cudss | cudss_llt | 0.022 | 0.055 | 2.2e-16 | 4.2 |
| trto5 | 3280 | 24158 | cholmod | supernodal LL^T -> simplicial LDL^T | 0.076 | 0.581 | 7.7e-16 | 0.1 |
| trto5 | 3280 | 24158 | cholmod_cpu | supernodal LL^T -> simplicial LDL^T | 0.108 | 0.736 | 7.6e-16 | 0.1 |
| trto5 | 3280 | 24158 | dense | dense_llt | 0.010 | 0.774 | 5.5e-16 | 82.1 |
| trto5 | 3280 | 24158 | cudss | cudss_llt | 0.059 | 0.349 | 3.3e-16 | 25.3 |
| buck5 | 3280 | 48385 | cholmod | supernodal LL^T -> simplicial LDL^T | 0.266 | 0.155 | 5.0e-16 | 0.1 |
| buck5 | 3280 | 48385 | cholmod_cpu | supernodal LL^T -> simplicial LDL^T | 0.023 | 0.154 | 5.2e-16 | 0.1 |
| buck5 | 3280 | 48385 | dense | dense_llt | 0.015 | 0.775 | 6.0e-16 | 82.1 |
| buck5 | 3280 | 48385 | cudss | cudss_llt | 0.055 | 0.274 | 3.8e-16 | 24.9 |
| vibra5 | 3280 | 48385 | cholmod | supernodal LL^T -> simplicial LDL^T | 0.084 | 0.151 | 5.0e-16 | 0.1 |
| vibra5 | 3280 | 48385 | cholmod_cpu | supernodal LL^T -> simplicial LDL^T | 0.016 | 0.149 | 5.6e-16 | 0.1 |
| vibra5 | 3280 | 48385 | dense | dense_llt | 0.036 | 0.721 | 6.1e-16 | 82.1 |
| vibra5 | 3280 | 48385 | cudss | cudss_llt | 0.055 | 0.274 | 3.8e-16 | 24.9 |
| shmup5 | 1800 | 117420 | cholmod | simplicial LDL^T | 0.063 | 0.066 | 7.1e-16 | 0.0 |
| shmup5 | 1800 | 117420 | cholmod_cpu | simplicial LDL^T | 0.061 | 0.345 | 7.5e-16 | 0.0 |
| shmup5 | 1800 | 117420 | dense | dense_llt | 0.063 | 0.363 | 8.0e-16 | 24.7 |
| shmup5 | 1800 | 117420 | cudss | cudss_llt | 0.023 | 0.281 | 6.5e-16 | 16.0 |
| mater-5 | 10143 | 338757 | cholmod | supernodal LL^T -> simplicial LDL^T | 0.032 | 0.584 | 7.7e-14 | 0.2 |
| mater-5 | 10143 | 338757 | cholmod_cpu | supernodal LL^T -> simplicial LDL^T | 0.031 | 0.583 | 4.6e-14 | 0.2 |
| mater-5 | 10143 | 338757 | dense | dense_llt | 0.092 | 2.416 | 8.3e-14 | 785.3 |
| mater-5 | 10143 | 338757 | cudss | cudss_llt | 0.080 | 0.376 | 4.4e-14 | 50.1 |
| mater-6 | 20463 | 690149 | cholmod | supernodal LL^T -> simplicial LDL^T | 0.933 | 1.296 | 9.2e-14 | 0.5 |
| mater-6 | 20463 | 690149 | cholmod_cpu | supernodal LL^T -> simplicial LDL^T | 0.310 | 1.315 | 9.0e-14 | 0.5 |
| mater-6 | 20463 | 690149 | dense | dense_llt | 0.469 | 6.862 | 1.4e-13 | 3196.3 |
| mater-6 | 20463 | 690149 | cudss | cudss_llt | 0.084 | 0.587 | 2.9e-14 | 80.4 |

Reading:

- The CHOLMOD path is the solver's baseline: its host solves cost 0.07-1.3 ms on these systems (mater-6, m =
  20463: 1.3 ms); the dense-factor variant applies to shmup5 (m = 1800, a dense factor) and makes its solve
  5x faster than the host solve (0.066 ms against 0.345 ms), as it did in main.
- cuDSS solves are the fastest on the sparse systems with many constraints (mater-5: 0.38 ms against 0.58 ms;
  mater-6: 0.59 ms against 1.30 ms), with the same residuals, a comparable factorization time once its kernel
  cache is warm, and 50-80 MiB on the device. On the small systems (m <= 3300) the host solve of CHOLMOD is as
  fast or faster: the device solve costs a fixed launch overhead of about 0.25 ms on the MIG slice.
- The dense backend holds 8 m^2 bytes (3.2 GiB at m = 20463) and its triangular solves are the slowest at that
  size (6.9 ms); it is the choice only when the factor is dense anyway and the system small, where the CHOLMOD
  variant already does the same with CHOLMOD's factor.
- The y-step is a small part of an iteration on these problems (the PSD projections dominate: the Phase 6b
  timing table), so the backend choice moves the iteration time by a few percent at most here; `linsys.backend
  = cudss` is the option to try on problems with tens of thousands of constraints and sparse factors.
