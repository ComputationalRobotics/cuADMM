/*

    psd_projector_bench.cu  (plan Section "PSD cone projection", "Integrating the composite polynomial filter"; the
    recorded run: c4eb89b:cuADMM-reorg/docs/phase6a_projector_benchmark.md)

    The FP16 composite polynomial filter (CompositeFilter) against the exact FP64 eigendecomposition (FullEvd) on
    synthetic dense symmetric blocks of size n in {2000, 3000, 5000, 10000, 20000}, called as the engine calls them
    today (src/legacy/solver.cu: SDPSolver::large_full_eig_project and SDPSolver::large_composite_project).

    Input per size n and spectrum family: Z = Q diag(lambda) Q^T, FP64, column-major, symmetrized as (Z + Z^T) / 2,
    with Q the orthogonal factor of the QR decomposition (cusolverDnDgeqrf + cusolverDnDorgqr) of a Gaussian matrix
    drawn by a counter-based generator (splitmix64 + Box-Muller) seeded by (seed, family, n). Everything but the
    Lanczos start vector is deterministic.

    Families (eigenvalues lambda_i, n values each; r = ceil(0.03 n)):
      full_rank     uniform in [-1, 1]
      low_pos_rank  r values uniform in [0.01, 1], the rest uniform in [-1, -0.01]
      low_neg_rank  r values uniform in [-1, -0.01], the rest uniform in [0.01, 1]
      clustered     c + noise with c in {-1, 0, 1} in equal thirds and noise uniform in [-1e-3, 1e-3]
      spread        sign * 10^u with u uniform in [-6, 0] (magnitudes log-uniform in [1e-6, 1], random signs): shows
                    the FP16 limits, since the filter cannot resolve eigenvalues below about 1e-3 of the scale

    Measured (cudaEvent pairs on the legacy default stream, which every call here uses; cudaDeviceSynchronize before
    the first event and cudaEventSynchronize on the last; one warm-up run, then the median of R runs: R = 5, R = 3
    at n >= 20000, R = 1 with --smoke; the host wall time of the same region is recorded next to it):
      (a) FullEvd: on a fresh copy of Z, cusolverDnXsyevd (FP64, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
          default cusolverDnParams) in place, W <- max(W, 0), T = Q diag(W) (cublasDdgmm), P = T Q^T (one cublasDgemm
          N/T, as the engine). Also the device workspace of the cusolverDnXsyevd_bufferSize query, w_evd(n), reported
          as w_evd(n) / (8 n^2) = doubles per n^2, the host workspace, and the device memory of the path:
          cudaMemGetInfo before and after allocating the workspace, W, info, T and P.
      (b) CompositeFilter: on a fresh copy of Z, ||Z||_F (cublasDnrm2); approximate_two_norm (FP64 Lanczos, 20 steps
          on Z^2, estimate up = sqrt(theta + ||Z^2 q - theta q||), lower bound lo = sqrt(theta); the vendored kernel
          draws its start vector from std::random_device, so the estimate changes from run to run and is recorded
          per run); scale s = 1.05 up (COMPOSITE_FP16_SCALE_MARGIN; s = 1 if up <= 0); A <- Z / s; composite_FP16
          (7 steps of a X + b X^3 + c X^5 with damping 1.01 on the input: 22 cublasGemmEx with FP16 inputs and FP32
          accumulation and output, then P = (1/2) A (I + X_7), symmetrized); A <- s A; ||P||_F; the finiteness check
          and the norm test ||P||_F <= (1 + 1e-2) ||Z||_F (COMPOSITE_NORM_SLACK). The time is split into Lanczos,
          filter (the two scalings included) and checks (the two norms). Not timed, following the memory model of
          the plan (Section "Large-block pipeline"): the copy of the input that today's engine keeps to restore the
          block on rejection, and the copy of the result into the P buffer (both O(n^2) copies, milliseconds).
          Accuracy of every run against the FP64 projection of the same Z: ||P_fp16 - P_fp64||_F / ||P_fp64||_F
          (median and maximum over the runs); for the last run, lambda_min(P_fp16) from one more cusolverDnXsyevd
          (FP64, CUSOLVER_EIG_MODE_NOVECTOR) of P_fp16, negative when the result is not PSD. Also the fraction of
          runs rejected by the checks (warm-up included) and the Lanczos estimate against the true
          ||Z||_2 = max |lambda_i(Z)| from the FP64 eigenvalues of Z (ranges over the runs of up / ||Z||_2,
          lo / ||Z||_2 and s / ||Z||_2). Memory of the path: cudaMemGetInfo before and after allocating the FP32
          workspace (3 n^2 floats, n^2 rounded up to a multiple of 4), the FP16 workspace (3 n^2 halves, same
          rounding) and the result buffer; the Lanczos kernel allocates its own (20 + 4) n doubles per call, which
          this does not see.
      (c) speedup = median FullEvd time / median CompositeFilter time, per size and family.

    Math modes (plan B1): cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) on the one cuBLAS handle, as the engine
    sets it on cublasH_composite_proj; CUBLAS_TENSOR_OP_MATH is never set. The filter sets its own compute type per
    call (cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_16F and CUBLAS_GEMM_DEFAULT_TENSOR_OP: FP16 Tensor Cores with
    FP32 accumulation). The FP64 GEMM of the FullEvd product, the Lanczos GEMVs and the construction of Z run under
    CUBLAS_DEFAULT_MATH (on Ampere, cuBLAS may use the FP64 Tensor Cores for DGEMM under this mode; nothing here
    changes that default).

    Time budget: --budget-min M (default 40). Before every case the time of the previous case at the same size
    (or, for the first case of a size, the previous size's last case scaled by (n / n_prev)^3) is the estimate; a
    case that would end past the budget is skipped and recorded as "skipped_budget". At n = 20000 the family order
    full_rank, low_pos_rank, low_neg_rank, clustered, spread drops the clustered and spread families first. Every
    case is appended to the CSVs as soon as it completes.

    usage: cuadmm_psd_projector_bench [--out DIR] [--smoke] [--sizes 2000,3000,...] [--families a,b,...]
                                      [--reps 5] [--reps-large 3] [--budget-min 40] [--seed 20261022]
      --smoke: n = 2000 only, one repetition after the warm-up
    Writes DIR/results.csv (one row per size and family), DIR/timings.csv (one row per run) and DIR/run_info.txt
    (GPU, driver, CUDA runtime, cuBLAS and cuSOLVER versions, build commit, options), and prints a Markdown table.

*/

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/check.h"
#include "cuadmm/version.h"
#include "cone/psd/kernels/composite_fp16.h"
#include "cone/psd/kernels/lanczos.h"

using cuadmm::cone::psd::approximate_two_norm;
using cuadmm::cone::psd::composite_fp16;

namespace
{

// the engine's constants (src/legacy/include/cuadmm_legacy/solver.h)
constexpr double kScaleMargin = 1.05; // COMPOSITE_FP16_SCALE_MARGIN
constexpr double kNormSlack = 1e-2;   // COMPOSITE_NORM_SLACK
constexpr int kLanczosSteps = 20;     // approximate_two_norm default, as the engine calls it
static unsigned long long lanczos_seed = 1; // one start vector per composite run (B8)
constexpr double kLanczosTol = 1e-10; // idem
constexpr int kMaxEvdBlock = 32768;   // CUADMM_MAX_EVD_BLOCK_SIZE (B21)

struct Options
{
    std::string out = ".";
    std::vector<int> sizes = {2000, 3000, 5000, 10000, 20000};
    std::vector<std::string> families = {"full_rank", "low_pos_rank", "low_neg_rank", "clustered", "spread"};
    int reps = 5;
    int reps_large = 3;
    int large_from = 20000;
    double budget_min = 40.0;
    std::uint64_t seed = 20261022ull;
    bool smoke = false;
};

// ------------------------------------------------------------------------------------------------ random numbers

__host__ __device__ inline std::uint64_t splitmix64(std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// uniform in [0, 1) from the 53 high bits of a hash
__host__ __device__ inline double hash_uniform(std::uint64_t h)
{
    return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0);
}

// counter-based host generator: value i of stream `seed` is a hash of (seed, i)
struct CounterRng
{
    std::uint64_t seed, counter = 0;
    explicit CounterRng(std::uint64_t s) : seed(s) {}
    double uniform() { return hash_uniform(splitmix64(seed + (++counter) * 0x9e3779b97f4a7c15ull)); }
};

inline std::uint64_t stream_seed(std::uint64_t seed, const std::string &family, int n, std::uint64_t salt)
{
    std::uint64_t h = splitmix64(seed ^ salt);
    for (unsigned char c : family)
        h = splitmix64(h ^ c);
    return splitmix64(h ^ static_cast<std::uint64_t>(n));
}

// standard normal entries, deterministic in (seed, index): Box-Muller on two hashes
__global__ void gaussian_fill_kernel(double *out, std::size_t count, std::uint64_t seed)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const std::uint64_t h1 = splitmix64(seed + (i + 1) * 0x9e3779b97f4a7c15ull);
    const std::uint64_t h2 = splitmix64(h1 ^ 0xda942042e4dd58b5ull);
    const double u1 = (static_cast<double>(h1 >> 11) + 1.0) * (1.0 / 9007199254740992.0); // (0, 1]
    const double u2 = hash_uniform(h2);                                                      // [0, 1)
    out[i] = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

__global__ void relu_kernel(double *w, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        w[i] = w[i] > 0.0 ? w[i] : 0.0;
}

// ------------------------------------------------------------------------------------------------ spectra

std::vector<double> make_spectrum(const std::string &family, int n, std::uint64_t seed)
{
    CounterRng rng(stream_seed(seed, family, n, 0x5bd1e995ull));
    std::vector<double> lam(n);
    const int r = std::max(1, static_cast<int>(std::ceil(0.03 * n)));
    if (family == "full_rank")
    {
        for (int i = 0; i < n; i++)
            lam[i] = -1.0 + 2.0 * rng.uniform();
    }
    else if (family == "low_pos_rank")
    {
        for (int i = 0; i < n; i++)
            lam[i] = i < r ? 0.01 + 0.99 * rng.uniform() : -1.0 + 0.99 * rng.uniform();
    }
    else if (family == "low_neg_rank")
    {
        for (int i = 0; i < n; i++)
            lam[i] = i < r ? -1.0 + 0.99 * rng.uniform() : 0.01 + 0.99 * rng.uniform();
    }
    else if (family == "clustered")
    {
        for (int i = 0; i < n; i++)
            lam[i] = static_cast<double>(i % 3 - 1) + 1e-3 * (2.0 * rng.uniform() - 1.0);
    }
    else if (family == "spread")
    {
        for (int i = 0; i < n; i++)
        {
            const double mag = std::pow(10.0, -6.0 + 6.0 * rng.uniform());
            lam[i] = rng.uniform() < 0.5 ? -mag : mag;
        }
    }
    else
        throw std::invalid_argument("unknown family " + family);
    return lam;
}

// ------------------------------------------------------------------------------------------------ device helpers

template <typename T>
T *device_alloc(std::size_t count)
{
    T *p = nullptr;
    CUADMM_CHECK_CUDA(cudaMalloc(&p, count * sizeof(T)));
    return p;
}

std::size_t device_free_bytes()
{
    std::size_t f = 0, t = 0;
    CUADMM_CHECK_CUDA(cudaMemGetInfo(&f, &t));
    return f;
}

void copy_d2d(double *dst, const double *src, std::size_t count)
{
    CUADMM_CHECK_CUDA(cudaMemcpy(dst, src, count * sizeof(double), cudaMemcpyDeviceToDevice));
}

struct Handles
{
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t sol = nullptr;
    cusolverDnParams_t params = nullptr;
    Handles()
    {
        CUADMM_CHECK_CUBLAS(cublasCreate(&blas));
        // B1: the default math mode, explicitly (the filter sets its compute type per call)
        CUADMM_CHECK_CUBLAS(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
        CUADMM_CHECK_CUSOLVER(cusolverDnCreate(&sol));
        CUADMM_CHECK_CUSOLVER(cusolverDnCreateParams(&params));
    }
    ~Handles()
    {
        if (params)
            CUADMM_CHECK_CUSOLVER_NOTHROW(cusolverDnDestroyParams(params));
        if (sol)
            CUADMM_CHECK_CUSOLVER_NOTHROW(cusolverDnDestroy(sol));
        if (blas)
            CUADMM_CHECK_CUBLAS_NOTHROW(cublasDestroy(blas));
    }
};

struct Events
{
    cudaEvent_t e[6];
    Events()
    {
        for (auto &x : e)
            CUADMM_CHECK_CUDA(cudaEventCreate(&x));
    }
    ~Events()
    {
        for (auto &x : e)
            CUADMM_CHECK_CUDA_NOTHROW(cudaEventDestroy(x));
    }
    void record(int k) { CUADMM_CHECK_CUDA(cudaEventRecord(e[k], 0)); }
    float ms(int a, int b)
    {
        float v = 0.f;
        CUADMM_CHECK_CUDA(cudaEventElapsedTime(&v, e[a], e[b]));
        return v;
    }
};

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0)
{
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

double median(std::vector<double> v)
{
    if (v.empty())
        return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// ------------------------------------------------------------------------------------------------ buffers per size

// Z, the working copy A and the QR workspace: common to both paths (not attributed to either)
struct BaseBuffers
{
    int n = 0;
    std::size_t n2 = 0;
    double *Z = nullptr, *A = nullptr, *lam = nullptr, *tau = nullptr, *qr_work = nullptr, *W_save = nullptr;
    int *qr_info = nullptr;
    int qr_lwork = 0;
    explicit BaseBuffers(Handles &h, int n_) : n(n_), n2(static_cast<std::size_t>(n_) * n_)
    {
        Z = device_alloc<double>(n2);
        A = device_alloc<double>(n2);
        lam = device_alloc<double>(n);
        tau = device_alloc<double>(n);
        W_save = device_alloc<double>(n);
        qr_info = device_alloc<int>(1);
        int l1 = 0, l2 = 0;
        CUADMM_CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(h.sol, n, n, A, n, &l1));
        CUADMM_CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(h.sol, n, n, n, A, n, tau, &l2));
        qr_lwork = std::max(l1, l2);
        qr_work = device_alloc<double>(static_cast<std::size_t>(qr_lwork));
    }
    ~BaseBuffers()
    {
        for (void *p : {static_cast<void *>(Z), static_cast<void *>(A), static_cast<void *>(lam), static_cast<void *>(tau),
                        static_cast<void *>(W_save), static_cast<void *>(qr_info), static_cast<void *>(qr_work)})
            if (p)
                CUADMM_CHECK_CUDA_NOTHROW(cudaFree(p));
    }
};

// the FullEvd path: the engine allocates the Xsyevd workspace (device and host), W, info, the tmp and P buffers
struct EvdPath
{
    int n = 0;
    std::size_t n2 = 0;
    std::size_t ws_bytes = 0, hws_bytes = 0, mem_bytes = 0;
    void *ws = nullptr;
    std::vector<char> hws;
    double *W = nullptr, *T = nullptr, *P = nullptr;
    int *info = nullptr;
    // the check EVD (eigenvalues only) of the FP16 result: its own workspace, not attributed to the path
    std::size_t chk_ws_bytes = 0, chk_hws_bytes = 0;
    void *chk_ws = nullptr;
    std::vector<char> chk_hws;

    EvdPath(Handles &h, int n_, double *probe) : n(n_), n2(static_cast<std::size_t>(n_) * n_)
    {
        const std::size_t before = device_free_bytes();
        W = device_alloc<double>(n);
        info = device_alloc<int>(1);
        CUADMM_CHECK_CUSOLVER(cusolverDnXsyevd_bufferSize(
            h.sol, h.params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, probe, n, CUDA_R_64F, W,
            CUDA_R_64F, &ws_bytes, &hws_bytes));
        ws = device_alloc<char>(ws_bytes);
        hws.resize(hws_bytes);
        T = device_alloc<double>(n2);
        P = device_alloc<double>(n2);
        CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
        mem_bytes = before - device_free_bytes();
        CUADMM_CHECK_CUSOLVER(cusolverDnXsyevd_bufferSize(
            h.sol, h.params, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, probe, n, CUDA_R_64F, W,
            CUDA_R_64F, &chk_ws_bytes, &chk_hws_bytes));
        chk_ws = device_alloc<char>(chk_ws_bytes);
        chk_hws.resize(chk_hws_bytes);
    }
    ~EvdPath()
    {
        for (void *p : {static_cast<void *>(W), static_cast<void *>(info), ws, static_cast<void *>(T), static_cast<void *>(P), chk_ws})
            if (p)
                CUADMM_CHECK_CUDA_NOTHROW(cudaFree(p));
    }

    // the engine's FullEvd: Xsyevd in place (A <- Q), W <- max(W, 0), T = Q diag(W), P = T Q^T; the eigenvalues of
    // the input are saved to W_save (before the max) when it is given
    void project(Handles &h, double *A, double *W_save)
    {
        CUADMM_CHECK_CUSOLVER(cusolverDnXsyevd(
            h.sol, h.params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, A, n, CUDA_R_64F, W,
            CUDA_R_64F, ws, ws_bytes, hws_bytes ? hws.data() : nullptr, hws_bytes, info));
        if (W_save)
            CUADMM_CHECK_CUDA(cudaMemcpyAsync(W_save, W, n * sizeof(double), cudaMemcpyDeviceToDevice, 0));
        relu_kernel<<<(n + 255) / 256, 256>>>(W, n);
        CUADMM_CHECK_LAUNCH();
        CUADMM_CHECK_CUBLAS(cublasDdgmm(h.blas, CUBLAS_SIDE_RIGHT, n, n, A, n, W, 1, T, n));
        const double one = 1.0, zero = 0.0;
        CUADMM_CHECK_CUBLAS(cublasDgemm(h.blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &one, T, n, A, n, &zero, P, n));
    }

    int host_info()
    {
        int v = 0;
        CUADMM_CHECK_CUDA(cudaMemcpy(&v, info, sizeof(int), cudaMemcpyDeviceToHost));
        return v;
    }

    // eigenvalues only of A (destroyed), into W
    void eigenvalues(Handles &h, double *A)
    {
        CUADMM_CHECK_CUSOLVER(cusolverDnXsyevd(
            h.sol, h.params, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, A, n, CUDA_R_64F, W,
            CUDA_R_64F, chk_ws, chk_ws_bytes, chk_hws_bytes ? chk_hws.data() : nullptr, chk_hws_bytes, info));
        CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
    }
};

// the CompositeFilter path: the FP32 and FP16 workspaces of the engine (3 n^2 each, n^2 rounded up to a multiple of
// 4) and the result buffer
struct CompositePath
{
    int n = 0;
    std::size_t n2 = 0, stride = 0, mem_bytes = 0;
    float *fws = nullptr;
    __half *hws = nullptr;
    double *P = nullptr;

    explicit CompositePath(int n_) : n(n_), n2(static_cast<std::size_t>(n_) * n_)
    {
        stride = n2 % 4 == 0 ? n2 : n2 + (4 - n2 % 4);
        const std::size_t before = device_free_bytes();
        fws = device_alloc<float>(3 * stride);
        hws = device_alloc<__half>(3 * stride);
        P = device_alloc<double>(n2);
        CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
        mem_bytes = before - device_free_bytes();
    }
    ~CompositePath()
    {
        for (void *p : {static_cast<void *>(fws), static_cast<void *>(hws), static_cast<void *>(P)})
            if (p)
                CUADMM_CHECK_CUDA_NOTHROW(cudaFree(p));
    }
};

struct CompositeRun
{
    float ms_total = 0, ms_lanczos = 0, ms_filter = 0, ms_checks = 0;
    double wall_ms = 0, lo = 0, up = 0, scale = 1, norm_in = 0, norm_out = 0, rel_err = 0;
    bool finite = false, accepted = false;
};

// the engine's large_composite_project on A (in place), without the input copy and the result copy
CompositeRun composite_project(Handles &h, int n, double *A, CompositePath &cp, Events &ev)
{
    const int nn = n * n;
    CompositeRun r;
    const auto t0 = Clock::now();
    ev.record(0);
    CUADMM_CHECK_CUBLAS(cublasDnrm2(h.blas, nn, A, 1, &r.norm_in));
    ev.record(1);
    // B8: the deterministic Lanczos start of Phase 6a, seeded per run (the vendored kernel seeded from the clock)
    approximate_two_norm(h.blas, h.sol, A, static_cast<std::size_t>(n), &r.lo, &r.up, static_cast<unsigned long long>(lanczos_seed++), kLanczosSteps, kLanczosTol);
    ev.record(2);
    r.scale = r.up > 0.0 ? kScaleMargin * r.up : 1.0;
    const double inv_scale = 1.0 / r.scale;
    CUADMM_CHECK_CUBLAS(cublasDscal(h.blas, nn, &inv_scale, A, 1));
    composite_fp16(h.blas, A, n, cp.fws, cp.hws);
    CUADMM_CHECK_CUBLAS(cublasDscal(h.blas, nn, &r.scale, A, 1));
    ev.record(3);
    CUADMM_CHECK_CUBLAS(cublasDnrm2(h.blas, nn, A, 1, &r.norm_out));
    r.finite = std::isfinite(r.norm_out) && std::isfinite(r.norm_in);
    r.accepted = r.finite && !(r.norm_out > (1.0 + kNormSlack) * r.norm_in);
    ev.record(4);
    CUADMM_CHECK_CUDA(cudaEventSynchronize(ev.e[4]));
    r.wall_ms = 1e3 * seconds_since(t0);
    r.ms_total = ev.ms(0, 4);
    r.ms_lanczos = ev.ms(1, 2);
    r.ms_filter = ev.ms(2, 3);
    r.ms_checks = ev.ms(0, 1) + ev.ms(3, 4);
    return r;
}

// Z = Q diag(lam) Q^T from the QR of a Gaussian matrix, symmetrized; uses A (Q) and T as workspaces
void build_z(Handles &h, BaseBuffers &b, double *T, const std::vector<double> &lam, std::uint64_t seed)
{
    const int n = b.n;
    CUADMM_CHECK_CUDA(cudaMemcpy(b.lam, lam.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    const unsigned blocks = static_cast<unsigned>((b.n2 + 255) / 256);
    gaussian_fill_kernel<<<blocks, 256>>>(b.A, b.n2, seed);
    CUADMM_CHECK_LAUNCH();
    CUADMM_CHECK_CUSOLVER(cusolverDnDgeqrf(h.sol, n, n, b.A, n, b.tau, b.qr_work, b.qr_lwork, b.qr_info));
    int info = 0;
    CUADMM_CHECK_CUDA(cudaMemcpy(&info, b.qr_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0)
        throw std::runtime_error("cusolverDnDgeqrf info = " + std::to_string(info));
    CUADMM_CHECK_CUSOLVER(cusolverDnDorgqr(h.sol, n, n, n, b.A, n, b.tau, b.qr_work, b.qr_lwork, b.qr_info));
    CUADMM_CHECK_CUDA(cudaMemcpy(&info, b.qr_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0)
        throw std::runtime_error("cusolverDnDorgqr info = " + std::to_string(info));
    const double one = 1.0, zero = 0.0, half = 0.5;
    CUADMM_CHECK_CUBLAS(cublasDdgmm(h.blas, CUBLAS_SIDE_RIGHT, n, n, b.A, n, b.lam, 1, T, n)); // T = Q diag(lam)
    CUADMM_CHECK_CUBLAS(cublasDgemm(h.blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &one, T, n, b.A, n, &zero, b.Z, n));
    CUADMM_CHECK_CUBLAS(cublasDgeam(h.blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, &half, b.Z, n, &half, b.Z, n, T, n));
    copy_d2d(b.Z, T, b.n2);
    CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
}

// ------------------------------------------------------------------------------------------------ results

struct CaseResult
{
    int n = 0;
    std::string family, status = "ok";
    int reps = 0, r_plus = 0, r_minus = 0;
    double true_norm = 0, build_s = 0;
    double evd_ms_median = 0, evd_ms_min = 0, evd_ms_max = 0, evd_wall_ms_median = 0;
    std::size_t evd_ws_bytes = 0, evd_hws_bytes = 0, evd_mem_bytes = 0;
    double comp_ms_median = 0, comp_ms_min = 0, comp_ms_max = 0, comp_wall_ms_median = 0;
    double lanczos_ms_median = 0, filter_ms_median = 0, checks_ms_median = 0;
    std::size_t comp_mem_bytes = 0;
    double rel_err_median = 0, rel_err_max = 0;
    double lambda_min_p16 = 0, lambda_max_p16 = 0;
    bool p16_finite = false;
    int rejected = 0, runs = 0;
    double up_ratio_min = 0, up_ratio_max = 0, lo_ratio_min = 0, lo_ratio_max = 0, scale_ratio_min = 0, scale_ratio_max = 0;
    double check_evd_s = 0, case_s = 0;
    double speedup() const { return comp_ms_median > 0 ? evd_ms_median / comp_ms_median : 0.0; }
    double violation() const { return p16_finite ? std::min(lambda_min_p16, 0.0) : std::numeric_limits<double>::quiet_NaN(); }
};

std::string g(double v, int digits = 6)
{
    std::ostringstream s;
    s << std::setprecision(digits) << v;
    return s.str();
}

void write_results_header(std::ofstream &f)
{
    f << "n,family,status,reps,r_plus,r_minus,true_norm2,build_s,"
      << "evd_ms_median,evd_ms_min,evd_ms_max,evd_wall_ms_median,evd_ws_dev_bytes,evd_ws_host_bytes,evd_ws_over_n2,evd_mem_bytes,evd_mem_over_n2,"
      << "comp_ms_median,comp_ms_min,comp_ms_max,comp_wall_ms_median,lanczos_ms_median,filter_ms_median,checks_ms_median,comp_mem_bytes,comp_mem_over_n2,"
      << "speedup,rel_fro_err_median,rel_fro_err_max,p16_finite,lambda_min_p16,lambda_max_p16,violation,violation_over_norm2,"
      << "rejected,runs,up_over_true_min,up_over_true_max,lo_over_true_min,lo_over_true_max,scale_over_true_min,scale_over_true_max,"
      << "check_evd_s,case_s\n";
}

void write_results_row(std::ofstream &f, const CaseResult &r)
{
    const double n2d = 8.0 * static_cast<double>(r.n) * r.n; // bytes of n^2 doubles
    f << r.n << ',' << r.family << ',' << r.status << ',' << r.reps << ',' << r.r_plus << ',' << r.r_minus << ','
      << g(r.true_norm, 10) << ',' << g(r.build_s) << ','
      << g(r.evd_ms_median) << ',' << g(r.evd_ms_min) << ',' << g(r.evd_ms_max) << ',' << g(r.evd_wall_ms_median) << ','
      << r.evd_ws_bytes << ',' << r.evd_hws_bytes << ',' << g(r.evd_ws_bytes / n2d) << ',' << r.evd_mem_bytes << ',' << g(r.evd_mem_bytes / n2d) << ','
      << g(r.comp_ms_median) << ',' << g(r.comp_ms_min) << ',' << g(r.comp_ms_max) << ',' << g(r.comp_wall_ms_median) << ','
      << g(r.lanczos_ms_median) << ',' << g(r.filter_ms_median) << ',' << g(r.checks_ms_median) << ',' << r.comp_mem_bytes << ',' << g(r.comp_mem_bytes / n2d) << ','
      << g(r.speedup()) << ',' << g(r.rel_err_median) << ',' << g(r.rel_err_max) << ',' << (r.p16_finite ? 1 : 0) << ','
      << g(r.lambda_min_p16, 10) << ',' << g(r.lambda_max_p16, 10) << ',' << g(r.violation(), 10) << ','
      << g(r.true_norm > 0 ? r.violation() / r.true_norm : std::numeric_limits<double>::quiet_NaN(), 10) << ','
      << r.rejected << ',' << r.runs << ','
      << g(r.up_ratio_min, 10) << ',' << g(r.up_ratio_max, 10) << ',' << g(r.lo_ratio_min, 10) << ',' << g(r.lo_ratio_max, 10) << ','
      << g(r.scale_ratio_min, 10) << ',' << g(r.scale_ratio_max, 10) << ',' << g(r.check_evd_s) << ',' << g(r.case_s) << '\n';
    f.flush();
}

void write_timings_header(std::ofstream &f)
{
    f << "n,family,path,rep,warmup,ms,wall_ms,lanczos_ms,filter_ms,checks_ms,lanczos_lo,lanczos_up,scale,norm_in,norm_out,finite,accepted,rel_fro_err\n";
}

// ------------------------------------------------------------------------------------------------ one case

CaseResult run_case(Handles &h, const Options &o, int n, const std::string &family, int reps, BaseBuffers &b, EvdPath &ep,
                    CompositePath &cp, std::ofstream &timings)
{
    const auto tcase = Clock::now();
    CaseResult r;
    r.n = n;
    r.family = family;
    r.reps = reps;
    r.evd_ws_bytes = ep.ws_bytes;
    r.evd_hws_bytes = ep.hws_bytes;
    r.evd_mem_bytes = ep.mem_bytes;
    r.comp_mem_bytes = cp.mem_bytes;
    Events ev;
    const int nn = n * n;

    // the input
    {
        const auto t0 = Clock::now();
        const std::vector<double> lam = make_spectrum(family, n, o.seed);
        build_z(h, b, ep.T, lam, stream_seed(o.seed, family, n, 0x9e3779b9ull));
        r.build_s = seconds_since(t0);
    }

    // (a) FullEvd: warm-up, then reps runs on fresh copies
    std::vector<double> evd_ms, evd_wall;
    for (int rep = 0; rep <= reps; rep++)
    {
        copy_d2d(b.A, b.Z, b.n2);
        CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
        const auto t0 = Clock::now();
        ev.record(0);
        ep.project(h, b.A, rep == 0 ? b.W_save : nullptr);
        ev.record(1);
        CUADMM_CHECK_CUDA(cudaEventSynchronize(ev.e[1]));
        const double wall = 1e3 * seconds_since(t0);
        const int info = ep.host_info();
        if (info != 0)
            throw std::runtime_error("cusolverDnXsyevd (FullEvd) info = " + std::to_string(info));
        const double ms = ev.ms(0, 1);
        timings << n << ',' << family << ",evd," << rep << ',' << (rep == 0 ? 1 : 0) << ',' << g(ms) << ',' << g(wall)
                << ",,,,,,,,,,,\n";
        if (rep > 0)
        {
            evd_ms.push_back(ms);
            evd_wall.push_back(wall);
        }
    }
    timings.flush();
    r.evd_ms_median = median(evd_ms);
    r.evd_ms_min = *std::min_element(evd_ms.begin(), evd_ms.end());
    r.evd_ms_max = *std::max_element(evd_ms.begin(), evd_ms.end());
    r.evd_wall_ms_median = median(evd_wall);

    // the true spectrum of Z (FP64 eigenvalues, ascending) and the reference norm
    {
        std::vector<double> w(n);
        CUADMM_CHECK_CUDA(cudaMemcpy(w.data(), b.W_save, n * sizeof(double), cudaMemcpyDeviceToHost));
        r.true_norm = std::max(std::abs(w.front()), std::abs(w.back()));
        for (double x : w)
        {
            if (x > 0)
                r.r_plus++;
            else if (x < 0)
                r.r_minus++;
        }
    }
    double norm_p64 = 0;
    CUADMM_CHECK_CUBLAS(cublasDnrm2(h.blas, nn, ep.P, 1, &norm_p64));

    // (b) CompositeFilter: warm-up, then reps runs on fresh copies; accuracy of every run against P_fp64
    std::vector<double> comp_ms, comp_wall, lan_ms, fil_ms, chk_ms, rel_errs;
    r.up_ratio_min = r.lo_ratio_min = r.scale_ratio_min = std::numeric_limits<double>::infinity();
    r.up_ratio_max = r.lo_ratio_max = r.scale_ratio_max = -std::numeric_limits<double>::infinity();
    for (int rep = 0; rep <= reps; rep++)
    {
        copy_d2d(b.A, b.Z, b.n2);
        CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
        CompositeRun cr = composite_project(h, n, b.A, cp, ev);
        // ||P_fp16 - P_fp64||_F / ||P_fp64||_F, T as the temporary
        copy_d2d(ep.T, b.A, b.n2);
        const double minus_one = -1.0;
        CUADMM_CHECK_CUBLAS(cublasDaxpy(h.blas, nn, &minus_one, ep.P, 1, ep.T, 1));
        double diff = 0;
        CUADMM_CHECK_CUBLAS(cublasDnrm2(h.blas, nn, ep.T, 1, &diff));
        cr.rel_err = norm_p64 > 0 ? diff / norm_p64 : diff;
        if (rep == reps)
            copy_d2d(cp.P, b.A, b.n2); // the last result, for the eigenvalue check
        r.runs++;
        if (!cr.accepted)
            r.rejected++;
        const double up_ratio = cr.up / r.true_norm, lo_ratio = cr.lo / r.true_norm, scale_ratio = cr.scale / r.true_norm;
        r.up_ratio_min = std::min(r.up_ratio_min, up_ratio);
        r.up_ratio_max = std::max(r.up_ratio_max, up_ratio);
        r.lo_ratio_min = std::min(r.lo_ratio_min, lo_ratio);
        r.lo_ratio_max = std::max(r.lo_ratio_max, lo_ratio);
        r.scale_ratio_min = std::min(r.scale_ratio_min, scale_ratio);
        r.scale_ratio_max = std::max(r.scale_ratio_max, scale_ratio);
        timings << n << ',' << family << ",composite," << rep << ',' << (rep == 0 ? 1 : 0) << ',' << g(cr.ms_total) << ','
                << g(cr.wall_ms) << ',' << g(cr.ms_lanczos) << ',' << g(cr.ms_filter) << ',' << g(cr.ms_checks) << ','
                << g(cr.lo, 10) << ',' << g(cr.up, 10) << ',' << g(cr.scale, 10) << ',' << g(cr.norm_in, 10) << ','
                << g(cr.norm_out, 10) << ',' << (cr.finite ? 1 : 0) << ',' << (cr.accepted ? 1 : 0) << ',' << g(cr.rel_err) << '\n';
        if (rep > 0)
        {
            comp_ms.push_back(cr.ms_total);
            comp_wall.push_back(cr.wall_ms);
            lan_ms.push_back(cr.ms_lanczos);
            fil_ms.push_back(cr.ms_filter);
            chk_ms.push_back(cr.ms_checks);
            rel_errs.push_back(cr.rel_err);
        }
        if (rep == reps)
            r.p16_finite = cr.finite;
    }
    timings.flush();
    r.comp_ms_median = median(comp_ms);
    r.comp_ms_min = *std::min_element(comp_ms.begin(), comp_ms.end());
    r.comp_ms_max = *std::max_element(comp_ms.begin(), comp_ms.end());
    r.comp_wall_ms_median = median(comp_wall);
    r.lanczos_ms_median = median(lan_ms);
    r.filter_ms_median = median(fil_ms);
    r.checks_ms_median = median(chk_ms);
    r.rel_err_median = median(rel_errs);
    r.rel_err_max = *std::max_element(rel_errs.begin(), rel_errs.end());

    // the eigenvalue violation of the last FP16 result: lambda_min(P_fp16) from an FP64 EVD (eigenvalues only)
    if (r.p16_finite)
    {
        const auto t0 = Clock::now();
        copy_d2d(b.A, cp.P, b.n2);
        ep.eigenvalues(h, b.A);
        const int info = ep.host_info();
        if (info != 0)
            throw std::runtime_error("cusolverDnXsyevd (check) info = " + std::to_string(info));
        CUADMM_CHECK_CUDA(cudaMemcpy(&r.lambda_min_p16, ep.W, sizeof(double), cudaMemcpyDeviceToHost));
        CUADMM_CHECK_CUDA(cudaMemcpy(&r.lambda_max_p16, ep.W + (n - 1), sizeof(double), cudaMemcpyDeviceToHost));
        r.check_evd_s = seconds_since(t0);
    }
    else
    {
        r.lambda_min_p16 = r.lambda_max_p16 = std::numeric_limits<double>::quiet_NaN();
    }
    r.case_s = seconds_since(tcase);
    return r;
}

// ------------------------------------------------------------------------------------------------ reporting

void print_case(const CaseResult &r)
{
    const double n2d = 8.0 * static_cast<double>(r.n) * r.n;
    std::printf("n=%d %-13s %s | build %.1f s | EVD %.1f ms (ws %.3f n^2) | composite %.1f ms (lanczos %.1f, filter %.1f, checks %.1f) | "
                "speedup %.2f | rel err %.2e (max %.2e) | lambda_min(P16) %.3e | rejected %d/%d | up/true %.5f..%.5f | %.0f s\n",
                r.n, r.family.c_str(), r.status.c_str(), r.build_s, r.evd_ms_median, r.evd_ws_bytes / n2d, r.comp_ms_median,
                r.lanczos_ms_median, r.filter_ms_median, r.checks_ms_median, r.speedup(), r.rel_err_median, r.rel_err_max,
                r.lambda_min_p16, r.rejected, r.runs, r.up_ratio_min, r.up_ratio_max, r.case_s);
    std::fflush(stdout);
}

void print_markdown(const std::vector<CaseResult> &results)
{
    std::printf("\n| n | family | FullEvd ms | Composite ms | Lanczos ms | filter ms | speedup | rel. Frobenius err (median / max) | "
                "lambda_min(P_fp16) | rejected | Lanczos up / true (min..max) | w_evd / n^2 | EVD mem GiB | composite mem GiB |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    const double gib = 1024.0 * 1024.0 * 1024.0;
    for (const CaseResult &r : results)
    {
        if (r.status != "ok")
        {
            std::printf("| %d | %s | %s | | | | | | | | | | | |\n", r.n, r.family.c_str(), r.status.c_str());
            continue;
        }
        const double n2d = 8.0 * static_cast<double>(r.n) * r.n;
        std::printf("| %d | %s | %.1f | %.1f | %.1f | %.1f | %.2f | %.2e / %.2e | %.2e | %d/%d | %.4f..%.4f | %.3f | %.2f | %.2f |\n",
                    r.n, r.family.c_str(), r.evd_ms_median, r.comp_ms_median, r.lanczos_ms_median, r.filter_ms_median, r.speedup(),
                    r.rel_err_median, r.rel_err_max, r.lambda_min_p16, r.rejected, r.runs, r.up_ratio_min, r.up_ratio_max,
                    r.evd_ws_bytes / n2d, r.evd_mem_bytes / gib, r.comp_mem_bytes / gib);
    }
    std::fflush(stdout);
}

void write_run_info(const Options &o, const std::string &path)
{
    cudaDeviceProp prop{};
    int dev = 0, driver = 0, runtime = 0, cublas_v = 0, cusolver_v = 0;
    CUADMM_CHECK_CUDA(cudaGetDevice(&dev));
    CUADMM_CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    CUADMM_CHECK_CUDA(cudaDriverGetVersion(&driver));
    CUADMM_CHECK_CUDA(cudaRuntimeGetVersion(&runtime));
    {
        Handles h;
        CUADMM_CHECK_CUBLAS(cublasGetVersion(h.blas, &cublas_v));
        CUADMM_CHECK_CUSOLVER(cusolverGetVersion(&cusolver_v));
    }
    std::size_t free_b = 0, total_b = 0;
    CUADMM_CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
    const std::time_t now = std::time(nullptr);
    char stamp[64];
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S %Z", std::localtime(&now));
    std::ostringstream s;
    s << "psd_projector_bench (cuADMM 2.0)\n"
      << "date: " << stamp << "\n"
      << "build commit: " << CUADMM_BUILD_COMMIT << (CUADMM_BUILD_DIRTY ? " (dirty)" : "") << "\n"
      << "gpu: " << prop.name << " (sm_" << prop.major << prop.minor << ", " << prop.multiProcessorCount << " SMs, "
      << (prop.totalGlobalMem >> 20) << " MiB, " << (free_b >> 20) << " MiB free at start)\n"
      << "driver version: " << driver / 1000 << '.' << (driver % 1000) / 10 << ", runtime version: " << runtime / 1000 << '.'
      << (runtime % 1000) / 10 << ", cuBLAS " << cublas_v << ", cuSOLVER " << cusolver_v << "\n"
      << "math mode: CUBLAS_DEFAULT_MATH on the handle; filter GEMMs: cublasGemmEx CUBLAS_COMPUTE_32F_FAST_16F\n"
      << "options: out=" << o.out << " smoke=" << (o.smoke ? 1 : 0) << " reps=" << o.reps << " reps_large=" << o.reps_large
      << " (n >= " << o.large_from << ") budget_min=" << o.budget_min << " seed=" << o.seed << " sizes=";
    for (std::size_t i = 0; i < o.sizes.size(); i++)
        s << (i ? "," : "") << o.sizes[i];
    s << " families=";
    for (std::size_t i = 0; i < o.families.size(); i++)
        s << (i ? "," : "") << o.families[i];
    s << "\nconstants: margin=" << kScaleMargin << " norm_slack=" << kNormSlack << " lanczos_steps=" << kLanczosSteps
      << " lanczos_tol=" << kLanczosTol << "\n";
    std::ofstream f(path);
    f << s.str();
    std::cout << s.str() << std::flush;
}

std::vector<std::string> split(const std::string &s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty())
            out.push_back(item);
    return out;
}

Options parse(int argc, char *argv[])
{
    Options o;
    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        auto value = [&](const char *name) -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument(std::string(name) + " needs a value");
            return argv[++i];
        };
        if (a == "--out")
            o.out = value("--out");
        else if (a == "--smoke")
            o.smoke = true;
        else if (a == "--sizes")
        {
            o.sizes.clear();
            for (const std::string &x : split(value("--sizes")))
                o.sizes.push_back(std::stoi(x));
        }
        else if (a == "--families")
            o.families = split(value("--families"));
        else if (a == "--reps")
            o.reps = std::stoi(value("--reps"));
        else if (a == "--reps-large")
            o.reps_large = std::stoi(value("--reps-large"));
        else if (a == "--budget-min")
            o.budget_min = std::stod(value("--budget-min"));
        else if (a == "--seed")
            o.seed = std::stoull(value("--seed"));
        else if (a == "--help" || a == "-h")
        {
            std::cout << "usage: cuadmm_psd_projector_bench [--out DIR] [--smoke] [--sizes 2000,3000,...] [--families a,b,...]\n"
                         "                                  [--reps 5] [--reps-large 3] [--budget-min 40] [--seed 20261022]\n";
            std::exit(0);
        }
        else
            throw std::invalid_argument("unknown argument " + a);
    }
    if (o.smoke)
    {
        o.sizes = {2000};
        o.reps = 1;
        o.reps_large = 1;
    }
    for (int n : o.sizes)
        if (n < 2 || n > kMaxEvdBlock)
            throw std::invalid_argument("size " + std::to_string(n) + " outside [2, " + std::to_string(kMaxEvdBlock) + "]");
    for (const std::string &f : o.families)
        make_spectrum(f, 2, 0); // validates the name
    if (o.reps < 1 || o.reps_large < 1)
        throw std::invalid_argument("reps and reps-large must be >= 1");
    return o;
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        const Options o = parse(argc, argv);
        const auto tstart = Clock::now();
        std::filesystem::create_directories(o.out);
        write_run_info(o, o.out + "/run_info.txt");
        std::ofstream results(o.out + "/results.csv"), timings(o.out + "/timings.csv");
        write_results_header(results);
        write_timings_header(timings);

        Handles h;
        std::vector<CaseResult> all;
        double last_case_s = 0;
        int last_n = 0;
        for (int n : o.sizes)
        {
            const int reps = n >= o.large_from ? o.reps_large : o.reps;
            std::printf("== n = %d (%d repetitions after one warm-up), %.1f min elapsed\n", n, reps, seconds_since(tstart) / 60);
            std::fflush(stdout);
            BaseBuffers b(h, n);
            EvdPath ep(h, n, b.A);
            CompositePath cp(n);
            std::printf("   Xsyevd workspace: device %zu bytes = %.4f n^2 doubles, host %zu bytes; EVD path %.3f GiB, composite path %.3f GiB\n",
                        ep.ws_bytes, ep.ws_bytes / (8.0 * n * static_cast<double>(n)), ep.hws_bytes,
                        ep.mem_bytes / 1073741824.0, cp.mem_bytes / 1073741824.0);
            std::fflush(stdout);
            bool first = true;
            for (const std::string &family : o.families)
            {
                double estimate = last_case_s;
                if (first && last_n > 0)
                    estimate = last_case_s * std::pow(static_cast<double>(n) / last_n, 3.0);
                const double elapsed_min = seconds_since(tstart) / 60;
                CaseResult r;
                if (elapsed_min + estimate / 60 > o.budget_min)
                {
                    r.n = n;
                    r.family = family;
                    r.status = "skipped_budget";
                    std::printf("n=%d %-13s skipped: %.1f min elapsed + %.1f min estimated > budget %.1f min\n", n, family.c_str(),
                                elapsed_min, estimate / 60, o.budget_min);
                    std::fflush(stdout);
                }
                else
                {
                    try
                    {
                        r = run_case(h, o, n, family, reps, b, ep, cp, timings);
                        last_case_s = r.case_s;
                        last_n = n;
                        first = false;
                    }
                    catch (const std::exception &e)
                    {
                        r = CaseResult();
                        r.n = n;
                        r.family = family;
                        r.status = std::string("error: ") + e.what();
                        std::replace(r.status.begin(), r.status.end(), ',', ';');
                        std::replace(r.status.begin(), r.status.end(), '\n', ' ');
                        std::printf("n=%d %-13s %s\n", n, family.c_str(), r.status.c_str());
                        std::fflush(stdout);
                    }
                    if (r.status == "ok")
                        print_case(r);
                }
                write_results_row(results, r);
                all.push_back(r);
            }
        }
        print_markdown(all);
        std::printf("\ntotal %.1f min\n", seconds_since(tstart) / 60);
        return 0;
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "psd_projector_bench: %s\n", e.what());
        return 1;
    }
}
