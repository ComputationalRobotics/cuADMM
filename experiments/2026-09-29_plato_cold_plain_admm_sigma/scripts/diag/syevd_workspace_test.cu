// Diagnostic for the initcheck reports inside cusolverDnXsyevd (PROTOCOL.md, Deviations 3-4).
// Calls cusolverDnXsyevd exactly as cuADMM's single_eig_cusolver does (default cusolverDnParams, jobz = VECTOR,
// uplo = LOWER, lda = n, the device workspace from cusolverDnXsyevd_bufferSize, a separate host workspace) on a
// deterministic random symmetric matrix whose every entry is written, with the device workspace
//   none     cudaMalloc only (as in cuADMM: never written by the caller)
//   zero     cudaMemset 0x00
//   nan      cudaMemset 0xFF (every double is a NaN)
//   garbage  pseudo-random bytes copied from the host
// and the eigenvalue output W and info also filled with the same pattern (cuADMM does not initialize them either).
// Prints FNV-1a hashes of the eigenvalues and eigenvectors: if they are identical for every fill, the results do
// not depend on the contents of the memory initcheck reports as read before being written.
// usage: syevd_workspace_test <n> <none|zero|nan|garbage> [repeats]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)
#define CS(x) do { cusolverStatus_t s = (x); if (s != CUSOLVER_STATUS_SUCCESS) { printf("cuSOLVER error %d at %d\n", (int)s, __LINE__); exit(2); } } while (0)

static uint64_t fnv(const void *p, size_t n)
{
    const unsigned char *c = (const unsigned char *)p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ULL; }
    return h;
}

static void fill(void *d, size_t bytes, const std::string &mode, uint64_t seed)
{
    if (mode == "none") return;
    if (mode == "zero") { CK(cudaMemset(d, 0x00, bytes)); return; }
    if (mode == "nan") { CK(cudaMemset(d, 0xFF, bytes)); return; }
    std::vector<unsigned char> h(bytes);
    uint64_t x = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    for (size_t i = 0; i < bytes; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; h[i] = (unsigned char)(x >> 24); }
    CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
}

int main(int argc, char **argv)
{
    if (argc < 3) { printf("usage: %s n none|zero|nan|garbage [repeats]\n", argv[0]); return 1; }
    const int n = atoi(argv[1]);
    const std::string mode = argv[2];
    const int reps = argc > 3 ? atoi(argv[3]) : 2;
    // deterministic symmetric matrix, both triangles written (as vector_to_matrices_kernel does)
    std::vector<double> A((size_t)n * n);
    uint64_t s = 12345;
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= j; i++)
        {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            const double v = ((double)(s >> 11) / 9007199254740992.0 - 0.5) * (i == j ? 4.0 : 1.0);
            A[(size_t)j * n + i] = A[(size_t)i * n + j] = v;
        }
    cusolverDnHandle_t h;
    CS(cusolverDnCreate(&h));
    cusolverDnParams_t par;
    CS(cusolverDnCreateParams(&par));
    double *dA, *dW;
    int *dinfo;
    CK(cudaMalloc(&dA, sizeof(double) * (size_t)n * n));
    CK(cudaMalloc(&dW, sizeof(double) * n));
    CK(cudaMalloc(&dinfo, sizeof(int)));
    size_t wd = 0, wh = 0;
    CS(cusolverDnXsyevd_bufferSize(h, par, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, dA, n, CUDA_R_64F, dW,
                                   CUDA_R_64F, &wd, &wh));
    void *dwork;
    CK(cudaMalloc(&dwork, wd));
    std::vector<unsigned char> hwork(wh > 0 ? wh : 1);
    std::vector<double> W(n), V((size_t)n * n);
    uint64_t first_hw = 0, first_hv = 0;
    bool stable = true;
    for (int r = 0; r < reps; r++)
    {
        // every repetition: fresh fill of the workspace/outputs, the same input matrix (cuADMM reuses the workspace
        // across iterations, so later calls see the previous call's leftovers)
        fill(dwork, wd, mode, 99 + r);
        fill(dW, sizeof(double) * n, mode, 7 + r);
        fill(dinfo, sizeof(int), mode, 3 + r);
        CK(cudaMemcpy(dA, A.data(), sizeof(double) * (size_t)n * n, cudaMemcpyHostToDevice));
        CS(cusolverDnXsyevd(h, par, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, dA, n, CUDA_R_64F, dW, CUDA_R_64F,
                            dwork, wd, hwork.data(), wh, dinfo));
        int info = -1;
        CK(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(W.data(), dW, sizeof(double) * n, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(V.data(), dA, sizeof(double) * (size_t)n * n, cudaMemcpyDeviceToHost));
        const uint64_t hw = fnv(W.data(), sizeof(double) * n), hv = fnv(V.data(), sizeof(double) * (size_t)n * n);
        bool finite = true;
        for (double x : W) finite = finite && std::isfinite(x);
        for (double x : V) finite = finite && std::isfinite(x);
        if (r == 0) { first_hw = hw; first_hv = hv; }
        else stable = stable && hw == first_hw && hv == first_hv;
        printf("n %d mode %s rep %d info %d finite %d lambda_min %.17g lambda_max %.17g hashW %016llx hashV %016llx workspace %zu bytes host %zu bytes\n",
               n, mode.c_str(), r, info, (int)finite, W[0], W[n - 1], (unsigned long long)hw, (unsigned long long)hv, wd, wh);
    }
    printf("RESULT n %d mode %s hashW %016llx hashV %016llx stable_across_reps %d\n", n, mode.c_str(), (unsigned long long)first_hw,
           (unsigned long long)first_hv, (int)stable);
    CS(cusolverDnDestroyParams(par));
    CS(cusolverDnDestroy(h));
    return 0;
}
