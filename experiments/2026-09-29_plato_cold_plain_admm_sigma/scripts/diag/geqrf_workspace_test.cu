// Diagnostic for the initcheck reports in cuSOLVER's geqr2_smem_domino_fast kernel, launched by cusolverDnDgeqrf from
// psd_projection's lobpcg (PROTOCOL.md, Deviation 6). Reproduces lobpcg's QR step exactly (src/lobpcg.cu): XRD =
// [X_k, R_k, Delta_X_k] (n x 3m, every entry written; in the first LOBPCG iteration Delta_X_k = X_k), then
// cusolverDnDgeqrf(n, 3m, XRD, n, tau, work, lwork) and cusolverDnDorgqr(n, 3m, 3m, XRD, n, tau, work, lwork) with
// lwork = max(geqrf, orgqr buffer sizes), tau and the workspace from cudaMalloc. The caller-provided tau, workspace
// and devInfo are left unwritten / zero / NaN (0xFF) / garbage; prints FNV-1a hashes of R (after geqrf) and Q (after
// orgqr) so that fills can be compared bitwise.
// usage: geqrf_workspace_test <n> <m> <none|zero|nan|garbage> <first|generic> [repeats]
#include <cstdio>
#include <cstdlib>
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
    if (argc < 5) { printf("usage: %s n m none|zero|nan|garbage first|generic [repeats]\n", argv[0]); return 1; }
    const int n = atoi(argv[1]), m = atoi(argv[2]);
    const std::string mode = argv[3], kind = argv[4];
    const int reps = argc > 5 ? atoi(argv[5]) : 2;
    const int c = 3 * m;
    // XRD: X (orthonormal-ish random block), R (random), D (= X in the first iteration, random otherwise)
    std::vector<double> XRD((size_t)n * c);
    uint64_t s = 4242;
    for (size_t i = 0; i < (size_t)n * 2 * m; i++)
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        XRD[i] = (double)(s >> 11) / 9007199254740992.0 - 0.5;
    }
    for (size_t i = 0; i < (size_t)n * m; i++)
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        XRD[(size_t)n * 2 * m + i] = kind == "first" ? XRD[i] : (double)(s >> 11) / 9007199254740992.0 - 0.5;
    }
    cusolverDnHandle_t h;
    CS(cusolverDnCreate(&h));
    double *dXRD, *dtau, *dwork;
    int *dinfo;
    CK(cudaMalloc(&dXRD, sizeof(double) * (size_t)n * c));
    CK(cudaMalloc(&dtau, sizeof(double) * c));
    CK(cudaMalloc(&dinfo, sizeof(int)));
    int lg = 0, lo = 0;
    CS(cusolverDnDgeqrf_bufferSize(h, n, c, dXRD, n, &lg));
    CS(cusolverDnDorgqr_bufferSize(h, n, c, c, dXRD, n, dtau, &lo));
    const int lwork = lg > lo ? lg : lo;
    CK(cudaMalloc(&dwork, sizeof(double) * (size_t)lwork));
    std::vector<double> Rm((size_t)n * c), Q((size_t)n * c);
    uint64_t hr0 = 0, hq0 = 0;
    bool stable = true;
    for (int r = 0; r < reps; r++)
    {
        fill(dtau, sizeof(double) * c, mode, 11 + r);
        fill(dwork, sizeof(double) * (size_t)lwork, mode, 23 + r);
        fill(dinfo, sizeof(int), mode, 5 + r);
        CK(cudaMemcpy(dXRD, XRD.data(), sizeof(double) * (size_t)n * c, cudaMemcpyHostToDevice));
        CS(cusolverDnDgeqrf(h, n, c, dXRD, n, dtau, dwork, lwork, dinfo));
        int info1 = -1;
        CK(cudaMemcpy(&info1, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(Rm.data(), dXRD, sizeof(double) * (size_t)n * c, cudaMemcpyDeviceToHost));
        CS(cusolverDnDorgqr(h, n, c, c, dXRD, n, dtau, dwork, lwork, dinfo));
        int info2 = -1;
        CK(cudaMemcpy(&info2, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(Q.data(), dXRD, sizeof(double) * (size_t)n * c, cudaMemcpyDeviceToHost));
        bool finite = true;
        for (double x : Q) finite = finite && std::isfinite(x);
        const uint64_t hr = fnv(Rm.data(), Rm.size() * sizeof(double)), hq = fnv(Q.data(), Q.size() * sizeof(double));
        if (r == 0) { hr0 = hr; hq0 = hq; } else stable = stable && hr == hr0 && hq == hq0;
        printf("n %d m %d kind %s mode %s rep %d info %d %d finite %d hashR %016llx hashQ %016llx lwork %d\n", n, m, kind.c_str(), mode.c_str(), r, info1, info2,
               (int)finite, (unsigned long long)hr, (unsigned long long)hq, lwork);
    }
    printf("RESULT n %d m %d kind %s mode %s hashR %016llx hashQ %016llx stable_across_reps %d\n", n, m, kind.c_str(), mode.c_str(), (unsigned long long)hr0,
           (unsigned long long)hq0, (int)stable);
    CS(cusolverDnDestroy(h));
    return 0;
}
