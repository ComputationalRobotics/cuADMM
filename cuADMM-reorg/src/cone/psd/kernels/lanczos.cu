/*

    cone/psd/kernels/lanczos.cu

    approximate_two_norm of the frozen psd_projection submodule (b566590), moved in unchanged apart from the checks,
    the namespace and the deterministic start vector (B8). The buffers are still allocated per call (B7: Phase 6b).

*/

#include "cone/psd/kernels/lanczos.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include "cone/psd/kernels/precision.h"
#include "core/check.h"
#include "core/device_buffer.h"

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost

namespace cuadmm
{
namespace cone
{
namespace psd
{

__global__ void fill_tridiagonal_kernel(double* T, const double* alpha, const double* beta, int nb_iter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nb_iter) {
        // Set diagonal
        T[i * nb_iter + i] = alpha[i];
        // Set upper diagonal
        if (i < nb_iter - 1) {
            T[i * nb_iter + (i + 1)] = beta[i];
            T[(i + 1) * nb_iter + i] = beta[i];
        }
    }
}

static void fill_tridiagonal(
    double* T, const double *d_alpha, const double *d_beta, int nb_iter, const int threadsPerBlock = 1024
) {
    // Launch kernel to fill the tridiagonal matrix T
    int blocks = (nb_iter + threadsPerBlock - 1) / threadsPerBlock;
    fill_tridiagonal_kernel<<<blocks, threadsPerBlock>>>(T, d_alpha, d_beta, nb_iter);
    
    // Check for errors in kernel launch
    CUADMM_CHECK_CUDA(cudaGetLastError());
}


namespace
{
struct LanczosLayout
{
    int lwork_eig = 0;
    std::size_t doubles = 0;
};

// The buffers of approximate_two_norm(n, max_iter) in carving order (max_iter already clamped to n).
LanczosLayout lanczos_layout(cusolverDnHandle_t cusolverH, std::size_t n, std::size_t max_iter)
{
    LanczosLayout L;
    core::DeviceBuffer<double> dummy(16);
    double *d = dummy.data();
    CUADMM_CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                                      (int)max_iter, d, (int)max_iter, d, &L.lwork_eig));
    L.doubles = n * max_iter + 4 * n                     // V, V_old, q, w, w1
                + max_iter * max_iter + 3 * max_iter     // T, d_beta, d_alpha, d_eigenvalues
                + (std::size_t)L.lwork_eig + 1           // the EVD workspace and devInfo
                + max_iter + 2 * n;                      // uk, y, ry
    return L;
}
} // namespace

std::size_t lanczos_workspace_doubles(cusolverDnHandle_t cusolverH, std::size_t n, std::size_t max_iter)
{
    return lanczos_layout(cusolverH, n, min(max_iter, n)).doubles;
}

void approximate_two_norm(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, size_t n,
    double* lo, double* up,
    unsigned long long seed,
    size_t max_iter, double tol, double *workspace
) {
    /* The workspace (B7: nothing is allocated per call when the caller passes one) */
    // constants
    const double zero = 0.0;
    const double one = 1.0;

    max_iter = min(max_iter, n);
    const LanczosLayout L = lanczos_layout(cusolverH, n, max_iter);
    core::DeviceBuffer<double> own_workspace;
    double *ws = workspace;
    if (ws == nullptr)
    {
        own_workspace.allocate(L.doubles);
        ws = own_workspace.data();
    }
    double *cursor = ws;
    auto carve = [&cursor](std::size_t count) { double *p = cursor; cursor += count; return p; };
    // storage
    double *V = carve(n * max_iter), *V_old = carve(n), *q = carve(n), *w = carve(n), *w1 = carve(n);
    double *T = carve(max_iter * max_iter), *d_beta = carve(max_iter), *d_alpha = carve(max_iter), *d_eigenvalues = carve(max_iter);
    const int lwork_eig = L.lwork_eig;
    double *d_work_eig = carve(L.lwork_eig);
    int *devInfo = reinterpret_cast<int *>(carve(1));
    double *uk = carve(max_iter), *y = carve(n), *ry = carve(n);

    std::vector<double> alpha(max_iter, 0.0);
    std::vector<double> beta(max_iter, 0.0);

    double minus_alpha, minus_beta_old = 0.0;

    /* Initial vector */
    // q = randn(n, 1)
    fill_deterministic(q, n, seed); // B8: deterministic start, seeded by (options.seed, block)

    // q = q / norm(q)
    double norm_q;
    CUADMM_CHECK_CUBLAS(cublasDnrm2(cublasH, n, q, 1, &norm_q));
    if (norm_q != 0.0) {
        double norm_q_inv = 1.0 / norm_q;
        CUADMM_CHECK_CUBLAS(cublasDscal(cublasH, n, &norm_q_inv, q, 1));
    }

    // V(:, 1) = q
    CUADMM_CHECK_CUBLAS(cublasDcopy(cublasH, n, q, 1, V, 1));
    // fill V_old with zeros
    CUADMM_CHECK_CUDA(cudaMemset(V_old, 0, n * sizeof(double)));

    /* Lanczos loop */
    int nb_iter = 0;
    for (int k = 0; k < max_iter; k++) {
        // w = A * q
        CUADMM_CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, n,
                                 &one, A, n, q, 1,
                                 &zero, w, 1));

        // w1 = At * w
        CUADMM_CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_T, n, n,
                                 &one, A, n, w, 1,
                                 &zero, w1, 1));
        // w = w1
        CUADMM_CHECK_CUBLAS(cublasDcopy(cublasH, n, w1, 1, w, 1));
        // hence w = A^T * A * q

        // alpha(k) = q^T * w
        CUADMM_CHECK_CUBLAS(cublasDdot(cublasH, n, q, 1, w, 1, &alpha[k]));

        // minus_alpha = -alpha[k]
        minus_alpha = -alpha[k];
        
        // w = w - alpha(k) * q - beta_old * V_old
        CUADMM_CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_alpha, q, 1, w, 1));
        CUADMM_CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_beta_old, V_old, 1, w, 1));
        
        // beta(k) = norm(w)
        CUADMM_CHECK_CUBLAS(cublasDnrm2(cublasH, n, w, 1, &beta[k]));

        // printf("alpha[k]: %5.4e, beta[k]: %5.4e \n", alpha[k], beta[k]);

        if (beta[k] <= tol * alpha[k] && k > 1)
            break;
            
        // V_old = q
        CUADMM_CHECK_CUBLAS(cublasDcopy(cublasH, n, q, 1, V_old, 1));
        // q = w / beta(k)
        CUADMM_CHECK_CUDA(cudaMemcpy(q, w, n * sizeof(double), cudaMemcpyDeviceToDevice));
        if (beta[k] != 0.0) {
            double beta_inv = 1.0 / beta[k];
            CUADMM_CHECK_CUBLAS(cublasDscal(cublasH, n, &beta_inv, q, 1));
        } else {
            // If beta is zero, we cannot proceed further
            // fprintf(stderr, "Lanczos iteration %d: beta is zero, stopping early.\n", k);
            break;
        }

        if (k < max_iter - 1) {
            // V(:, k+1) = q
            CUADMM_CHECK_CUBLAS(cublasDcopy(cublasH, n, q, 1, V + (k + 1) * n, 1));
        }
        // minus_beta_old = -beta[k]
        minus_beta_old = -beta[k];

        nb_iter++;
    }

    if (nb_iter == 0) {
        // in this case, the matrix is an all-zero matrix
        *lo = 0.0;
        *up = 1.0;

        CUADMM_CHECK_CUDA(cudaDeviceSynchronize());

        return;
    }

    /* Tridiagonal T */
    // T = diag(alpha) + diag(beta(2:end),1) + diag(beta(2:end),-1);
    CUADMM_CHECK_CUDA(cudaMemcpy(d_alpha, alpha.data(), nb_iter * sizeof(double), H2D));
    CUADMM_CHECK_CUDA(cudaMemcpy(d_beta,  beta.data(),  nb_iter * sizeof(double), H2D));
    // fill_tridiagonal only writes the three diagonals: zero the rest of T
    CUADMM_CHECK_CUDA(cudaMemset(T, 0, nb_iter * nb_iter * sizeof(double)));
    fill_tridiagonal(
        T, d_alpha, d_beta, nb_iter
    );

    /* Largest Ritz pair */
    // the eigenvalue decomposition of T (nb_iter <= max_iter: the workspace was sized for max_iter)

    // compute eigenvalues and eigenvectors
    CUADMM_CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                    nb_iter, T, nb_iter, d_eigenvalues,
                                    d_work_eig, lwork_eig, devInfo));

    CUADMM_CHECK_CUDA(cudaDeviceSynchronize());

    // print eigenvalues for debugging
    // std::vector<double> eigenvalues(nb_iter);
    // CUADMM_CHECK_CUDA(cudaMemcpy(eigenvalues.data(), d_eigenvalues, nb_iter * sizeof(double), cudaMemcpyDeviceToHost));
    // std::cout << "Eigenvalues: ";
    // for (const auto& ev : eigenvalues) {
    //     std::cout << ev << "\n";
    // }
    // std::cout << std::endl;

    // retrieve the max eigenvalue and corresponding eigenvector
    // int idx_max;
    // CUADMM_CHECK_CUBLAS(cublasIdamax(cublasH, nb_iter, d_eigenvalues, 1, &idx_max));
    // idx_max--; // convert to 0-based index
    int idx_max = nb_iter - 1;

    double theta;
    CUADMM_CHECK_CUDA(cudaMemcpy(&theta, d_eigenvalues + idx_max, sizeof(double), D2H));

    // uk = T(:, idx_max)
    CUADMM_CHECK_CUBLAS(cublasDcopy(cublasH, nb_iter, T + idx_max * nb_iter, 1, uk, 1));
    // y = V(:,1:nb_iter) * uk
    CUADMM_CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, nb_iter, &one, V, n, uk, 1, &zero, y, 1));

    // ry = A * y
    CUADMM_CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_N, n, n,
                                &one, A, n, y, 1,
                                &zero, ry, 1));
    // w1 = At * ry
    CUADMM_CHECK_CUBLAS(cublasDgemv(cublasH, CUBLAS_OP_T, n, n,
                                &one, A, n, ry, 1,
                                &zero, w1, 1));
    // ry = w1
    CUADMM_CHECK_CUBLAS(cublasDcopy(cublasH, n, w1, 1, ry, 1));
    // hence ry = A^T * A * y

    // ry = ry - theta * y
    double minus_theta = -theta;
    CUADMM_CHECK_CUBLAS(cublasDaxpy(cublasH, n, &minus_theta, y, 1, ry, 1));

    /* Output */
    // lo = sqrt(theta)
    *lo = std::sqrt(theta);

    // up = sqrt(theta + norm(ry))
    double norm_ry;
    CUADMM_CHECK_CUBLAS(cublasDnrm2(cublasH, n, ry, 1, &norm_ry));
    // printf("theta: %5.4e, norm_ry: %5.4e \n", theta, norm_ry);
    *up = std::sqrt(theta + norm_ry);
    // printf("up: %5.4e \n", *up);

    CUADMM_CHECK_CUDA(cudaDeviceSynchronize());

    // std::cout << "Upper bound: " << *up << std::endl;

    return;
}


} // namespace psd
} // namespace cone
} // namespace cuadmm
