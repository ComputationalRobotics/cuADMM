#ifndef PSD_PROJECTION_LANCZOS_H
#define PSD_PROJECTION_LANCZOS_H

/// @brief Approximates the two-norm of a symmetric matrix using the Lanczos method.
/// @param A Dense square matrix stored on device
/// @param n Size of the matrix (n x n)
/// @param lo Certified lower bound on ‖A‖₂
/// @param up Certified upper bound on ‖A‖₂
/// @param max_iter Maximum number of Lanczos iterations
/// @param tol Relative residual tolerance for convergence
void approximate_two_norm(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, size_t n,
    double* lo, double* up,
    size_t max_iter = 20, double tol = 1e-10
);

/// @brief WIP: Compute the k largest eigenpairs of a symmetric matrix using the Lanczos method. It performs a two-step post-cleaning, dropping paris whose residual > tol and dropping host eigenvectors that lose orthogonality.
/// @param cublasH cuBLAS handle
/// @param cusolverH cuSOLVER handle
/// @param A symmetric matrix of size n x n
/// @param n size of A
/// @param k target number of extremal eigenpairs to compute
/// @param r actual number of eigenpairs computed (output parameter, r <= k)
/// @param eigenvalues rx1 vector of Ritz values after cleaning
/// @param eigenvectors rxn matrix of Ritz vectors after cleaning
/// @param upper_bound_only if true, only computes an upper bound on the spectral norm (default false)
/// @param max_iter maximum number of Lanczos iterations (default n)
/// @param tol residual tolerance for step-1 cleaning (default 1e-10)
/// @param ortho_tol orthogonality tolerance for step-2 cleaning (default 1e-3)
/// @return A sign-robust upper bound on ‖A‖₂
double compute_eigenpairs(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, size_t n,
    const size_t k,
    size_t *r,
    double* eigenvalues, double* eigenvectors,
    const bool upper_bound_only = false,
    const size_t max_iter = 0, const double tol = 1e-10, const double ortho_tol = 1e-5,
    const bool verbose = false
);

#endif // PSD_PROJECTION_LANCZOS_H