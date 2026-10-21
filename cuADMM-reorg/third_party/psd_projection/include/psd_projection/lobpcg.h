#ifndef PSD_PROJECTION_LOBPCG_H
#define PSD_PROJECTION_LOBPCG_H

#include <limits>

/// @brief Convergence information returned by `lobpcg`.
struct LobpcgInfo {
    int iterations;       ///< number of Rayleigh-Ritz updates performed
    double residual_norm; ///< max_j ||A v_j - d_j v_j||_2 over the checked returned pairs (see conv_threshold and min_checked)
    bool converged;       ///< residual_norm < tol and every cuSOLVER call succeeded
};

/// @brief Computes the largest `m` eigenpairs of a symmetric matrix `A` using the LOBPCG algorithm.
/// @param cublasH cuBLAS handle
/// @param cusolverH cuSOLVER handle for QR and eigenvalue decompositions
/// @param A n x n symmetric matrix (device pointer)
/// @param V n x m matrix to store the eigenvectors (device pointer)
/// @param D m x m matrix to store the eigenvalues (device pointer)
/// @param n size of the matrix A
/// @param m number of eigenpairs to compute
/// @param warmstart if true, warm start the algorithm using the values in `V` and `D`; in this case, `V` is assumed to be orthonormal
/// @param maxiter maximum number of iterations
/// @param tol convergence tolerance
/// @param verbose if true, print verbose output
/// @param info optional output: iterations, residual norm of the returned pairs, and convergence flag. If null and a cuSOLVER factorization fails, a std::runtime_error is thrown.
/// @param conv_threshold only the Ritz pairs with value > conv_threshold take part in the convergence test (default: all pairs)
/// @param min_checked the first min_checked Ritz pairs always take part in the convergence test, whatever their value
///        (Ritz values are lower bounds of the eigenvalues: without it, a start subspace whose Ritz values are all below
///        conv_threshold, e.g. a random one, is reported as converged after 0 iterations)
/// @note Convergence is only declared after at least one Rayleigh-Ritz update, so that the residual block has
///       expanded the search subspace once (with maxiter = 0 the result is reported as not converged). An eigenvector
///       (numerically) orthogonal to the search subspace, e.g. to an exactly invariant warm start, cannot be detected
///       by any residual or Ritz-value test.
/// @note The eigenpairs are returned in decreasing order of the eigenvalues. The convergence test uses the largest
///       per-column residual norm ||A v_j - d_j v_j||_2 among the checked pairs, evaluated on the returned pairs.
void lobpcg(
    cublasHandle_t cublasH,
    cusolverDnHandle_t cusolverH,
    const double* A, // n x n, device pointer
    double* V,       // n x m, device pointer (output eigenvectors)
    double* D,       // m x m, device pointer (output eigenvalues, diagonal)
    const int n,
    const int m ,    // number of eigenpairs
    const bool warmstart = false,
    const int maxiter = 100, // maximum iterations
    const double tol = 1e-8,  // convergence tolerance
    const bool verbose = false, // verbosity flag
    LobpcgInfo* info = nullptr, // optional convergence information
    const double conv_threshold = -std::numeric_limits<double>::infinity(),
    const int min_checked = 0
);

#endif // PSD_PROJECTION_LOBPCG_H