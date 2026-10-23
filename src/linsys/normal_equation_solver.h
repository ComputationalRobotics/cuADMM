/*

    linsys/normal_equation_solver.h

    NormalEquationSolver (plan Section 11.3): the backends of the y-step behind one interface, written for exact and
    inexact solves. analyze() and factorize() run once at setup (the system does not depend on sigma); solve() runs
    once per plain-ADMM iteration (twice for sGS-ADMM) with device vectors on the solver's stream.

    Backends (plan Section 11.4): CholmodSolver (always built; today's behavior moved out of SDPSolver, bitwise),
    CudssSolver (optional, CUADMM_WITH_CUDSS), DenseCusolverSolver (cuSOLVER dense), and the reserved PcgSolver.

    Contract of solve(): rhs and y are device arrays of m doubles; the solver may synchronize the stream for a
    host solve; y is written on return; with req.rel_tol == 0 the solve is exact (direct backends) and the report
    carries iterations = 0, converged = true; a direct backend may report the relative residual only when it
    computes it (bench and tests), otherwise 0.

*/

#ifndef CUADMM_LINSYS_NORMAL_EQUATION_SOLVER_H
#define CUADMM_LINSYS_NORMAL_EQUATION_SOLVER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include <cuda_runtime_api.h>

#include "linsys/normal_system.h"

namespace cuadmm
{
namespace linsys
{

struct SolveRequest
{
    double rel_tol = 0.0;           // 0: exact (direct backends); > 0: an inexact solve is allowed
    int max_iter = 0;               // iterative backends only
    bool use_initial_guess = false; // warm start from the content of y (e.g. the previous y-step)
};

struct SolveReport
{
    int iterations = 0;
    double rel_residual = 0.0; // ||(A A^T + eps I) y - rhs|| / ||rhs|| when computed, else 0
    bool converged = true;
};

struct Capabilities
{
    bool exact = true;              // the solve is a direct solve
    bool device_solves = false;     // the solve runs on the device without a host synchronization
    bool saddle_point = false;      // the saddle-point form of plan Section 9.4 (reserved)
    bool sigma_dependent_ok = false; // update() supports a changed system (reserved)
};

struct SymbolicInfo
{
    double nnz_L = 0.0;           // nonzeros of the factor (both triangles counted once)
    std::size_t device_bytes = 0; // memory the factorization will need on the device
    std::size_t host_bytes = 0;   // and on the host
    std::string ordering;         // e.g. "amd", "metis", "cudss_default"
    double seconds = 0.0;
};

struct FactorInfo
{
    double min_pivot = 0.0, max_pivot = 0.0; // pivots D_jj (LDL^T) or L_jj^2 (LL^T) when known
    int tiny_pivots = 0;                     // |D_jj| < 1e-12 max |D|: dependent constraints
    int nonpositive_pivots = 0;
    bool supernodal = false;                 // CHOLMOD: the supernodal path was used
    std::string kind;                        // "ldlt", "llt", "dense_llt", "dense_ldlt", ...
    double seconds = 0.0;
};

class NormalEquationSolver
{
public:
    virtual ~NormalEquationSolver() = default;
    virtual std::string_view name() const = 0;
    virtual Capabilities capabilities() const = 0;
    /// Ordering and symbolic analysis; may be called once before factorize().
    virtual SymbolicInfo analyze(const NormalSystem &sys) = 0;
    /// Numerical factorization (a no-op for matrix-free solvers); analyze() is called by factorize() if needed.
    virtual FactorInfo factorize(const NormalSystem &sys) = 0;
    /// Future sigma-dependent systems (plan Section 11.3); the default throws.
    virtual void update(const NormalSystem &sys);
    /// y <- (A A^T + eps I)^{-1} rhs; device arrays of sys.m() doubles; the work is ordered on `stream`.
    virtual SolveReport solve(const double *rhs, double *y, int m, const SolveRequest &req, cudaStream_t stream) = 0;
    /// Device and host memory held by the factorization after factorize() (for the memory plan and the report).
    virtual std::size_t device_bytes() const { return 0; }
    virtual std::size_t host_bytes() const { return 0; }
};

/// Relative residual ||(A A^T + eps I) y - rhs|| / ||rhs|| computed on the host from device vectors (tests, bench).
double relative_residual_host(const NormalSystem &sys, const double *d_rhs, const double *d_y, cudaStream_t stream);

} // namespace linsys
} // namespace cuadmm

#endif // CUADMM_LINSYS_NORMAL_EQUATION_SOLVER_H
