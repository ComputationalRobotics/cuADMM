/*

    linsys/direct/cholmod/cholmod_solver.h

    CholmodSolver (plan Section 11.4): the normal-equation backend of main behind the NormalEquationSolver
    interface, bit for bit. factorize() runs CHOLMOD on the host as SDPSolver::init() did (the engine's
    CholeskySolverCPU: A A^T + eps I, the supernodal LL^T attempt then the simplicial LDL^T, the pivot statistics)
    and keeps the factor's fill-reducing permutation on the device; when the factor is dense enough and small
    enough (today's rule: m >= 2000, nnz(L) > 0.3 m (m + 1) / 2, 8 m^2 bytes within a quarter of the free
    device memory, a simplicial LDL^T), its L and D are copied to the device and the solves are two cuBLAS
    triangular solves (the "cholmod_dense_gpu" variant). solve() is the y-step of main: permute the right-hand
    side, solve with the factor (CHOLMOD on the host through the pinned-free host vectors, or the dense
    triangular solves on the device), permute back.

*/

#ifndef CUADMM_LINSYS_DIRECT_CHOLMOD_CHOLMOD_SOLVER_H
#define CUADMM_LINSYS_DIRECT_CHOLMOD_CHOLMOD_SOLVER_H

#include <memory>
#include <string>
#include <vector>

#include <cublas_v2.h>

#include "core/device_buffer.h"
#include "linsys/normal_equation_solver.h"

class CholeskySolverCPU;

namespace cuadmm
{
namespace linsys
{

class CholmodSolver final : public NormalEquationSolver
{
public:
    struct Options
    {
        bool dense_gpu_auto = true;         // today's rule for the dense-factor variant
        int dense_min_m = 2000;             // the variant needs at least this many constraints
        double dense_min_fill = 0.3;        // and nnz(L) above this fraction of m (m + 1) / 2
        double dense_memory_fraction = 0.25; // and 8 m^2 bytes within this fraction of the free device memory
        bool force_dense_gpu = false;       // the variant whatever the rule (tests); throws when the factor is not simplicial LDL^T
    };

    /// cublas: the handle of the dense triangular solves (the engine's main handle: the solves run on its stream).
    explicit CholmodSolver(cublasHandle_t cublas);
    CholmodSolver(cublasHandle_t cublas, const Options &options);
    ~CholmodSolver() override;
    CholmodSolver(const CholmodSolver &) = delete;
    CholmodSolver &operator=(const CholmodSolver &) = delete;

    std::string_view name() const override { return dense_gpu_ ? "cholmod_dense_gpu" : "cholmod"; }
    Capabilities capabilities() const override;
    SymbolicInfo analyze(const NormalSystem &sys) override;
    FactorInfo factorize(const NormalSystem &sys) override;
    SolveReport solve(const double *rhs, double *y, int m, const SolveRequest &req, cudaStream_t stream) override;
    std::size_t device_bytes() const override;
    std::size_t host_bytes() const override;

    bool dense_gpu() const { return dense_gpu_; }
    const FactorInfo &factor_info() const { return factor_; }
    const SymbolicInfo &symbolic_info() const { return symbolic_; }
    /// The CHOLMOD object of the factorization (the copied tests of main read its statistics).
    CholeskySolverCPU &cholmod();

private:
    cublasHandle_t cublas_;
    Options options_;
    std::unique_ptr<CholeskySolverCPU> chol_;
    FactorInfo factor_;
    SymbolicInfo symbolic_;
    int m_ = 0;
    bool factorized_ = false;
    bool dense_gpu_ = false;
    core::DeviceBuffer<int> perm_, perm_inv_;
    core::DeviceBuffer<double> rhs_perm_, y_perm_, L_dense_, D_dense_;
};

} // namespace linsys
} // namespace cuadmm

#endif // CUADMM_LINSYS_DIRECT_CHOLMOD_CHOLMOD_SOLVER_H
