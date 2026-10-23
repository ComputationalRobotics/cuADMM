/*

    linsys/direct/cholmod/cholmod_solver.cu

    The factorization block of SDPSolver::init() and SDPSolver::solve_AAt_perm() of Milestone 1, with the
    permutations of the y-step, moved without a change of their calls.

*/

#include "linsys/direct/cholmod/cholmod_solver.h"

#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "core/check.h"
#include "cuadmm_legacy/check.h"
#include "cuadmm_legacy/cholesky_cpu.h"
#include "cuadmm_legacy/kernels.h"
#include "cuadmm_legacy/utils.h"

namespace cuadmm
{
namespace linsys
{

CholmodSolver::CholmodSolver(cublasHandle_t cublas) : CholmodSolver(cublas, Options()) {}

CholmodSolver::CholmodSolver(cublasHandle_t cublas, const Options &options) : cublas_(cublas), options_(options) {}

CholmodSolver::~CholmodSolver() = default;

Capabilities CholmodSolver::capabilities() const
{
    Capabilities c;
    c.exact = true;
    c.device_solves = dense_gpu_;
    return c;
}

CholeskySolverCPU &CholmodSolver::cholmod()
{
    if (!chol_)
        throw std::logic_error("CholmodSolver: no factorization yet");
    return *chol_;
}

SymbolicInfo CholmodSolver::analyze(const NormalSystem &sys)
{
    // CHOLMOD analyzes and factorizes in one go (the supernodal attempt decides the ordering's use): the symbolic
    // information is known after factorize()
    (void)sys;
    symbolic_.ordering = "cholmod (AMD/COLAMD, the supernodal LL^T attempt then simplicial LDL^T)";
    return symbolic_;
}

FactorInfo CholmodSolver::factorize(const NormalSystem &sys)
{
    const auto start = std::chrono::steady_clock::now();
    m_ = sys.m();
    chol_.reset(new CholeskySolverCPU());
    // the engine's call took A in CSC (its A^T in CSR, from cuSPARSE's conversion: the entries of a row of A^T in
    // increasing column order); the same arrays from the host transpose of the NormalSystem's A^T in CSC
    const int m = sys.m(), n = sys.n();
    const int *tp = sys.at_colptr(), *ti = sys.at_rowind();
    const double *tx = sys.at_val();
    const int nnz = tp[m];
    std::vector<int> a_colptr(n + 1, 0), a_rowind(nnz), fill(n, 0);
    std::vector<double> a_val(nnz);
    for (int k = 0; k < nnz; k++)
        a_colptr[ti[k] + 1]++;
    for (int i = 0; i < n; i++)
        a_colptr[i + 1] += a_colptr[i];
    for (int j = 0; j < m; j++)
        for (int k = tp[j]; k < tp[j + 1]; k++)
        {
            const int i = ti[k];
            const int dst = a_colptr[i] + fill[i]++;
            a_rowind[dst] = j;
            a_val[dst] = tx[k];
        }
    chol_->get_A(a_colptr.data(), a_rowind.data(), a_val.data(), m, n, nnz, false, sys.eps());
    chol_->factorize();
    factor_ = FactorInfo();
    factor_.min_pivot = chol_->min_pivot;
    factor_.max_pivot = chol_->max_pivot;
    factor_.tiny_pivots = chol_->tiny_pivots;
    factor_.nonpositive_pivots = chol_->nonpositive_pivots;
    factor_.supernodal = chol_->supernodal_used;
    factor_.kind = chol_->supernodal_used ? "supernodal LL^T -> simplicial LDL^T" : "simplicial LDL^T";
    symbolic_.nnz_L = chol_->cc.lnz;
    symbolic_.host_bytes = static_cast<std::size_t>(chol_->cc.lnz) * (sizeof(double) + sizeof(int));

    // With a dense factor (few constraints, e.g. neosfrbr25: nnz(L) = m^2/2 = 1e8, 119 ms per CPU solve), the y-step
    // solves are two dense triangular solves on the GPU (4 ms), with the same factor. Sparse factors stay on the CPU:
    // cuSPARSE triangular solves were measured slower than CHOLMOD's there (0.1-1.3x).
    const int con_num = m_;
    {
        size_t free_mem = 0, total_mem = 0;
        CUADMM_CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
        const double m = con_num;
        const cholmod_factor *F = chol_->chol_fac_L;
        const bool simplicial_ldlt = !F->is_super && !F->is_ll;
        dense_gpu_ = options_.dense_gpu_auto && con_num >= options_.dense_min_m && m * m <= double(std::numeric_limits<int>::max()) &&
                     8.0 * m * m <= options_.dense_memory_fraction * double(free_mem) &&
                     chol_->cc.lnz > options_.dense_min_fill * m * (m + 1) / 2 && simplicial_ldlt;
        if (options_.force_dense_gpu)
        {
            if (!simplicial_ldlt)
                throw std::runtime_error("CholmodSolver: the dense-factor variant needs a simplicial LDL^T factor");
            dense_gpu_ = true;
        }
        if (dense_gpu_)
        {
            // simplicial LDL^T: column j holds D_jj first, then the entries L(i, j), i > j (Lnz[j] entries in all)
            const int *Lp = (const int *)F->p, *Li = (const int *)F->i, *Lnz = (const int *)F->nz;
            const double *Lx = (const double *)F->x;
            std::vector<double> L_dense(size_t(con_num) * con_num, 0.0), D(con_num);
            for (int j = 0; j < con_num; j++)
            {
                if (Lnz[j] < 1 || Li[Lp[j]] != j)
                    throw std::runtime_error("unexpected layout of the CHOLMOD factor: the diagonal is not first");
                D[j] = Lx[Lp[j]];
                L_dense[size_t(j) * con_num + j] = 1.0;
                for (int k = Lp[j] + 1; k < Lp[j] + Lnz[j]; k++)
                    L_dense[size_t(j) * con_num + Li[k]] = Lx[k];
            }
            L_dense_.allocate(size_t(con_num) * con_num);
            D_dense_.allocate(con_num);
            CUADMM_CHECK_CUDA(cudaMemcpy(L_dense_.data(), L_dense.data(), sizeof(double) * L_dense.size(), cudaMemcpyHostToDevice));
            CUADMM_CHECK_CUDA(cudaMemcpy(D_dense_.data(), D.data(), sizeof(double) * con_num, cudaMemcpyHostToDevice));
            factor_.kind = "simplicial LDL^T, dense L and D on the device";
        }
    }

    // the fill-reducing permutation of the factor and its inverse, on the device, with the permuted vectors
    perm_.allocate(con_num);
    perm_inv_.allocate(con_num);
    rhs_perm_.allocate(con_num);
    y_perm_.allocate(con_num);
    CUADMM_CHECK_CUDA(cudaMemcpy(perm_.data(), chol_->chol_fac_L->Perm, sizeof(int) * con_num, cudaMemcpyHostToDevice));
    std::vector<int> perm_tmp(con_num, 0);
    std::vector<int> perm_inv_tmp;
    std::memcpy(perm_tmp.data(), chol_->chol_fac_L->Perm, sizeof(int) * con_num);
    get_inverse_permutation(perm_inv_tmp, perm_tmp);
    CUADMM_CHECK_CUDA(cudaMemcpy(perm_inv_.data(), perm_inv_tmp.data(), sizeof(int) * con_num, cudaMemcpyHostToDevice));

    factor_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    symbolic_.seconds = factor_.seconds;
    factorized_ = true;
    return factor_;
}

SolveReport CholmodSolver::solve(const double *rhs, double *y, int m, const SolveRequest &req, cudaStream_t stream)
{
    (void)req;
    if (!factorized_)
        throw std::logic_error("CholmodSolver::solve before factorize");
    if (m != m_)
        throw std::invalid_argument("CholmodSolver::solve: " + std::to_string(m) + " unknowns for a system of size " + std::to_string(m_));
    // the y-step of main: rhs_perm <- P^{-1} rhs, solve, y <- P y_perm (the kernels and cuBLAS on the given stream;
    // the legacy default stream in Milestone 2)
    perform_permutation(rhs_perm_.data(), rhs, perm_inv_.data(), m, stream);
    if (dense_gpu_)
    {
        // y_perm <- L^{-T} D^{-1} L^{-1} rhs_perm
        CUADMM_CHECK_CUDA(cudaMemcpyAsync(y_perm_.data(), rhs_perm_.data(), sizeof(double) * m, cudaMemcpyDeviceToDevice, stream));
        CUADMM_CHECK_CUBLAS(cublasDtrsv(cublas_, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, m, L_dense_.data(), m, y_perm_.data(), 1));
        dense_vector_div_dense_vector_mul_scalar(y_perm_.data(), D_dense_.data(), 1.0, m, stream);
        CUADMM_CHECK_CUBLAS(cublasDtrsv(cublas_, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT, m, L_dense_.data(), m, y_perm_.data(), 1));
    }
    else
    {
        CUADMM_CHECK_CUDA(cudaStreamSynchronize(stream));
        CUADMM_CHECK_CUDA(cudaMemcpy(chol_->chol_dn_rhs->x, rhs_perm_.data(), sizeof(double) * m, cudaMemcpyDeviceToHost));
        chol_->solve();
        CUADMM_CHECK_CUDA(cudaMemcpy(y_perm_.data(), chol_->chol_dn_res->x, sizeof(double) * m, cudaMemcpyHostToDevice));
    }
    perform_permutation(y, y_perm_.data(), perm_.data(), m, stream);
    SolveReport r;
    r.iterations = 1;
    r.converged = true;
    return r;
}

std::size_t CholmodSolver::device_bytes() const
{
    return (perm_.size() + perm_inv_.size()) * sizeof(int) + (rhs_perm_.size() + y_perm_.size() + L_dense_.size() + D_dense_.size()) * sizeof(double);
}

std::size_t CholmodSolver::host_bytes() const { return symbolic_.host_bytes; }

} // namespace linsys
} // namespace cuadmm
