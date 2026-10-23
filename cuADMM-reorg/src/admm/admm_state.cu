/*

    admm/admm_state.cu

    The steps of SDPSolver::solve() of main (src/solver.cu, Steps 1-5), moved here unchanged: the same kernels in the
    same order on the same buffers, so that the iterates are bit for bit those of the references.

*/

#include "admm/admm_state.h"

#include <cmath>
#include <iomanip>
#include <iostream>

#include "cuadmm_legacy/solver.h"

namespace cuadmm
{
namespace admm
{


AdmmState::AdmmState(SDPSolver &engine, cone::ConeProduct &cones) : engine_(engine), cones_(cones)
{
    ctx_.stream = nullptr; // the legacy default stream of the engine's kernels
    ctx_.cublas = engine.cublasH.cublas_handle;
    ctx_.cusparse = engine.cusparseH.cusparse_handle;
}

void AdmmState::y_step(double sigma)
{
    SDPSolver &e = engine_;
    /* r_s^{k+1/2} = b/sigma - A(X/sigma + S - C) */
    // rhsy <-- -1.0 * A * SmC + 0.0 * rhsy
    SpMV_cusparse(e.cusparseH, e.A_csr, e.SmC, e.rhsy, -1.0, 0.0, e.SpMV_AX_buffer);
    // rhsy <-- 1/sig * Rp + rhsy
    axpy_cublas(e.cublasH, e.Rp, e.rhsy, 1 / sigma);
    /* y^{k+1/2} = (AA^T)^{-1} r_s^{k+1/2} */
    e.linsys->solve(e.rhsy.vals, e.y.vals, e.con_num, cuadmm::linsys::SolveRequest(), nullptr);
    y_solves_++;
}

void AdmmState::cone_step(double sigma, int k)
{
    SDPSolver &e = engine_;
    // Aty <-- 1.0 * At * y + 0.0 * Aty
    SpMV_cusparse(e.cusparseH, e.At_csr, e.y, e.Aty, 1.0, 0.0, e.SpMV_Aty_buffer);
    // Rd1 <-- Aty; Rd1 <-- (-1.0) * C + 1.0 * Rd1, hence Rd1 = A^T y^{k+1/2} - C
    CHECK_CUDA(cudaMemcpy(e.Rd1.vals, e.Aty.vals, sizeof(double) * e.vec_len, D2D));
    axpby_cusparse(e.cusparseH, e.C, e.Rd1, -1.0, 1.0);
    // Xinput <-- -(Rd1 + 1/sig * X)
    dense_vector_plus_dense_vector_mul_scalar(e.Xinput, e.Rd1, e.X, 1.0 / sigma);
    dense_vector_negate(e.Xinput);
    // S = Pi_K*(Xinput): the PSD cone (unpack, the tiers of iteration k, pack), the nonnegative and the free cones
    cones_.project_dual(core::DeviceSpan<const double>(e.Xinput.vals, e.vec_len), core::DeviceSpan<double>(e.S.vals, e.vec_len), k, ctx_);
}

void AdmmState::update_smc()
{
    SDPSolver &e = engine_;
    // SmC <-- S; SmC <-- -1.0 * C + 1.0 * SmC, hence SmC = S^{k+1} - C
    CHECK_CUDA(cudaMemcpy(e.SmC.vals, e.S.vals, sizeof(double) * e.vec_len, D2D));
    axpby_cusparse(e.cusparseH, e.C, e.SmC, -1.0, 1.0);
}

void AdmmState::second_y_step(double sigma)
{
    SDPSolver &e = engine_;
    // rhsy <-- -1.0 * A * SmC + 0.0 * rhsy, hence rhsy = - A(S - C)
    SpMV_cusparse(e.cusparseH, e.A_csr, e.SmC, e.rhsy, -1.0, 0.0, e.SpMV_AX_buffer);
    // rhsy <-- 1/sig * Rp + rhsy, hence rhsy = 1/sigma b - A(X^k /sigma + S^{k+1} - C)
    axpy_cublas(e.cublasH, e.Rp, e.rhsy, 1 / sigma);
    // y <-- linsys(rhsy)
    e.linsys->solve(e.rhsy.vals, e.y.vals, e.con_num, cuadmm::linsys::SolveRequest(), nullptr);
    y_solves_++;
    // Aty <-- 1.0 * At * y + 0.0 * Aty, hence Aty = A^T y^{k+1}
    SpMV_cusparse(e.cusparseH, e.At_csr, e.y, e.Aty, 1.0, 0.0, e.SpMV_Aty_buffer);
    // Rd1 <-- Aty; Rd1 <-- (-1.0) * C + 1.0 * Rd1, hence Rd1 = A^T y^{k+1} - C
    CHECK_CUDA(cudaMemcpy(e.Rd1.vals, e.Aty.vals, sizeof(double) * e.vec_len, D2D));
    axpby_cusparse(e.cusparseH, e.C, e.Rd1, -1.0, 1.0);
}

void AdmmState::multiplier_step(double sigma, double tau)
{
    SDPSolver &e = engine_;
    // Rd <-- 1.0 * Rd1 + 1.0 * S, hence Rd = A^T y^{k+1} - C + S
    dense_vector_add_dense_vector(e.Rd, e.Rd1, e.S, 1.0, 1.0);
    // X <-- X + (tau * sig) * Rd
    dense_vector_add_dense_vector(e.X, e.Rd, 1.0, tau * sigma);
}

KktResiduals AdmmState::evaluate_kkt()
{
    SDPSolver &e = engine_;
    // Rp <-- -1.0 * A * X + 0.0 * Rp; Rp <-- 1.0 * b + 1.0 * Rp, hence Rp = b - A X
    SpMV_cusparse(e.cusparseH, e.A_csr, e.X, e.Rp, -1.0, 0.0, e.SpMV_AX_buffer);
    axpby_cusparse(e.cusparseH, e.b, e.Rp, 1.0, 1.0);
    // primal error and objective (original units)
    dense_vector_mul_dense_vector_mul_scalar(e.Rporg, e.normA, e.Rp, e.bscale);
    e.errRp = e.Rporg.get_norm(e.cublasH) / e.norm_borg;
    e.pobj = SparseVV_cusparse(e.cusparseH, e.C, e.X, e.SpVV_CtX_buffer) * e.objscale;
    // dual error and objective
    dense_vector_mul_scalar(e.Rdorg, e.Rd, e.Cscale);
    e.errRd = e.Rdorg.get_norm(e.cublasH) / e.norm_Corg;
    e.dobj = SparseVV_cusparse(e.cusparseH, e.b, e.y, e.SpVV_bty_buffer) * e.objscale;
    e.maxfeas = std::max(e.errRp, e.errRd);
    e.relgap = std::abs(e.pobj - e.dobj) / (1 + std::abs(e.pobj) + std::abs(e.dobj));
    KktResiduals r;
    r.eta_p = e.errRp;
    r.eta_d = e.errRd;
    r.eta_g = e.relgap;
    r.pobj = e.pobj;
    r.dobj = e.dobj;
    return r;
}

KktResiduals AdmmState::evaluate_kkt_initial()
{
    SDPSolver &e = engine_;
    // the starting point: SmC = S - C, Rp = b - A X, Aty = A^T y, Rd = Aty + SmC (the rescale path of solve() of main)
    CHECK_CUDA(cudaMemcpy(e.SmC.vals, e.S.vals, sizeof(double) * e.vec_len, D2D));
    axpby_cusparse(e.cusparseH, e.C, e.SmC, -1.0, 1.0);
    SpMV_cusparse(e.cusparseH, e.A_csr, e.X, e.Rp, -1.0, 0.0, e.SpMV_AX_buffer);
    axpby_cusparse(e.cusparseH, e.b, e.Rp, 1.0, 1.0);
    SpMV_cusparse(e.cusparseH, e.At_csr, e.y, e.Aty, 1.0, 0.0, e.SpMV_Aty_buffer);
    dense_vector_add_dense_vector(e.Rd, e.Aty, e.SmC);
    dense_vector_mul_dense_vector_mul_scalar(e.Rporg, e.normA, e.Rp, e.bscale);
    dense_vector_mul_scalar(e.Rdorg, e.Rd, e.Cscale);
    e.errRp = e.Rporg.get_norm(e.cublasH) / e.norm_borg;
    e.errRd = e.Rdorg.get_norm(e.cublasH) / e.norm_Corg;
    e.maxfeas = std::max(e.errRp, e.errRd);
    e.pobj = SparseVV_cusparse(e.cusparseH, e.C, e.X, e.SpVV_CtX_buffer) * e.objscale;
    e.dobj = SparseVV_cusparse(e.cusparseH, e.b, e.y, e.SpVV_bty_buffer) * e.objscale;
    e.relgap = std::abs(e.pobj - e.dobj) / (1 + std::abs(e.pobj) + std::abs(e.dobj));
    KktResiduals r;
    r.eta_p = e.errRp;
    r.eta_d = e.errRd;
    r.eta_g = e.relgap;
    r.pobj = e.pobj;
    r.dobj = e.dobj;
    return r;
}

void AdmmState::scale_iterate()
{
    SDPSolver &e = engine_;
    // X, y, S given in original units: scale them as init() and the rescale path of solve() of main do
    dense_vector_mul_dense_vector(e.y, e.normA);
    dense_vector_div_scalar(e.X, e.bscale);
    dense_vector_div_scalar(e.S, e.Cscale);
    dense_vector_div_scalar(e.y, e.Cscale);
}

void AdmmState::unscale_iterate()
{
    SDPSolver &e = engine_;
    // recover the original solution by unscaling
    dense_vector_mul_scalar(e.X, e.bscale);
    dense_vector_div_dense_vector_mul_scalar(e.y, e.normA, e.Cscale);
    dense_vector_mul_scalar(e.S, e.Cscale);
}

void AdmmState::begin_iteration(int k, double eta, bool verbose)
{
    cones_.begin_iteration(k, eta, verbose);
}

} // namespace admm
} // namespace cuadmm
