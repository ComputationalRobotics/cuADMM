/*

    termination/best_iterate_tracker.cu

*/

#include "termination/best_iterate_tracker.h"

#include <cstdio>
#include <limits>

#include "cuadmm_legacy/solver.h"

namespace cuadmm
{
namespace termination
{

void BestIterateTracker::reset()
{
    best_kkt_ = std::numeric_limits<double>::infinity(); // no best iterate saved yet
    best_iteration_ = 0;
    engine_.best_KKT = best_kkt_;
}

void BestIterateTracker::observe(int k, const admm::KktResiduals &r)
{
    SDPSolver &e = engine_;
    // the residuals describe exactly the current (X, y, S), and the copies are issued on the legacy default stream
    // so that they are ordered with the kernels updating X, y, S
    const double kkt = std::max(std::max(r.eta_p, r.eta_d), r.eta_g);
    if (kkt < best_kkt_)
    {
        best_kkt_ = kkt;
        best_iteration_ = k;
        e.best_KKT = kkt;
        e.best_errRp = r.eta_p;
        e.best_errRd = r.eta_d;
        e.best_relgap = r.eta_g;
        e.best_pobj = r.pobj;
        e.best_dobj = r.dobj;
        CHECK_CUDA(cudaMemcpyAsync(e.X_best.vals, e.X.vals, sizeof(double) * e.vec_len, D2D, (cudaStream_t)0));
        CHECK_CUDA(cudaMemcpyAsync(e.y_best.vals, e.y.vals, sizeof(double) * e.con_num, D2D, (cudaStream_t)0));
        CHECK_CUDA(cudaMemcpyAsync(e.S_best.vals, e.S.vals, sizeof(double) * e.vec_len, D2D, (cudaStream_t)0));
    }
}

bool BestIterateTracker::restore_if_better()
{
    SDPSolver &e = engine_;
    /* Return the best iterate if it is strictly better than the last one */
    if (!(best_kkt_ < std::max(e.maxfeas, e.relgap)))
        return false;
    printf(" Returning the best iterate: max KKT residual = %2.1e (last iterate: %2.1e)\n", best_kkt_, std::max(e.maxfeas, e.relgap));
    CHECK_CUDA(cudaMemcpyAsync(e.X.vals, e.X_best.vals, sizeof(double) * e.vec_len, D2D, (cudaStream_t)0));
    CHECK_CUDA(cudaMemcpyAsync(e.y.vals, e.y_best.vals, sizeof(double) * e.con_num, D2D, (cudaStream_t)0));
    CHECK_CUDA(cudaMemcpyAsync(e.S.vals, e.S_best.vals, sizeof(double) * e.vec_len, D2D, (cudaStream_t)0));
    // the reported residuals and objectives describe the returned iterate
    e.errRp = e.best_errRp;
    e.errRd = e.best_errRd;
    e.relgap = e.best_relgap;
    e.pobj = e.best_pobj;
    e.dobj = e.best_dobj;
    e.maxfeas = std::max(e.errRp, e.errRd);
    return true;
}

} // namespace termination
} // namespace cuadmm
