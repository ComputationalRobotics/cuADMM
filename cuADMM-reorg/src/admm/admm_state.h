/*

    admm/admm_state.h

    AdmmState (plan Section 7.1): the iterate (X, y, S), the cached residual vectors and the three steps of the
    iteration (2)-(4) in scaled units, implemented on the data and workspaces of the ported engine (SDPSolver of
    src/legacy/, which still owns the problem data, the scaling, the factorization and the PSD tiers: Phases 5-7).
    The code of each step is the code of SDPSolver::solve() of main, moved here unchanged (bitwise acceptance).

*/

#ifndef CUADMM_ADMM_STATE_H
#define CUADMM_ADMM_STATE_H

#include <string>

#include "admm/penalty_policy.h" // KktResiduals
#include "cone/cone_product.h"
#include "core/gpu_context.h"

class SDPSolver;

namespace cuadmm
{
namespace admm
{

class AdmmState
{
public:
    AdmmState(SDPSolver &engine, cone::ConeProduct &cones);

    // (2): rhsy = Rp / sigma - A (S - C); y = (AA^T)^{-1} rhsy (CHOLMOD on the CPU, or the dense factor on the GPU)
    void y_step(double sigma);
    // (3): Aty = A^T y; Rd1 = Aty - C; Xinput = -(Rd1 + X / sigma); S = Pi_K*(Xinput) through the product cone (the
    // PSD tiers with the large-block schedule of iteration k, the nonnegative and the free blocks)
    void cone_step(double sigma, int k);
    // SmC = S - C (read by the next y-step)
    void update_smc();
    // sGS-ADMM only: the second normal-equation solve y = (AA^T)^{-1} [Rp / sigma - A (S^{k+1} - C)], then Aty and Rd1
    void second_y_step(double sigma);
    // (4): Rd = Rd1 + S; X += tau sigma Rd
    void multiplier_step(double sigma, double tau);
    // Rp = b - A X and the KKT residuals and objectives of the current iterate in original units (Rd is the one of
    // the multiplier step)
    KktResiduals evaluate_kkt();
    // The same for a starting point (X, y, S) in scaled units: SmC, Rp, Aty and Rd are formed first (the rescale path
    // of solve() of main)
    KktResiduals evaluate_kkt_initial();
    // Scaling of an iterate between original and scaled units (plan: the Scaling object of Phase 5+)
    void scale_iterate();
    void unscale_iterate();

    // The projection-method schedule of today (switched_proj_method): at iteration k with the residuals of the
    // current iterate, switch to the final projection method (the end of the warm-up, the start of the low-rank
    // analysis) when k > switch_iter or eta < switch_tol, k > 1.
    void begin_iteration(int k, double eta, int switch_iter, double switch_tol);

    long y_solves() const { return y_solves_; }
    void reset_counters() { y_solves_ = 0; }
    SDPSolver &engine() { return engine_; }
    const SDPSolver &engine() const { return engine_; }
    cone::ConeProduct &cones() { return cones_; }
    core::ExecContext &context() { return ctx_; }

private:
    SDPSolver &engine_;
    cone::ConeProduct &cones_;
    core::ExecContext ctx_; // Phase 5: the legacy default stream and the engine's handles
    long y_solves_ = 0;
};

} // namespace admm
} // namespace cuadmm

#endif // CUADMM_ADMM_STATE_H
