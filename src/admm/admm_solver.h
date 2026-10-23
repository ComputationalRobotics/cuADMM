/*

    admm/admm_solver.h

    AdmmSolver::run (plan Section 7.1): the iteration loop as the mathematics,

        r = kkt.evaluate(s)                                   // starting point
        for k = 1, 2, ...:
            if (stop = stopping.check(k - 1, r)) return finish(stop)
            scheme.iterate(s, sigma, tau)                     // y-step(s), cone step, multiplier step
            r = kkt.evaluate(s)                               // post-step residuals
            sigma = penalty.update(k, r, sigma)
            observers.on_iteration(k, r, sigma)

    on the components of Phases 4 (AdmmState, IterationScheme, PenaltyPolicy, StoppingRule, BestIterateTracker,
    observers). The loop reproduces solve() of main bit for bit; the ported engine (SDPSolver) still holds the data.

*/

#ifndef CUADMM_ADMM_SOLVER_H
#define CUADMM_ADMM_SOLVER_H

#include <memory>
#include <string>

#include "admm/admm_state.h"
#include "admm/iteration_scheme.h"
#include "admm/penalty_policy.h"
#include "report/observers.h"
#include "termination/best_iterate_tracker.h"
#include "termination/stopping_rule.h"

namespace cuadmm
{
namespace admm
{

struct RunConfig
{
    int max_iter = 1000000;
    double tol = 1e-4;                          // the internal tolerance (also read by the legacy tau reduction)
    int switch_admm = 0;                        // sGS-ADMM for k < switch_admm, plain ADMM after (0: ADMM throughout)
    double tau_admm = 1.618;
    double tau_sgs = 1.95;
    bool legacy_sgs_tau_reduction = false;      // today's hidden rule: tau = max(1.618, tau / 1.1) when eta_d < tol
    bool verbose = true;
};

struct RunOutcome
{
    std::string stop_reason;
    bool converged = false;
    bool time_limit_reached = false;
    bool returned_best_iterate = false; // the internal best iterate was restored
    bool nonfinite = false;
    int iterations = 0;
    double solve_time_s = 0.0;
    double sigma_final = 0.0;
};

class AdmmSolver
{
public:
    AdmmSolver(AdmmState &state, PenaltyPolicy &penalty, termination::StoppingRule &stopping, termination::BestIterateTracker &tracker,
               report::ObserverSet &observers, const RunConfig &config);

    // Runs the iterations from the current (X, y, S) of the engine: scaled units if `rescale` is false (the cold start
    // of init()), original units otherwise (a warm start or a second solve). Returns the unscaled iterate in the
    // engine with its residuals, cone violations and certificate quantities (as solve() of main).
    RunOutcome run(double sigma0, bool rescale);

private:
    void finish(const termination::StopDecision &stop, double elapsed_s, bool nonfinite);

    AdmmState &state_;
    PenaltyPolicy &penalty_;
    termination::StoppingRule &stopping_;
    termination::BestIterateTracker &tracker_;
    report::ObserverSet &observers_;
    RunConfig config_;
    AdmmScheme admm_;
    SgsAdmmScheme sgs_;
};

} // namespace admm
} // namespace cuadmm

#endif // CUADMM_ADMM_SOLVER_H
