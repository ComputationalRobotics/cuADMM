/*

    termination/best_iterate_tracker.h

    BestIterateTracker (plan Section 7.4): keeps the best iterate of a run by its internal KKT residual and restores
    it at the end when the last iterate is worse (the "Bug 2.2" return of main). In validation mode the best
    externally evaluated iterate is kept by ValidatedRule, ranked by ValidationResult::better_than (the practical
    merit, then max |DIMACS| in the strict stage; plan B23).

*/

#ifndef CUADMM_TERMINATION_BEST_ITERATE_TRACKER_H
#define CUADMM_TERMINATION_BEST_ITERATE_TRACKER_H

#include "admm/penalty_policy.h" // KktResiduals

class SDPSolver;

namespace cuadmm
{
namespace termination
{

class BestIterateTracker
{
public:
    explicit BestIterateTracker(SDPSolver &engine) : engine_(engine) {}
    void reset();
    // after iteration k with residuals r: keep a copy of (X, y, S) when its KKT residual is the smallest so far
    void observe(int k, const admm::KktResiduals &r);
    // at termination: restore the best iterate if it is strictly better than the returned one; true if restored
    bool restore_if_better();
    double best_kkt() const { return best_kkt_; }
    int best_iteration() const { return best_iteration_; }

private:
    SDPSolver &engine_;
    double best_kkt_;
    int best_iteration_ = 0;
};

} // namespace termination
} // namespace cuadmm

#endif // CUADMM_TERMINATION_BEST_ITERATE_TRACKER_H
