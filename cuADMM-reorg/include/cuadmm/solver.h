/*

    solver.h

    The Solver facade (plan Section 6.3): options in, ProblemData in, SolveResult out. One GPU per solver.

        cuadmm::SolverOptions options;
        options.penalty.policy = cuadmm::PenaltyPolicyKind::Fixed;
        cuadmm::Solver solver(options);
        solver.setup(problem);                  // scale, factorize, plan the cones, check memory
        cuadmm::SolveResult result = solver.solve();   // resets the penalty-policy state

    Phase 2: the facade drives the ported SDPSolver (src/legacy/) and reproduces its iterates bit for bit for
    equivalent options; Phase 4 replaces the loop by AdmmSolver and adds observers.

*/

#ifndef CUADMM_SOLVER_FACADE_H
#define CUADMM_SOLVER_FACADE_H

#include <memory>

#include "cuadmm/options.h"
#include "cuadmm/problem.h"
#include "cuadmm/result.h"

namespace cuadmm
{

class Solver
{
public:
    explicit Solver(SolverOptions options); // validates the options
    ~Solver();
    Solver(const Solver &) = delete;
    Solver &operator=(const Solver &) = delete;

    // Presolve (off in Milestone 1), scale, factorize AA^T, plan the cones, check the memory; throws on an
    // unsupported cone, a problem that does not fit the 32-bit indexing of the current engine, or a GPU error.
    void setup(const ProblemData &problem);

    // Runs the iterations from the cold start, or from the warm start (original units); resets the penalty-policy
    // state. setup() must have been called.
    SolveResult solve(const WarmStart *warm_start = nullptr);

    const SolverOptions &options() const;

    // Setup summary (after setup()): the normal-equation backend chosen and the factorization statistics.
    std::string linsys_backend() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cuadmm

#endif // CUADMM_SOLVER_FACADE_H
