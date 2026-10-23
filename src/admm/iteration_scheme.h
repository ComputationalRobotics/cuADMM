/*

    admm/iteration_scheme.h

    IterationScheme (plan Section 7.2): one iteration of (2)-(4) as plain ADMM, or its sGS variant with the second
    normal-equation solve. Both run the step implementations of AdmmState.

*/

#ifndef CUADMM_ADMM_ITERATION_SCHEME_H
#define CUADMM_ADMM_ITERATION_SCHEME_H

#include <string_view>

#include "admm/admm_state.h"

namespace cuadmm
{
namespace admm
{

class IterationScheme
{
public:
    virtual ~IterationScheme() = default;
    virtual std::string_view name() const = 0;
    // One iteration k with penalty sigma and step length tau: y-step(s), cone step, multiplier step.
    virtual void iterate(AdmmState &s, int k, double sigma, double tau) = 0;
    virtual int normal_solves_per_iteration() const = 0; // 1 for ADMM, 2 for sGS-ADMM
};

/// @brief (2) -> (3) -> (4): the plain two-block ADMM (switch_admm = 0 of main).
class AdmmScheme final : public IterationScheme
{
public:
    std::string_view name() const override { return "admm"; }
    void iterate(AdmmState &s, int k, double sigma, double tau) override
    {
        s.y_step(sigma);
        s.cone_step(sigma, k);
        s.update_smc();
        s.multiplier_step(sigma, tau);
    }
    int normal_solves_per_iteration() const override { return 1; }
};

/// @brief (2) -> (3) -> (2') -> (4): sGS-ADMM (the iterations before switch_admm of main).
class SgsAdmmScheme final : public IterationScheme
{
public:
    std::string_view name() const override { return "sgs_admm"; }
    void iterate(AdmmState &s, int k, double sigma, double tau) override
    {
        s.y_step(sigma);
        s.cone_step(sigma, k);
        s.update_smc();
        s.second_y_step(sigma);
        s.multiplier_step(sigma, tau);
    }
    int normal_solves_per_iteration() const override { return 2; }
};

} // namespace admm
} // namespace cuadmm

#endif // CUADMM_ADMM_ITERATION_SCHEME_H
