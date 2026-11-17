/*

    penalty_policy.cc

    FixedPenalty and BalanceFreezePenalty (see penalty_policy.h).

*/

#include "admm/penalty_policy.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "report/writers.h"

namespace cuadmm
{
namespace admm
{

bool KktResiduals::finite() const
{
    return std::isfinite(eta_p) && std::isfinite(eta_d) && std::isfinite(eta_g) && std::isfinite(pobj) && std::isfinite(dobj);
}

// FixedPenalty ------------------------------------------------------------------------------------------------------

void FixedPenalty::reset(double sigma0)
{
    sigma0_ = sigma0;
    log_.clear();
}

double FixedPenalty::update(int k, const KktResiduals &, double sigma)
{
    if (sigma != sigma0_)
        throw std::logic_error("fixed sigma policy violated at iteration " + std::to_string(k) + ": sigma = " + std::to_string(sigma) +
                               ", sigma_0 = " + std::to_string(sigma0_));
    return sigma;
}

std::string FixedPenalty::parameters_json() const { return "{\"sigma\": " + report::json_number(sigma0_) + "}"; }

// BalanceFreezePenalty ----------------------------------------------------------------------------------------------

BalanceFreezePenalty::BalanceFreezePenalty(const PenaltyOptions &o)
    : factor_(o.factor), sigma_min_(o.sigma_min), sigma_max_(o.sigma_max), freeze_iteration_(o.freeze_iteration)
{
    if (!(sigma_min_ > 0.0 && sigma_min_ <= sigma_max_))
        throw std::invalid_argument("penalty.sigma_min must be positive and not exceed penalty.sigma_max");
    if (!(factor_ > 1.0))
        throw std::invalid_argument("penalty.factor must be greater than 1");
    if (freeze_iteration_ < 0)
        throw std::invalid_argument("penalty.freeze_iteration must be non-negative");
}

bool BalanceFreezePenalty::is_check(int k)
{
    return (k <= 200 && (k % 10) == 1) || (k > 200 && k <= 1000 && (k % 25) == 1) || (k > 1000 && k <= 5000 && (k % 50) == 1) ||
           (k > 5000 && k <= 10000 && (k % 200) == 1) || (k > 10000 && (k % 1000) == 1);
}

void BalanceFreezePenalty::reset(double sigma0)
{
    sigma_ = std::min(std::max(sigma0, sigma_min_), sigma_max_);
    prim_win_ = dual_win_ = 0;
    frozen_ = false;
    freeze_at_ = -1;
    log_.clear();
}

double BalanceFreezePenalty::update(int k, const KktResiduals &r, double sigma)
{
    sigma_ = sigma;
    if (frozen_)
        return sigma_;
    // the vote
    const double feasratio = r.eta_p / r.eta_d;
    if (feasratio < 1)
        prim_win_ += 1;
    else
        dual_win_ += 1;
    const int prim_before = prim_win_, dual_before = dual_win_;
    if (is_check(k))
    {
        int direction = 0;
        if (prim_win_ > std::max(1.0, 1.35 * dual_win_))
        {
            prim_win_ = 0;
            direction = +1;
        }
        else if (dual_win_ > std::max(1.0, 1.35 * prim_win_))
        {
            dual_win_ = 0;
            direction = -1;
        }
        if (direction != 0)
        {
            const double before = sigma_;
            const double moved = direction > 0 ? sigma_ * factor_ : sigma_ / factor_;
            sigma_ = std::min(std::max(moved, sigma_min_), sigma_max_);
            if (sigma_ != before)
                log_.push_back({k, before, sigma_, "admm_schedule", r.eta_p, r.eta_d, r.eta_g, feasratio, prim_before, dual_before});
        }
    }
    if (k >= freeze_iteration_)
    {
        // the freeze keeps the current value (clamped into the bounds); it is logged even when sigma does not move
        const double before = sigma_;
        sigma_ = std::min(std::max(sigma_, sigma_min_), sigma_max_);
        frozen_ = true;
        freeze_at_ = k;
        log_.push_back({k, before, sigma_, "freeze_freeze_iteration", r.eta_p, r.eta_d, r.eta_g, feasratio, prim_before, dual_before});
    }
    return sigma_;
}

std::string BalanceFreezePenalty::parameters_json() const
{
    std::ostringstream o;
    o << "{\"factor\": " << report::json_number(factor_) << ", \"sigma_min\": " << report::json_number(sigma_min_)
      << ", \"sigma_max\": " << report::json_number(sigma_max_) << ", \"freeze_iteration\": " << freeze_iteration_
      << ", \"win_ratio\": 1.35, \"schedule\": \"200:10,1000:25,5000:50,10000:200,inf:1000\"}";
    return o.str();
}

std::unique_ptr<PenaltyPolicy> make_penalty_policy(const PenaltyOptions &o)
{
    switch (o.policy)
    {
    case PenaltyPolicyKind::Fixed:
        return std::make_unique<FixedPenalty>();
    case PenaltyPolicyKind::BalanceFreeze:
        return std::make_unique<BalanceFreezePenalty>(o);
    }
    throw std::invalid_argument("unknown penalty policy");
}

} // namespace admm
} // namespace cuadmm
