/*

    penalty_policy.h

    The penalty policies: FixedPenalty (sigma = sigma_0) and BalanceFreezePenalty (residual balancing on a fixed check
    schedule, frozen at a set iteration). Pure host code, driven by the residual sequence and testable on recorded
    sequences without a GPU.

    Protocol: after iteration k (1-based) has produced the residuals r of the new iterate,
    sigma_next = policy.update(k, r, sigma) is the sigma of iteration k + 1. Every change is recorded in log().
    reset(sigma0) starts a solve (warm starts included).

*/

#ifndef CUADMM_ADMM_PENALTY_POLICY_H
#define CUADMM_ADMM_PENALTY_POLICY_H

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "cuadmm/options.h"
#include "cuadmm/result.h"

namespace cuadmm
{
namespace admm
{

/// @brief KKT residuals of an iterate and its objectives.
struct KktResiduals
{
    double eta_p = 0.0, eta_d = 0.0, eta_g = 0.0, pobj = 0.0, dobj = 0.0;
    double eta() const { return std::max(std::max(eta_p, eta_d), eta_g); }
    bool finite() const;
};

class PenaltyPolicy
{
public:
    virtual ~PenaltyPolicy() = default;
    virtual std::string_view name() const = 0;
    virtual void reset(double sigma0) = 0;
    // k: the iteration just completed; r: its residuals; sigma: the sigma it used. Returns the sigma of iteration k + 1.
    virtual double update(int k, const KktResiduals &r, double sigma) = 0;
    virtual double sigma_min() const = 0;
    virtual double sigma_max() const = 0;
    // the parameters as the summary reports them (JSON object)
    virtual std::string parameters_json() const = 0;
    virtual bool frozen() const { return false; }
    virtual int freeze_iteration() const { return -1; }
    virtual std::string freeze_reason() const { return ""; }

    const std::vector<SigmaChange> &log() const { return log_; }
    void clear_log() { log_.clear(); }

protected:
    std::vector<SigmaChange> log_;
};

/// @brief sigma_k = sigma_0.
class FixedPenalty final : public PenaltyPolicy
{
public:
    std::string_view name() const override { return "fixed"; }
    void reset(double sigma0) override;
    double update(int k, const KktResiduals &r, double sigma) override;
    double sigma_min() const override { return sigma0_; }
    double sigma_max() const override { return sigma0_; }
    std::string parameters_json() const override;

private:
    double sigma0_ = 0.0;
};

/// @brief balance_freeze. After every iteration the vote counts a primal win if eta_p / eta_d < 1, a dual win
/// otherwise. At the check iterations
///     k in {k <= 200: k = 1 mod 10} u {200 < k <= 1000: k = 1 mod 25} u {1000 < k <= 5000: k = 1 mod 50}
///          u {5000 < k <= 10000: k = 1 mod 200} u {k > 10000: k = 1 mod 1000}
/// sigma is multiplied by `factor` when the primal wins exceed max(1, 1.35 * dual wins), divided by it when the dual
/// wins exceed max(1, 1.35 * primal wins); the winner's count is reset, the loser's kept; sigma stays in
/// [sigma_min, sigma_max]. From iteration freeze_iteration on, sigma keeps its current value.
class BalanceFreezePenalty final : public PenaltyPolicy
{
public:
    explicit BalanceFreezePenalty(const PenaltyOptions &options);
    std::string_view name() const override { return "balance_freeze"; }
    void reset(double sigma0) override;
    double update(int k, const KktResiduals &r, double sigma) override;
    double sigma_min() const override { return sigma_min_; }
    double sigma_max() const override { return sigma_max_; }
    std::string parameters_json() const override;
    bool frozen() const override { return frozen_; }
    int freeze_iteration() const override { return freeze_at_; }
    std::string freeze_reason() const override { return frozen_ ? "freeze_iteration" : ""; }

    // the check iterations of the schedule above
    static bool is_check(int k);

private:
    double factor_, sigma_min_, sigma_max_;
    int freeze_iteration_;
    // state
    double sigma_ = 0.0;
    int prim_win_ = 0, dual_win_ = 0;
    bool frozen_ = false;
    int freeze_at_ = -1;
};

/// @brief The policy of the options.
std::unique_ptr<PenaltyPolicy> make_penalty_policy(const PenaltyOptions &options);

} // namespace admm
} // namespace cuadmm

#endif // CUADMM_ADMM_PENALTY_POLICY_H
