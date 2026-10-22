/*

    penalty_policy.h

    The penalty policies of plan Section 7.3: FixedPenalty, LegacyPenalty (the rule of 9bb4ba5 bit for bit, Monitor1
    included) and ResidualBalancingPenalty (the October 13 rule of 2d8a755, and with the freeze options the planned
    balance_freeze rule of plan Section 3.5). Pure host code: driven by the residual sequence, testable on scripted
    sequences without a GPU.

    Protocol (plan Section 7.1): after iteration k (1-based) has produced the residuals r of the new iterate,
    sigma_next = policy.update(k, r, sigma, sgs_step) is the sigma of iteration k + 1. Every change is recorded in
    log(). reset(sigma0) starts a solve (warm starts included); state() / restore() allow an exact resume.

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

/// @brief KKT residuals of an iterate in original units (plan Section 3.4) and its objectives.
struct KktResiduals
{
    double eta_p = 0.0, eta_d = 0.0, eta_g = 0.0, pobj = 0.0, dobj = 0.0;
    double eta() const { return std::max(std::max(eta_p, eta_d), eta_g); }
    bool finite() const;
};

/// @brief Everything a policy needs to resume exactly from a checkpoint.
struct PolicyState
{
    int k = 0;                   // last iteration seen
    double sigma = 0.0;          // sigma of the next iteration
    int prim_win = 0, dual_win = 0;
    bool frozen = false;
    int freeze_iteration = -1;
    std::string freeze_reason;
    double step = 0.0;           // balance_freeze: current factor
    std::vector<int> dirs;       // balance_freeze: directions of the changes
    std::vector<double> sigmas;  // balance_freeze: sigma after each change
    // legacy_adaptive: the Monitor1 queues (the difference queues are the consecutive differences)
    std::vector<double> monitor_pobj, monitor_dobj, monitor_obj_diff, monitor_ratio;
};

class PenaltyPolicy
{
public:
    virtual ~PenaltyPolicy() = default;
    virtual std::string_view name() const = 0;
    virtual void reset(double sigma0) = 0;
    // k: the iteration just completed; r: its residuals; sgs_step: iteration k was an sGS-ADMM step (the adaptive
    // policies of the references apply the sGS-phase rule then)
    virtual double update(int k, const KktResiduals &r, double sigma, bool sgs_step) = 0;
    virtual PolicyState state() const = 0;
    virtual void restore(const PolicyState &st) = 0;
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
    void record(int k, double old_sigma, double new_sigma, const char *rule, const KktResiduals &r, double feasratio, int prim_win,
                int dual_win);
    std::vector<SigmaChange> log_;
};

/// @brief sigma_k = sigma_0.
class FixedPenalty final : public PenaltyPolicy
{
public:
    std::string_view name() const override { return "fixed"; }
    void reset(double sigma0) override;
    double update(int k, const KktResiduals &r, double sigma, bool sgs_step) override;
    PolicyState state() const override;
    void restore(const PolicyState &st) override;
    double sigma_min() const override { return sigma0_; }
    double sigma_max() const override { return sigma0_; }
    std::string parameters_json() const override;

private:
    double sigma0_ = 0.0;
    int k_ = 0;
};

/// @brief The rule of 9bb4ba5 bit for bit: the vote every iteration, the plain-ADMM schedule with factor 2 and no
/// guard in [1e-6, 1e6], the Monitor1 "chasing" correction after iteration 5000 every 25 iterations, and the sGS-phase
/// rule (factor sgs_factor every sgs_update_stage_1 / _2 iterations, hysteresis 1.2) for sGS steps.
class LegacyPenalty final : public PenaltyPolicy
{
public:
    explicit LegacyPenalty(const PenaltyOptions &options);
    ~LegacyPenalty() override;
    std::string_view name() const override { return "legacy_adaptive"; }
    void reset(double sigma0) override;
    double update(int k, const KktResiduals &r, double sigma, bool sgs_step) override;
    PolicyState state() const override;
    void restore(const PolicyState &st) override;
    double sigma_min() const override { return sigma_min_; }
    double sigma_max() const override { return sigma_max_; }
    std::string parameters_json() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    double sigma_min_ = 1e-6, sigma_max_ = 1e6;
    int sgs_threshold_, sgs_stage_1_, sgs_stage_2_;
    double sgs_factor_;
    double residual_weight_ = 1.0;
};

/// @brief The October 13 rule of 2d8a755 (residual_balancing: guard max(1, win_ratio * other), factor, bounds
/// [1e-3, 1e3], no Monitor1, the same sGS-phase rule as LegacyPenalty) and, with freeze enabled, the planned
/// balance_freeze rule: freeze at freeze_iteration or on ping-pong, band-centre value, bias, step schedule, sigma_0
/// clamped into the bounds (plan Section 3.5, Table 2).
class ResidualBalancingPenalty final : public PenaltyPolicy
{
public:
    // freeze: false for residual_balancing, true for balance_freeze
    ResidualBalancingPenalty(const PenaltyOptions &options, bool freeze);
    std::string_view name() const override { return freeze_enabled_ ? "balance_freeze" : "residual_balancing"; }
    void reset(double sigma0) override;
    double update(int k, const KktResiduals &r, double sigma, bool sgs_step) override;
    PolicyState state() const override;
    void restore(const PolicyState &st) override;
    double sigma_min() const override { return sigma_min_; }
    double sigma_max() const override { return sigma_max_; }
    std::string parameters_json() const override;
    bool frozen() const override { return frozen_; }
    int freeze_iteration() const override { return freeze_at_; }
    std::string freeze_reason() const override { return freeze_reason_; }

    // the check iterations of the schedule (k = 1 mod period while k <= up to)
    bool is_check(int k) const;

private:
    void change(int k, int direction, const KktResiduals &r, double feasratio, int prim_before, int dual_before);
    void freeze(int k, const char *reason, const KktResiduals &r, double feasratio, int prim_before, int dual_before);

    PenaltyOptions options_;
    bool freeze_enabled_;
    bool guard_;
    double factor_, win_ratio_, residual_weight_, sigma_min_, sigma_max_, freeze_bias_;
    int freeze_iteration_opt_;
    // state
    double sigma_ = 0.0;
    int k_ = 0;
    int prim_win_ = 0, dual_win_ = 0;
    bool frozen_ = false;
    int freeze_at_ = -1;
    std::string freeze_reason_;
    double step_ = 2.0;
    std::vector<int> dirs_;
    std::vector<double> sigmas_;
};

/// @brief The policy of the options (plan Section 7.3): fixed, legacy_adaptive, residual_balancing, balance_freeze.
std::unique_ptr<PenaltyPolicy> make_penalty_policy(const PenaltyOptions &options);

} // namespace admm
} // namespace cuadmm

#endif // CUADMM_ADMM_PENALTY_POLICY_H
