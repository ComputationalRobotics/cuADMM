/*

    penalty_policy.cc

    The plain-ADMM schedule of the reference rules (9bb4ba5 and 2d8a755, src/solver.cu of main):

        check at k in {k <= 200: k = 1 mod 10} u {200 < k <= 1000: k = 1 mod 25} u {1000 < k <= 5000: k = 1 mod 50}
                   u {5000 < k <= 10000: k = 1 mod 200} u {k > 10000: k = 1 mod 1000}

    with, every iteration, feasratio = residual_weight * eta_p / eta_d; prim_win += 1 if feasratio < 1 else dual_win += 1
    (the counters are incremented before the check of the same iteration; the winner's counter is reset, the loser's
    kept). The sGS-phase rule of both references: at k <= threshold with k = 1 mod stage_1, or k > threshold with
    k = 1 mod stage_2: prim_win > 1.2 dual_win raises sigma by sgs_factor (and resets prim_win), dual_win > 1.2 prim_win
    lowers it.

*/

#include "admm/penalty_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "cuadmm_legacy/monitors.h"
#include "report/writers.h"

namespace cuadmm
{
namespace admm
{

bool KktResiduals::finite() const
{
    return std::isfinite(eta_p) && std::isfinite(eta_d) && std::isfinite(eta_g) && std::isfinite(pobj) && std::isfinite(dobj);
}

void PenaltyPolicy::record(int k, double old_sigma, double new_sigma, const char *rule, const KktResiduals &r, double feasratio,
                           int prim_win, int dual_win)
{
    if (new_sigma != old_sigma)
        log_.push_back({k, old_sigma, new_sigma, rule, r.eta_p, r.eta_d, r.eta_g, feasratio, prim_win, dual_win});
}

namespace
{
// the schedule shared by the reference rules
bool reference_check(int k)
{
    return (k <= 200 && (k % 10) == 1) || (k > 200 && k <= 1000 && (k % 25) == 1) || (k > 1000 && k <= 5000 && (k % 50) == 1) ||
           (k > 5000 && k <= 10000 && (k % 200) == 1) || (k > 10000 && (k % 1000) == 1);
}

bool sgs_check(int k, int threshold, int stage_1, int stage_2)
{
    return (k <= threshold && (k % stage_1) == 1) || (k > threshold && (k % stage_2) == 1);
}

std::string json_num(double v) { return report::json_number(v); }

std::string schedule_string(const std::vector<std::pair<int, int>> &v)
{
    std::string out;
    for (std::size_t i = 0; i < v.size(); i++)
        out += (i ? "," : "") + (v[i].first == std::numeric_limits<int>::max() ? std::string("inf") : std::to_string(v[i].first)) + ":" +
               std::to_string(v[i].second);
    return out;
}
} // namespace

// FixedPenalty ------------------------------------------------------------------------------------------------------

void FixedPenalty::reset(double sigma0)
{
    sigma0_ = sigma0;
    k_ = 0;
    log_.clear();
}

double FixedPenalty::update(int k, const KktResiduals &, double sigma, bool)
{
    k_ = k;
    if (sigma != sigma0_)
        throw std::logic_error("fixed sigma policy violated at iteration " + std::to_string(k) + ": sigma = " + std::to_string(sigma) +
                               ", sigma_0 = " + std::to_string(sigma0_));
    return sigma;
}

PolicyState FixedPenalty::state() const
{
    PolicyState s;
    s.k = k_;
    s.sigma = sigma0_;
    return s;
}

void FixedPenalty::restore(const PolicyState &st)
{
    k_ = st.k;
    sigma0_ = st.sigma;
}

std::string FixedPenalty::parameters_json() const
{
    return "{\"factor\": 1, \"sigma_min\": " + json_num(sigma0_) + ", \"sigma_max\": " + json_num(sigma0_) +
           ", \"win_ratio\": 1.35, \"guard\": false, \"residual_weight\": 1, \"monitor1\": false, \"sgs_rule\": \"\"}";
}

// LegacyPenalty -----------------------------------------------------------------------------------------------------

struct LegacyPenalty::Impl
{
    double sigma = 0.0;
    int k = 0;
    int prim_win = 0, dual_win = 0;
    Monitor1 monitor; // fresh at every reset (a local of solve() at 9bb4ba5)
};

LegacyPenalty::LegacyPenalty(const PenaltyOptions &o)
    : impl_(new Impl), sgs_threshold_(o.sgs_update_threshold), sgs_stage_1_(o.sgs_update_stage_1), sgs_stage_2_(o.sgs_update_stage_2),
      sgs_factor_(o.sgs_factor), residual_weight_(o.residual_weight)
{
    if (o.sigma_min > 0)
        sigma_min_ = o.sigma_min;
    if (o.sigma_max > 0)
        sigma_max_ = o.sigma_max;
    if (!(sigma_min_ <= sigma_max_))
        throw std::invalid_argument("penalty.sigma_min must not exceed penalty.sigma_max");
}

LegacyPenalty::~LegacyPenalty() = default;

void LegacyPenalty::reset(double sigma0)
{
    impl_->sigma = sigma0; // not clamped (9bb4ba5)
    impl_->k = 0;
    impl_->prim_win = impl_->dual_win = 0;
    impl_->monitor = Monitor1();
    log_.clear();
}

double LegacyPenalty::update(int k, const KktResiduals &r, double sigma, bool sgs_step)
{
    Impl &s = *impl_;
    s.k = k;
    s.sigma = sigma;
    // the vote (solver.cu:2001-2005 at 9bb4ba5)
    const double feasratio = residual_weight_ * r.eta_p / r.eta_d;
    if (feasratio < 1)
        s.prim_win += 1;
    else
        s.dual_win += 1;
    const double sig_before = s.sigma;
    const int prim_before = s.prim_win, dual_before = s.dual_win;
    if (sgs_step)
    {
        if (sgs_check(k, sgs_threshold_, sgs_stage_1_, sgs_stage_2_))
        {
            if (s.prim_win > 1.2 * s.dual_win)
            {
                s.prim_win = 0;
                s.sigma = std::min(sigma_max_, s.sigma * sgs_factor_);
            }
            else if (s.dual_win > 1.2 * s.prim_win)
            {
                s.dual_win = 0;
                s.sigma = std::max(sigma_min_, s.sigma / sgs_factor_);
            }
        }
        record(k, sig_before, s.sigma, "sgs_rule", r, feasratio, prim_before, dual_before);
        return s.sigma;
    }
    if (reference_check(k))
    {
        // 9bb4ba5: sigscale = max(2 exp(-k / 50000), 2) == 2 for every k >= 1 (B6), no guard
        const double factor = 2.0;
        if (s.prim_win > 1.35 * s.dual_win)
        {
            s.prim_win = 0;
            s.sigma = std::min(sigma_max_, s.sigma * factor);
        }
        else if (s.dual_win > 1.35 * s.prim_win)
        {
            s.dual_win = 0;
            s.sigma = std::max(sigma_min_, s.sigma / factor);
        }
    }
    record(k, sig_before, s.sigma, "admm_schedule", r, feasratio, prim_before, dual_before);
    // the Monitor1 "chasing" correction (9bb4ba5: k > 5000 and k % 25 == 0, on the global iteration)
    if (k > 5000 && k % static_cast<int>(s.monitor.update_interval) == 0)
    {
        const double sig_mon = s.sigma;
        s.monitor.push(r.pobj, r.dobj, r.eta_p, r.eta_d, r.eta_g);
        if (s.monitor.if_full())
            s.sigma = s.monitor.chase_update_sig(s.sigma);
        record(k, sig_mon, s.sigma, "monitor1", r, feasratio, prim_before, dual_before);
    }
    return s.sigma;
}

PolicyState LegacyPenalty::state() const
{
    const Impl &s = *impl_;
    PolicyState st;
    st.k = s.k;
    st.sigma = s.sigma;
    st.prim_win = s.prim_win;
    st.dual_win = s.dual_win;
    st.monitor_pobj.assign(s.monitor.buffer_pobj.q1.begin(), s.monitor.buffer_pobj.q1.end());
    st.monitor_dobj.assign(s.monitor.buffer_dobj.q1.begin(), s.monitor.buffer_dobj.q1.end());
    st.monitor_obj_diff.assign(s.monitor.buffer_obj_diff.q1.begin(), s.monitor.buffer_obj_diff.q1.end());
    st.monitor_ratio.assign(s.monitor.buffer_relgap_feas_ratio.q1.begin(), s.monitor.buffer_relgap_feas_ratio.q1.end());
    return st;
}

void LegacyPenalty::restore(const PolicyState &st)
{
    Impl &s = *impl_;
    s.k = st.k;
    s.sigma = st.sigma;
    s.prim_win = st.prim_win;
    s.dual_win = st.dual_win;
    s.monitor = Monitor1();
    // re-push the queues: the difference queues follow exactly (value - previous value)
    auto refill = [](SlidingDiffQueue<double> &q, const std::vector<double> &values)
    {
        for (double v : values)
            q.push(v);
    };
    refill(s.monitor.buffer_pobj, st.monitor_pobj);
    refill(s.monitor.buffer_dobj, st.monitor_dobj);
    refill(s.monitor.buffer_obj_diff, st.monitor_obj_diff);
    refill(s.monitor.buffer_relgap_feas_ratio, st.monitor_ratio);
}

std::string LegacyPenalty::parameters_json() const
{
    return "{\"factor\": 2, \"sigma_min\": " + json_num(sigma_min_) + ", \"sigma_max\": " + json_num(sigma_max_) +
           ", \"win_ratio\": 1.35, \"guard\": false, \"residual_weight\": " + json_num(residual_weight_) +
           ", \"monitor1\": true, \"sgs_rule\": \"legacy_sgs\", \"sgs_factor\": " + json_num(sgs_factor_) + "}";
}

// ResidualBalancingPenalty ------------------------------------------------------------------------------------------

ResidualBalancingPenalty::ResidualBalancingPenalty(const PenaltyOptions &o, bool freeze)
    : options_(o), freeze_enabled_(freeze), guard_(freeze ? o.guard : true), factor_(o.factor), win_ratio_(freeze ? o.win_ratio : 1.35),
      residual_weight_(o.residual_weight), sigma_min_(o.sigma_min > 0 ? o.sigma_min : 1e-3), sigma_max_(o.sigma_max > 0 ? o.sigma_max : 1e3),
      freeze_bias_(o.freeze_bias), freeze_iteration_opt_(o.freeze_iteration >= 0 ? o.freeze_iteration : (freeze ? 10000 : -1))
{
    if (!(sigma_min_ <= sigma_max_))
        throw std::invalid_argument("penalty.sigma_min must not exceed penalty.sigma_max");
    if (!(factor_ > 1.0))
        throw std::invalid_argument("penalty.factor must be greater than 1");
}

bool ResidualBalancingPenalty::is_check(int k) const
{
    if (!freeze_enabled_)
        return reference_check(k); // the fixed schedule of 2d8a755
    for (const std::pair<int, int> &entry : options_.schedule)
        if (k <= entry.first)
            return (k % entry.second) == 1;
    return (k % options_.schedule.back().second) == 1;
}

void ResidualBalancingPenalty::reset(double sigma0)
{
    // balance_freeze clamps sigma_0 (plan pseudocode); residual_balancing does not (2d8a755, DECISIONS 12)
    sigma_ = freeze_enabled_ ? std::min(std::max(sigma0, sigma_min_), sigma_max_) : sigma0;
    k_ = 0;
    prim_win_ = dual_win_ = 0;
    frozen_ = false;
    freeze_at_ = -1;
    freeze_reason_.clear();
    step_ = factor_;
    dirs_.clear();
    sigmas_.clear();
    log_.clear();
}

void ResidualBalancingPenalty::change(int k, int d, const KktResiduals &r, double feasratio, int prim_before, int dual_before)
{
    const double before = sigma_;
    if (freeze_enabled_ && options_.step_schedule == StepSchedule::HalveOnReversal && !dirs_.empty() && d == -dirs_.back())
        step_ = std::sqrt(step_);
    const double moved = d > 0 ? sigma_ * step_ : sigma_ / step_;
    sigma_ = std::min(std::max(moved, sigma_min_), sigma_max_);
    dirs_.push_back(d);
    sigmas_.push_back(sigma_);
    record(k, before, sigma_, "admm_schedule", r, feasratio, prim_before, dual_before);
    if (freeze_enabled_ && options_.step_schedule == StepSchedule::HalveOnReversal && std::log2(step_) < 1.0 / 16.0)
        freeze(k, "step", r, feasratio, prim_before, dual_before);
}

void ResidualBalancingPenalty::freeze(int k, const char *reason, const KktResiduals &r, double feasratio, int prim_before, int dual_before)
{
    if (frozen_)
        return;
    const double before = sigma_;
    const int W = options_.pingpong_window;
    if (static_cast<int>(dirs_.size()) >= W)
    {
        int net = 0;
        for (std::size_t i = dirs_.size() - W; i < dirs_.size(); i++)
            net += dirs_[i];
        if (std::abs(net) <= options_.pingpong_net)
        {
            // sigma bounces in a band: freeze at its centre (geometric mean of the last W values)
            if (options_.freeze_value == FreezeValue::BandCentre)
            {
                double log_sum = 0.0;
                for (std::size_t i = sigmas_.size() - W; i < sigmas_.size(); i++)
                    log_sum += std::log(sigmas_[i]);
                sigma_ = std::exp(log_sum / W);
            }
        }
    }
    sigma_ = std::min(std::max(freeze_bias_ * sigma_, sigma_min_), sigma_max_);
    frozen_ = true;
    freeze_at_ = k;
    freeze_reason_ = reason;
    // the freeze is logged even when the value does not move (plan: "log(k, reason, sigma)")
    log_.push_back({k, before, sigma_, std::string("freeze_") + reason, r.eta_p, r.eta_d, r.eta_g, feasratio, prim_before, dual_before});
}

double ResidualBalancingPenalty::update(int k, const KktResiduals &r, double sigma, bool sgs_step)
{
    k_ = k;
    sigma_ = sigma;
    if (frozen_)
        return sigma_;
    // the vote, in the reference form (feasratio < 1 rather than w * eta_p < eta_d: the division is what 2d8a755 rounds)
    const double feasratio = residual_weight_ * r.eta_p / r.eta_d;
    if (feasratio < 1)
        prim_win_ += 1;
    else
        dual_win_ += 1;
    const double sig_before = sigma_;
    const int prim_before = prim_win_, dual_before = dual_win_;
    if (sgs_step)
    {
        // the sGS-phase rule of the references (2d8a755 shares it with legacy_adaptive)
        if (sgs_check(k, options_.sgs_update_threshold, options_.sgs_update_stage_1, options_.sgs_update_stage_2))
        {
            if (prim_win_ > 1.2 * dual_win_)
            {
                prim_win_ = 0;
                sigma_ = std::min(sigma_max_, sigma_ * options_.sgs_factor);
            }
            else if (dual_win_ > 1.2 * prim_win_)
            {
                dual_win_ = 0;
                sigma_ = std::max(sigma_min_, sigma_ / options_.sgs_factor);
            }
        }
        record(k, sig_before, sigma_, "sgs_rule", r, feasratio, prim_before, dual_before);
        return sigma_;
    }
    if (is_check(k))
    {
        const double threshold_p = guard_ ? std::max(1.0, win_ratio_ * dual_win_) : win_ratio_ * dual_win_;
        const double threshold_d = guard_ ? std::max(1.0, win_ratio_ * prim_win_) : win_ratio_ * prim_win_;
        if (prim_win_ > threshold_p)
        {
            prim_win_ = 0;
            if (freeze_enabled_)
                change(k, +1, r, feasratio, prim_before, dual_before);
            else
            {
                sigma_ = std::min(sigma_max_, sigma_ * factor_); // 2d8a755 (the constant 2 of main as penalty.factor)
                record(k, sig_before, sigma_, "admm_schedule", r, feasratio, prim_before, dual_before);
            }
        }
        else if (dual_win_ > threshold_d)
        {
            dual_win_ = 0;
            if (freeze_enabled_)
                change(k, -1, r, feasratio, prim_before, dual_before);
            else
            {
                sigma_ = std::max(sigma_min_, sigma_ / factor_);
                record(k, sig_before, sigma_, "admm_schedule", r, feasratio, prim_before, dual_before);
            }
        }
    }
    if (freeze_enabled_ && !frozen_)
    {
        if (freeze_iteration_opt_ >= 0 && k >= freeze_iteration_opt_)
            freeze(k, "freeze_iteration", r, feasratio, prim_before, dual_before);
        else if (k >= options_.pingpong_min_iter && static_cast<int>(dirs_.size()) >= options_.pingpong_window)
        {
            int net = 0;
            for (std::size_t i = dirs_.size() - options_.pingpong_window; i < dirs_.size(); i++)
                net += dirs_[i];
            if (std::abs(net) <= options_.pingpong_net)
                freeze(k, "ping_pong", r, feasratio, prim_before, dual_before);
        }
    }
    return sigma_;
}

PolicyState ResidualBalancingPenalty::state() const
{
    PolicyState st;
    st.k = k_;
    st.sigma = sigma_;
    st.prim_win = prim_win_;
    st.dual_win = dual_win_;
    st.frozen = frozen_;
    st.freeze_iteration = freeze_at_;
    st.freeze_reason = freeze_reason_;
    st.step = step_;
    st.dirs = dirs_;
    st.sigmas = sigmas_;
    return st;
}

void ResidualBalancingPenalty::restore(const PolicyState &st)
{
    k_ = st.k;
    sigma_ = st.sigma;
    prim_win_ = st.prim_win;
    dual_win_ = st.dual_win;
    frozen_ = st.frozen;
    freeze_at_ = st.freeze_iteration;
    freeze_reason_ = st.freeze_reason;
    step_ = st.step;
    dirs_ = st.dirs;
    sigmas_ = st.sigmas;
}

std::string ResidualBalancingPenalty::parameters_json() const
{
    std::ostringstream o;
    o << "{\"factor\": " << json_num(factor_) << ", \"sigma_min\": " << json_num(sigma_min_) << ", \"sigma_max\": " << json_num(sigma_max_)
      << ", \"win_ratio\": " << json_num(win_ratio_) << ", \"guard\": " << (guard_ ? "true" : "false")
      << ", \"residual_weight\": " << json_num(residual_weight_) << ", \"monitor1\": false, \"sgs_rule\": \"legacy_sgs\""
      << ", \"sgs_factor\": " << json_num(options_.sgs_factor);
    if (freeze_enabled_)
    {
        o << ", \"step_schedule\": \"" << name_of(options_.step_schedule) << "\", \"freeze_iteration\": " << freeze_iteration_opt_
          << ", \"pingpong_window\": " << options_.pingpong_window << ", \"pingpong_net\": " << options_.pingpong_net
          << ", \"pingpong_min_iter\": " << options_.pingpong_min_iter << ", \"freeze_value\": \"" << name_of(options_.freeze_value)
          << "\", \"freeze_bias\": " << json_num(freeze_bias_) << ", \"schedule\": \"" << schedule_string(options_.schedule) << "\"";
    }
    o << "}";
    return o.str();
}

std::unique_ptr<PenaltyPolicy> make_penalty_policy(const PenaltyOptions &o)
{
    switch (o.policy)
    {
    case PenaltyPolicyKind::Fixed:
        return std::make_unique<FixedPenalty>();
    case PenaltyPolicyKind::LegacyAdaptive:
        return std::make_unique<LegacyPenalty>(o);
    case PenaltyPolicyKind::ResidualBalancing:
        return std::make_unique<ResidualBalancingPenalty>(o, false);
    case PenaltyPolicyKind::BalanceFreeze:
        return std::make_unique<ResidualBalancingPenalty>(o, true);
    }
    throw std::invalid_argument("unknown penalty policy");
}

} // namespace admm
} // namespace cuadmm
