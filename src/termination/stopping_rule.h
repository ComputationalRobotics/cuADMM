/*

    termination/stopping_rule.h

    StoppingRule (plan Section 7.4): InternalKktRule (the internal residual, the iteration and time limits, non-finite
    residuals) and ValidatedRule (today's validation-aware stopping with the ExternalValidator on the original data,
    and its strict DIMACS stage). The rules decide; the solver executes the stop.

    Both keep the records of today in the engine (validation_history, best_validation, first_* fields) so that the
    copied tests and the facade read them unchanged (Phase 5+ moves the records into SolveResult).

*/

#ifndef CUADMM_TERMINATION_STOPPING_RULE_H
#define CUADMM_TERMINATION_STOPPING_RULE_H

#include <optional>
#include <set>
#include <string>
#include <string_view>

#include "admm/penalty_policy.h" // KktResiduals

class SDPSolver;
struct ValidationRecord;

namespace cuadmm
{
namespace termination
{

struct StopDecision
{
    std::string reason;  // "internal_kkt", "max_iter", "time_limit", "non_finite", "externally_validated", "strict_validated"
    std::string message; // the final line of the console report
    bool converged = false;
    bool time_limit = false;
};

class StoppingRule
{
public:
    virtual ~StoppingRule() = default;
    virtual std::string_view name() const = 0;
    virtual void on_start() {}
    // done: completed iterations; r: the residuals of the current iterate; elapsed_s: seconds since the start of the
    // iterations; at_cap: the iteration limit is reached
    virtual std::optional<StopDecision> check(int done, const admm::KktResiduals &r, double elapsed_s, bool at_cap) = 0;
    // after the stop: restore the iterate the rule wants returned (the validated / best externally evaluated one)
    virtual void finish(SDPSolver &engine, bool nonfinite) { (void)engine; (void)nonfinite; }
    virtual bool validating() const { return false; }
};

/// @brief Stop on the internal residual eta < tol, at the iteration cap or the time limit; non-finite residuals throw.
class InternalKktRule final : public StoppingRule
{
public:
    InternalKktRule(double tol, double time_limit_s) : tol_(tol), time_limit_(time_limit_s) {}
    std::string_view name() const override { return "internal_kkt"; }
    std::optional<StopDecision> check(int done, const admm::KktResiduals &r, double elapsed_s, bool at_cap) override;

private:
    double tol_, time_limit_;
};

/// @brief Today's validation-aware stopping (ValidationConfig of the engine): snapshots every `interval` iterations,
/// at downward crossings of tol by the internal residual, at the checkpoints and at termination; stops only on a
/// validated snapshot (or, with strict_dimacs_tol > 0, on a validated snapshot with max_abs_dimacs below it); returns
/// the best externally evaluated iterate otherwise.
class ValidatedRule final : public StoppingRule
{
public:
    ValidatedRule(SDPSolver &engine, double tol, double time_limit_s);
    std::string_view name() const override { return strict_ ? "strict_dimacs" : "validated"; }
    void on_start() override;
    std::optional<StopDecision> check(int done, const admm::KktResiduals &r, double elapsed_s, bool at_cap) override;
    void finish(SDPSolver &engine, bool nonfinite) override;
    bool validating() const override { return true; }

    // Snapshot of the current iterate (unscaled on the host), external validation, best-iterate bookkeeping and the
    // user callback; the record goes into the engine's validation_history (validate_current_iterate of main).
    ValidationRecord validate_current_iterate(int iter, double time_s, const std::string &trigger, bool checkpoint);

private:
    SDPSolver &engine_;
    double tol_, time_limit_;
    bool strict_;
    std::set<int> checkpoints_;
    bool prev_internal_ok_ = false;
};

} // namespace termination
} // namespace cuadmm

#endif // CUADMM_TERMINATION_STOPPING_RULE_H
