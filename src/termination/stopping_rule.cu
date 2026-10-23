/*

    termination/stopping_rule.cu

    The Step 0 of solve() of main (src/solver.cu), split by rule; the validation snapshot is validate_current_iterate
    of main, unchanged.

*/

#include "termination/stopping_rule.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "cuadmm_legacy/solver.h"

namespace cuadmm
{
namespace termination
{

// InternalKktRule ---------------------------------------------------------------------------------------------------

std::optional<StopDecision> InternalKktRule::check(int done, const admm::KktResiduals &r, double elapsed_s, bool at_cap)
{
    if (!r.finite())
        throw std::runtime_error("non-finite KKT residuals at iteration " + std::to_string(done) + ": errRp = " + std::to_string(r.eta_p) +
                                 ", errRd = " + std::to_string(r.eta_d) + ", pobj = " + std::to_string(r.pobj) + ", dobj = " + std::to_string(r.dobj));
    StopDecision d;
    if (r.eta() < tol_)
    {
        d.reason = "internal_kkt";
        d.message = "Solver ended: converged.";
        d.converged = true;
        return d;
    }
    if (at_cap)
    {
        d.reason = "max_iter";
        d.message = "Solver ended: maximum iteration reached";
        return d;
    }
    if (elapsed_s > time_limit_)
    {
        d.reason = "time_limit";
        d.message = "Solver ended: time limit reached";
        d.time_limit = true;
        return d;
    }
    return std::nullopt;
}

// ValidatedRule -----------------------------------------------------------------------------------------------------

ValidatedRule::ValidatedRule(SDPSolver &engine, double tol, double time_limit_s)
    : engine_(engine), tol_(tol), time_limit_(time_limit_s), strict_(engine.validation.strict_dimacs_tol > 0),
      checkpoints_(engine.validation.checkpoint_iterations.begin(), engine.validation.checkpoint_iterations.end())
{
}

void ValidatedRule::on_start()
{
    SDPSolver &e = engine_;
    prev_internal_ok_ = false;
    if (!e.external_validator)
    {
        e.external_validator.reset(new ExternalValidator(
            e.vec_len, e.con_num,
            e.host_At_col_ptrs.data(), e.host_At_row_ids.data(), e.host_At_vals.data(),
            (int)e.host_b_idx.size(), e.host_b_idx.data(), e.host_b_vals.data(),
            (int)e.host_C_idx.size(), e.host_C_idx.data(), e.host_C_vals.data(),
            e.host_blk_types, e.host_blk_sizes));
        if (!e.dimacs_kinds.empty())
            e.external_validator->set_dimacs_kinds(e.dimacs_kinds);
        // the validator keeps its own copy of the problem data
        std::vector<int>().swap(e.host_At_col_ptrs);
        std::vector<int>().swap(e.host_At_row_ids);
        std::vector<double>().swap(e.host_At_vals);
        std::vector<int>().swap(e.host_b_idx);
        std::vector<double>().swap(e.host_b_vals);
        std::vector<int>().swap(e.host_C_idx);
        std::vector<double>().swap(e.host_C_vals);
    }
}

std::optional<StopDecision> ValidatedRule::check(int done, const admm::KktResiduals &r, double elapsed_s, bool at_cap)
{
    SDPSolver &e = engine_;
    const double internal_kkt = r.eta();
    StopDecision d;
    // the internal residual only triggers a validation; only a validated snapshot stops the solve
    const bool internal_ok = internal_kkt < tol_;
    const bool at_time = elapsed_s > time_limit_;
    std::string trigger;
    if (!r.finite())
    {
        // the iterate is not usable: stop and return the best externally evaluated one
        if (e.validation_history.empty())
            throw std::runtime_error("non-finite KKT residuals at iteration " + std::to_string(done) + " before any external validation");
        d.reason = "non_finite";
        d.message = "Solver ended: non-finite residuals (not validated)";
        return d;
    }
    else if (done > 0 || at_cap || at_time || internal_ok)
    {
        // the starting point (done == 0) only when the solve ends at once or it already satisfies the internal
        // tolerance (a warm start); an iteration that is also a requested checkpoint is flagged
        if (at_cap)
            trigger = "max_iter";
        else if (at_time)
            trigger = "time_limit";
        else if (internal_ok && !prev_internal_ok_)
            trigger = "internal_crossing";
        else if (done > 0 && checkpoints_.count(done))
            trigger = "checkpoint";
        else if (done > 0 && done % e.validation.interval == 0)
            trigger = "interval";
    }
    prev_internal_ok_ = internal_ok;
    bool stop = false;
    if (!trigger.empty())
    {
        const ValidationRecord rec = validate_current_iterate(done, elapsed_s, trigger, done > 0 && checkpoints_.count(done) > 0);
        if (rec.result.validated)
        {
            const double strict_tol = e.validation.strict_dimacs_tol;
            if (!e.practical_validated)
            {
                // the practical criteria: the first time (recorded; the solve stops here unless a strict stage is set)
                e.practical_validated = true;
                e.first_practical_iteration = done;
                e.first_practical_time_s = elapsed_s; // (the solve clock, as solve() of main after the validation)
                e.converged = true;
                e.externally_validated_converged = true;
                e.first_validated_iteration = done;
                if (strict_tol > 0)
                    printf(" practical criteria validated at iteration %d (%.3f s, max_abs_DIMACS %.2e): continuing toward %.1e\n", done,
                           e.first_practical_time_s, rec.result.max_abs_dimacs, strict_tol);
            }
            const bool strict_ok = strict_tol > 0 && std::isfinite(rec.result.max_abs_dimacs) && rec.result.max_abs_dimacs <= strict_tol;
            if (strict_tol <= 0 || strict_ok)
            {
                stop = true;
                d.message = strict_ok ? "Solver ended: strict DIMACS target validated" : "Solver ended: externally validated";
                d.reason = strict_ok ? "strict_validated" : "externally_validated";
                d.converged = true;
                if (strict_ok)
                {
                    e.strict_validated = true;
                    e.first_strict_iteration = done;
                    e.first_strict_time_s = elapsed_s;
                }
            }
        }
        else if (trigger == "internal_crossing")
            printf(" internal KKT %.2e < %.1e at iteration %d, but the external validation failed (%s): continuing\n", internal_kkt, tol_, done,
                   rec.result.failed.c_str());
    }
    if (!stop && at_cap)
    {
        stop = true;
        d.reason = "max_iter";
        d.message = "Solver ended: maximum iteration reached (safety cap, not validated)";
    }
    if (!stop && at_time)
    {
        stop = true;
        d.reason = "time_limit";
        d.message = "Solver ended: time limit reached (not validated)";
        d.time_limit = true;
    }
    if (stop)
        return d;
    return std::nullopt;
}

void ValidatedRule::finish(SDPSolver &e, bool nonfinite)
{
    /* Validation-aware stopping without a validated snapshot: return the best externally evaluated iterate if it is
       strictly better than the last one (the last snapshot is the final iterate), or whenever the final iterate is
       non-finite */
    if (!e.strict_validated && !(e.externally_validated_converged && e.validation.strict_dimacs_tol <= 0) && !e.validation_history.empty())
    {
        const ValidationRecord &last = e.validation_history.back();
        const ValidationRecord &best = e.best_validation;
        if (nonfinite || (best.iter != last.iter && best.result.better_than(last.result)))
        {
            e.returned_best_external = true;
            if (nonfinite)
                printf(" Returning the best externally evaluated iterate: iteration %d, max |DIMACS| %2.1e, merit %2.1e (the final iterate is non-finite)\n",
                       best.iter, best.result.max_abs_dimacs, best.result.merit());
            else
                printf(" Returning the best externally evaluated iterate: iteration %d, max |DIMACS| %2.1e, merit %2.1e (final iterate: %2.1e, %2.1e)\n",
                       best.iter, best.result.max_abs_dimacs, best.result.merit(), last.result.max_abs_dimacs, last.result.merit());
            CHECK_CUDA(cudaMemcpyAsync(e.X.vals, e.X_best.vals, sizeof(double) * e.vec_len, D2D, (cudaStream_t)0));
            CHECK_CUDA(cudaMemcpyAsync(e.y.vals, e.y_best.vals, sizeof(double) * e.con_num, D2D, (cudaStream_t)0));
            CHECK_CUDA(cudaMemcpyAsync(e.S.vals, e.S_best.vals, sizeof(double) * e.vec_len, D2D, (cudaStream_t)0));
            // the solver's own residuals and objectives of that snapshot
            e.errRp = best.internal_errRp;
            e.errRd = best.internal_errRd;
            e.relgap = best.internal_relgap;
            e.pobj = best.internal_pobj;
            e.dobj = best.internal_dobj;
            e.maxfeas = std::max(e.errRp, e.errRd);
        }
    }
}

ValidationRecord ValidatedRule::validate_current_iterate(int iter, double time_s, const std::string &trigger, bool checkpoint)
{
    SDPSolver &e = engine_;
    const auto t0 = std::chrono::steady_clock::now();
    ValidationRecord rec;
    rec.iter = iter;
    rec.time_s = time_s;
    rec.sigma = e.sig;
    rec.internal_kkt = std::max(e.maxfeas, e.relgap);
    rec.internal_errRp = e.errRp;
    rec.internal_errRd = e.errRd;
    rec.internal_relgap = e.relgap;
    rec.internal_pobj = e.pobj;
    rec.internal_dobj = e.dobj;
    rec.trigger = trigger;
    rec.checkpoint = checkpoint;

    // snapshot of the scaled iterate, unscaled exactly like at the end of solve()
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<double> X(e.vec_len), y(e.con_num), S(e.vec_len), nA(e.con_num);
    CHECK_CUDA(cudaMemcpy(X.data(), e.X.vals, sizeof(double) * e.vec_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(y.data(), e.y.vals, sizeof(double) * e.con_num, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(S.data(), e.S.vals, sizeof(double) * e.vec_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(nA.data(), e.normA.vals, sizeof(double) * e.con_num, cudaMemcpyDeviceToHost));
    for (int i = 0; i < e.vec_len; i++)
    {
        X[i] = X[i] * e.bscale;
        S[i] = S[i] * e.Cscale;
    }
    for (int j = 0; j < e.con_num; j++)
        y[j] = y[j] / nA[j] * e.Cscale;

    rec.result = e.external_validator->evaluate(X.data(), y.data(), S.data(), e.validation.tol, e.validation.threads);

    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
        e.peak_gpu_mem_bytes = std::max(e.peak_gpu_mem_bytes, total_mem - free_mem);

    // best externally evaluated iterate (scaled copies on the device, restored at termination if needed)
    if (e.validation_history.empty() || rec.result.better_than(e.best_validation.result))
    {
        CHECK_CUDA(cudaMemcpy(e.X_best.vals, e.X.vals, sizeof(double) * e.vec_len, D2D));
        CHECK_CUDA(cudaMemcpy(e.y_best.vals, e.y.vals, sizeof(double) * e.con_num, D2D));
        CHECK_CUDA(cudaMemcpy(e.S_best.vals, e.S.vals, sizeof(double) * e.vec_len, D2D));
        e.best_validation = rec; // validation_s is set below
    }
    rec.validation_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (e.best_validation.iter == rec.iter)
        e.best_validation.validation_s = rec.validation_s;
    e.validation_time_s += rec.validation_s;
    e.validation_history.push_back(rec);

    const ValidationResult &r = rec.result;
    printf(" [validation] it %7d %-17s | ext KKT %.2e (p %.1e d %.1e g %.1e) | X cone %.1e | dual cone %.1e | sigma %.3e | %s (%.0f ms)\n",
           iter, (trigger + (checkpoint && trigger != "checkpoint" ? "+checkpoint" : "")).c_str(), r.kkt_full, r.primal_res,
           r.dual_res, r.relgap, r.X_cone_violation, r.dual_cone_violation, rec.sigma,
           r.validated ? "VALIDATED" : ("failed: " + r.failed).c_str(), 1e3 * rec.validation_s);
    if (e.validation.on_validation)
    {
        // timed separately; a failure (e.g. a checkpoint that cannot be written) is reported, not fatal
        const auto t1 = std::chrono::steady_clock::now();
        try
        {
            e.validation.on_validation(rec, X, y, S);
        }
        catch (const std::exception &ex)
        {
            e.callback_failures++;
            printf(" WARNING: validation callback failed at iteration %d: %s\n", iter, ex.what());
        }
        rec.callback_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        e.callback_time_s += rec.callback_s;
        e.validation_history.back().callback_s = rec.callback_s;
    }
    return rec;
}

} // namespace termination
} // namespace cuadmm
