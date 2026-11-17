/*

    admm/admm_solver.cu

    The ADMM loop on the components of the algorithm layer: the steps of AdmmState, the penalty policy, the stopping
    rule, the best-iterate tracker and the observers.

*/

#include "admm/admm_solver.h"

#include "termination/certificate.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "cuadmm_legacy/solver.h"

namespace cuadmm
{
namespace admm
{

AdmmSolver::AdmmSolver(AdmmState &state, PenaltyPolicy &penalty, termination::StoppingRule &stopping, termination::BestIterateTracker &tracker,
                       report::ObserverSet &observers, const RunConfig &config)
    : state_(state), penalty_(penalty), stopping_(stopping), tracker_(tracker), observers_(observers), config_(config)
{
}

RunOutcome AdmmSolver::run(double sigma0, bool rescale)
{
    SDPSolver &e = state_.engine();
    const RunConfig &c = config_;
    const bool validating = stopping_.validating();
    RunOutcome out;

    // per-solve state of the engine (bookkeeping fields read by the facade and the copied tests)
    e.stop_reason.clear();
    e.externally_validated_converged = false;
    e.internal_converged_at_return = false;
    e.first_validated_iteration = -1;
    e.first_internal_below_1e_3 = -1;
    e.first_internal_below_tol = -1;
    e.validation_history.clear();
    e.best_validation = ValidationRecord();
    e.returned_best_external = false;
    e.validation_time_s = 0.0;
    e.y_solves = 0;
    e.sigma_log.clear();
    e.practical_validated = e.strict_validated = false;
    e.first_practical_iteration = e.first_strict_iteration = -1;
    e.first_practical_time_s = e.first_strict_time_s = -1.0;
    e.callback_time_s = 0.0;
    e.callback_failures = 0;
    e.peak_gpu_mem_bytes = 0;
    e.sigma0 = sigma0;
    e.sig = sigma0;
    e.sigma_changes = 0;
    e.iterations = 0;
    e.lobpcg_calls = e.lobpcg_fallbacks = e.composite_fallbacks = e.jacobi_not_converged = 0;
    e.info_iter_num = 0;
    e.converged = false;
    e.returned_best_iterate = false;
    e.time_limit_reached = false;
    e.stop_tol = c.tol;
    state_.reset_counters();
    tracker_.reset();
    auto sample_gpu_mem = [&e]()
    {
        size_t free_mem = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
            e.peak_gpu_mem_bytes = std::max(e.peak_gpu_mem_bytes, total_mem - free_mem);
    };
    sample_gpu_mem();
    stopping_.on_start();

    if (c.verbose)
    {
        std::cout << std::endl;
        std::cout << " Algorithm: plain ADMM" << std::endl;
        if (validating && e.validation.strict_dimacs_tol > 0)
            std::cout << " Strict stage: after the practical criteria, continue until max_abs_DIMACS <= " << e.validation.strict_dimacs_tol
                      << " or a safety limit" << std::endl;
        std::cout << " Sigma policy: " << penalty_.name() << ", initial sigma " << sigma0 << " " << penalty_.parameters_json() << std::endl;
        std::cout << " Maximum iterations: " << c.max_iter << ", internal KKT tolerance: " << c.tol << ", tau " << c.tau << std::endl;
        if (validating)
        {
            const ValidationTolerances &t = e.validation.tol;
            std::cout << " External validation: every " << e.validation.interval
                      << " iterations and at each crossing of the internal tolerance; stop only when eta_p <= " << t.primal << ", eta_d <= " << t.dual
                      << ", eta_g <= " << t.gap << ", X cone <= " << t.cone << ", dual cone <= " << t.dual_cone << " (recomputed on the CPU, all finite)"
                      << std::endl;
        }
        else
            std::cout << " External validation: off (stopping on the internal KKT residual)" << std::endl;
        if (std::isfinite(e.time_limit))
            std::cout << " Time limit: " << e.time_limit << " s" << std::endl;
        std::cout << "                   use LOBPCG: " << (e.use_lobpcg ? "true" : "false") << std::endl;
        std::cout << "    initial projection method: " << get_projection_method_name(e.initial_proj_method, false) << std::endl;
        std::cout << "      final projection method: " << get_projection_method_name(e.final_proj_method, false) << std::endl;
        std::cout << std::endl;
        std::cout << " ------------------------------------------------------------------------------" << std::endl;
        std::cout << "                                   cuADMM" << std::endl;
        std::cout << " ------------------------------------------------------------------------------" << std::endl;
    }

    // the starting point: scaled from original units for a warm start or a second solve (the rescale path of main)
    if (rescale)
        state_.scale_iterate();
    KktResiduals r = state_.evaluate_kkt_initial();
    if (rescale)
        CHECK_CUDA(cudaEventRecord(e.start)); // the timer of the console table restarts at a second solve, as today
    penalty_.reset(sigma0);
    double sigma = sigma0;
    observers_.on_start(r, sigma);
    if (c.verbose)
        std::cout << " --------------- Starting with projection method " << get_projection_method_name(e.current_proj_method) << "------" << std::endl;

    const auto solve_start = std::chrono::steady_clock::now();
    auto seconds_since_start = [&solve_start]()
    { return std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count(); };
    bool nonfinite = false;

    for (int k = 1; k <= c.max_iter + 1; k++)
    {
        /* Step 0: termination check and logging (the stopping rule decides) */
        const int done = k - 1;
        const double internal_kkt = r.eta();
        if (done > 0 && internal_kkt < 1e-3 && e.first_internal_below_1e_3 < 0)
            e.first_internal_below_1e_3 = done;
        if (done > 0 && internal_kkt < c.tol && e.first_internal_below_tol < 0)
            e.first_internal_below_tol = done;
        const double elapsed = seconds_since_start();
        const std::optional<termination::StopDecision> stop = stopping_.check(done, r, elapsed, k > c.max_iter);
        {
            // the console table (the time column: the engine's event clock, as today)
            float milliseconds = 0.0f;
            cudaEventRecord(e.stop);
            cudaEventSynchronize(e.stop);
            cudaEventElapsedTime(&milliseconds, e.start, e.stop);
            observers_.on_check(done, r, sigma, milliseconds / 1000, stop.has_value());
        }
        if (stop)
        {
            e.stop_reason = stop->reason;
            e.converged = stop->converged;
            e.time_limit_reached = stop->time_limit;
            out.stop_reason = stop->reason;
            out.converged = stop->converged;
            out.time_limit_reached = stop->time_limit;
            out.nonfinite = nonfinite;
            finish(*stop, validating ? seconds_since_start() : elapsed, nonfinite, done);
            break;
        }

        // the projection-method switch (warm-up end / low-rank start), on the residuals of the current iterate
        state_.begin_iteration(k, internal_kkt, c.verbose);

        /* Steps 1-4: y-step, cone step (with S - C), multiplier step */
        const double tau = c.tau;
        state_.y_step(sigma);
        state_.cone_step(sigma, k);
        state_.update_smc();
        state_.multiplier_step(sigma, tau);
        e.iterations = k;
        e.y_solves = state_.y_solves();

        /* Step 5: residuals of the new iterate, best iterate, sigma */
        r = state_.evaluate_kkt();
        if (!r.finite())
        {
            if (!validating)
                throw std::runtime_error("non-finite KKT residuals at iteration " + std::to_string(k) + ": errRp = " + std::to_string(r.eta_p) +
                                         ", errRd = " + std::to_string(r.eta_d) + ", pobj = " + std::to_string(r.pobj) + ", dobj = " + std::to_string(r.dobj));
            nonfinite = true; // Step 0 stops and returns the best externally evaluated iterate
        }
        // (internal stopping only) the best iterate so far
        if (!validating)
            tracker_.observe(k, r);

        const double sigma_next = penalty_.update(k, r, sigma);
        if (sigma_next != sigma)
            e.sigma_changes++;
        e.sig = sigma_next;
        e.tau = tau;

        report::IterationEvent event;
        event.k = k;
        event.r = r;
        event.sigma_used = sigma;
        event.sigma_next = sigma_next;
        event.tau = tau;
        event.time_s = seconds_since_start();
        observers_.on_iteration(event);
        sigma = sigma_next;
    }

    // the sigma log of the engine (the copied tests and the facade read it)
    for (const SigmaChange &u : penalty_.log())
        e.sigma_log.push_back({u.iteration, u.old_sigma, u.new_sigma, u.rule, u.eta_p, u.eta_d, u.eta_g, u.feasratio, u.prim_win, u.dual_win});

    // recover the original solution by unscaling
    state_.unscale_iterate();
    out.iterations = e.iterations;
    out.solve_time_s = e.solve_time;
    out.sigma_final = e.sig;
    out.returned_best_iterate = e.returned_best_iterate;
    return out;
}

void AdmmSolver::finish(const termination::StopDecision &stop, double solve_time_s, bool nonfinite, int iterations)
{
    SDPSolver &e = state_.engine();
    const RunConfig &c = config_;
    const bool validating = stopping_.validating();
    e.solve_time = solve_time_s;
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
        e.peak_gpu_mem_bytes = std::max(e.peak_gpu_mem_bytes, total_mem - free_mem);

    // validation mode: the best externally evaluated iterate when nothing was validated
    stopping_.finish(e, nonfinite);
    // the best internal iterate if it is strictly better than the returned one
    e.returned_best_iterate = tracker_.restore_if_better();

    /* Cone violations of the returned X and S over all blocks, and certificate of C - A^T y (Phase 5: through the
       product cone's measures) */
    cone::ConeProduct &cones = state_.cones();
    core::ExecContext &ctx = state_.context();
    const double min_X = cones.measure(core::DeviceSpan<const double>(e.X.vals, e.vec_len), cone::Side::Primal, ctx).min_block_min_psd_nonneg();
    const double min_S = cones.measure(core::DeviceSpan<const double>(e.S.vals, e.vec_len), cone::Side::Dual, ctx).min_block_min_psd_nonneg();
    // scale them back and make them relative to 1 + ||b|| and 1 + ||C||
    e.errPSD_X = -min_X * e.bscale / e.norm_borg;
    e.errPSD_S = -min_S * e.Cscale / e.norm_Corg;
    e.internal_converged_at_return = std::max(e.maxfeas, e.relgap) < c.tol;

    // Z = C - A^T y for the returned y (scaled units: C - A^T y = Cscale * (C_s - A_s^T y_s))
    SpMV_cusparse(e.cusparseH, e.At_csr, e.y, e.Aty, 1.0, 0.0, e.SpMV_Aty_buffer);
    CHECK_CUDA(cudaMemcpy(e.Rd1.vals, e.Aty.vals, sizeof(double) * e.vec_len, D2D));
    dense_vector_negate(e.Rd1);
    axpby_cusparse(e.cusparseH, e.C, e.Rd1, 1.0, 1.0);
    const termination::CertificateResult cert = termination::Certificate::measure(cones, core::DeviceSpan<const double>(e.Rd1.vals, e.vec_len), e.Cscale, ctx);
    e.dual_block_min = cert.dual_block_min;
    e.dual_cone_coef = cert.dual_cone_coef;
    const double dual_cone_coef_sum = cert.coef_sum;

    float milliseconds = 0.0f;
    cudaEventRecord(e.stop);
    cudaEventSynchronize(e.stop);
    cudaEventElapsedTime(&milliseconds, e.start, e.stop);
    e.total_time = milliseconds / 1000;
    if (!c.verbose)
        return;
    printf(" ------------------------------------------------------------------------------\n\n");
    std::cout << stop.message << std::endl;
    printf("\n primal infeasibility = %2.1e \n dual   infeasibility = %2.1e \n relative gap         = %2.1e", e.errRp, e.errRd, e.relgap);
    printf("\n PSD violation X      = %2.1e \n PSD violation S      = %2.1e", e.errPSD_X, e.errPSD_S);
    printf("\n certificate: <b,y> + R * %.6e is a lower bound on the optimal value if every block of an"
           "\n              optimal X has trace (or entries) bounded by R, see certified_lower_bound()",
           dual_cone_coef_sum);
    if (e.use_lobpcg)
        printf("\n\n LOBPCG calls = %d, certificate fallbacks to cuSOLVER = %d", e.lobpcg_calls, e.lobpcg_fallbacks);
    if (e.jacobi_not_converged > 0)
        printf("\n batched Jacobi EVDs below their tolerance (warning only) = %d", e.jacobi_not_converged);
    if (e.composite_fallbacks > 0)
        printf("\n non-finite or norm-test composite projections replaced by cuSOLVER = %d", e.composite_fallbacks);
    printf("\n\n primal objective = %- 9.8e \n dual   objective = %- 9.8e", e.pobj, e.dobj);
    printf("\n\n algorithm: plain ADMM, iterations = %d\n sigma policy: %s, initial sigma %.6e, final sigma %.6e, %d change(s)", iterations,
           std::string(penalty_.name()).c_str(), e.sigma0, e.sig, e.sigma_changes);
    if (penalty_.frozen())
        printf(" (frozen at iteration %d: %s)", penalty_.freeze_iteration(), penalty_.freeze_reason().c_str());
    printf("\n stop reason: %s", e.stop_reason.c_str());
    if (validating)
    {
        printf("\n external validations: %zu (%.2f s); first validated iteration: %d; returned the %s", e.validation_history.size(),
               e.validation_time_s, e.first_validated_iteration,
               e.externally_validated_converged ? "validated iterate" : (e.returned_best_external ? "best externally evaluated iterate" : "final iterate"));
        const ValidationRecord &b = e.best_validation;
        if (!e.validation_history.empty())
            printf("\n best external iterate: iteration %d, KKT %.2e, X cone %.2e, dual cone %.2e, merit %.2e", b.iter, b.result.kkt_full,
                   b.result.X_cone_violation, b.result.dual_cone_violation, b.result.merit());
        if (e.callback_failures > 0)
            printf("\n WARNING: %d validation callback(s) failed (see above)", e.callback_failures);
    }
    printf("\n\n time per iteration = %2.4fs \n total time         = %2.1fs (init %2.1fs, iterations %2.1fs)", e.solve_time / std::max(1, iterations),
           e.total_time, e.init_time, e.solve_time);
    printf("\n -------------------------------------------------------------------------------\n\n");
}

} // namespace admm
} // namespace cuadmm
