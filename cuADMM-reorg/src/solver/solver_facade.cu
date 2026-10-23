/*

    solver_facade.cu

    The Solver facade of Phase 2 over the ported SDPSolver (src/legacy/). It maps SolverOptions and ProblemData onto
    init()/solve() of the engine exactly as today's CLI did, so that the iterates are bit for bit those of the
    references for equivalent options, and collects a SolveResult. Phase 4 replaces the engine's loop.

    Option mapping (today's CLI, src/legacy/main.cu of main):
      algorithm.scheme admm / sgs_admm        -> switch_admm 0 / INT_MAX; scheme_switch_iteration n -> switch_admm n
      algorithm.tau                           -> the step length of both schemes (the references used 1.618 / 1.95)
      penalty.*                               -> sigma_policy, sig, penalty_factor, penalty_sigma_min/max, the sGS rule
      psd.large.warmup none                   -> initial = final = EIG_FP64; the low-rank analysis (LOBPCG phase)
                                                 starts at psd.lowrank.start_iter (switch_proj_max_iter = start - 1,
                                                 switch_proj_tol = 0): today's CLI default (0, 1e-2) starts it at 2
      psd.large.warmup composite_fp16         -> initial COMPOSITE_FP16, final EIG_FP64, warm-up end at
                                                 psd.large.warmup.switch_iter / switch_tol (today's C++ API)
      psd.lowrank.enabled, psd.medium_streams, psd.medium_batch_min -> use_lobpcg, eig_stream_num_per_gpu, medium_batch_min_count
      validation.*                            -> ValidationConfig (checkpoints and the final snapshot written here)

*/

#include "cuadmm/solver.h"

#include "cone/cone_product.h"
#include "termination/certificate.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "admm/admm_solver.h"
#include "cuadmm_legacy/solver.h"
#include "report/observers.h"
#include "report/writers.h"
#include "termination/best_iterate_tracker.h"
#include "termination/stopping_rule.h"

namespace cuadmm
{

struct Solver::Impl
{
    SolverOptions options;
    std::unique_ptr<SDPSolver> engine;
    bool is_setup = false;
    bool first_solve = true;
    // problem data as the engine takes it (32-bit), kept for the lifetime of the solver
    int vec_len = 0, con_num = 0;
    std::vector<int> at_colptr, at_rowind, b_index, c_index, blk_sizes;
    std::vector<double> at_val, b_val, c_val;
    std::vector<char> blk_types;
    Sense sense = Sense::Minimize;
    double objective_offset = 0.0;
    std::string source_format;
    std::size_t num_blocks = 0;
    bool has_free_block = false;
    // engine call parameters (fixed at setup)
    int switch_admm = 0;
    int switch_proj_max_iter = 0;
    double switch_proj_tol = 0.0;
    std::vector<std::shared_ptr<report::IterationObserver>> observers;
    std::unique_ptr<admm::PenaltyPolicy> penalty; // built at setup from the options; reset at every solve
    std::unique_ptr<cone::ConeProduct> cones;      // the product cone of the problem (Phase 5), on the engine's tiers
};

namespace
{

int checked_int(std::int64_t v, const char *what)
{
    if (v < 0 || v > std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string(what) + " = " + std::to_string(v) +
                                    " does not fit the 32-bit indexing of the current engine (plan: 64-bit sizes are a later phase)");
    return static_cast<int>(v);
}

// the engine's own policy field (read by its summary helpers); balance_freeze is a ResidualBalancingPenalty with the
// freeze options and has no engine value
SigmaPolicy to_engine_policy(PenaltyPolicyKind k)
{
    switch (k)
    {
    case PenaltyPolicyKind::Fixed:
        return SigmaPolicy::FIXED;
    case PenaltyPolicyKind::LegacyAdaptive:
        return SigmaPolicy::LEGACY_ADAPTIVE;
    default:
        return SigmaPolicy::RESIDUAL_BALANCING;
    }
}

std::vector<double> device_to_host(const DeviceDenseVector<double> &v)
{
    std::vector<double> h(v.size);
    if (v.size > 0)
        CHECK_CUDA(cudaMemcpy(h.data(), v.vals, sizeof(double) * v.size, cudaMemcpyDeviceToHost));
    return h;
}

std::string validation_csv(const SDPSolver &s)
{
    std::ostringstream f;
    f << std::setprecision(17)
      << "iter,time_s,sigma,trigger,checkpoint,internal_kkt,internal_errRp,internal_errRd,internal_relgap,primal_res,dual_res,"
         "relgap,kkt_full,pobj,dobj,X_cone_violation,X_cone_violation_normb,dual_cone_violation,min_eig_X,min_eig_Z,"
         "X_negative_blocks,finite,validated,failed,validation_s,callback_s,S_cone_violation,min_eig_S,"
         "dimacs1,dimacs2,dimacs3,dimacs4,dimacs5,dimacs6,max_abs_dimacs,err6_defined,err4_z,err6_z\n";
    for (const ValidationRecord &v : s.validation_history)
    {
        const ValidationResult &r = v.result;
        f << v.iter << "," << v.time_s << "," << v.sigma << "," << v.trigger << "," << (v.checkpoint ? 1 : 0) << ","
          << v.internal_kkt << "," << v.internal_errRp << "," << v.internal_errRd << "," << v.internal_relgap << "," << r.primal_res << ","
          << r.dual_res << "," << r.relgap << "," << r.kkt_full << "," << r.pobj << "," << r.dobj << "," << r.X_cone_violation << ","
          << r.X_cone_violation_normb << "," << r.dual_cone_violation << "," << r.min_eig_X << "," << r.min_eig_Z << ","
          << r.X_negative_blocks << "," << (r.finite ? 1 : 0) << "," << (r.validated ? 1 : 0) << ",\"" << r.failed << "\","
          << v.validation_s << "," << v.callback_s << "," << r.S_cone_violation << "," << r.min_eig_S << ","
          << r.dimacs[0] << "," << r.dimacs[1] << "," << r.dimacs[2] << "," << r.dimacs[3] << "," << r.dimacs[4] << ","
          << r.dimacs[5] << "," << r.max_abs_dimacs << "," << (r.err6_defined ? 1 : 0) << "," << r.err4_z << "," << r.err6_z << "\n";
    }
    return f.str();
}

} // namespace

Solver::Solver(SolverOptions options) : impl_(new Impl)
{
    options.validate();
    impl_->options = std::move(options);
}

Solver::~Solver() = default;

const SolverOptions &Solver::options() const { return impl_->options; }

void Solver::add_observer(std::shared_ptr<report::IterationObserver> observer) { impl_->observers.push_back(std::move(observer)); }

std::string Solver::linsys_backend() const
{
    if (!impl_->is_setup)
        return "";
    return impl_->engine->AAt_dense_gpu ? "cholmod_dense_gpu" : "cholmod";
}

void Solver::setup(const ProblemData &problem)
{
    Impl &im = *impl_;
    const SolverOptions &o = im.options;
    problem.validate();
    im.penalty = admm::make_penalty_policy(o.penalty); // validates the policy parameters

    // 32-bit arrays of the engine
    im.vec_len = checked_int(problem.vec_len(), "vec_len");
    im.con_num = checked_int(problem.num_constraints(), "the number of constraints");
    checked_int(problem.At.nnz(), "nnz(At)");
    im.at_colptr.assign(problem.At.colptr.begin(), problem.At.colptr.end());
    im.at_rowind.assign(problem.At.rowind.begin(), problem.At.rowind.end());
    im.at_val = problem.At.val;
    im.b_index.assign(problem.b.index.begin(), problem.b.index.end());
    im.b_val = problem.b.val;
    im.c_index.assign(problem.C.index.begin(), problem.C.index.end());
    im.c_val = problem.C.val;
    im.blk_types.clear();
    im.blk_sizes.clear();
    im.has_free_block = false;
    for (const ConeSpec &c : problem.cones)
    {
        im.blk_types.push_back(cone_code(c.kind)); // throws for the cones of Phase 9
        im.blk_sizes.push_back(checked_int(c.size, "a cone size"));
        im.has_free_block = im.has_free_block || c.kind == ConeKind::Free;
    }
    im.num_blocks = problem.cones.size();
    im.sense = problem.sense;
    im.objective_offset = problem.objective_offset;
    im.source_format = problem.source.format;
    if (!o.certificate.block_bounds.empty() && o.certificate.block_bounds.size() != im.num_blocks)
        throw std::invalid_argument("certificate.block_bounds: " + std::to_string(o.certificate.block_bounds.size()) + " bounds for " +
                                    std::to_string(im.num_blocks) + " blocks");

    // scheme and projection schedule
    im.switch_admm = o.algorithm.scheme == SchemeKind::SgsAdmm ? std::numeric_limits<int>::max() : o.algorithm.scheme_switch_iteration;
    ProjectionMethod initial = ProjectionMethod::EIG_FP64;
    if (o.psd.warmup == WarmupKind::CompositeFp16)
    {
        initial = ProjectionMethod::COMPOSITE_FP16;
        im.switch_proj_max_iter = o.psd.warmup_switch_iter;
        im.switch_proj_tol = o.psd.warmup_switch_tol;
    }
    else
    {
        // no warm-up: the switch of the engine only starts the low-rank analysis (plan Q2: iteration 2 by default)
        im.switch_proj_max_iter = o.psd.lowrank_start_iter - 1;
        im.switch_proj_tol = 0.0;
    }

    CHECK_CUDA(cudaSetDevice(o.gpu_device));
    im.engine.reset(new SDPSolver());
    SDPSolver &e = *im.engine;
    e.medium_batch_min_count = o.psd.medium_batch_min;
    e.init(o.psd.medium_streams, im.vec_len, im.con_num,
           im.at_colptr.data(), im.at_rowind.data(), im.at_val.data(), static_cast<int>(im.at_val.size()),
           im.b_index.data(), im.b_val.data(), static_cast<int>(im.b_val.size()),
           im.c_index.data(), im.c_val.data(), static_cast<int>(im.c_val.size()),
           im.blk_types.data(), im.blk_sizes.data(), static_cast<int>(im.num_blocks),
           initial, ProjectionMethod::EIG_FP64,
           nullptr, nullptr, nullptr, // the warm start is handed over by solve()
           o.penalty.sigma0, o.psd.lowrank_enabled);
    im.cones.reset(new cone::ConeProduct(im.blk_types, im.blk_sizes, cone::ConeBuildContext{im.vec_len, &e}));
    e.sigma_policy = to_engine_policy(o.penalty.policy);
    e.penalty_factor = o.penalty.factor;
    e.penalty_sigma_min = o.penalty.sigma_min;
    e.penalty_sigma_max = o.penalty.sigma_max;
    e.legacy_sgs_tau_reduction = o.algorithm.sgs_tau_reduction;
    e.time_limit = o.algorithm.time_limit;
    e.validation.interval = o.validation.interval;
    e.validation.tol = ValidationTolerances::all(o.validation.tol > 0 ? o.validation.tol : o.algorithm.tol);
    e.validation.threads = o.validation.threads;
    e.validation.checkpoint_iterations = o.validation.checkpoint_iterations;
    e.validation.strict_dimacs_tol = o.validation.strict_dimacs_tol;
    e.dimacs_kinds = problem.source.dimacs_kinds;
    if (!o.validation.checkpoint_dir.empty() || !o.validation.final_save_dir.empty())
    {
        const std::string checkpoint_dir = o.validation.checkpoint_dir, final_dir = o.validation.final_save_dir;
        e.validation.on_validation = [checkpoint_dir, final_dir](const ValidationRecord &rec, const std::vector<double> &X,
                                                                 const std::vector<double> &y, const std::vector<double> &S)
        {
            // the final snapshot of a solve that ends at a safety limit (the returned iterate is the best one)
            if (!final_dir.empty() && (rec.trigger == "time_limit" || rec.trigger == "max_iter"))
            {
                if (!report::write_snapshot_txt(final_dir, X, y, S))
                    throw std::runtime_error("could not write the final iterate to " + final_dir);
                std::ofstream(final_dir + "/iteration.txt") << rec.iter << "\n";
            }
            if (!rec.checkpoint || checkpoint_dir.empty())
                return;
            const std::string dir = checkpoint_dir + "/iter_" + std::to_string(rec.iter);
            if (!report::write_snapshot_txt(dir, X, y, S))
                throw std::runtime_error("could not write the checkpoint " + dir); // reported and counted by the engine
        };
    }
    im.is_setup = true;
    im.first_solve = true;
}

SolveResult Solver::solve(const WarmStart *warm_start)
{
    Impl &im = *impl_;
    if (!im.is_setup)
        throw std::logic_error("Solver::solve() before setup()");
    const SolverOptions &o = im.options;
    SDPSolver &e = *im.engine;

    bool rescale = !im.first_solve;
    if (warm_start)
    {
        if (static_cast<int>(warm_start->X.size()) != im.vec_len || static_cast<int>(warm_start->S.size()) != im.vec_len ||
            static_cast<int>(warm_start->y.size()) != im.con_num)
            throw std::invalid_argument("warm start sizes do not match the problem");
        CHECK_CUDA(cudaMemcpy(e.X.vals, warm_start->X.data(), sizeof(double) * im.vec_len, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(e.y.vals, warm_start->y.data(), sizeof(double) * im.con_num, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(e.S.vals, warm_start->S.data(), sizeof(double) * im.vec_len, cudaMemcpyHostToDevice));
        rescale = true; // original units: solve() scales them and recomputes the residuals
        e.sig = warm_start->sigma > 0 ? warm_start->sigma : o.penalty.sigma0;
    }
    else if (!im.first_solve)
    {
        // a cold start again: zero iterate in original units, rescaled by solve()
        CHECK_CUDA(cudaMemset(e.X.vals, 0, sizeof(double) * im.vec_len));
        CHECK_CUDA(cudaMemset(e.y.vals, 0, sizeof(double) * im.con_num));
        CHECK_CUDA(cudaMemset(e.S.vals, 0, sizeof(double) * im.vec_len));
        e.sig = o.penalty.sigma0;
    }
    im.first_solve = false;

    // the components of the algorithm layer (plan Section 7) on the engine's data
    e.stop_tol = o.algorithm.tol;
    e.switch_admm = im.switch_admm;
    admm::AdmmState state(e, *im.cones);
    termination::BestIterateTracker tracker(e);
    std::unique_ptr<termination::StoppingRule> rule;
    if (o.validation.interval > 0)
        rule.reset(new termination::ValidatedRule(e, o.algorithm.tol, o.algorithm.time_limit));
    else
        rule.reset(new termination::InternalKktRule(o.algorithm.tol, o.algorithm.time_limit));
    report::ObserverSet observers;
    observers.add(std::make_shared<report::ConsoleObserver>(o.output.verbosity));
    std::shared_ptr<report::HistoryObserver> history = std::make_shared<report::HistoryObserver>();
    observers.add(history);
    for (const std::shared_ptr<report::IterationObserver> &ob : im.observers)
        observers.add(ob);
    admm::RunConfig cfg;
    cfg.max_iter = o.algorithm.max_iter;
    cfg.tol = o.algorithm.tol;
    cfg.switch_admm = im.switch_admm;
    cfg.tau_admm = o.algorithm.tau;
    cfg.tau_sgs = o.algorithm.sgs_tau >= 0 ? o.algorithm.sgs_tau : o.algorithm.tau;
    cfg.legacy_sgs_tau_reduction = o.algorithm.sgs_tau_reduction;
    cfg.switch_proj_max_iter = im.switch_proj_max_iter;
    cfg.switch_proj_tol = im.switch_proj_tol;
    cfg.verbose = o.output.verbosity >= 1;
    admm::AdmmSolver loop(state, *im.penalty, *rule, tracker, observers, cfg);
    loop.run(e.sig, rescale);

    SolveResult r;
    r.stop_reason = e.stop_reason;
    r.converged = e.converged;
    if (e.stop_reason == "internal_kkt")
        r.status = Status::Converged;
    else if (e.stop_reason == "externally_validated")
        r.status = Status::Validated;
    else if (e.stop_reason == "strict_validated")
        r.status = Status::StrictValidated;
    else if (e.stop_reason == "time_limit")
        r.status = Status::TimeLimit;
    else if (e.stop_reason == "non_finite")
        r.status = Status::NonFinite;
    else
        r.status = Status::MaxIter;
    r.returned_best_iterate = e.returned_best_iterate || e.returned_best_external;
    r.best.X = device_to_host(e.X);
    r.best.y = device_to_host(e.y);
    r.best.S = device_to_host(e.S);
    r.best.eta_p = e.errRp;
    r.best.eta_d = e.errRd;
    r.best.eta_g = e.relgap;
    r.best.pobj = e.pobj;
    r.best.dobj = e.dobj;
    r.final = r.best; // the engine keeps only the returned iterate (Phase 4: BestIterateTracker keeps both)
    r.eta_p = e.errRp;
    r.eta_d = e.errRd;
    r.eta_g = e.relgap;
    const double sign = im.sense == Sense::Maximize ? -1.0 : 1.0;
    r.pobj = sign * e.pobj + im.objective_offset;
    r.dobj = sign * e.dobj + im.objective_offset;
    r.cone_violation_X = e.errPSD_X;
    r.cone_violation_S = e.errPSD_S;
    r.y_solves = e.y_solves;
    r.setup_time_s = e.init_time;
    r.solve_time_s = e.solve_time;
    r.total_time_s = e.total_time;
    r.validation_time_s = e.validation_time_s;
    r.peak_gpu_memory_bytes = e.peak_gpu_mem_bytes;
    r.penalty_policy = std::string(im.penalty->name());
    r.penalty_parameters_json = im.penalty->parameters_json();
    r.freeze_iteration = im.penalty->freeze_iteration();
    r.freeze_reason = im.penalty->freeze_reason();
    r.sigma0 = e.sigma0;
    r.sigma_final = e.sig;
    r.sigma0_original = e.sigma0 * e.bscale / e.Cscale;
    r.sigma_final_original = e.sig * e.bscale / e.Cscale;
    r.sigma_changes = e.sigma_changes;
    for (const SDPSolver::SigmaUpdate &u : e.sigma_log)
    {
        r.sigma_log.push_back({u.iter, u.old_sigma, u.new_sigma, u.rule, u.errRp, u.errRd, u.relgap, u.feasratio, u.prim_win, u.dual_win});
        if (u.iter > 10000)
            r.sigma_changes_after_1e4++;
        r.sigma_total_variation_log2 += std::abs(std::log2(u.new_sigma / u.old_sigma));
    }
    r.scheme = name_of(o.algorithm.scheme);
    r.sgs_iterations = e.sgs_phase_iterations;
    r.admm_iterations = e.admm_phase_iterations;
    r.switch_iteration = e.switch_iteration;
    r.dual_block_min = e.dual_block_min;
    r.dual_cone_coef = e.dual_cone_coef;
    // the certificate (Phase 5: termination::Certificate on the product cone's measures, explicit per-block bounds, B24)
    {
        const std::vector<ConeKind> &kinds = im.cones->kinds();
        termination::CertificateResult cert;
        cert.dual_block_min = e.dual_block_min;
        cert.dual_cone_coef = e.dual_cone_coef;
        termination::Certificate::evaluate(cert, e.dobj, termination::Certificate::bounds_from_options(o.certificate, kinds, im.blk_sizes, false), kinds);
        r.certified_lower_bound = cert.lower_bound;
        r.certificate_missing_bounds = cert.missing_bounds;
        r.certificate_note = cert.note;
        termination::CertificateResult cert_scaled = cert;
        termination::Certificate::evaluate(cert_scaled, e.dobj, termination::Certificate::bounds_from_options(o.certificate, kinds, im.blk_sizes, true), kinds);
        r.certified_lower_bound_scaled = cert_scaled.lower_bound;
        // B24: with a free block the bound is valid only if |X_u| <= R_u holds for the bound applied to it
        r.certificate_conditional_on_free_bounds = cert.conditional_on_free_bounds || cert_scaled.conditional_on_free_bounds;
    }
    r.externally_validated = e.externally_validated_converged;
    r.practical_validated = e.practical_validated;
    r.strict_validated = e.strict_validated;
    r.first_validated_iteration = e.first_validated_iteration;
    r.first_practical_iteration = e.first_practical_iteration;
    r.first_strict_iteration = e.first_strict_iteration;
    r.validations = static_cast<int>(e.validation_history.size());
    r.best_max_abs_dimacs = e.validation_history.empty() ? std::numeric_limits<double>::quiet_NaN() : e.best_validation.result.max_abs_dimacs;
    if (!e.validation_history.empty())
        r.validation_history_csv = validation_csv(e);
    r.linsys_backend = linsys_backend();
    r.aat_supernodal = e.cpu_AAt_solver.supernodal_used;
    r.aat_nnz_L = e.cpu_AAt_solver.cc.lnz;
    r.aat_tiny_pivots = e.cpu_AAt_solver.tiny_pivots;
    r.aat_nonpositive_pivots = e.cpu_AAt_solver.nonpositive_pivots;
    r.lobpcg_calls = e.lobpcg_calls;
    r.lobpcg_fallbacks = e.lobpcg_fallbacks;
    r.composite_fallbacks = e.composite_fallbacks;
    r.jacobi_not_converged = e.jacobi_not_converged;
    // history of this solve, from the observer
    r.history = history->take();
    r.iterations = static_cast<int>(r.history.size());
    r.best.iteration = r.iterations;
    r.final.iteration = r.iterations;
    return r;
}

const char *name_of(Status s)
{
    switch (s)
    {
    case Status::Converged:
        return "converged";
    case Status::MaxIter:
        return "max_iter";
    case Status::TimeLimit:
        return "time_limit";
    case Status::NonFinite:
        return "non_finite";
    case Status::Validated:
        return "validated";
    case Status::StrictValidated:
        return "strict_validated";
    }
    return "?";
}

} // namespace cuadmm
