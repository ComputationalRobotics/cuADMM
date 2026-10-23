/*

    writers.cc

*/

#include "report/writers.h"

#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "cuadmm/version.h"

namespace cuadmm
{
namespace report
{

// the shortest decimal that round-trips exactly (std::to_chars): exact replays, readable values (plan B25); integral
// values below 1e16 in fixed notation (1000000, not 1e+06), as Python prints them
std::string json_number(double x)
{
    if (!std::isfinite(x))
        return "null";
    char buf[64];
    const std::to_chars_result r = (x == std::trunc(x) && std::fabs(x) < 1e16) ? std::to_chars(buf, buf + sizeof(buf), x, std::chars_format::fixed)
                                                                               : std::to_chars(buf, buf + sizeof(buf), x);
    return std::string(buf, r.ptr);
}

std::string json_string(const std::string &s)
{
    std::string out = "\"";
    for (char c : s)
    {
        if (c == '"' || c == '\\')
            out += '\\';
        else if (c == '\n')
        {
            out += "\\n";
            continue;
        }
        out += c;
    }
    return out + "\"";
}

namespace
{
std::ofstream open_or_throw(const std::string &path, std::ios::openmode mode = std::ios::out)
{
    std::ofstream f(path, mode);
    if (!f.is_open())
        throw std::runtime_error("could not open " + path + " for writing");
    return f;
}
} // namespace

void write_history_csv(const std::string &path, const SolveResult &result)
{
    std::ofstream f = open_or_throw(path);
    f << std::setprecision(17) << "iter,time_s,pobj,dobj,eta_p,eta_d,eta_g,sigma,tau,phase\n";
    for (const HistoryRow &r : result.history)
        f << r.iteration << "," << r.time_s << "," << r.pobj << "," << r.dobj << "," << r.eta_p << "," << r.eta_d << "," << r.eta_g << ","
          << r.sigma << "," << r.tau << "," << (r.sgs ? "sgs" : "admm") << "\n";
}

void write_sigma_log_csv(const std::string &path, const SolveResult &result)
{
    std::ofstream f = open_or_throw(path);
    f << std::setprecision(17) << "iter,old_sigma,new_sigma,rule,eta_p,eta_d,eta_g,feasratio,prim_win,dual_win\n";
    for (const SigmaChange &u : result.sigma_log)
        f << u.iteration << "," << u.old_sigma << "," << u.new_sigma << "," << u.rule << "," << u.eta_p << "," << u.eta_d << "," << u.eta_g
          << "," << u.feasratio << "," << u.prim_win << "," << u.dual_win << "\n";
}

void write_validation_history_csv(const std::string &path, const SolveResult &result)
{
    std::ofstream f = open_or_throw(path);
    f << result.validation_history_csv;
}

bool write_snapshot_txt(const std::string &directory, const std::vector<double> &X, const std::vector<double> &y,
                        const std::vector<double> &S)
{
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec)
        return false;
    auto write = [&](const std::string &name, const std::vector<double> &v)
    {
        std::ofstream f(directory + "/" + name);
        f << std::setprecision(17);
        for (double x : v)
            f << x << "\n";
        f.close();
        return !f.fail();
    };
    return write("X.txt", X) && write("y.txt", y) && write("S.txt", S);
}

void append_summary_json(const std::string &path, const std::string &problem_path, const ProblemData &problem,
                         const SolverOptions &o, const SolveResult &r, double load_time_s, double cuda_init_time_s)
{
    std::ofstream f = open_or_throw(path, std::ios::app);
    const bool b = true;
    auto B = [](bool v) { return v ? "true" : "false"; };
    (void)b;
    const int switch_admm = o.algorithm.scheme == SchemeKind::SgsAdmm ? std::numeric_limits<int>::max() : o.algorithm.scheme_switch_iteration;
    const int sgs_requested = o.algorithm.scheme_switch_iteration > 1 ? o.algorithm.scheme_switch_iteration - 1 : -1; // -1: not a hybrid (as today)
    std::string algorithm_name;
    if (o.algorithm.scheme == SchemeKind::SgsAdmm)
        algorithm_name = "pure sGS-ADMM (Algorithm 1)";
    else if (o.algorithm.scheme_switch_iteration <= 1)
        algorithm_name = "plain ADMM";
    else
        algorithm_name = "hybrid: sGS-ADMM for iterations 1-" + std::to_string(o.algorithm.scheme_switch_iteration - 1) + ", then plain ADMM";
    const std::string status = r.converged ? "converged" : (r.status == Status::TimeLimit ? "time_limit" : (r.status == Status::NonFinite ? "non_finite" : "max_iter"));
    f << std::setprecision(17)
      << "{\"problem\": " << json_string(problem_path) << ", \"tag\": " << json_string(o.output.tag)
      << ", \"format\": " << json_string(problem.source.format) << ", \"source_path\": " << json_string(problem.source.path)
      << ", \"sense\": " << json_string(name_of(problem.sense))
      << ", \"objective_offset\": " << json_number(problem.objective_offset)
      << ", \"vec_len\": " << problem.vec_len() << ", \"con_num\": " << problem.num_constraints() << ", \"blocks\": " << problem.cones.size()
      << ", \"stop_tol\": " << json_number(o.algorithm.tol) << ", \"max_iter\": " << o.algorithm.max_iter
      << ", \"scheme\": " << json_string(name_of(o.algorithm.scheme)) << ", \"tau\": " << json_number(o.algorithm.tau)
      << ", \"switch_admm\": " << switch_admm << ", \"sgs_iterations_requested\": " << sgs_requested
      << ", \"algorithm\": " << json_string(algorithm_name)
      << ", \"sigma_policy\": " << json_string(r.penalty_policy) << ", \"sigma0\": " << json_number(r.sigma0)
      << ", \"sigma_policy_parameters\": " << r.penalty_parameters_json
      << ", \"sigma_final_scaled\": " << json_number(r.sigma_final) << ", \"sigma_final_original\": " << json_number(r.sigma_final_original)
      << ", \"sigma0_original\": " << json_number(r.sigma0_original)
      << ", \"sigma_changes\": " << r.sigma_changes << ", \"sigma_changes_after_1e4\": " << r.sigma_changes_after_1e4
      << ", \"sigma_total_variation_log2\": " << json_number(r.sigma_total_variation_log2)
      << ", \"freeze_iteration\": " << r.freeze_iteration << ", \"freeze_reason\": " << json_string(r.freeze_reason)
      << ", \"sigma_log_entries\": " << r.sigma_log.size()
      << ", \"sgs_phase_iterations\": " << r.sgs_iterations << ", \"admm_phase_iterations\": " << r.admm_iterations
      << ", \"switch_iteration\": " << r.switch_iteration
      << ", \"sig0\": " << json_number(o.penalty.sigma0) << ", \"use_lobpcg\": " << B(o.psd.lowrank_enabled)
      << ", \"warm_start\": " << B(o.io.warm_start || !o.io.warm_start_dir.empty())
      << ", \"status\": " << json_string(status) << ", \"converged\": " << B(r.converged)
      << ", \"returned_best_iterate\": " << B(r.returned_best_iterate)
      << ", \"iterations\": " << r.iterations << ", \"time_s\": " << json_number(r.total_time_s)
      << ", \"init_time_s\": " << json_number(r.setup_time_s) << ", \"solve_time_s\": " << json_number(r.solve_time_s)
      << ", \"time_limit\": " << json_number(o.algorithm.time_limit)
      << ", \"t_load_s\": " << json_number(load_time_s) << ", \"t_cuda_init_s\": " << json_number(cuda_init_time_s)
      << ", \"stop_reason\": " << json_string(r.stop_reason)
      << ", \"validation_interval\": " << o.validation.interval
      << ", \"validation_tol\": " << json_number(o.validation.tol < 0 ? o.algorithm.tol : o.validation.tol)
      << ", \"validation_threads\": " << o.validation.threads << ", \"validation_count\": " << r.validations
      << ", \"validation_time_s\": " << json_number(r.validation_time_s)
      << ", \"externally_validated_converged\": " << B(r.externally_validated)
      << ", \"practical_validated\": " << B(r.practical_validated) << ", \"first_practical_iteration\": " << r.first_practical_iteration
      << ", \"strict_validated\": " << B(r.strict_validated) << ", \"first_strict_iteration\": " << r.first_strict_iteration
      << ", \"first_validated_iteration\": " << r.first_validated_iteration
      << ", \"strict_dimacs_tol\": " << json_number(o.validation.strict_dimacs_tol)
      << ", \"best_max_abs_dimacs\": " << json_number(r.best_max_abs_dimacs)
      << ", \"y_solves\": " << r.y_solves << ", \"y_solves_per_iteration\": " << json_number(r.iterations > 0 ? double(r.y_solves) / r.iterations : 0.0)
      << ", \"peak_gpu_mem_mib\": " << json_number(r.peak_gpu_memory_bytes / 1048576.0)
      << ", \"linsys_backend\": " << json_string(r.linsys_backend) << ", \"aat_supernodal\": " << B(r.aat_supernodal)
      << ", \"aat_nnz_L\": " << json_number(r.aat_nnz_L) << ", \"aat_tiny_pivots\": " << r.aat_tiny_pivots
      << ", \"aat_nonpositive_pivots\": " << r.aat_nonpositive_pivots
      << ", \"errRp\": " << json_number(r.eta_p) << ", \"errRd\": " << json_number(r.eta_d) << ", \"relgap\": " << json_number(r.eta_g)
      << ", \"eta_p\": " << json_number(r.eta_p) << ", \"eta_d\": " << json_number(r.eta_d) << ", \"eta_g\": " << json_number(r.eta_g)
      << ", \"pobj\": " << json_number(r.pobj) << ", \"dobj\": " << json_number(r.dobj)
      << ", \"errPSD_X\": " << json_number(r.cone_violation_X) << ", \"errPSD_S\": " << json_number(r.cone_violation_S)
      << ", \"lobpcg_calls\": " << r.lobpcg_calls << ", \"lobpcg_fallbacks\": " << r.lobpcg_fallbacks
      << ", \"composite_fallbacks\": " << r.composite_fallbacks << ", \"jacobi_not_converged\": " << r.jacobi_not_converged
      << ", \"final_sig\": " << json_number(r.sigma_final)
      << ", \"trace_bound\": " << json_number(o.certificate.trace_bound);
    if (o.certificate.trace_bound >= 0 || !o.certificate.block_bounds.empty())
        f << ", \"certified_lower_bound\": " << json_number(r.certified_lower_bound);
    if (!o.certificate.block_bounds.empty())
    {
        f << ", \"certificate_block_bounds\": [";
        for (std::size_t b = 0; b < o.certificate.block_bounds.size(); b++)
            f << (b ? ", " : "") << json_number(o.certificate.block_bounds[b]);
        f << "]";
    }
    if (!r.certificate_missing_bounds.empty())
    {
        f << ", \"certificate_missing_bounds\": [";
        for (std::size_t b = 0; b < r.certificate_missing_bounds.size(); b++)
            f << (b ? ", " : "") << r.certificate_missing_bounds[b];
        f << "], \"certificate_note\": " << json_string(r.certificate_note);
    }
    f << ", \"trace_bound_scale\": " << json_number(o.certificate.trace_bound_scale);
    if (o.certificate.trace_bound_scale >= 0)
        f << ", \"certified_lower_bound_scaled\": " << json_number(r.certified_lower_bound_scaled);
    f << ", \"certificate_conditional_on_free_bounds\": " << B(r.certificate_conditional_on_free_bounds)
      << ", \"conversion_report\": [";
    for (std::size_t k = 0; k < problem.source.report.size(); k++)
        f << (k ? ", " : "") << json_string(problem.source.report[k]);
    f << "]"
      << ", \"options\": " << options_json(o)
      << ", \"build_project\": " << json_string(CUADMM_BUILD_PROJECT) << ", \"build_commit\": " << json_string(CUADMM_BUILD_COMMIT)
      << ", \"build_dirty\": " << B(CUADMM_BUILD_DIRTY)
      << "}" << std::endl;
}

} // namespace report
} // namespace cuadmm
