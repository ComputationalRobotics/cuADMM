#include <cusparse.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <chrono>
#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

#include "cuadmm/solver.h"
#include "cuadmm/io.h"
#include "cuadmm/problem.h"

// Command-line options; the defaults reproduce the historical behaviour of cuadmm_exe
// (plain ADMM from the first iteration, LOBPCG enabled, stopping tolerance 1e-4).
struct Options
{
    std::string prefix;
    double stop_tol = 1e-4;
    int max_iter = 1000000;
    int switch_admm = 0; // sGS-ADMM for iter < switch_admm, then plain ADMM (0: plain ADMM throughout)
    bool switch_admm_set = false;
    std::string algorithm;                      // "admm" or "sgs" (sets switch_admm); empty: use switch_admm
    int sgs_iterations = -1;                    // hybrid mode: N sGS iterations, then plain ADMM (sets switch_admm = N + 1)
    bool sgs_iterations_set = false;
    std::string sigma_policy = "legacy_adaptive"; // "fixed", "legacy_adaptive" (the historical behaviour) or
                                                  // "fixed_sgs_then_legacy_admm"
    double sig = 1e2;
    bool use_lobpcg = true;
    int switch_proj_max_iter = 0;
    double switch_proj_tol = 1e-2;
    int sig_update_threshold = 500;
    int sig_update_stage_1 = 50;
    int sig_update_stage_2 = 100;
    double sigscale = 2.0;
    int eig_stream_num_per_gpu = 15;
    bool warm_start = false;
    std::string warm_start_dir; // read X.txt, y.txt, S.txt from this directory
    std::string save_dir;       // write X.txt, y.txt, S.txt and certificate.txt to this directory
    std::string summary_file; // append one JSON line with the results
    std::string history_file; // write the per-iteration history as CSV
    std::string tag;          // free-form label copied to the summary
    double time_limit = -1;        // seconds of iterations after which the solver stops (<= 0: none)
    double trace_bound = -1;       // bound R on the trace of every block of an optimal X (certificate, paper eq. 30)
    double trace_bound_scale = -1; // R_b = f * n_b for PSD blocks, f for the entries of 'l'/'u' blocks (see below)
    // validation-aware stopping (SDPSolver::validation); off by default (stop on the internal KKT residual)
    int validate_interval = 0;
    double validate_tol = -1;           // tolerance of every external criterion (default: --tol)
    int validate_threads = 1;
    std::vector<int> checkpoint_iters;  // extra validation snapshots, saved to checkpoint_dir if given
    std::string checkpoint_dir;         // X.txt, y.txt, S.txt of the requested checkpoints
    std::string validation_history_file; // CSV with one line per external validation
    double strict_dimacs_tol = 0.0;      // strict stage after the practical criteria (0: off)
    std::string save_final_dir;          // X, y, S of the final snapshot of a solve that ends at a safety limit
    std::string sigma_log_file;          // CSV with one line per sigma change
    double t_load_s = 0, t_cuda_init_s = 0; // measured in main()
};

// Per-block bounds R_b of the certificate: R_b = f * n_b for PSD blocks of size n_b, and f for the entries of 'l' and
// 'u' blocks. With f = 1 this is the bound of the paper (Appendix C, Theorem 2) for moment matrices of POP variables
// scaled to [-1, 1] (and localizing matrices of constraints with |g| <= 1 there). The paper's s(n, kappa) R^2 does not
// hold for R != 1 (the trace of a moment matrix grows like R^{2 kappa}), so f = R^2 is not a valid bound in general.
static std::vector<double> size_scaled_bounds(const SDPSolver &solver, double f)
{
    std::vector<double> bounds(solver.blk_info.size());
    for (size_t b = 0; b < bounds.size(); b++)
        bounds[b] = solver.blk_info[b].type == 's' ? f * solver.blk_info[b].size : f;
    return bounds;
}

static void print_usage(const char *exe)
{
    std::cerr
        << "usage: " << exe << " <problem directory> [options]\n"
        << "  The directory contains blk.txt, At.txt, b.txt, C.txt, con_num.txt (and X.txt, y.txt, S.txt with --warm-start).\n"
        << "options (defaults in brackets):\n"
        << "  --tol <x>                  KKT stopping tolerance [1e-4]\n"
        << "  --max-iter <n>             maximum number of iterations [1000000]\n"
        << "  --time-limit <s>           stop after <s> seconds of iterations, returning the best iterate [none]\n"
        << "  --switch-admm <n>          sGS-ADMM for the first n-1 iterations, then plain ADMM;\n"
        << "                             0 = plain ADMM throughout; use n > max-iter for pure sGS-ADMM [0]\n"
        << "  --algorithm <admm|sgs>     plain ADMM throughout, or pure sGS-ADMM throughout (the switch to plain ADMM\n"
        << "                             cannot happen); alternative to --switch-admm [not set]\n"
        << "  --sgs-iterations <n>       hybrid mode (not Algorithm 1): exactly n sGS-ADMM iterations, then plain ADMM from\n"
        << "                             the same X, y, S (= --switch-admm n+1; n = 0: plain ADMM); alternative to\n"
        << "                             --switch-admm and --algorithm [not set]\n"
        << "  --sigma-policy <p>         'fixed': sigma = --sig for every iteration (Algorithm 1 of the paper);\n"
        << "                             'legacy_adaptive': the historical adaptive sigma rules [legacy_adaptive];\n"
        << "                             'fixed_sgs_then_legacy_admm': fixed during the sGS phase, then the legacy plain-ADMM\n"
        << "                             rules counted from the first plain-ADMM iteration\n"
        << "  --validate-interval <n>    validation-aware stopping: validate the iterate externally (CPU, original data)\n"
        << "                             every n iterations and when the internal KKT residual crosses --tol; stop only\n"
        << "                             when eta_p, eta_d, eta_g, the X cone and the dual cone violations are <= the\n"
        << "                             tolerance (0: off, stop on the internal KKT residual) [0]\n"
        << "  --validate-tol <x>         tolerance of every external criterion [--tol]\n"
        << "  --validate-threads <n>     CPU threads of the external validation [1]\n"
        << "  --checkpoint-iters <list>  comma-separated iterations with an extra validation snapshot (e.g. 10000)\n"
        << "  --checkpoint-dir <dir>     save X.txt, y.txt, S.txt of those snapshots in <dir>/iter_<k> (a failed write\n"
        << "                             is reported and counted, the solve continues)\n"
        << "  --validation-history <f>   write one CSV line per external validation\n"
        << "  --strict-dimacs-tol <x>    after the validated criteria, continue until max |DIMACS error| <= x (0: stop there) [0]\n"
        << "  --save-final-dir <dir>     save X, y, S of the final snapshot of a solve that ends at a safety limit\n"
        << "  --sigma-log <f>            write one CSV line per sigma change (iteration, old, new, rule, residuals)\n"
        << "  --sig <x>                  initial penalty sigma [1e2]\n"
        << "  --lobpcg <0|1>             use LOBPCG for low-rank large blocks [1]\n"
        << "  --switch-proj-iter <n>     iteration after which the final projection method is used [0]\n"
        << "  --switch-proj-tol <x>      KKT residual below which the final projection method is used [1e-2]\n"
        << "  --sig-update-threshold <n> [500]   --sig-update-stage-1 <n> [50]   --sig-update-stage-2 <n> [100]\n"
        << "  --sigscale <x>             sigma update (period, factor) of the sGS phase, with --sigma-policy\n"
        << "                             legacy_adaptive only; the plain ADMM phase has its own schedule [2]\n"
        << "  --streams <n>              CUDA streams for the medium eigendecompositions [15]\n"
        << "  --warm-start               read X.txt, y.txt, S.txt from the problem directory\n"
        << "  --warm-start-dir <dir>     read X.txt, y.txt, S.txt (dense, one value per entry) from <dir>\n"
        << "  --save-solution <dir>      write the returned X.txt, y.txt, S.txt (readable by --warm-start-dir) and\n"
        << "                             certificate.txt (per block: type, size, lambda_min and c_b of C - A^T y) to <dir>\n"
        << "  --summary <file>           append a one-line JSON summary of the run to <file>\n"
        << "  --history <file>           write the per-iteration history (pobj, dobj, residuals, sigma) as CSV\n"
        << "  --tag <label>              label copied to the JSON summary\n"
        << "  --trace-bound <R>          report the certified lower bound <b,y> + R sum_b c_b (paper eq. 30), valid if\n"
        << "                             every PSD block of an optimal X has trace <= R ('l'/'u' blocks: entries in [0,R] / [-R,R])\n"
        << "  --trace-bound-scale <f>    same with R_b = f * n_b for a PSD block of size n_b (f for 'l'/'u' entries): with f = 1\n"
        << "                             this is the bound of the paper (Thm. 2) for moment matrices of variables scaled to [-1,1]\n"
        << "                             (localizing matrices: only if |g| <= 1 there); valid only if tr(X_b) <= f n_b, and\n"
        << "                             f = R^2 is NOT valid for variables in [-R,R] with R > 1 (see README)\n";
}

// Parses the command line; returns false (after printing the usage) on an error.
// Parses a whole token as an int (std::stoi alone accepts e.g. "1e4" as 1); throws std::invalid_argument.
static int parse_int_strict(const std::string &token)
{
    size_t used = 0;
    const int v = std::stoi(token, &used);
    if (used != token.size())
        throw std::invalid_argument("not an integer: " + token);
    return v;
}

// A double as a JSON number, or null if it is not finite (JSON has no NaN or infinity).
static std::string json_number(double x)
{
    if (!std::isfinite(x))
        return "null";
    std::ostringstream o;
    o << std::setprecision(10) << x;
    return o.str();
}

static bool parse_options(int argc, char *argv[], Options &opt)
{
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")
    {
        print_usage(argv[0]);
        return false;
    }
    opt.prefix = argv[1];
    for (int i = 2; i < argc; i++)
    {
        const std::string key = argv[i];
        if (key == "--warm-start")
        {
            opt.warm_start = true;
            continue;
        }
        if (i + 1 >= argc)
        {
            std::cerr << "ERROR: missing value for option " << key << std::endl;
            print_usage(argv[0]);
            return false;
        }
        const std::string val = argv[++i];
        try
        {
            if (key == "--tol")
                opt.stop_tol = std::stod(val);
            else if (key == "--max-iter")
                opt.max_iter = parse_int_strict(val);
            else if (key == "--time-limit")
                opt.time_limit = std::stod(val);
            else if (key == "--switch-admm")
            {
                opt.switch_admm = parse_int_strict(val);
                opt.switch_admm_set = true;
            }
            else if (key == "--sgs-iterations")
            {
                opt.sgs_iterations = parse_int_strict(val);
                opt.sgs_iterations_set = true;
            }
            else if (key == "--algorithm")
                opt.algorithm = val;
            else if (key == "--sigma-policy")
                opt.sigma_policy = val;
            else if (key == "--validate-interval")
                opt.validate_interval = parse_int_strict(val);
            else if (key == "--validate-tol")
                opt.validate_tol = std::stod(val);
            else if (key == "--validate-threads")
                opt.validate_threads = parse_int_strict(val);
            else if (key == "--checkpoint-iters")
            {
                size_t pos = 0;
                while (pos < val.size())
                {
                    size_t comma = val.find(',', pos);
                    if (comma == std::string::npos)
                        comma = val.size();
                    opt.checkpoint_iters.push_back(parse_int_strict(val.substr(pos, comma - pos)));
                    pos = comma + 1;
                }
            }
            else if (key == "--checkpoint-dir")
                opt.checkpoint_dir = val;
            else if (key == "--strict-dimacs-tol")
                opt.strict_dimacs_tol = std::stod(val);
            else if (key == "--save-final-dir")
                opt.save_final_dir = val;
            else if (key == "--sigma-log")
                opt.sigma_log_file = val;
            else if (key == "--validation-history")
                opt.validation_history_file = val;
            else if (key == "--sig")
                opt.sig = std::stod(val);
            else if (key == "--lobpcg")
                opt.use_lobpcg = std::stoi(val) != 0;
            else if (key == "--switch-proj-iter")
                opt.switch_proj_max_iter = std::stoi(val);
            else if (key == "--switch-proj-tol")
                opt.switch_proj_tol = std::stod(val);
            else if (key == "--sig-update-threshold")
                opt.sig_update_threshold = std::stoi(val);
            else if (key == "--sig-update-stage-1")
                opt.sig_update_stage_1 = std::stoi(val);
            else if (key == "--sig-update-stage-2")
                opt.sig_update_stage_2 = std::stoi(val);
            else if (key == "--sigscale")
                opt.sigscale = std::stod(val);
            else if (key == "--streams")
                opt.eig_stream_num_per_gpu = std::stoi(val);
            else if (key == "--summary")
                opt.summary_file = val;
            else if (key == "--history")
                opt.history_file = val;
            else if (key == "--tag")
                opt.tag = val;
            else if (key == "--trace-bound")
                opt.trace_bound = std::stod(val);
            else if (key == "--trace-bound-scale")
                opt.trace_bound_scale = std::stod(val);
            else if (key == "--warm-start-dir")
                opt.warm_start_dir = val;
            else if (key == "--save-solution")
                opt.save_dir = val;
            else
            {
                std::cerr << "ERROR: unknown option " << key << std::endl;
                print_usage(argv[0]);
                return false;
            }
        }
        catch (const std::exception &)
        {
            std::cerr << "ERROR: invalid value '" << val << "' for option " << key << std::endl;
            return false;
        }
    }
    SigmaPolicy policy;
    if (!parse_sigma_policy(opt.sigma_policy, policy))
    {
        std::cerr << "ERROR: --sigma-policy must be 'fixed', 'legacy_adaptive' or 'fixed_sgs_then_legacy_admm'" << std::endl;
        return false;
    }
    if (opt.sgs_iterations_set)
    {
        if (opt.switch_admm_set || !opt.algorithm.empty() || opt.sgs_iterations < 0 ||
            opt.sgs_iterations > std::numeric_limits<int>::max() - 2)
        {
            std::cerr << "ERROR: --sgs-iterations must be a non-negative integer and cannot be combined with --switch-admm or --algorithm"
                      << std::endl;
            return false;
        }
        opt.switch_admm = opt.sgs_iterations == 0 ? 0 : opt.sgs_iterations + 1; // iterations 1..n are sGS-ADMM
    }
    if (!opt.algorithm.empty())
    {
        if (opt.switch_admm_set)
        {
            std::cerr << "ERROR: use either --algorithm or --switch-admm, not both" << std::endl;
            return false;
        }
        if (opt.algorithm == "admm")
            opt.switch_admm = 0;
        else if (opt.algorithm == "sgs")
            opt.switch_admm = std::numeric_limits<int>::max(); // the switch to plain ADMM can never happen
        else
        {
            std::cerr << "ERROR: --algorithm must be 'admm' or 'sgs'" << std::endl;
            return false;
        }
    }
    if (opt.validate_interval < 0 || opt.validate_threads <= 0 || (opt.validate_tol <= 0 && opt.validate_tol != -1))
    {
        std::cerr << "ERROR: --validate-interval must be non-negative, --validate-threads and --validate-tol positive" << std::endl;
        return false;
    }
    if (opt.validate_interval == 0 && (opt.validate_tol != -1 || opt.validate_threads != 1 || !opt.checkpoint_iters.empty() ||
                                       !opt.checkpoint_dir.empty() || !opt.validation_history_file.empty() ||
                                       opt.strict_dimacs_tol != 0.0 || !opt.save_final_dir.empty()))
    {
        std::cerr << "ERROR: --validate-tol, --validate-threads, --checkpoint-iters, --checkpoint-dir, --validation-history, "
                  << "--strict-dimacs-tol and --save-final-dir require --validate-interval > 0" << std::endl;
        return false;
    }
    if (!(opt.strict_dimacs_tol >= 0.0) || !std::isfinite(opt.strict_dimacs_tol))
    {
        std::cerr << "ERROR: --strict-dimacs-tol must be a finite non-negative number" << std::endl;
        return false;
    }
#ifdef CUADMM_PLAIN_ADMM_ONLY
    // plain-ADMM-only build: every option that would run an sGS-ADMM step is rejected
    {
        const char *why = nullptr;
        if (opt.algorithm == "sgs")
            why = "--algorithm sgs";
        else if (opt.switch_admm_set && opt.switch_admm > 1)
            why = "--switch-admm > 1";
        else if (opt.sgs_iterations_set && opt.sgs_iterations > 0)
            why = "--sgs-iterations > 0";
        else if (opt.sigma_policy == "fixed_sgs_then_legacy_admm")
            why = "--sigma-policy fixed_sgs_then_legacy_admm";
        if (why)
        {
            std::cerr << "ERROR: " << why << " requests sGS-ADMM, which is disabled in this plain-ADMM-only build "
                      << "(CUADMM_PLAIN_ADMM_ONLY, branch experiment/plato-cold-plain-admm-sigma); use plain ADMM "
                      << "(--algorithm admm, the default)" << std::endl;
            return false;
        }
    }
#endif
    for (int c : opt.checkpoint_iters)
        if (c <= 0)
        {
            std::cerr << "ERROR: --checkpoint-iters must be positive iteration numbers" << std::endl;
            return false;
        }
    if (opt.max_iter > std::numeric_limits<int>::max() - 2)
    {
        std::cerr << "ERROR: --max-iter is too large" << std::endl;
        return false;
    }
    if (opt.stop_tol <= 0 || opt.max_iter < 0 || opt.switch_admm < 0 || opt.sig <= 0 || opt.eig_stream_num_per_gpu <= 0 ||
        opt.sig_update_threshold < 0 || opt.sig_update_stage_1 <= 0 || opt.sig_update_stage_2 <= 0 || opt.sigscale <= 0)
    {
        std::cerr << "ERROR: --tol, --sig, --streams, --sigscale and --sig-update-stage-1/2 must be positive, "
                  << "--max-iter, --switch-admm and --sig-update-threshold non-negative" << std::endl;
        return false;
    }
    return true;
}

// Escapes a string for a JSON value.
static std::string json_string(const std::string &s)
{
    std::string out = "\"";
    for (char c : s)
    {
        if (c == '"' || c == '\\')
            out += '\\';
        out += c;
    }
    return out + "\"";
}

// Sum over the blocks of the certificate coefficients c_b of C - A^T y, and smallest eigenvalue over the PSD blocks.
static void dual_certificate_stats(const SDPSolver &solver, double &coef_sum, double &min_eig)
{
    coef_sum = 0.0;
    min_eig = std::numeric_limits<double>::infinity();
    for (size_t b = 0; b < solver.dual_cone_coef.size(); b++)
    {
        coef_sum += solver.dual_cone_coef[b];
        if (solver.blk_info[b].type == 's')
            min_eig = std::min(min_eig, solver.dual_block_min[b]);
    }
    if (!std::isfinite(min_eig))
        min_eig = 0.0; // no PSD block
}

static void write_summary(const Options &opt, const Problem &problem, const SDPSolver &solver)
{
    double coef_sum, min_eig;
    dual_certificate_stats(solver, coef_sum, min_eig);
    std::ofstream f(opt.summary_file, std::ios::app);
    if (!f.is_open())
    {
        std::cerr << "WARNING: could not open " << opt.summary_file << " for the summary" << std::endl;
        return;
    }
    f << std::setprecision(10)
      << "{\"problem\": " << json_string(opt.prefix) << ", \"tag\": " << json_string(opt.tag)
      << ", \"vec_len\": " << problem.vec_len << ", \"con_num\": " << problem.con_num
      << ", \"stop_tol\": " << opt.stop_tol << ", \"max_iter\": " << opt.max_iter
      << ", \"switch_admm\": " << opt.switch_admm
      << ", \"sigma_policy\": " << json_string(sigma_policy_name(solver.sigma_policy)) << ", \"sigma0\": " << solver.sigma0
      << ", \"sigma_changes\": " << solver.sigma_changes << ", \"admm_phase_iterations\": " << solver.admm_phase_iterations
      << ", \"sgs_iterations_requested\": " << opt.sgs_iterations << ", \"sgs_phase_iterations\": " << solver.sgs_phase_iterations
      << ", \"switch_iteration\": " << solver.switch_iteration << ", \"sgs_phase_time_s\": " << json_number(solver.sgs_phase_time_s)
      << ", \"admm_phase_time_s\": " << json_number(solver.admm_phase_time_s)
      << ", \"sgs_phase_validation_s\": " << json_number(solver.sgs_phase_validation_s)
      << ", \"admm_phase_validation_s\": " << json_number(solver.admm_phase_validation_s)
      << ", \"kkt_at_switch\": " << json_number(solver.kkt_at_switch) << ", \"sigma_at_switch\": " << json_number(solver.sigma_at_switch)
      << ", \"tau_last_sgs\": " << json_number(solver.tau_last_sgs)
      << ", \"sigma_changes_sgs_phase\": " << solver.sigma_changes_sgs_phase
      << ", \"sigma_changes_admm_phase\": " << solver.sigma_changes_admm_phase
      << ", \"algorithm\": " << json_string(admm_mode_name(opt.switch_admm, opt.max_iter)) << ", \"sig0\": " << opt.sig
      << ", \"use_lobpcg\": " << (opt.use_lobpcg ? "true" : "false")
      << ", \"sigscale\": " << opt.sigscale
      << ", \"warm_start\": " << ((opt.warm_start || !opt.warm_start_dir.empty()) ? "true" : "false")
      << ", \"status\": " << json_string(solver.converged ? "converged" : (solver.time_limit_reached ? "time_limit" : (solver.stop_reason == "non_finite" ? "non_finite" : "max_iter")))
      << ", \"converged\": " << (solver.converged ? "true" : "false")
      << ", \"returned_best_iterate\": " << (solver.returned_best_iterate ? "true" : "false")
      << ", \"iterations\": " << solver.info_iter_num << ", \"time_s\": " << solver.total_time
      << ", \"init_time_s\": " << solver.init_time << ", \"solve_time_s\": " << solver.solve_time
      << ", \"time_limit\": " << opt.time_limit
      << ", \"t_load_s\": " << opt.t_load_s << ", \"t_cuda_init_s\": " << opt.t_cuda_init_s
      << ", \"stop_reason\": " << json_string(solver.stop_reason)
      << ", \"solver_reported_converged\": " << (solver.internal_converged_at_return ? "true" : "false")
      << ", \"externally_validated_converged\": " << (solver.externally_validated_converged ? "true" : "false")
      << ", \"validation_interval\": " << solver.validation.interval
      << ", \"validation_tol\": " << solver.validation.tol.primal
      << ", \"validation_threads\": " << solver.validation.threads
      << ", \"validation_count\": " << solver.validation_history.size()
      << ", \"validation_time_s\": " << solver.validation_time_s
      << ", \"callback_time_s\": " << solver.callback_time_s << ", \"callback_failures\": " << solver.callback_failures
#ifdef CUADMM_PLAIN_ADMM_ONLY
      << ", \"plain_admm_only_build\": true"
#else
      << ", \"plain_admm_only_build\": false"
#endif
      << ", \"y_solves\": " << solver.y_solves << ", \"y_solves_per_iteration\": "
      << json_number(solver.info_iter_num > 0 ? (double)solver.y_solves / solver.info_iter_num : 0.0)
      << ", \"sigma_log_entries\": " << solver.sigma_log.size()
      << ", \"strict_dimacs_tol\": " << json_number(solver.validation.strict_dimacs_tol)
      << ", \"practical_validated\": " << (solver.practical_validated ? "true" : "false")
      << ", \"first_practical_iteration\": " << solver.first_practical_iteration
      << ", \"first_practical_time_s\": " << json_number(solver.first_practical_time_s)
      << ", \"strict_validated\": " << (solver.strict_validated ? "true" : "false")
      << ", \"first_strict_iteration\": " << solver.first_strict_iteration
      << ", \"first_strict_time_s\": " << json_number(solver.first_strict_time_s)
      << ", \"first_validated_iteration\": " << solver.first_validated_iteration
      << ", \"first_internal_below_1e-3\": " << solver.first_internal_below_1e_3
      << ", \"first_internal_below_tol\": " << solver.first_internal_below_tol
      << ", \"returned_best_external\": " << (solver.returned_best_external ? "true" : "false")
      << ", \"peak_gpu_mem_mib\": " << solver.peak_gpu_mem_bytes / 1048576.0;
    if (!solver.validation_history.empty())
    {
        auto rec = [&](const char *name, const ValidationRecord &v)
        {
            const ValidationResult &r = v.result;
            f << ", \"" << name << "_iter\": " << v.iter << ", \"" << name << "_time_s\": " << json_number(v.time_s)
              << ", \"" << name << "_validation_s\": " << json_number(v.validation_s)
              << ", \"" << name << "_kkt_full\": " << json_number(r.kkt_full)
              << ", \"" << name << "_primal_res\": " << json_number(r.primal_res) << ", \"" << name << "_dual_res\": " << json_number(r.dual_res)
              << ", \"" << name << "_relgap\": " << json_number(r.relgap) << ", \"" << name << "_X_cone_violation\": " << json_number(r.X_cone_violation)
              << ", \"" << name << "_dual_cone_violation\": " << json_number(r.dual_cone_violation)
              << ", \"" << name << "_merit\": " << json_number(r.merit())
              << ", \"" << name << "_pobj\": " << json_number(r.pobj) << ", \"" << name << "_dobj\": " << json_number(r.dobj)
              << ", \"" << name << "_internal_kkt\": " << json_number(v.internal_kkt)
              << ", \"" << name << "_S_cone_violation\": " << json_number(r.S_cone_violation)
              << ", \"" << name << "_max_abs_dimacs\": " << json_number(r.max_abs_dimacs)
              << ", \"" << name << "_dimacs\": [" << json_number(r.dimacs[0]) << ", " << json_number(r.dimacs[1]) << ", "
              << json_number(r.dimacs[2]) << ", " << json_number(r.dimacs[3]) << ", " << json_number(r.dimacs[4]) << ", "
              << json_number(r.dimacs[5]) << "]"
              << ", \"" << name << "_err4_z\": " << json_number(r.err4_z) << ", \"" << name << "_err6_z\": " << json_number(r.err6_z)
              << ", \"" << name << "_failed\": " << json_string(r.failed)
              << ", \"" << name << "_validated\": " << (r.validated ? "true" : "false");
        };
        rec("best", solver.best_validation);
        rec("final", solver.validation_history.back());
    }
    f << ", \"aat_supernodal\": " << (solver.cpu_AAt_solver.supernodal_used ? "true" : "false")
      << ", \"aat_nnz_L\": " << solver.cpu_AAt_solver.cc.lnz
      << ", \"aat_tiny_pivots\": " << solver.cpu_AAt_solver.tiny_pivots
      << ", \"aat_nonpositive_pivots\": " << solver.cpu_AAt_solver.nonpositive_pivots
      << ", \"errRp\": " << json_number(solver.errRp) << ", \"errRd\": " << json_number(solver.errRd) << ", \"relgap\": " << json_number(solver.relgap)
      << ", \"pobj\": " << json_number(solver.pobj) << ", \"dobj\": " << json_number(solver.dobj)
      << ", \"errPSD_X\": " << json_number(solver.errPSD_X) << ", \"errPSD_S\": " << json_number(solver.errPSD_S)
      << ", \"lobpcg_calls\": " << solver.lobpcg_calls << ", \"lobpcg_fallbacks\": " << solver.lobpcg_fallbacks
      << ", \"final_sig\": " << solver.sig
      << ", \"dual_cone_coef_sum\": " << coef_sum << ", \"dual_min_eig\": " << min_eig
      << ", \"trace_bound\": " << opt.trace_bound;
    if (opt.trace_bound >= 0)
        f << ", \"certified_lower_bound\": " << solver.certified_lower_bound(opt.trace_bound);
    f << ", \"trace_bound_scale\": " << opt.trace_bound_scale;
    if (opt.trace_bound_scale >= 0)
        f << ", \"certified_lower_bound_scaled\": " << solver.certified_lower_bound(size_scaled_bounds(solver, opt.trace_bound_scale));
    f << "}" << std::endl;
}

// Writes a device vector as text, one value per line, with full precision.
static void write_device_vector(const std::string &filename, const DeviceDenseVector<double> &v)
{
    std::vector<double> h(v.size);
    CHECK_CUDA(cudaMemcpy(h.data(), v.vals, sizeof(double) * v.size, cudaMemcpyDeviceToHost));
    std::ofstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("could not open " + filename + " for writing");
    f << std::setprecision(17);
    for (double x : h)
        f << x << "\n";
}

static void save_solution(const Options &opt, const SDPSolver &solver)
{
    std::filesystem::create_directories(opt.save_dir);
    write_device_vector(opt.save_dir + "/X.txt", solver.X);
    write_device_vector(opt.save_dir + "/y.txt", solver.y);
    write_device_vector(opt.save_dir + "/S.txt", solver.S);
    std::ofstream f(opt.save_dir + "/certificate.txt");
    f << std::setprecision(17) << "# b_y = " << solver.dobj << "\n"
      << "# block type size lambda_min c_b   (C - A^T y, original units; lower bound = b_y + sum_b R_b c_b)\n";
    for (size_t b = 0; b < solver.blk_info.size(); b++)
        f << b << " " << solver.blk_info[b].type << " " << solver.blk_info[b].size << " "
          << solver.dual_block_min[b] << " " << solver.dual_cone_coef[b] << "\n";
}

// One CSV line per external validation of the last solve().
static void write_validation_history(const Options &opt, const SDPSolver &solver)
{
    std::ofstream f(opt.validation_history_file);
    if (!f.is_open())
    {
        std::cerr << "WARNING: could not open " << opt.validation_history_file << " for the validation history" << std::endl;
        return;
    }
    f << std::setprecision(12)
      << "iter,time_s,sigma,trigger,checkpoint,internal_kkt,internal_errRp,internal_errRd,internal_relgap,primal_res,dual_res,"
         "relgap,kkt_full,pobj,dobj,X_cone_violation,X_cone_violation_normb,dual_cone_violation,min_eig_X,min_eig_Z,"
         "X_negative_blocks,finite,validated,failed,validation_s,callback_s,S_cone_violation,min_eig_S,"
         "dimacs1,dimacs2,dimacs3,dimacs4,dimacs5,dimacs6,max_abs_dimacs,err6_defined,err4_z,err6_z\n";
    for (const ValidationRecord &v : solver.validation_history)
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
}

// Writes a host vector as text, one value per line, with full precision; returns false if the file cannot be
// written completely.
static bool write_host_vector(const std::string &filename, const std::vector<double> &v)
{
    std::ofstream f(filename);
    f << std::setprecision(17);
    for (double x : v)
        f << x << "\n";
    f.close();
    return !f.fail();
}

// One line per sigma change of the adaptive rules (SDPSolver::sigma_log).
static void write_sigma_log(const Options &opt, const SDPSolver &solver)
{
    std::ofstream f(opt.sigma_log_file);
    if (!f.is_open())
    {
        std::cerr << "WARNING: could not open " << opt.sigma_log_file << " for the sigma log" << std::endl;
        return;
    }
    f << std::setprecision(12) << "iter,old_sigma,new_sigma,rule,errRp,errRd,relgap,feasratio,prim_win,dual_win\n";
    for (const SDPSolver::SigmaUpdate &u : solver.sigma_log)
        f << u.iter << "," << u.old_sigma << "," << u.new_sigma << "," << u.rule << "," << u.errRp << "," << u.errRd << ","
          << u.relgap << "," << u.feasratio << "," << u.prim_win << "," << u.dual_win << "\n";
}

static void write_history(const Options &opt, const SDPSolver &solver)
{
    std::ofstream f(opt.history_file);
    if (!f.is_open())
    {
        std::cerr << "WARNING: could not open " << opt.history_file << " for the history" << std::endl;
        return;
    }
    // the info arrays accumulate over solve() calls: the last info_iter_num entries belong to the last one
    const size_t first = solver.info_pobj_arr.size() - solver.info_iter_num;
    // phase: the update of that iteration (sgs for iter < switch_admm, admm otherwise); tau: its step length
    f << std::setprecision(10) << "iter,time_s,pobj,dobj,errRp,errRd,relgap,sig,tau,phase\n";
    for (int k = 0; k < solver.info_iter_num; k++)
        f << k + 1 << "," << solver.info_time_arr[first + k] << "," << solver.info_pobj_arr[first + k] << ","
          << solver.info_dobj_arr[first + k] << "," << solver.info_errRp_arr[first + k] << ","
          << solver.info_errRd_arr[first + k] << "," << solver.info_relgap_arr[first + k] << ","
          << solver.info_sig_arr[first + k] << "," << solver.info_tau_arr[first + k] << ","
          << (k + 1 < opt.switch_admm ? "sgs" : "admm") << "\n";
}

int main(int argc, char *argv[])
{
    Options opt;
    if (!parse_options(argc, argv, opt))
        return 1;

    try
    {
        const auto t_load = std::chrono::steady_clock::now();
        Problem problem;
        problem.from_txt(opt.prefix, opt.warm_start);
        opt.t_load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
        if (!opt.warm_start_dir.empty())
        {
            read_dense_vector_data(opt.warm_start_dir + "/X.txt", problem.X_vals);
            read_dense_vector_data(opt.warm_start_dir + "/y.txt", problem.y_vals);
            read_dense_vector_data(opt.warm_start_dir + "/S.txt", problem.S_vals);
            if ((int)problem.X_vals.size() != problem.vec_len || (int)problem.S_vals.size() != problem.vec_len ||
                (int)problem.y_vals.size() != problem.con_num)
                throw std::invalid_argument("warm start in " + opt.warm_start_dir + " does not match the problem sizes");
        }

        // extract the second element of each tuple in blk_vals
        std::vector<char> blk_sizes;
        std::vector<int> blk_vals;
        for (const auto &blk : problem.blk_vals)
        {
            blk_sizes.push_back(std::get<0>(blk));
            blk_vals.push_back(std::get<1>(blk));
        }

        const auto t_cuda = std::chrono::steady_clock::now();
        CHECK_CUDA(cudaFree(0)); // CUDA context creation, timed separately from init()
        opt.t_cuda_init_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_cuda).count();

        SDPSolver solver;
        solver.init(
            opt.eig_stream_num_per_gpu,
            problem.vec_len, problem.con_num,
            problem.At_csc_col_ptrs.data(), problem.At_csc_row_ids.data(), problem.At_csc_vals.data(), problem.At_nnz,
            problem.b_indices.data(), problem.b_vals.data(), problem.b_nnz,
            problem.C_indices.data(), problem.C_vals.data(), problem.C_nnz,
            blk_sizes.data(), blk_vals.data(), problem.mat_num,
            ProjectionMethod::EIG_FP64,
            ProjectionMethod::EIG_FP64,
            // no warm start unless the vectors were read: init() treats any non-null pointer as a warm start
            problem.X_vals.empty() ? nullptr : problem.X_vals.data(),
            problem.y_vals.empty() ? nullptr : problem.y_vals.data(),
            problem.S_vals.empty() ? nullptr : problem.S_vals.data(),
            opt.sig, opt.use_lobpcg);

        if (opt.time_limit > 0)
            solver.time_limit = opt.time_limit;
        parse_sigma_policy(opt.sigma_policy, solver.sigma_policy); // validated in parse_options
        solver.validation.interval = opt.validate_interval;
        solver.validation.tol = ValidationTolerances::all(opt.validate_tol > 0 ? opt.validate_tol : opt.stop_tol);
        solver.validation.threads = opt.validate_threads;
        solver.validation.checkpoint_iterations = opt.checkpoint_iters;
        solver.validation.strict_dimacs_tol = opt.strict_dimacs_tol;
        {
            // block kinds of the original data for the DIMACS norms, written by the SDPA converter (optional)
            std::ifstream kf(opt.prefix + (opt.prefix.back() == '/' ? "" : "/") + "dimacs_blocks.txt");
            char kind;
            int size;
            while (kf >> kind >> size)
                solver.dimacs_kinds.push_back(kind);
            if (!solver.dimacs_kinds.empty() && (int)solver.dimacs_kinds.size() != problem.mat_num)
                throw std::invalid_argument("dimacs_blocks.txt lists " + std::to_string(solver.dimacs_kinds.size()) + " blocks, blk.txt " +
                                            std::to_string(problem.mat_num));
            if (!solver.dimacs_kinds.empty())
                std::cout << " DIMACS block kinds from dimacs_blocks.txt (" << solver.dimacs_kinds.size() << " blocks)" << std::endl;
        }
        if (!opt.checkpoint_dir.empty() || !opt.save_final_dir.empty())
            solver.validation.on_validation = [&opt](const ValidationRecord &rec, const std::vector<double> &X,
                                                     const std::vector<double> &y, const std::vector<double> &S)
            {
                // the final snapshot of a solve that ends at a safety limit (the returned iterate is the best one)
                if (!opt.save_final_dir.empty() && (rec.trigger == "time_limit" || rec.trigger == "max_iter"))
                {
                    std::error_code ec;
                    std::filesystem::create_directories(opt.save_final_dir, ec);
                    if (ec || !write_host_vector(opt.save_final_dir + "/X.txt", X) || !write_host_vector(opt.save_final_dir + "/y.txt", y) ||
                        !write_host_vector(opt.save_final_dir + "/S.txt", S))
                        throw std::runtime_error("could not write the final iterate to " + opt.save_final_dir);
                    std::ofstream(opt.save_final_dir + "/iteration.txt") << rec.iter << "\n";
                }
                // the requested checkpoints (the validated iterate is the returned solution, see --save-solution)
                if (!rec.checkpoint || opt.checkpoint_dir.empty())
                    return;
                const std::string dir = opt.checkpoint_dir + "/iter_" + std::to_string(rec.iter);
                std::error_code ec;
                std::filesystem::create_directories(dir, ec);
                if (ec || !write_host_vector(dir + "/X.txt", X) || !write_host_vector(dir + "/y.txt", y) ||
                    !write_host_vector(dir + "/S.txt", S))
                    // reported and counted by the solver (callback_failures), which continues
                    throw std::runtime_error("could not write the checkpoint " + dir + (ec ? ": " + ec.message() : ""));
            };

        // sGS-ADMM for iter < switch_admm, then standard ADMM
        solver.solve(
            opt.max_iter, opt.stop_tol,
            opt.sig_update_threshold, opt.sig_update_stage_1, opt.sig_update_stage_2,
            opt.switch_admm, opt.switch_proj_max_iter, opt.switch_proj_tol, opt.sigscale);

        if (opt.trace_bound >= 0)
            printf(" certified lower bound (trace bound R = %g): %.10e\n", opt.trace_bound, solver.certified_lower_bound(opt.trace_bound));
        if (opt.trace_bound_scale >= 0)
            printf(" certified lower bound (R_b = %g * n_b, valid if tr(X_b) <= R_b for every block): %.10e\n", opt.trace_bound_scale,
                   solver.certified_lower_bound(size_scaled_bounds(solver, opt.trace_bound_scale)));
        if (!opt.summary_file.empty())
            write_summary(opt, problem, solver);
        if (!opt.history_file.empty())
            write_history(opt, solver);
        if (!opt.validation_history_file.empty())
            write_validation_history(opt, solver);
        if (!opt.sigma_log_file.empty())
            write_sigma_log(opt, solver);
        if (!opt.save_dir.empty())
            save_solution(opt, solver);
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
