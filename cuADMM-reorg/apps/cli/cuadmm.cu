/*

    cuadmm.cu

    The command-line front-end of cuADMM-reorg (plan Phase 2): every flag comes from the option table of
    include/cuadmm/options.h (`--print-options` lists it; today's flags are accepted as aliases for one release).

    usage: cuadmm_exe <problem> [--option value ...]
      <problem>: a legacy TXT directory (blk.txt, At.txt, b.txt, C.txt, con_num.txt), a native .h5 file, an SDPA
      .dat-s / .dat-s.gz file, or a SeDuMi / SDPT3 / MOSEK prob .mat file (plan Section 12; --formats lists what
      this build reads; --format forces one). The conversion report of the reader is printed and goes into the
      summary.

*/

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

#include "cuadmm/cuadmm.h"
#include "cuadmm/version.h"
#include "report/writers.h"

using namespace cuadmm;

namespace
{

void print_usage(const char *exe)
{
    std::string formats;
    for (const std::string &f : io::available_formats())
        formats += (formats.empty() ? "" : ", ") + f;
    std::cerr << "usage: " << exe << " <problem> [options]\n"
              << "  <problem>: a TXT directory (blk.txt, At.txt, b.txt, C.txt, con_num.txt), a native .h5 file, an SDPA .dat-s[.gz] file,\n"
              << "             or a SeDuMi / SDPT3 / MOSEK prob .mat file; formats compiled in: " << formats << "\n"
              << "options (--flag value; booleans take 0/1; every option of the table, with today's flags as aliases):\n"
              << option_table_help()
              << "\n  --help                     this text\n"
              << "  --print-options [markdown] the option table (plain or as a Markdown table)\n"
              << "  --formats                  the input formats compiled into this build\n"
              << "  --version                  the build commit\n";
}

struct Parsed
{
    std::string problem;
    SolverOptions options;
    std::set<std::string> given; // option paths set on the command line
};

// Parses the command line; throws std::invalid_argument on an error.
Parsed parse(int argc, char *argv[])
{
    Parsed p;
    if (argc < 2)
        throw std::invalid_argument("missing problem path");
    int i = 1;
    if (argv[1][0] != '-')
    {
        p.problem = argv[1];
        i = 2;
    }
    for (; i < argc; i++)
    {
        const std::string key = argv[i];
        if (key == "--warm-start") // a flag without a value in today's CLI
        {
            set_option(p.options, "io.warm_start", "1");
            p.given.insert("io.warm_start");
            continue;
        }
        const OptionSpec *spec = find_flag(key);
        if (!spec)
            throw std::invalid_argument("unknown option " + key + " (see --help)");
        if (i + 1 >= argc)
            throw std::invalid_argument("missing value for option " + key);
        set_option(p.options, spec->path, argv[++i]);
        p.given.insert(spec->path);
    }
    if (p.problem.empty())
        throw std::invalid_argument("missing problem path");
    // combinations that today's CLI rejects
    const bool sgs_count = p.given.count("algorithm.sgs_iterations") > 0;
    const bool switch_given = p.given.count("algorithm.scheme_switch_iteration") > 0;
    const bool scheme_given = p.given.count("algorithm.scheme") > 0;
    if (sgs_count && (switch_given || scheme_given))
        throw std::invalid_argument("--sgs-iterations cannot be combined with --scheme-switch-iteration/--switch-admm or --scheme/--algorithm");
    if (switch_given && scheme_given)
        throw std::invalid_argument("use either --scheme/--algorithm or --scheme-switch-iteration/--switch-admm, not both");
    if (p.options.algorithm.scheme == SchemeKind::SgsAdmm && p.given.count("algorithm.tau") == 0)
        p.options.algorithm.tau = 1.95; // the engine's sGS step length until Phase 4 (plan Phase 4 criterion)
    p.options.validate();
    return p;
}

} // namespace

int main(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help")
        {
            print_usage(argv[0]);
            return 1;
        }
        if (a == "--version")
        {
            std::cout << CUADMM_BUILD_PROJECT << " " << CUADMM_BUILD_COMMIT << (CUADMM_BUILD_DIRTY ? " (local changes)" : "") << std::endl;
            return 0;
        }
        if (a == "--formats")
        {
            for (const std::string &f : io::available_formats())
                std::cout << f << "\n";
            return 0;
        }
        if (a == "--print-options")
        {
            const bool markdown = i + 1 < argc && std::string(argv[i + 1]) == "markdown";
            std::cout << (markdown ? option_table_markdown() : option_table_help());
            return 0;
        }
    }
    Parsed p;
    try
    {
        p = parse(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    try
    {
        const SolverOptions &o = p.options;
        const auto t_load = std::chrono::steady_clock::now();
        ProblemData problem = io::read_problem(p.problem, o.io);
        WarmStart warm;
        bool has_warm = false;
        if (o.io.warm_start)
        {
            warm = io::read_warm_start(p.problem, problem);
            has_warm = true;
        }
        else if (!o.io.warm_start_dir.empty())
        {
            warm = io::read_warm_start(o.io.warm_start_dir, problem);
            has_warm = true;
        }
        const double t_load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
        if (o.output.verbosity > 0)
        {
            if (problem.source.format != "txt") // the TXT reader prints its own summary (as today)
                std::cout << "Loaded " << problem.source.format << " problem from " << problem.source.path << " in " << t_load_s << " s\n"
                          << "              vector length: " << problem.vec_len() << "\n"
                          << "      number of constraints: " << problem.num_constraints() << "\n"
                          << "           number of blocks: " << problem.cones.size() << "\n"
                          << "  number of non-zeros in At: " << problem.At.nnz() << "\n"
                          << "  number of non-zeros in  b: " << problem.b.nnz() << "\n"
                          << "  number of non-zeros in  C: " << problem.C.nnz() << "\n"
                          << "            objective sense: " << name_of(problem.sense) << ", offset " << problem.objective_offset << std::endl;
            for (const std::string &line : problem.source.report)
                std::cout << "  [io] " << line << std::endl;
            if (!problem.source.dimacs_kinds.empty())
                std::cout << " DIMACS block kinds of the original data (" << problem.source.dimacs_kinds.size() << " blocks)" << std::endl;
        }

        const auto t_cuda = std::chrono::steady_clock::now();
        if (cudaSetDevice(o.gpu_device) != cudaSuccess || cudaFree(0) != cudaSuccess) // CUDA context creation, timed separately
            throw std::runtime_error("could not initialize CUDA device " + std::to_string(o.gpu_device));
        const double t_cuda_init_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_cuda).count();

        Solver solver(o);
        solver.setup(problem);
        const SolveResult result = solver.solve(has_warm ? &warm : nullptr);

        if (o.certificate.trace_bound >= 0)
            printf(" certified lower bound (trace bound R = %g): %.10e\n", o.certificate.trace_bound, result.certified_lower_bound);
        if (o.certificate.trace_bound_scale >= 0)
            printf(" certified lower bound (R_b = %g * n_b, valid if tr(X_b) <= R_b for every block): %.10e\n", o.certificate.trace_bound_scale,
                   result.certified_lower_bound_scaled);
        if (result.certificate_conditional_on_free_bounds)
            printf(" (the problem has free blocks: the bound is valid only if their entries respect the bound applied to them, B24)\n");
        if (!o.output.summary.empty())
            report::append_summary_json(o.output.summary, p.problem, problem, o, result, t_load_s, t_cuda_init_s);
        if (!o.output.history.empty())
            report::write_history_csv(o.output.history, result);
        if (!o.validation.history_file.empty())
            report::write_validation_history_csv(o.validation.history_file, result);
        if (!o.output.sigma_log.empty())
            report::write_sigma_log_csv(o.output.sigma_log, result);
        if (!o.io.save_dir.empty())
            io::write_solution(o.io.save_dir, problem, result, o.io.save_format);
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
