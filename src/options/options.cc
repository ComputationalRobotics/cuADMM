/*

    options.cc

    The option table (single source of truth), the path-based setter and getter, validation, documentation.

*/

#include "cuadmm/options.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace cuadmm
{

namespace
{

template <typename T, std::size_t N>
const char *enum_name(T v, const char *const (&names)[N])
{
    const std::size_t i = static_cast<std::size_t>(v);
    return i < N ? names[i] : "?";
}

template <typename T, std::size_t N>
T enum_parse(const std::string &s, const char *const (&names)[N], const char *what)
{
    for (std::size_t i = 0; i < N; i++)
        if (s == names[i])
            return static_cast<T>(i);
    std::string allowed;
    for (std::size_t i = 0; i < N; i++)
        allowed += std::string(i ? ", " : "") + names[i];
    throw std::invalid_argument(std::string("unknown ") + what + " '" + s + "' (allowed: " + allowed + ")");
}

const char *const kPolicyNames[] = {"fixed", "balance_freeze"};
const char *const kBackendNames[] = {"auto", "cholmod", "cudss", "dense"};
const char *const kWarmupNames[] = {"none", "composite_fp16"};
const char *const kFormatNames[] = {"auto", "h5", "sdpa", "sedumi", "sdpt3", "mosek", "txt"};
const char *const kSenseNames[] = {"min", "max"};

// textual helpers -------------------------------------------------------------------------------------------------

bool parse_bool(const std::string &s)
{
    if (s == "1" || s == "true" || s == "on" || s == "yes")
        return true;
    if (s == "0" || s == "false" || s == "off" || s == "no")
        return false;
    throw std::invalid_argument("not a boolean: '" + s + "' (use 0/1, true/false)");
}

int parse_int(const std::string &s)
{
    std::size_t used = 0;
    long long v;
    try
    {
        v = std::stoll(s, &used);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument("not an integer: '" + s + "'");
    }
    if (used != s.size() || v > std::numeric_limits<int>::max() || v < std::numeric_limits<int>::min())
        throw std::invalid_argument("not an integer in range: '" + s + "'");
    return static_cast<int>(v);
}

double parse_double(const std::string &s)
{
    if (s == "inf" || s == "none")
        return std::numeric_limits<double>::infinity();
    std::size_t used = 0;
    double v;
    try
    {
        v = std::stod(s, &used);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument("not a number: '" + s + "'");
    }
    if (used != s.size())
        throw std::invalid_argument("not a number: '" + s + "'");
    return v;
}

std::vector<std::string> split(const std::string &s, char sep)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : s)
    {
        if (c == sep)
        {
            out.push_back(cur);
            cur.clear();
        }
        else
            cur += c;
    }
    if (!s.empty())
        out.push_back(cur);
    return out;
}

// shortest representation that round-trips exactly (std::to_chars): 1.618 prints as 1.618, not 1.6180000000000001
std::string fmt(double v)
{
    if (std::isinf(v))
        return v > 0 ? "inf" : "-inf";
    if (std::isnan(v))
        return "nan";
    char buf[64];
    const std::to_chars_result r = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, r.ptr);
}

std::string fmt(int v) { return std::to_string(v); }
std::string fmt(bool v) { return v ? "true" : "false"; }

std::string json_string(const std::string &s)
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

} // namespace

// enum names ----------------------------------------------------------------------------------------------------

const char *name_of(PenaltyPolicyKind v) { return enum_name(v, kPolicyNames); }
const char *name_of(LinsysBackend v) { return enum_name(v, kBackendNames); }
const char *name_of(WarmupKind v) { return enum_name(v, kWarmupNames); }
const char *name_of(IoFormat v) { return enum_name(v, kFormatNames); }
const char *name_of(Sense v) { return enum_name(v, kSenseNames); }
PenaltyPolicyKind parse_penalty_policy(const std::string &s) { return enum_parse<PenaltyPolicyKind>(s, kPolicyNames, "penalty policy"); }
LinsysBackend parse_linsys_backend(const std::string &s) { return enum_parse<LinsysBackend>(s, kBackendNames, "linear-system backend"); }
WarmupKind parse_warmup(const std::string &s) { return enum_parse<WarmupKind>(s, kWarmupNames, "PSD warm-up"); }
IoFormat parse_io_format(const std::string &s) { return enum_parse<IoFormat>(s, kFormatNames, "input format"); }
Sense parse_sense(const std::string &s)
{
    if (s == "minimize" || s == "minimise")
        return Sense::Minimize;
    if (s == "maximize" || s == "maximise")
        return Sense::Maximize;
    return enum_parse<Sense>(s, kSenseNames, "objective sense");
}

// the table ----------------------------------------------------------------------------------------------------

const std::vector<OptionSpec> &option_table()
{
    static const std::vector<OptionSpec> table = {
        // algorithm
        {"algorithm.tau", "--tau", "double", "", "1.618", "multiplier step length, in (0, 2)"},
        {"algorithm.max_iter", "--max-iter", "int", "", "1000000", "maximum number of iterations"},
        {"algorithm.time_limit", "--time-limit", "double", "", "inf", "seconds of iterations after which the solve stops, returning the best iterate"},
        {"algorithm.tol", "--tol", "double", "", "1e-4", "internal stopping tolerance on eta = max(eta_p, eta_d, eta_g)"},
        // penalty
        {"penalty.policy", "--policy", "enum", "fixed, balance_freeze", "balance_freeze", "balance_freeze: sigma follows the residual vote at the checks of the schedule 200:10,1000:25,5000:50,10000:200 (up to iteration: period; one check per 1000 iterations after that) and is frozen at penalty.freeze_iteration; fixed: sigma = penalty.sigma0 throughout"},
        {"penalty.sigma0", "--sigma0", "double", "", "100", "initial sigma (scaled units)"},
        {"penalty.sigma_min", "--sigma-min", "double", "", "1e-3", "lower bound of the updates (balance_freeze)"},
        {"penalty.sigma_max", "--sigma-max", "double", "", "1e3", "upper bound of the updates (balance_freeze)"},
        {"penalty.factor", "--factor", "double", "", "1.35", "step of an update: sigma * factor up, sigma / factor down (balance_freeze)"},
        {"penalty.freeze_iteration", "--freeze-iteration", "int", "", "10000", "sigma keeps its current value from this iteration on (balance_freeze)"},
        // linsys
        {"linsys.backend", "--backend", "enum", "auto, cholmod, cudss, dense", "auto", "normal-equation backend: auto = cholmod (CHOLMOD on the host; a dense factor on the GPU when it fits), dense (cuSOLVER Cholesky on the device), cudss (cuDSS, when compiled in)"},
        {"linsys.regularization", "--regularization", "double", "", "1e-15", "added to the diagonal of AA^T"},
        {"linsys.presolve", "--presolve", "bool", "", "false", "remove dependent constraint rows before the factorization"},
        {"linsys.memory_fraction", "--memory-fraction", "double", "", "0.25", "budget of a dense factor as a fraction of free memory"},
        // certificate
        {"certificate.block_bounds", "--block-bounds", "double_list", "", "none", "per-block bounds R_b of the certified lower bound (comma-separated, one per block; a negative entry: no bound for that block, the bound is then unavailable)"},
        {"certificate.trace_bound", "--trace-bound", "double", "", "none", "the same bound R for every block (paper eq. 30)"},
        {"certificate.trace_bound_scale", "--trace-bound-scale", "double", "", "none", "R_b = f n_b for PSD blocks of size n_b, f for the entries of l and u blocks"},
        // validation
        {"validation.interval", "--validation-interval", "int", "", "0", "validation-aware stopping: validate every n iterations on the CPU with the original data and stop only on a validated snapshot (0: off)"},
        {"validation.tol", "--validation-tol", "double", "", "algorithm.tol", "tolerance of every external criterion"},
        {"validation.threads", "--validation-threads", "int", "", "1", "CPU threads of the external validation"},
        {"validation.strict_dimacs_tol", "--strict-dimacs-tol", "double", "", "0", "after the validated criteria, continue until max |DIMACS error| <= x (0: stop at the criteria)"},
        {"validation.checkpoint_iterations", "--checkpoint-iterations", "int_list", "", "none", "iterations with an extra validation snapshot"},
        {"validation.checkpoint_dir", "--checkpoint-dir", "string", "", "none", "save X.txt, y.txt, S.txt of those snapshots in <dir>/iter_<k>"},
        {"validation.final_save_dir", "--final-save-dir", "string", "", "none", "save the final snapshot of a solve that ends at a safety limit"},
        {"validation.history_file", "--validation-history", "string", "", "none", "one CSV line per external validation"},
        // io
        {"io.format", "--format", "enum", "auto, h5, sdpa, sedumi, sdpt3, mosek, txt", "auto", "input format (auto: from the extension and the .mat variables)"},
        {"io.sense", "--sense", "enum", "min, max", "from the file", "objective sense (required for MOSEK files without a sense variable)"},
        {"io.simplify", "--simplify", "bool", "", "true", "1x1 and diagonal-only PSD blocks become l blocks (reported); never for TXT input"},
        {"io.warm_start_dir", "--warm-start-dir", "string", "", "none", "read X.txt, y.txt, S.txt (dense, one value per line) from this directory"},
        {"io.warm_start", "--warm-start", "bool", "", "false", "read X.txt, y.txt, S.txt from the problem directory (TXT input)"},
        {"io.save_dir", "--save-dir", "string", "", "none", "write the returned X, y, S (and certificate.txt) in the input's convention"},
        {"io.save_format", "--save-format", "string", "", "input's", "convention of the written solution: txt, h5, sdpa, sedumi, sdpt3, mosek (default: the input's)"},
        // output
        {"output.verbosity", "--verbosity", "int", "", "1", "0: quiet, 1: the iteration table, 2: details"},
        {"output.history", "--history", "string", "", "none", "per-iteration CSV (17 significant digits)"},
        {"output.summary", "--summary", "string", "", "none", "append a one-line JSON summary of the run"},
        {"output.sigma_log", "--sigma-log", "string", "", "none", "one CSV line per sigma change"},
        {"output.tag", "--tag", "string", "", "none", "label copied to the summary"},
        // psd
        {"psd.small_max", "--psd-small-max", "int", "", "32", "largest small PSD block (batched Jacobi; at most 32)"},
        {"psd.medium_max", "--psd-medium-max", "int", "", "1000", "largest medium PSD block (batched or streamed EVD); the larger blocks form the large tier"},
        {"psd.medium_batch_min", "--psd-medium-batch-min", "int", "", "2", "medium size groups with at least this many blocks use the batched EVD (0: never)"},
        {"psd.medium_streams", "--psd-medium-streams", "int", "", "15", "CUDA streams of the medium eigendecompositions (>= 1)"},
        {"psd.large.memory_budget", "--psd-large-memory-budget", "double", "", "0.9", "fraction of free memory for the large tier"},
        {"psd.large.warmup", "--psd-warmup", "enum", "none, composite_fp16", "composite_fp16", "low-precision warm-up of the large blocks of at least warmup.min_size: the FP16 composite filter until eta < switch_tol"},
        {"psd.large.warmup.min_size", "--psd-warmup-min-size", "int", "", "5000", "smallest large block that gets the warm-up"},
        {"psd.large.warmup.switch_tol", "--psd-warmup-switch-tol", "double", "", "1e-2", "the warm-up ends at the first iteration with eta below this"},
        {"psd.large.warmup.switch_iter", "--psd-warmup-switch-iter", "int", "", "5000", "or after this many iterations"},
        {"psd.medium.max_batch", "--psd-medium-max-batch", "int", "", "0", "blocks per batched EVD call of a medium size group (0: the whole group)"},
        {"psd.lowrank.enabled", "--psd-lowrank-enabled", "bool", "", "true", "LOBPCG for large blocks of low positive or negative rank"},
        {"psd.lowrank.start_iter", "--psd-lowrank-start-iter", "int", "", "2", "first iteration of the low-rank analysis for blocks without a warm-up (warm-up blocks: the end of the warm-up)"},
        {"psd.lowrank.reevaluate", "--psd-lowrank-reevaluate", "int", "", "100", "exact EVD with rank analysis every this many iterations"},
        {"psd.lowrank.ratio", "--psd-lowrank-ratio", "double", "", "0.05", "low-rank threshold as a fraction of n (LOBPCG while 0 < rank < ratio * n); sizes the warm start, ceil(n * ratio * relaxation) columns"},
        {"psd.lowrank.relaxation", "--psd-lowrank-relaxation", "double", "", "1.2", "eigenpairs requested: ceil(relaxation * rank)"},
        {"psd.lowrank.max_iter", "--psd-lowrank-max-iter", "int", "", "20", "LOBPCG iterations"},
        {"psd.lowrank.tol", "--psd-lowrank-tol", "double", "", "1e-5", "LOBPCG convergence tolerance (absolute)"},
        {"psd.lowrank.rank_rtol", "--psd-lowrank-rank-rtol", "double", "", "1e-8", "rank tolerance relative to max |lambda|"},
        {"psd.lowrank.rank_stop_factor", "--psd-lowrank-rank-stop-factor", "double", "", "1e-3", "rank tolerance floor as a fraction of the stopping tolerance"},
        {"psd.composite.margin", "--psd-composite-margin", "double", "", "1.05", "safety margin on the Lanczos estimate of the composite filter"},
        {"psd.composite.lanczos_steps", "--psd-composite-lanczos-steps", "int", "", "20", "Lanczos steps of the norm estimate"},
        {"psd.composite.norm_slack", "--psd-composite-norm-slack", "double", "", "1e-2", "norm test slack of the composite result"},
        {"psd.trace", "--psd-trace", "string", "", "none", "one JSON line per rank analysis of a large PSD block (eigenvalue counts of the projection input, mode), LOBPCG and composite rejection and warm-up switch, and per-block totals at the end"},
        // misc
        {"seed", "--seed", "int", "", "0", "seed of the deterministic generators (the Lanczos start vectors)"},
        {"gpu.device", "--device", "int", "", "0", "CUDA device of the solver"},
    };
    return table;
}

const OptionSpec *find_flag(const std::string &flag)
{
    for (const OptionSpec &spec : option_table())
        if (spec.flag == flag)
            return &spec;
    return nullptr;
}

namespace
{
const OptionSpec *find_path(const std::string &path)
{
    for (const OptionSpec &spec : option_table())
        if (spec.path == path)
            return &spec;
    return nullptr;
}

std::vector<int> parse_int_list(const std::string &s)
{
    std::vector<int> out;
    if (s == "none" || s.empty())
        return out;
    for (const std::string &t : split(s, ','))
        out.push_back(parse_int(t));
    return out;
}

std::vector<double> parse_double_list(const std::string &s)
{
    std::vector<double> out;
    if (s == "none" || s.empty())
        return out;
    for (const std::string &t : split(s, ','))
        out.push_back(parse_double(t));
    return out;
}

template <typename T>
std::string fmt_list(const std::vector<T> &v)
{
    if (v.empty())
        return "none";
    std::string out;
    for (std::size_t i = 0; i < v.size(); i++)
        out += (i ? "," : "") + fmt(v[i]);
    return out;
}

std::string fmt_optional_double(double v)
{
    return v < 0 ? "none" : fmt(v);
}
} // namespace

void set_option(SolverOptions &o, const std::string &path, const std::string &value)
{
    const OptionSpec *spec = find_path(path);
    if (!spec)
        throw std::invalid_argument("unknown option '" + path + "'");
    try
    {
        const std::string &p = path;
        if (p == "algorithm.tau") o.algorithm.tau = parse_double(value);
        else if (p == "algorithm.max_iter") o.algorithm.max_iter = parse_int(value);
        else if (p == "algorithm.time_limit") o.algorithm.time_limit = parse_double(value);
        else if (p == "algorithm.tol") o.algorithm.tol = parse_double(value);
        else if (p == "penalty.policy") o.penalty.policy = parse_penalty_policy(value);
        else if (p == "penalty.sigma0") o.penalty.sigma0 = parse_double(value);
        else if (p == "penalty.sigma_min") o.penalty.sigma_min = parse_double(value);
        else if (p == "penalty.sigma_max") o.penalty.sigma_max = parse_double(value);
        else if (p == "penalty.factor") o.penalty.factor = parse_double(value);
        else if (p == "penalty.freeze_iteration") o.penalty.freeze_iteration = parse_int(value);
        else if (p == "linsys.backend") o.linsys.backend = parse_linsys_backend(value);
        else if (p == "linsys.regularization") o.linsys.regularization = parse_double(value);
        else if (p == "linsys.presolve") o.linsys.presolve = parse_bool(value);
        else if (p == "linsys.memory_fraction") o.linsys.memory_fraction = parse_double(value);
        else if (p == "certificate.block_bounds") o.certificate.block_bounds = parse_double_list(value);
        else if (p == "certificate.trace_bound") o.certificate.trace_bound = value == "none" ? -1.0 : parse_double(value);
        else if (p == "certificate.trace_bound_scale") o.certificate.trace_bound_scale = value == "none" ? -1.0 : parse_double(value);
        else if (p == "validation.interval") o.validation.interval = parse_int(value);
        else if (p == "validation.tol") o.validation.tol = value == "algorithm.tol" ? -1.0 : parse_double(value);
        else if (p == "validation.threads") o.validation.threads = parse_int(value);
        else if (p == "validation.strict_dimacs_tol") o.validation.strict_dimacs_tol = parse_double(value);
        else if (p == "validation.checkpoint_iterations") o.validation.checkpoint_iterations = parse_int_list(value);
        else if (p == "validation.checkpoint_dir") o.validation.checkpoint_dir = value == "none" ? "" : value;
        else if (p == "validation.final_save_dir") o.validation.final_save_dir = value == "none" ? "" : value;
        else if (p == "validation.history_file") o.validation.history_file = value == "none" ? "" : value;
        else if (p == "io.format") o.io.format = parse_io_format(value);
        else if (p == "io.sense")
        {
            if (value == "from the file" || value == "file")
                o.io.sense.reset();
            else
                o.io.sense = parse_sense(value);
        }
        else if (p == "io.simplify") o.io.simplify = parse_bool(value);
        else if (p == "io.warm_start_dir") o.io.warm_start_dir = value == "none" ? "" : value;
        else if (p == "io.warm_start") o.io.warm_start = parse_bool(value);
        else if (p == "io.save_dir") o.io.save_dir = value == "none" ? "" : value;
        else if (p == "io.save_format") o.io.save_format = value == "input's" ? "" : value;
        else if (p == "output.verbosity") o.output.verbosity = parse_int(value);
        else if (p == "output.history") o.output.history = value == "none" ? "" : value;
        else if (p == "output.summary") o.output.summary = value == "none" ? "" : value;
        else if (p == "output.sigma_log") o.output.sigma_log = value == "none" ? "" : value;
        else if (p == "output.tag") o.output.tag = value == "none" ? "" : value;
        else if (p == "psd.small_max") o.psd.small_max = parse_int(value);
        else if (p == "psd.medium_max") o.psd.medium_max = parse_int(value);
        else if (p == "psd.medium_batch_min") o.psd.medium_batch_min = parse_int(value);
        else if (p == "psd.medium_streams") o.psd.medium_streams = parse_int(value);
        else if (p == "psd.large.memory_budget") o.psd.large_memory_budget = parse_double(value);
        else if (p == "psd.large.warmup") o.psd.warmup = parse_warmup(value);
        else if (p == "psd.large.warmup.min_size") o.psd.warmup_min_size = parse_int(value);
        else if (p == "psd.large.warmup.switch_tol") o.psd.warmup_switch_tol = parse_double(value);
        else if (p == "psd.large.warmup.switch_iter") o.psd.warmup_switch_iter = parse_int(value);
        else if (p == "psd.medium.max_batch") o.psd.medium_max_batch = parse_int(value);
        else if (p == "psd.lowrank.enabled") o.psd.lowrank_enabled = parse_bool(value);
        else if (p == "psd.lowrank.start_iter") o.psd.lowrank_start_iter = parse_int(value);
        else if (p == "psd.lowrank.reevaluate") o.psd.lowrank_reevaluate = parse_int(value);
        else if (p == "psd.lowrank.ratio") o.psd.lowrank_ratio = parse_double(value);
        else if (p == "psd.trace") o.psd.trace = value == "none" ? "" : value;
        else if (p == "psd.lowrank.relaxation") o.psd.lowrank_relaxation = parse_double(value);
        else if (p == "psd.lowrank.max_iter") o.psd.lowrank_max_iter = parse_int(value);
        else if (p == "psd.lowrank.tol") o.psd.lowrank_tol = parse_double(value);
        else if (p == "psd.lowrank.rank_rtol") o.psd.lowrank_rank_rtol = parse_double(value);
        else if (p == "psd.lowrank.rank_stop_factor") o.psd.lowrank_rank_stop_factor = parse_double(value);
        else if (p == "psd.composite.margin") o.psd.composite_margin = parse_double(value);
        else if (p == "psd.composite.lanczos_steps") o.psd.composite_lanczos_steps = parse_int(value);
        else if (p == "psd.composite.norm_slack") o.psd.composite_norm_slack = parse_double(value);
        else if (p == "seed")
        {
            const long long v = std::stoll(value);
            if (v < 0)
                throw std::invalid_argument("seed must be non-negative");
            o.seed = static_cast<std::uint64_t>(v);
        }
        else if (p == "gpu.device") o.gpu_device = parse_int(value);
        else
            throw std::logic_error("option '" + path + "' is in the table but has no setter");
    }
    catch (const std::invalid_argument &e)
    {
        throw std::invalid_argument("option " + path + ": " + e.what());
    }
}

std::string get_option(const SolverOptions &o, const std::string &p)
{
    if (!find_path(p))
        throw std::invalid_argument("unknown option '" + p + "'");
    if (p == "algorithm.tau") return fmt(o.algorithm.tau);
    if (p == "algorithm.max_iter") return fmt(o.algorithm.max_iter);
    if (p == "algorithm.time_limit") return fmt(o.algorithm.time_limit);
    if (p == "algorithm.tol") return fmt(o.algorithm.tol);
    if (p == "penalty.policy") return name_of(o.penalty.policy);
    if (p == "penalty.sigma0") return fmt(o.penalty.sigma0);
    if (p == "penalty.sigma_min") return fmt(o.penalty.sigma_min);
    if (p == "penalty.sigma_max") return fmt(o.penalty.sigma_max);
    if (p == "penalty.factor") return fmt(o.penalty.factor);
    if (p == "penalty.freeze_iteration") return fmt(o.penalty.freeze_iteration);
    if (p == "linsys.backend") return name_of(o.linsys.backend);
    if (p == "linsys.regularization") return fmt(o.linsys.regularization);
    if (p == "linsys.presolve") return fmt(o.linsys.presolve);
    if (p == "linsys.memory_fraction") return fmt(o.linsys.memory_fraction);
    if (p == "certificate.block_bounds") return fmt_list(o.certificate.block_bounds);
    if (p == "certificate.trace_bound") return fmt_optional_double(o.certificate.trace_bound);
    if (p == "certificate.trace_bound_scale") return fmt_optional_double(o.certificate.trace_bound_scale);
    if (p == "validation.interval") return fmt(o.validation.interval);
    if (p == "validation.tol") return o.validation.tol < 0 ? "algorithm.tol" : fmt(o.validation.tol);
    if (p == "validation.threads") return fmt(o.validation.threads);
    if (p == "validation.strict_dimacs_tol") return fmt(o.validation.strict_dimacs_tol);
    if (p == "validation.checkpoint_iterations") return fmt_list(o.validation.checkpoint_iterations);
    if (p == "validation.checkpoint_dir") return o.validation.checkpoint_dir.empty() ? "none" : o.validation.checkpoint_dir;
    if (p == "validation.final_save_dir") return o.validation.final_save_dir.empty() ? "none" : o.validation.final_save_dir;
    if (p == "validation.history_file") return o.validation.history_file.empty() ? "none" : o.validation.history_file;
    if (p == "io.format") return name_of(o.io.format);
    if (p == "io.sense") return o.io.sense ? name_of(*o.io.sense) : "from the file";
    if (p == "io.simplify") return fmt(o.io.simplify);
    if (p == "io.warm_start_dir") return o.io.warm_start_dir.empty() ? "none" : o.io.warm_start_dir;
    if (p == "io.warm_start") return fmt(o.io.warm_start);
    if (p == "io.save_dir") return o.io.save_dir.empty() ? "none" : o.io.save_dir;
    if (p == "io.save_format") return o.io.save_format.empty() ? "input's" : o.io.save_format;
    if (p == "output.verbosity") return fmt(o.output.verbosity);
    if (p == "output.history") return o.output.history.empty() ? "none" : o.output.history;
    if (p == "output.summary") return o.output.summary.empty() ? "none" : o.output.summary;
    if (p == "output.sigma_log") return o.output.sigma_log.empty() ? "none" : o.output.sigma_log;
    if (p == "output.tag") return o.output.tag.empty() ? "none" : o.output.tag;
    if (p == "psd.small_max") return fmt(o.psd.small_max);
    if (p == "psd.medium_max") return fmt(o.psd.medium_max);
    if (p == "psd.medium_batch_min") return fmt(o.psd.medium_batch_min);
    if (p == "psd.medium_streams") return fmt(o.psd.medium_streams);
    if (p == "psd.large.memory_budget") return fmt(o.psd.large_memory_budget);
    if (p == "psd.large.warmup") return name_of(o.psd.warmup);
    if (p == "psd.large.warmup.min_size") return fmt(o.psd.warmup_min_size);
    if (p == "psd.large.warmup.switch_tol") return fmt(o.psd.warmup_switch_tol);
    if (p == "psd.large.warmup.switch_iter") return fmt(o.psd.warmup_switch_iter);
    if (p == "psd.medium.max_batch") return fmt(o.psd.medium_max_batch);
    if (p == "psd.lowrank.enabled") return fmt(o.psd.lowrank_enabled);
    if (p == "psd.lowrank.start_iter") return fmt(o.psd.lowrank_start_iter);
    if (p == "psd.lowrank.reevaluate") return fmt(o.psd.lowrank_reevaluate);
    if (p == "psd.lowrank.ratio") return fmt(o.psd.lowrank_ratio);
    if (p == "psd.trace") return o.psd.trace.empty() ? "none" : o.psd.trace;
    if (p == "psd.lowrank.relaxation") return fmt(o.psd.lowrank_relaxation);
    if (p == "psd.lowrank.max_iter") return fmt(o.psd.lowrank_max_iter);
    if (p == "psd.lowrank.tol") return fmt(o.psd.lowrank_tol);
    if (p == "psd.lowrank.rank_rtol") return fmt(o.psd.lowrank_rank_rtol);
    if (p == "psd.lowrank.rank_stop_factor") return fmt(o.psd.lowrank_rank_stop_factor);
    if (p == "psd.composite.margin") return fmt(o.psd.composite_margin);
    if (p == "psd.composite.lanczos_steps") return fmt(o.psd.composite_lanczos_steps);
    if (p == "psd.composite.norm_slack") return fmt(o.psd.composite_norm_slack);
    if (p == "seed") return std::to_string(o.seed);
    if (p == "gpu.device") return fmt(o.gpu_device);
    throw std::logic_error("option '" + p + "' is in the table but has no getter");
}

// validation --------------------------------------------------------------------------------------------------------

void SolverOptions::validate() const
{
    auto fail = [](const std::string &path, const std::string &why)
    { throw std::invalid_argument("option " + path + ": " + why); };

    if (!(algorithm.tau > 0.0 && algorithm.tau < 2.0))
        fail("algorithm.tau", "must be in (0, 2), got " + fmt(algorithm.tau));
    if (algorithm.max_iter < 0 || algorithm.max_iter > std::numeric_limits<int>::max() - 2)
        fail("algorithm.max_iter", "must be in [0, INT_MAX - 2]");
    if (!(algorithm.time_limit > 0.0))
        fail("algorithm.time_limit", "must be positive (inf: none)");
    if (!(algorithm.tol > 0.0) || !std::isfinite(algorithm.tol))
        fail("algorithm.tol", "must be a positive finite number");

    if (!(penalty.sigma0 > 0.0) || !std::isfinite(penalty.sigma0))
        fail("penalty.sigma0", "must be a positive finite number");
    if (!(penalty.sigma_min > 0.0) || !std::isfinite(penalty.sigma_min))
        fail("penalty.sigma_min", "must be a positive finite number");
    if (!(penalty.sigma_max >= penalty.sigma_min) || !std::isfinite(penalty.sigma_max))
        fail("penalty.sigma_max", "must be finite and not below penalty.sigma_min");
    if (!(penalty.factor > 1.0) || !std::isfinite(penalty.factor))
        fail("penalty.factor", "must be a finite number greater than 1");
    if (penalty.freeze_iteration < 0)
        fail("penalty.freeze_iteration", "must be non-negative");

    if (!(linsys.regularization >= 0.0) || linsys.regularization > 1e-3)
        fail("linsys.regularization", "must be in [0, 1e-3]");
    if (!(linsys.memory_fraction > 0.0 && linsys.memory_fraction <= 1.0))
        fail("linsys.memory_fraction", "must be in (0, 1]");

    if (certificate.trace_bound >= 0.0 && !std::isfinite(certificate.trace_bound))
        fail("certificate.trace_bound", "must be finite");
    if (certificate.trace_bound_scale >= 0.0 && !std::isfinite(certificate.trace_bound_scale))
        fail("certificate.trace_bound_scale", "must be finite");
    for (double r : certificate.block_bounds)
        if (std::isnan(r) || r == std::numeric_limits<double>::infinity())
            fail("certificate.block_bounds", "bounds must be finite (a negative entry means: no bound for that block)");

    if (validation.interval < 0)
        fail("validation.interval", "must be non-negative");
    if (validation.tol >= 0.0 && !(validation.tol > 0.0))
        fail("validation.tol", "must be positive");
    if (validation.threads < 1)
        fail("validation.threads", "must be at least 1");
    if (!(validation.strict_dimacs_tol >= 0.0) || !std::isfinite(validation.strict_dimacs_tol))
        fail("validation.strict_dimacs_tol", "must be a finite non-negative number");
    for (int c : validation.checkpoint_iterations)
        if (c <= 0)
            fail("validation.checkpoint_iterations", "iterations must be positive");
    if (validation.interval == 0 &&
        (validation.tol >= 0.0 || validation.threads != 1 || !validation.checkpoint_iterations.empty() || !validation.checkpoint_dir.empty() ||
         !validation.history_file.empty() || validation.strict_dimacs_tol != 0.0 || !validation.final_save_dir.empty()))
        fail("validation.interval", "validation.tol, .threads, .strict_dimacs_tol, .checkpoint_iterations, .checkpoint_dir, .final_save_dir and .history_file need validation.interval > 0");

    if (!io.save_format.empty() && io.save_format != "txt" && io.save_format != "h5" && io.save_format != "sdpa" && io.save_format != "sedumi" &&
        io.save_format != "sdpt3" && io.save_format != "mosek")
        fail("io.save_format", "must be one of txt, h5, sdpa, sedumi, sdpt3, mosek (got '" + io.save_format + "')");
    if (io.warm_start && !io.warm_start_dir.empty())
        fail("io.warm_start", "use either io.warm_start or io.warm_start_dir");

    if (output.verbosity < 0 || output.verbosity > 2)
        fail("output.verbosity", "must be 0, 1 or 2");

    if (psd.small_max < 1 || psd.small_max > 32)
        fail("psd.small_max", "must be in [1, 32]: the small tier is cusolverDnDsyevjBatched, which supports n <= 32");
    if (psd.medium_max <= psd.small_max || psd.medium_max > 46340)
        fail("psd.medium_max", "must be above psd.small_max and at most 46340 (n * n fits in an int)");
    if (psd.medium_batch_min < 0)
        fail("psd.medium_batch_min", "must be non-negative");
    if (psd.medium_streams < 1)
        fail("psd.medium_streams", "must be at least 1");
    if (psd.medium_max_batch < 0)
        fail("psd.medium.max_batch", "must be non-negative (0: the whole size group)");
    if (!(psd.large_memory_budget > 0.0 && psd.large_memory_budget <= 1.0))
        fail("psd.large.memory_budget", "must be in (0, 1]");
    if (psd.warmup_min_size < 1)
        fail("psd.large.warmup.min_size", "must be positive");
    if (!(psd.warmup_switch_tol >= 0.0))
        fail("psd.large.warmup.switch_tol", "must be non-negative");
    if (psd.warmup_switch_iter < 0)
        fail("psd.large.warmup.switch_iter", "must be non-negative");
    if (psd.lowrank_start_iter < 2)
        fail("psd.lowrank.start_iter", "must be at least 2 (the first iteration after the starting point)");
    if (psd.lowrank_reevaluate < 1)
        fail("psd.lowrank.reevaluate", "must be at least 1");
    if (!(psd.lowrank_ratio > 0.0 && psd.lowrank_ratio <= 1.0))
        fail("psd.lowrank.ratio", "must be in (0, 1]");
    if (!(psd.lowrank_relaxation >= 1.0))
        fail("psd.lowrank.relaxation", "must be at least 1");
    if (!(psd.lowrank_rank_rtol >= 0.0) || !(psd.lowrank_rank_stop_factor >= 0.0))
        fail("psd.lowrank.rank_rtol", "the rank tolerances must be non-negative");
    if (!(psd.lowrank_tol > 0.0) || psd.lowrank_max_iter < 1)
        fail("psd.lowrank.tol", "the LOBPCG tolerance must be positive and the iteration count at least 1");
    if (!(psd.composite_margin >= 1.0))
        fail("psd.composite.margin", "must be at least 1");
    if (psd.composite_lanczos_steps < 1)
        fail("psd.composite.lanczos_steps", "must be at least 1");
    if (!(psd.composite_norm_slack >= 0.0))
        fail("psd.composite.norm_slack", "must be non-negative");

    if (gpu_device < 0)
        fail("gpu.device", "must be non-negative");
}

// documentation ---------------------------------------------------------------------------------------------------

std::string option_table_markdown()
{
    std::ostringstream o;
    o << "# Options of cuADMM\n\nGenerated from the option table of `src/options/options.cc` (`cuadmm_exe --print-options markdown`). "
         "The command line, MATLAB and the C++ API use the same table; unknown names are errors.\n\n"
         "| Option | CLI flag | Type | Default | Meaning |\n|---|---|---|---|---|\n";
    for (const OptionSpec &s : option_table())
        o << "| `" << s.path << "` | `" << s.flag << "` | " << s.type << (s.values.empty() ? "" : " (" + s.values + ")") << " | `"
          << s.default_value << "` | " << s.help << " |\n";
    return o.str();
}

std::string option_table_help()
{
    std::ostringstream o;
    std::string group;
    for (const OptionSpec &s : option_table())
    {
        const std::string g = s.path.substr(0, s.path.find('.'));
        if (g != group)
        {
            group = g;
            o << "\n  [" << group << "]\n";
        }
        o << "  " << s.flag << " <" << s.type << ">" << (s.values.empty() ? "" : " {" + s.values + "}") << "  [" << s.default_value << "]\n      "
          << s.help << "\n";
    }
    return o.str();
}

// true when v is a JSON number: -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?
bool is_json_number(const std::string &v)
{
    std::size_t i = 0;
    const std::size_t n = v.size();
    if (i < n && v[i] == '-')
        i++;
    if (i >= n || !std::isdigit(static_cast<unsigned char>(v[i])))
        return false;
    if (v[i] == '0')
        i++;
    else
        while (i < n && std::isdigit(static_cast<unsigned char>(v[i])))
            i++;
    if (i < n && v[i] == '.')
    {
        i++;
        if (i >= n || !std::isdigit(static_cast<unsigned char>(v[i])))
            return false;
        while (i < n && std::isdigit(static_cast<unsigned char>(v[i])))
            i++;
    }
    if (i < n && (v[i] == 'e' || v[i] == 'E'))
    {
        i++;
        if (i < n && (v[i] == '+' || v[i] == '-'))
            i++;
        if (i >= n || !std::isdigit(static_cast<unsigned char>(v[i])))
            return false;
        while (i < n && std::isdigit(static_cast<unsigned char>(v[i])))
            i++;
    }
    return i == n;
}

std::string options_json(const SolverOptions &options)
{
    std::ostringstream o;
    o << "{";
    bool first = true;
    for (const OptionSpec &s : option_table())
    {
        const std::string v = get_option(options, s.path);
        o << (first ? "" : ", ") << json_string(s.path) << ": ";
        first = false;
        // numbers and booleans verbatim; every other textual value (the symbolic defaults "none", "inf",
        // "algorithm.tol", ...) as a JSON string, so that the summary is always valid JSON
        if ((s.type == "int" || s.type == "double") && is_json_number(v))
            o << v;
        else if (s.type == "bool" && (v == "true" || v == "false"))
            o << v;
        else
            o << json_string(v);
    }
    o << "}";
    return o.str();
}

} // namespace cuadmm
