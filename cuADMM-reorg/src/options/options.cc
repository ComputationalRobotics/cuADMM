/*

    options.cc

    The option table (single source of truth), the path-based setter and getter, validation, documentation.

*/

#include "cuadmm/options.h"

#include <algorithm>
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

const char *const kSchemeNames[] = {"admm", "sgs_admm"};
const char *const kPolicyNames[] = {"fixed", "legacy_adaptive", "residual_balancing", "balance_freeze"};
const char *const kStepScheduleNames[] = {"constant", "halve_on_reversal"};
const char *const kFreezeValueNames[] = {"last", "band_centre"};
const char *const kBackendNames[] = {"auto", "cholmod", "cudss", "dense"};
const char *const kDeterminismNames[] = {"strict", "fast"};
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

const char *name_of(SchemeKind v) { return enum_name(v, kSchemeNames); }
const char *name_of(PenaltyPolicyKind v) { return enum_name(v, kPolicyNames); }
const char *name_of(StepSchedule v) { return enum_name(v, kStepScheduleNames); }
const char *name_of(FreezeValue v) { return enum_name(v, kFreezeValueNames); }
const char *name_of(LinsysBackend v) { return enum_name(v, kBackendNames); }
const char *name_of(Determinism v) { return enum_name(v, kDeterminismNames); }
const char *name_of(WarmupKind v) { return enum_name(v, kWarmupNames); }
const char *name_of(IoFormat v) { return enum_name(v, kFormatNames); }
const char *name_of(Sense v) { return enum_name(v, kSenseNames); }
SchemeKind parse_scheme(const std::string &s) { return enum_parse<SchemeKind>(s, kSchemeNames, "scheme"); }
PenaltyPolicyKind parse_penalty_policy(const std::string &s) { return enum_parse<PenaltyPolicyKind>(s, kPolicyNames, "penalty policy"); }
StepSchedule parse_step_schedule(const std::string &s) { return enum_parse<StepSchedule>(s, kStepScheduleNames, "step schedule"); }
FreezeValue parse_freeze_value(const std::string &s) { return enum_parse<FreezeValue>(s, kFreezeValueNames, "freeze value"); }
LinsysBackend parse_linsys_backend(const std::string &s) { return enum_parse<LinsysBackend>(s, kBackendNames, "linear-system backend"); }
Determinism parse_determinism(const std::string &s) { return enum_parse<Determinism>(s, kDeterminismNames, "determinism"); }
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
        {"algorithm.scheme", "--scheme", {"--algorithm"}, "enum", "admm, sgs_admm", "admm",
         "iteration scheme: plain two-block ADMM, or sGS-ADMM (one extra normal-equation solve per iteration); the alias "
         "--algorithm accepts admm and sgs", ""},
        {"algorithm.tau", "--tau", {}, "double", "", "1.618", "multiplier step length, in (0, 2); the legacy sGS path used 1.95", ""},
        {"algorithm.sgs_tau", "--sgs-tau", {}, "double", "", "tau", "step length of the sGS iterations of the hybrid mode (the sGS phase of main used 1.95, then 1.618); tau: the same as algorithm.tau", ""},
        {"algorithm.scheme_switch_iteration", "--scheme-switch-iteration", {"--switch-admm"}, "int", "", "0",
         "hybrid mode (plan Q5): sGS-ADMM for the iterations before this one, then plain ADMM from the same X, y, S; 0: off", ""},
        {"algorithm.sgs_iterations", "--sgs-iterations", {}, "int", "", "0",
         "hybrid mode as a count: exactly n sGS-ADMM iterations, then plain ADMM (= scheme_switch_iteration n + 1); 0: plain ADMM", ""},
        {"algorithm.max_iter", "--max-iter", {}, "int", "", "1000000", "maximum number of iterations", ""},
        {"algorithm.time_limit", "--time-limit", {}, "double", "", "inf", "seconds of iterations after which the solve stops, returning the best iterate", ""},
        {"algorithm.tol", "--tol", {}, "double", "", "1e-4", "internal stopping tolerance on eta = max(eta_p, eta_d, eta_g)", ""},
        {"algorithm.sgs_tau_reduction", "--sgs-tau-reduction", {}, "bool", "", "false",
         "today's hidden sGS rule, tau = max(1.618, tau / 1.1) at sGS iterations whose incoming eta_d is below tol; off: tau is the option (plan)", ""},
        // penalty
        {"penalty.policy", "--policy", {"--sigma-policy"}, "enum", "fixed, legacy_adaptive, residual_balancing, balance_freeze", "residual_balancing",
         "fixed: sigma_k = sigma0; legacy_adaptive: the rule of 9bb4ba5 (no guard, factor 2, [1e-6, 1e6], Monitor1); "
         "residual_balancing: the October 13 rule of 2d8a755 (guard, factor, [1e-3, 1e3]); balance_freeze: residual "
         "balancing that freezes sigma (plan Section 3.5; Phase 4)", ""},
        {"penalty.sigma0", "--sigma0", {"--sig"}, "double", "", "100", "initial sigma (scaled units)", ""},
        {"penalty.sigma_min", "--sigma-min", {}, "double", "", "policy", "lower bound of the adaptive updates (policy: the policy's own)", ""},
        {"penalty.sigma_max", "--sigma-max", {}, "double", "", "policy", "upper bound of the adaptive updates (policy: the policy's own)", ""},
        {"penalty.factor", "--factor", {"--penalty-factor"}, "double", "", "2", "multiplicative step of residual_balancing and balance_freeze (1.2 = the October 20 edit of main)", ""},
        {"penalty.step_schedule", "--step-schedule", {}, "enum", "constant, halve_on_reversal", "constant", "balance_freeze: constant factor, or the factor's square root at every reversal", "4"},
        {"penalty.win_ratio", "--win-ratio", {}, "double", "", "1.35", "hysteresis of the win counts (legacy_adaptive and residual_balancing: fixed at 1.35)", ""},
        {"penalty.guard", "--guard", {}, "bool", "", "true", "winner above max(1, win_ratio * other) (true) or above win_ratio * other (false); legacy_adaptive: false, residual_balancing: true", ""},
        {"penalty.residual_weight", "--residual-weight", {}, "double", "", "1", "weight on eta_p in the vote (today's ratioconst)", ""},
        {"penalty.schedule", "--schedule", {}, "schedule", "", "200:10,1000:25,5000:50,10000:200,inf:1000",
         "check iterations as (up to, period) pairs: a check at every k = 1 mod period while k <= up to (balance_freeze; the reference policies use their own fixed schedule)", "4"},
        {"penalty.freeze_iteration", "--freeze-iteration", {}, "int", "", "policy", "balance_freeze: freeze sigma at this iteration at the latest (policy: 10000); residual_balancing: off", "4"},
        {"penalty.pingpong_window", "--pingpong-window", {}, "int", "", "8", "balance_freeze: window W of the ping-pong test", "4"},
        {"penalty.pingpong_net", "--pingpong-net", {}, "int", "", "2", "balance_freeze: freeze when |net movement over the window| <= this", "4"},
        {"penalty.pingpong_min_iter", "--pingpong-min-iter", {}, "int", "", "1000", "balance_freeze: no ping-pong freeze before this iteration", "4"},
        {"penalty.freeze_value", "--freeze-value", {}, "enum", "last, band_centre", "band_centre", "balance_freeze: frozen value when sigma bounces in a band", "4"},
        {"penalty.freeze_bias", "--freeze-bias", {}, "double", "", "1", "balance_freeze: factor applied to the frozen value", "4"},
        {"penalty.sgs_update_threshold", "--sgs-update-threshold", {"--sig-update-threshold"}, "int", "", "500", "sGS-phase rule of the adaptive policies: checks every stage_1 iterations up to this iteration, every stage_2 after", ""},
        {"penalty.sgs_update_stage_1", "--sgs-update-stage-1", {"--sig-update-stage-1"}, "int", "", "50", "sGS-phase rule: period before the threshold", ""},
        {"penalty.sgs_update_stage_2", "--sgs-update-stage-2", {"--sig-update-stage-2"}, "int", "", "100", "sGS-phase rule: period after the threshold", ""},
        {"penalty.sgs_factor", "--sgs-factor", {"--sigscale"}, "double", "", "2", "sGS-phase rule: factor of the updates", ""},
        // scaling
        {"scaling.enabled", "--scaling", {}, "bool", "", "true", "row normalization of A and the two scalars b_s, C_s (Milestone 1: always on)", ""},
        // linsys
        {"linsys.backend", "--backend", {}, "enum", "auto, cholmod, cudss, dense", "auto", "normal-equation backend (Milestone 1: auto = cholmod with today's dense-factor rule)", "7"},
        {"linsys.regularization", "--regularization", {}, "double", "", "1e-15", "added to the diagonal of AA^T (Milestone 1: fixed)", "7"},
        {"linsys.presolve", "--presolve", {}, "bool", "", "false", "dependent-row removal (Phase 7; off in the bitwise phases)", "7"},
        {"linsys.memory_fraction", "--memory-fraction", {}, "double", "", "0.25", "budget of a dense or cuDSS factor as a fraction of free memory", "7"},
        {"linsys.autotune", "--autotune", {}, "bool", "", "false", "time the two best backends and keep the faster", "7"},
        // certificate
        {"certificate.block_bounds", "--block-bounds", {}, "double_list", "", "none", "per-block bounds R_b of the certified lower bound (comma-separated, one per block)", ""},
        {"certificate.trace_bound", "--trace-bound", {}, "double", "", "none", "the same bound R for every block (paper eq. 30)", ""},
        {"certificate.trace_bound_scale", "--trace-bound-scale", {}, "double", "", "none", "R_b = f n_b for PSD blocks of size n_b, f for the entries of l and u blocks", ""},
        // validation
        {"validation.interval", "--validation-interval", {"--validate-interval"}, "int", "", "0", "validation-aware stopping: validate every n iterations on the CPU with the original data and stop only on a validated snapshot (0: off)", ""},
        {"validation.tol", "--validation-tol", {"--validate-tol"}, "double", "", "algorithm.tol", "tolerance of every external criterion", ""},
        {"validation.threads", "--validation-threads", {"--validate-threads"}, "int", "", "1", "CPU threads of the external validation", ""},
        {"validation.strict_dimacs_tol", "--strict-dimacs-tol", {}, "double", "", "0", "after the validated criteria, continue until max |DIMACS error| <= x (0: stop at the criteria)", ""},
        {"validation.checkpoint_iterations", "--checkpoint-iterations", {"--checkpoint-iters"}, "int_list", "", "none", "iterations with an extra validation snapshot", ""},
        {"validation.checkpoint_dir", "--checkpoint-dir", {}, "string", "", "none", "save X.txt, y.txt, S.txt of those snapshots in <dir>/iter_<k>", ""},
        {"validation.final_save_dir", "--final-save-dir", {"--save-final-dir"}, "string", "", "none", "save the final snapshot of a solve that ends at a safety limit", ""},
        {"validation.history_file", "--validation-history", {}, "string", "", "none", "one CSV line per external validation", ""},
        // io
        {"io.format", "--format", {}, "enum", "auto, h5, sdpa, sedumi, sdpt3, mosek, txt", "auto", "input format (auto: from the extension and the .mat variables)", ""},
        {"io.sense", "--sense", {}, "enum", "min, max", "from the file", "objective sense (required for MOSEK files without a sense variable)", ""},
        {"io.simplify", "--simplify", {}, "bool", "", "true", "1x1 and diagonal-only PSD blocks become l blocks (reported); never for TXT input", ""},
        {"io.warm_start_dir", "--warm-start-dir", {}, "string", "", "none", "read X.txt, y.txt, S.txt (dense, one value per line) from this directory", ""},
        {"io.warm_start", "--warm-start", {}, "bool", "", "false", "read X.txt, y.txt, S.txt from the problem directory (TXT input)", ""},
        {"io.save_dir", "--save-dir", {"--save-solution"}, "string", "", "none", "write the returned X, y, S (and certificate.txt) in the input's convention", ""},
        {"io.save_format", "--save-format", {}, "string", "", "input's", "convention of the written solution: txt, h5, sdpa, sedumi, sdpt3, mosek (default: the input's)", ""},
        // output
        {"output.verbosity", "--verbosity", {}, "int", "", "1", "0: quiet, 1: the iteration table, 2: details", ""},
        {"output.history", "--history", {}, "string", "", "none", "per-iteration CSV (17 significant digits)", ""},
        {"output.summary", "--summary", {}, "string", "", "none", "append a one-line JSON summary of the run", ""},
        {"output.sigma_log", "--sigma-log", {}, "string", "", "none", "one CSV line per sigma change", ""},
        {"output.tag", "--tag", {}, "string", "", "none", "label copied to the summary", ""},
        // psd
        {"psd.small_max", "--psd-small-max", {}, "int", "", "32", "largest small PSD block (batched Jacobi); Milestone 1: fixed", "6a"},
        {"psd.medium_max", "--psd-medium-max", {}, "int", "", "1000", "largest medium PSD block; Milestone 1: fixed", "6a"},
        {"psd.medium_batch_min", "--psd-medium-batch-min", {}, "int", "", "2", "medium size groups with at least this many blocks use the batched EVD (0: never)", ""},
        {"psd.medium_streams", "--psd-medium-streams", {"--streams"}, "int", "", "15", "CUDA streams of the medium eigendecompositions (>= 1)", ""},
        {"psd.jacobi_tol", "--psd-jacobi-tol", {}, "double", "", "1e-16", "tolerance of the batched Jacobi EVD; Milestone 1: fixed", "6a"},
        {"psd.jacobi_max_sweeps", "--psd-jacobi-max-sweeps", {}, "int", "", "15", "sweeps of the batched Jacobi EVD; Milestone 1: fixed", "6a"},
        {"psd.large.memory_budget", "--psd-large-memory-budget", {}, "double", "", "0.9", "fraction of free memory for the large tier (Phase 6b)", "6b"},
        {"psd.large.warmup", "--psd-warmup", {}, "enum", "none, composite_fp16", "none", "low-precision warm-up of the large blocks (composite_fp16 becomes the default in Phase 6a)", ""},
        {"psd.large.warmup.min_size", "--psd-warmup-min-size", {}, "int", "", "5000", "smallest block that gets the warm-up (Milestone 1: the warm-up applies to every large block, as the C++ API did)", "6a"},
        {"psd.large.warmup.switch_tol", "--psd-warmup-switch-tol", {"--switch-proj-tol"}, "double", "", "1e-2", "the warm-up ends at the first iteration with eta below this", ""},
        {"psd.large.warmup.switch_iter", "--psd-warmup-switch-iter", {"--switch-proj-iter"}, "int", "", "5000", "or after this many iterations", ""},
        {"psd.lowrank.enabled", "--psd-lowrank-enabled", {"--lobpcg"}, "bool", "", "true", "LOBPCG for large blocks of low positive or negative rank", ""},
        {"psd.lowrank.start_iter", "--psd-lowrank-start-iter", {}, "int", "", "2", "first iteration of the low-rank analysis for blocks without a warm-up (warm-up blocks: the end of the warm-up)", ""},
        {"psd.lowrank.reevaluate", "--psd-lowrank-reevaluate", {}, "int", "", "100", "exact EVD with rank analysis every this many iterations; Milestone 1: fixed", "6a"},
        {"psd.lowrank.ratio", "--psd-lowrank-ratio", {}, "double", "", "0.03", "low-rank threshold as a fraction of n; Milestone 1: fixed", "6a"},
        {"psd.lowrank.relaxation", "--psd-lowrank-relaxation", {}, "double", "", "1.2", "eigenpairs requested: ceil(relaxation * rank); Milestone 1: fixed", "6a"},
        {"psd.lowrank.max_iter", "--psd-lowrank-max-iter", {}, "int", "", "20", "LOBPCG iterations; Milestone 1: fixed", "6a"},
        {"psd.lowrank.tol", "--psd-lowrank-tol", {}, "double", "", "1e-5", "LOBPCG tolerance (absolute today, B13); Milestone 1: fixed", "6a"},
        {"psd.lowrank.rank_rtol", "--psd-lowrank-rank-rtol", {}, "double", "", "1e-8", "rank tolerance relative to max |lambda|; Milestone 1: fixed", "6a"},
        {"psd.lowrank.rank_stop_factor", "--psd-lowrank-rank-stop-factor", {}, "double", "", "1e-3", "rank tolerance floor as a fraction of the stopping tolerance; Milestone 1: fixed", "6a"},
        {"psd.composite.margin", "--psd-composite-margin", {}, "double", "", "1.05", "safety margin on the Lanczos estimate; Milestone 1: fixed", "6a"},
        {"psd.composite.lanczos_steps", "--psd-composite-lanczos-steps", {}, "int", "", "20", "Lanczos steps of the norm estimate; Milestone 1: fixed", "6a"},
        {"psd.composite.norm_slack", "--psd-composite-norm-slack", {}, "double", "", "1e-2", "norm test slack of the composite result (B22); Milestone 1: fixed", "6a"},
        // misc
        {"seed", "--seed", {}, "int", "", "0", "seed of the deterministic generators (the Lanczos start from Phase 6a)", ""},
        {"determinism", "--determinism", {}, "enum", "strict, fast", "strict", "strict: deterministic SpMV and reductions; fast: opt-in nondeterministic modes (none yet)", ""},
        {"gpu.device", "--device", {}, "int", "", "0", "CUDA device of the solver", ""},
    };
    return table;
}

const OptionSpec *find_flag(const std::string &flag)
{
    for (const OptionSpec &spec : option_table())
    {
        if (spec.flag == flag)
            return &spec;
        for (const std::string &a : spec.aliases)
            if (a == flag)
                return &spec;
    }
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

std::vector<std::pair<int, int>> parse_schedule(const std::string &s)
{
    std::vector<std::pair<int, int>> out;
    for (const std::string &t : split(s, ','))
    {
        const std::vector<std::string> parts = split(t, ':');
        if (parts.size() != 2)
            throw std::invalid_argument("schedule entries are upto:period, got '" + t + "'");
        const int upto = parts[0] == "inf" ? std::numeric_limits<int>::max() : parse_int(parts[0]);
        const int period = parse_int(parts[1]);
        if (period < 1 || upto < 1)
            throw std::invalid_argument("schedule entries need upto >= 1 and period >= 1, got '" + t + "'");
        if (!out.empty() && upto <= out.back().first)
            throw std::invalid_argument("schedule entries must have increasing 'up to' iterations");
        out.push_back({upto, period});
    }
    if (out.empty())
        throw std::invalid_argument("empty schedule");
    return out;
}

std::string fmt_schedule(const std::vector<std::pair<int, int>> &v)
{
    std::string out;
    for (std::size_t i = 0; i < v.size(); i++)
        out += (i ? "," : "") + (v[i].first == std::numeric_limits<int>::max() ? std::string("inf") : std::to_string(v[i].first)) + ":" + std::to_string(v[i].second);
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
        if (p == "algorithm.scheme")
        {
            // the legacy alias --algorithm accepts "sgs" for sgs_admm
            o.algorithm.scheme = value == "sgs" ? SchemeKind::SgsAdmm : parse_scheme(value);
        }
        else if (p == "algorithm.tau") o.algorithm.tau = parse_double(value);
        else if (p == "algorithm.sgs_tau") o.algorithm.sgs_tau = value == "tau" ? -1.0 : parse_double(value);
        else if (p == "algorithm.scheme_switch_iteration") o.algorithm.scheme_switch_iteration = parse_int(value);
        else if (p == "algorithm.sgs_iterations")
        {
            const int n = parse_int(value);
            if (n < 0 || n > std::numeric_limits<int>::max() - 2)
                throw std::invalid_argument("sgs_iterations must be a non-negative count");
            o.algorithm.scheme_switch_iteration = n == 0 ? 0 : n + 1;
        }
        else if (p == "algorithm.max_iter") o.algorithm.max_iter = parse_int(value);
        else if (p == "algorithm.time_limit") o.algorithm.time_limit = parse_double(value);
        else if (p == "algorithm.tol") o.algorithm.tol = parse_double(value);
        else if (p == "algorithm.sgs_tau_reduction") o.algorithm.sgs_tau_reduction = parse_bool(value);
        else if (p == "penalty.policy") o.penalty.policy = parse_penalty_policy(value);
        else if (p == "penalty.sigma0") o.penalty.sigma0 = parse_double(value);
        else if (p == "penalty.sigma_min") o.penalty.sigma_min = value == "policy" ? -1.0 : parse_double(value);
        else if (p == "penalty.sigma_max") o.penalty.sigma_max = value == "policy" ? -1.0 : parse_double(value);
        else if (p == "penalty.factor") o.penalty.factor = parse_double(value);
        else if (p == "penalty.step_schedule") o.penalty.step_schedule = parse_step_schedule(value);
        else if (p == "penalty.win_ratio") o.penalty.win_ratio = parse_double(value);
        else if (p == "penalty.guard") o.penalty.guard = parse_bool(value);
        else if (p == "penalty.residual_weight") o.penalty.residual_weight = parse_double(value);
        else if (p == "penalty.schedule") o.penalty.schedule = parse_schedule(value);
        else if (p == "penalty.freeze_iteration") o.penalty.freeze_iteration = value == "policy" ? -1 : parse_int(value);
        else if (p == "penalty.pingpong_window") o.penalty.pingpong_window = parse_int(value);
        else if (p == "penalty.pingpong_net") o.penalty.pingpong_net = parse_int(value);
        else if (p == "penalty.pingpong_min_iter") o.penalty.pingpong_min_iter = parse_int(value);
        else if (p == "penalty.freeze_value") o.penalty.freeze_value = parse_freeze_value(value);
        else if (p == "penalty.freeze_bias") o.penalty.freeze_bias = parse_double(value);
        else if (p == "penalty.sgs_update_threshold") o.penalty.sgs_update_threshold = parse_int(value);
        else if (p == "penalty.sgs_update_stage_1") o.penalty.sgs_update_stage_1 = parse_int(value);
        else if (p == "penalty.sgs_update_stage_2") o.penalty.sgs_update_stage_2 = parse_int(value);
        else if (p == "penalty.sgs_factor") o.penalty.sgs_factor = parse_double(value);
        else if (p == "scaling.enabled") o.scaling.enabled = parse_bool(value);
        else if (p == "linsys.backend") o.linsys.backend = parse_linsys_backend(value);
        else if (p == "linsys.regularization") o.linsys.regularization = parse_double(value);
        else if (p == "linsys.presolve") o.linsys.presolve = parse_bool(value);
        else if (p == "linsys.memory_fraction") o.linsys.memory_fraction = parse_double(value);
        else if (p == "linsys.autotune") o.linsys.autotune = parse_bool(value);
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
        else if (p == "psd.jacobi_tol") o.psd.jacobi_tol = parse_double(value);
        else if (p == "psd.jacobi_max_sweeps") o.psd.jacobi_max_sweeps = parse_int(value);
        else if (p == "psd.large.memory_budget") o.psd.large_memory_budget = parse_double(value);
        else if (p == "psd.large.warmup") o.psd.warmup = parse_warmup(value);
        else if (p == "psd.large.warmup.min_size") o.psd.warmup_min_size = parse_int(value);
        else if (p == "psd.large.warmup.switch_tol") o.psd.warmup_switch_tol = parse_double(value);
        else if (p == "psd.large.warmup.switch_iter") o.psd.warmup_switch_iter = parse_int(value);
        else if (p == "psd.lowrank.enabled") o.psd.lowrank_enabled = parse_bool(value);
        else if (p == "psd.lowrank.start_iter") o.psd.lowrank_start_iter = parse_int(value);
        else if (p == "psd.lowrank.reevaluate") o.psd.lowrank_reevaluate = parse_int(value);
        else if (p == "psd.lowrank.ratio") o.psd.lowrank_ratio = parse_double(value);
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
        else if (p == "determinism") o.determinism = parse_determinism(value);
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
    if (p == "algorithm.scheme") return name_of(o.algorithm.scheme);
    if (p == "algorithm.tau") return fmt(o.algorithm.tau);
    if (p == "algorithm.sgs_tau") return o.algorithm.sgs_tau < 0 ? "tau" : fmt(o.algorithm.sgs_tau);
    if (p == "algorithm.scheme_switch_iteration") return fmt(o.algorithm.scheme_switch_iteration);
    if (p == "algorithm.sgs_iterations") return fmt(o.algorithm.scheme_switch_iteration > 1 ? o.algorithm.scheme_switch_iteration - 1 : 0);
    if (p == "algorithm.max_iter") return fmt(o.algorithm.max_iter);
    if (p == "algorithm.time_limit") return fmt(o.algorithm.time_limit);
    if (p == "algorithm.tol") return fmt(o.algorithm.tol);
    if (p == "algorithm.sgs_tau_reduction") return fmt(o.algorithm.sgs_tau_reduction);
    if (p == "penalty.policy") return name_of(o.penalty.policy);
    if (p == "penalty.sigma0") return fmt(o.penalty.sigma0);
    if (p == "penalty.sigma_min") return o.penalty.sigma_min < 0 ? "policy" : fmt(o.penalty.sigma_min);
    if (p == "penalty.sigma_max") return o.penalty.sigma_max < 0 ? "policy" : fmt(o.penalty.sigma_max);
    if (p == "penalty.factor") return fmt(o.penalty.factor);
    if (p == "penalty.step_schedule") return name_of(o.penalty.step_schedule);
    if (p == "penalty.win_ratio") return fmt(o.penalty.win_ratio);
    if (p == "penalty.guard") return fmt(o.penalty.guard);
    if (p == "penalty.residual_weight") return fmt(o.penalty.residual_weight);
    if (p == "penalty.schedule") return fmt_schedule(o.penalty.schedule);
    if (p == "penalty.freeze_iteration") return o.penalty.freeze_iteration < 0 ? "policy" : fmt(o.penalty.freeze_iteration);
    if (p == "penalty.pingpong_window") return fmt(o.penalty.pingpong_window);
    if (p == "penalty.pingpong_net") return fmt(o.penalty.pingpong_net);
    if (p == "penalty.pingpong_min_iter") return fmt(o.penalty.pingpong_min_iter);
    if (p == "penalty.freeze_value") return name_of(o.penalty.freeze_value);
    if (p == "penalty.freeze_bias") return fmt(o.penalty.freeze_bias);
    if (p == "penalty.sgs_update_threshold") return fmt(o.penalty.sgs_update_threshold);
    if (p == "penalty.sgs_update_stage_1") return fmt(o.penalty.sgs_update_stage_1);
    if (p == "penalty.sgs_update_stage_2") return fmt(o.penalty.sgs_update_stage_2);
    if (p == "penalty.sgs_factor") return fmt(o.penalty.sgs_factor);
    if (p == "scaling.enabled") return fmt(o.scaling.enabled);
    if (p == "linsys.backend") return name_of(o.linsys.backend);
    if (p == "linsys.regularization") return fmt(o.linsys.regularization);
    if (p == "linsys.presolve") return fmt(o.linsys.presolve);
    if (p == "linsys.memory_fraction") return fmt(o.linsys.memory_fraction);
    if (p == "linsys.autotune") return fmt(o.linsys.autotune);
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
    if (p == "psd.jacobi_tol") return fmt(o.psd.jacobi_tol);
    if (p == "psd.jacobi_max_sweeps") return fmt(o.psd.jacobi_max_sweeps);
    if (p == "psd.large.memory_budget") return fmt(o.psd.large_memory_budget);
    if (p == "psd.large.warmup") return name_of(o.psd.warmup);
    if (p == "psd.large.warmup.min_size") return fmt(o.psd.warmup_min_size);
    if (p == "psd.large.warmup.switch_tol") return fmt(o.psd.warmup_switch_tol);
    if (p == "psd.large.warmup.switch_iter") return fmt(o.psd.warmup_switch_iter);
    if (p == "psd.lowrank.enabled") return fmt(o.psd.lowrank_enabled);
    if (p == "psd.lowrank.start_iter") return fmt(o.psd.lowrank_start_iter);
    if (p == "psd.lowrank.reevaluate") return fmt(o.psd.lowrank_reevaluate);
    if (p == "psd.lowrank.ratio") return fmt(o.psd.lowrank_ratio);
    if (p == "psd.lowrank.relaxation") return fmt(o.psd.lowrank_relaxation);
    if (p == "psd.lowrank.max_iter") return fmt(o.psd.lowrank_max_iter);
    if (p == "psd.lowrank.tol") return fmt(o.psd.lowrank_tol);
    if (p == "psd.lowrank.rank_rtol") return fmt(o.psd.lowrank_rank_rtol);
    if (p == "psd.lowrank.rank_stop_factor") return fmt(o.psd.lowrank_rank_stop_factor);
    if (p == "psd.composite.margin") return fmt(o.psd.composite_margin);
    if (p == "psd.composite.lanczos_steps") return fmt(o.psd.composite_lanczos_steps);
    if (p == "psd.composite.norm_slack") return fmt(o.psd.composite_norm_slack);
    if (p == "seed") return std::to_string(o.seed);
    if (p == "determinism") return name_of(o.determinism);
    if (p == "gpu.device") return fmt(o.gpu_device);
    throw std::logic_error("option '" + p + "' is in the table but has no getter");
}

// validation --------------------------------------------------------------------------------------------------------

void SolverOptions::validate() const
{
    auto fail = [](const std::string &path, const std::string &why)
    { throw std::invalid_argument("option " + path + ": " + why); };
    const SolverOptions defaults;

    if (!(algorithm.tau > 0.0 && algorithm.tau < 2.0))
        fail("algorithm.tau", "must be in (0, 2), got " + fmt(algorithm.tau));
    if (algorithm.sgs_tau >= 0.0 && !(algorithm.sgs_tau > 0.0 && algorithm.sgs_tau < 2.0))
        fail("algorithm.sgs_tau", "must be in (0, 2), got " + fmt(algorithm.sgs_tau));
    if (algorithm.scheme_switch_iteration < 0)
        fail("algorithm.scheme_switch_iteration", "must be non-negative");
    if (algorithm.scheme == SchemeKind::SgsAdmm && algorithm.scheme_switch_iteration > 0)
        fail("algorithm.scheme_switch_iteration", "the hybrid mode starts with sGS-ADMM by itself: use scheme admm with the switch iteration, or scheme sgs_admm without it");
    if (algorithm.max_iter < 0 || algorithm.max_iter > std::numeric_limits<int>::max() - 2)
        fail("algorithm.max_iter", "must be in [0, INT_MAX - 2]");
    if (!(algorithm.time_limit > 0.0))
        fail("algorithm.time_limit", "must be positive (inf: none)");
    if (!(algorithm.tol > 0.0) || !std::isfinite(algorithm.tol))
        fail("algorithm.tol", "must be a positive finite number");

    if (!(penalty.sigma0 > 0.0) || !std::isfinite(penalty.sigma0))
        fail("penalty.sigma0", "must be a positive finite number");
    if (penalty.sigma_min >= 0.0 && !(penalty.sigma_min > 0.0))
        fail("penalty.sigma_min", "must be positive");
    if (penalty.sigma_max >= 0.0 && !(penalty.sigma_max > 0.0))
        fail("penalty.sigma_max", "must be positive");
    if (penalty.sigma_min > 0.0 && penalty.sigma_max > 0.0 && penalty.sigma_min > penalty.sigma_max)
        fail("penalty.sigma_min", "must not exceed penalty.sigma_max");
    if (!(penalty.factor > 1.0) || !std::isfinite(penalty.factor))
        fail("penalty.factor", "must be greater than 1");
    if (!(penalty.win_ratio >= 1.0))
        fail("penalty.win_ratio", "must be at least 1");
    if (!(penalty.residual_weight > 0.0))
        fail("penalty.residual_weight", "must be positive");
    if (penalty.schedule.empty())
        fail("penalty.schedule", "must not be empty");
    if (penalty.pingpong_window < 2 || penalty.pingpong_net < 0 || penalty.pingpong_min_iter < 0)
        fail("penalty.pingpong_window", "window >= 2, net >= 0, min_iter >= 0");
    if (!(penalty.freeze_bias > 0.0))
        fail("penalty.freeze_bias", "must be positive");
    if (penalty.sgs_update_threshold < 0 || penalty.sgs_update_stage_1 < 1 || penalty.sgs_update_stage_2 < 1)
        fail("penalty.sgs_update_threshold", "threshold >= 0 and stages >= 1");
    if (!(penalty.sgs_factor > 0.0))
        fail("penalty.sgs_factor", "must be positive");
    // the reference policies have their own fixed hysteresis (plan: legacy_adaptive bit for bit, residual_balancing =
    // 2d8a755): win_ratio, guard, residual_weight and the schedule are read by balance_freeze only, and an explicit
    // value that contradicts a reference rule is rejected
    if (penalty.policy == PenaltyPolicyKind::LegacyAdaptive || penalty.policy == PenaltyPolicyKind::ResidualBalancing)
    {
        if (penalty.win_ratio != 1.35)
            fail("penalty.win_ratio", "is fixed at 1.35 for the reference policies legacy_adaptive and residual_balancing");
        if (penalty.schedule != defaults.penalty.schedule)
            fail("penalty.schedule", "the reference policies use the fixed schedule 200:10,1000:25,5000:50,10000:200,inf:1000");
        if (penalty.residual_weight != 1.0)
            fail("penalty.residual_weight", "is fixed at 1 for the reference policies");
        if (penalty.freeze_iteration >= 0)
            fail("penalty.freeze_iteration", "only balance_freeze freezes sigma");
        if (penalty.policy == PenaltyPolicyKind::LegacyAdaptive && penalty.factor != 2.0)
            fail("penalty.factor", "is fixed at 2 for legacy_adaptive (9bb4ba5)");
        if (penalty.policy == PenaltyPolicyKind::ResidualBalancing && !penalty.guard)
            fail("penalty.guard", "residual_balancing is the guarded rule of 2d8a755; legacy_adaptive is the unguarded one");
    }

    if (!scaling.enabled)
        fail("scaling.enabled", "Milestone 1: the ported engine always scales (false is not implemented yet)");
    if (linsys.backend != LinsysBackend::Auto && linsys.backend != LinsysBackend::Cholmod)
        fail("linsys.backend", std::string(name_of(linsys.backend)) + " comes with Phase 7; Milestone 1 offers auto (= cholmod)");
    if (linsys.regularization != 1e-15)
        fail("linsys.regularization", "is fixed at 1e-15 until Phase 7");
    if (linsys.presolve)
        fail("linsys.presolve", "comes with Phase 7 (off in the bitwise phases)");
    if (linsys.autotune)
        fail("linsys.autotune", "comes with Phase 7");
    if (!(linsys.memory_fraction > 0.0 && linsys.memory_fraction <= 1.0))
        fail("linsys.memory_fraction", "must be in (0, 1]");

    if (certificate.trace_bound >= 0.0 && !std::isfinite(certificate.trace_bound))
        fail("certificate.trace_bound", "must be finite");
    if (certificate.trace_bound_scale >= 0.0 && !std::isfinite(certificate.trace_bound_scale))
        fail("certificate.trace_bound_scale", "must be finite");
    for (double r : certificate.block_bounds)
        if (!(r >= 0.0) || !std::isfinite(r))
            fail("certificate.block_bounds", "bounds must be finite and non-negative");

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

    if (psd.small_max != 32 || psd.medium_max != 1000)
        fail("psd.small_max", "the tier limits are fixed at 32 and 1000 until Phase 6a");
    if (psd.medium_batch_min < 0)
        fail("psd.medium_batch_min", "must be non-negative");
    if (psd.medium_streams < 1)
        fail("psd.medium_streams", "must be at least 1 (B17)");
    if (psd.jacobi_tol != 1e-16 || psd.jacobi_max_sweeps != 15)
        fail("psd.jacobi_tol", "the Jacobi parameters are fixed until Phase 6a");
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
    if (psd.lowrank_reevaluate != 100 || psd.lowrank_ratio != 0.03 || psd.lowrank_relaxation != 1.2 || psd.lowrank_max_iter != 20 ||
        psd.lowrank_tol != 1e-5 || psd.lowrank_rank_rtol != 1e-8 || psd.lowrank_rank_stop_factor != 1e-3)
        fail("psd.lowrank.reevaluate", "the LOBPCG parameters are fixed until Phase 6a");
    if (psd.composite_margin != 1.05 || psd.composite_lanczos_steps != 20 || psd.composite_norm_slack != 1e-2)
        fail("psd.composite.margin", "the composite parameters are fixed until Phase 6a");
    if (psd.warmup == WarmupKind::CompositeFp16 && psd.warmup_min_size != defaults.psd.warmup_min_size)
        fail("psd.large.warmup.min_size", "Milestone 1 applies the warm-up to every large block (the per-size threshold comes with Phase 6a)");

    if (determinism != Determinism::Strict)
        fail("determinism", "fast has no nondeterministic mode yet (Milestone 1)");
    if (gpu_device < 0)
        fail("gpu.device", "must be non-negative");
}

// documentation ---------------------------------------------------------------------------------------------------

std::string option_table_markdown()
{
    std::ostringstream o;
    o << "# Options of cuADMM-reorg\n\nGenerated from the option table of `src/options/options.cc` (`cuadmm --print-options`). "
         "Every front-end uses the same table; unknown names are errors.\n\n"
         "| Option | CLI flag (aliases) | Type | Default | Phase | Meaning |\n|---|---|---|---|---|---|\n";
    for (const OptionSpec &s : option_table())
    {
        std::string flags = "`" + s.flag + "`";
        for (const std::string &a : s.aliases)
            flags += " (`" + a + "`)";
        o << "| `" << s.path << "` | " << flags << " | " << s.type << (s.values.empty() ? "" : " (" + s.values + ")") << " | `"
          << s.default_value << "` | " << (s.phase.empty() ? "now" : s.phase) << " | " << s.help << " |\n";
    }
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
        std::string flags = s.flag;
        for (const std::string &a : s.aliases)
            flags += ", " + a;
        o << "  " << flags << " <" << s.type << ">" << (s.values.empty() ? "" : " {" + s.values + "}") << "  [" << s.default_value << "]"
          << (s.phase.empty() ? "" : "  (phase " + s.phase + ")") << "\n      " << s.help << "\n";
    }
    return o.str();
}

std::string options_json(const SolverOptions &options)
{
    std::ostringstream o;
    o << "{";
    bool first = true;
    for (const OptionSpec &s : option_table())
    {
        if (s.path == "algorithm.sgs_iterations")
            continue; // derived from scheme_switch_iteration
        const std::string v = get_option(options, s.path);
        o << (first ? "" : ", ") << json_string(s.path) << ": ";
        first = false;
        if (s.type == "int" && v != "policy" && v != "none")
            o << v;
        else if (s.type == "double" && v != "policy" && v != "none" && v != "algorithm.tol" && v != "inf" && v != "-inf")
            o << v;
        else if (s.type == "bool")
            o << v;
        else
            o << json_string(v);
    }
    o << "}";
    return o.str();
}

} // namespace cuadmm
