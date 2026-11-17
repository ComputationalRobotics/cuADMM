/*

    options.h

    SolverOptions: the single source of the defaults of cuADMM. Every front-end (CLI, MATLAB, C++) builds one of
    these; the CLI flags and the MATLAB fields come from the option table of option_table() (options.cc), which also
    prints the documentation (docs/OPTIONS.md).

    Option names are lower snake_case paths (`penalty.policy`); unknown names and unknown values are errors.

*/

#ifndef CUADMM_OPTIONS_H
#define CUADMM_OPTIONS_H

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace cuadmm
{

enum class PenaltyPolicyKind { Fixed, BalanceFreeze };
enum class LinsysBackend { Auto, Cholmod, Cudss, Dense };
enum class WarmupKind { None, CompositeFp16 };
enum class IoFormat { Auto, H5, Sdpa, Sedumi, Sdpt3, Mosek, Txt };
enum class Sense { Minimize, Maximize };

struct AlgorithmOptions
{
    double tau = 1.618; // multiplier step length, in (0, 2)
    int max_iter = 1000000;
    double time_limit = std::numeric_limits<double>::infinity(); // seconds of iterations; inf: none
    double tol = 1e-4;                                           // internal stopping tolerance on eta
};

struct PenaltyOptions
{
    // balance_freeze: sigma follows the residual vote (one step of `factor` at the checks of a fixed schedule) and is
    // frozen at freeze_iteration; fixed: sigma = sigma0 throughout
    PenaltyPolicyKind policy = PenaltyPolicyKind::BalanceFreeze;
    double sigma0 = 100.0;      // scaled units
    double sigma_min = 1e-3;    // bounds of the updates (balance_freeze)
    double sigma_max = 1e3;
    double factor = 1.35;       // sigma * factor up, sigma / factor down (balance_freeze)
    int freeze_iteration = 10000; // sigma keeps its value from this iteration on (balance_freeze)
};

struct LinsysOptions
{
    LinsysBackend backend = LinsysBackend::Auto; // auto = cholmod
    double regularization = 1e-15;
    bool presolve = false;
    double memory_fraction = 0.25;
};

struct CertificateOptions
{
    std::vector<double> block_bounds; // per-block bounds R_b (empty: none)
    double trace_bound = -1.0;        // today's --trace-bound (< 0: none)
    double trace_bound_scale = -1.0;  // today's --trace-bound-scale (< 0: none)
};

struct ValidationOptions
{
    int interval = 0;               // external validation every `interval` iterations (0: off)
    double tol = -1.0;              // < 0: algorithm.tol
    int threads = 1;
    double strict_dimacs_tol = 0.0; // 0: off
    std::vector<int> checkpoint_iterations;
    std::string checkpoint_dir;
    std::string final_save_dir;
    std::string history_file; // one CSV line per external validation
};

struct IoOptions
{
    IoFormat format = IoFormat::Auto;
    std::optional<Sense> sense;  // required for MOSEK files without a `sense` variable
    bool simplify = true;        // never applied to legacy TXT input
    std::string warm_start_dir;  // X.txt, y.txt, S.txt
    bool warm_start = false;     // read the warm start from the problem directory (TXT)
    std::string save_dir;        // solution in the input's convention
    std::string save_format;     // empty: the input's
};

struct OutputOptions
{
    int verbosity = 1;
    std::string history;   // per-iteration CSV, 17 significant digits
    std::string summary;   // JSON summary (one object appended per run)
    std::string sigma_log; // one CSV line per sigma change
    std::string tag;       // label copied to the summary
};

struct PsdOptions
{
    int small_max = 32;    // largest small block (batched Jacobi; at most 32)
    int medium_max = 1000; // largest medium block (batched/streamed EVD); larger blocks are large
    int medium_batch_min = 2;
    int medium_streams = 15; // must be >= 1
    double large_memory_budget = 0.9; // fraction of free memory
    WarmupKind warmup = WarmupKind::CompositeFp16; // the FP16 composite filter for the large blocks of at least warmup_min_size
    int medium_max_batch = 0; // blocks per batched EVD call (0: the whole size group)
    int warmup_min_size = 5000;       // the large blocks of at least this size get the warm-up
    double warmup_switch_tol = 1e-2;
    int warmup_switch_iter = 5000;
    bool lowrank_enabled = true;
    int lowrank_start_iter = 2; // blocks without a warm-up; warm-up blocks: the end of the warm-up
    int lowrank_reevaluate = 100;
    double lowrank_ratio = 0.05;      // low-rank threshold as a fraction of n; sizes the warm start
    double lowrank_relaxation = 1.2;  // eigenpairs requested: ceil(relaxation * rank)
    int lowrank_max_iter = 20;        // LOBPCG iterations
    double lowrank_tol = 1e-5;        // LOBPCG tolerance (absolute)
    double lowrank_rank_rtol = 1e-8;  // rank tolerance relative to max |lambda|
    double lowrank_rank_stop_factor = 1e-3; // rank tolerance floor as a fraction of the stopping tolerance
    double composite_margin = 1.05;   // safety margin on the Lanczos estimate
    int composite_lanczos_steps = 20; // Lanczos steps of the norm estimate
    double composite_norm_slack = 1e-2; // norm test slack of the composite result
    std::string trace;                  // per-block trace of the large blocks (JSON lines; empty: off)
};

struct SolverOptions
{
    AlgorithmOptions algorithm;
    PenaltyOptions penalty;
    LinsysOptions linsys;
    CertificateOptions certificate;
    ValidationOptions validation;
    IoOptions io;
    OutputOptions output;
    PsdOptions psd;
    std::uint64_t seed = 0;
    int gpu_device = 0;

    // Checks every option (ranges and combinations); throws std::invalid_argument with the option path and the reason.
    void validate() const;
};

// Names of the enumerations (lower snake_case); parse_* throws std::invalid_argument on an unknown name.
const char *name_of(PenaltyPolicyKind v);
const char *name_of(LinsysBackend v);
const char *name_of(WarmupKind v);
const char *name_of(IoFormat v);
const char *name_of(Sense v);
PenaltyPolicyKind parse_penalty_policy(const std::string &s);
LinsysBackend parse_linsys_backend(const std::string &s);
WarmupKind parse_warmup(const std::string &s);
IoFormat parse_io_format(const std::string &s);
Sense parse_sense(const std::string &s);

/// @brief One row of the option table: path, CLI flag, type, allowed values, default, help.
struct OptionSpec
{
    std::string path;          // e.g. "penalty.policy"
    std::string flag;          // e.g. "--policy"
    std::string type;          // "int", "double", "bool", "string", "enum", "int_list", "double_list"
    std::string values;        // allowed values of an enum, "" otherwise
    std::string default_value; // as printed in the documentation
    std::string help;
};

/// @brief The option table (single source of truth for the CLI, MATLAB and the documentation).
const std::vector<OptionSpec> &option_table();

/// @brief Sets the option `path` from its textual value; throws std::invalid_argument for an unknown path or a value
/// that does not parse. Enum values are their names, booleans 0/1/true/false, lists comma-separated.
void set_option(SolverOptions &options, const std::string &path, const std::string &value);

/// @brief The textual value of the option `path` (as set_option accepts it); throws on an unknown path.
std::string get_option(const SolverOptions &options, const std::string &path);

/// @brief Looks up a CLI flag in the table; nullptr if unknown.
const OptionSpec *find_flag(const std::string &flag);

/// @brief The option table as Markdown (docs/OPTIONS.md) and as the CLI help text.
std::string option_table_markdown();
std::string option_table_help();

/// @brief All options as a JSON object (for the summary).
std::string options_json(const SolverOptions &options);

} // namespace cuadmm

#endif // CUADMM_OPTIONS_H
