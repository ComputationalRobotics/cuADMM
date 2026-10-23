/*

    options.h

    SolverOptions: the single source of defaults of cuADMM-reorg (plan Section 15, Table 5; PSD options of
    Section 9.9). Every front-end (CLI, MATLAB, C++) builds one of these; the CLI flags and the MATLAB fields are
    generated from the option table of option_table() (options.cc), which also prints the documentation.

    Option names are lower snake_case paths (`penalty.policy`); unknown names and unknown values are errors, never
    aliases (plan principle 4). Milestone 1 keeps the options that the ported solver honours; options of later phases
    (presolve, cuDSS, the PSD memory model, ...) are listed with their phase and rejected with a clear message when
    set to a value the current engine cannot honour.

*/

#ifndef CUADMM_OPTIONS_H
#define CUADMM_OPTIONS_H

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cuadmm
{

enum class SchemeKind { Admm, SgsAdmm };
enum class PenaltyPolicyKind { Fixed, LegacyAdaptive, ResidualBalancing, BalanceFreeze };
enum class StepSchedule { Constant, HalveOnReversal };
enum class FreezeValue { Last, BandCentre };
enum class LinsysBackend { Auto, Cholmod, Cudss, Dense };
enum class Determinism { Strict, Fast };
enum class WarmupKind { None, CompositeFp16 };
enum class FinalProductKind { GemmCompat, Syrk };        // the final product of the large blocks
enum class BatchedProductKind { ScaledGemm, GemmCompat }; // the final product of the small and medium blocks
enum class IoFormat { Auto, H5, Sdpa, Sedumi, Sdpt3, Mosek, Txt };
enum class Sense { Minimize, Maximize };

struct AlgorithmOptions
{
    SchemeKind scheme = SchemeKind::Admm; // admm or sgs_admm
    double tau = 1.618;                   // multiplier step length, validated in (0, 2)
    double sgs_tau = -1.0;                // step length of the sGS iterations of the hybrid mode (< 0: tau); main used 1.95
    int scheme_switch_iteration = 0;      // hybrid mode (until Q5): sGS-ADMM for iterations < this, then ADMM; 0: off
    int max_iter = 1000000;
    double time_limit = std::numeric_limits<double>::infinity(); // seconds of iterations; inf: none
    double tol = 1e-4;                                           // internal stopping tolerance on eta
    // today's hidden sGS rule (tau = max(1.618, tau / 1.1) at sGS iterations whose incoming eta_d < tol); the plan removes
    // it (tau is the option); on only to reproduce the sGS histories of main
    bool sgs_tau_reduction = false;
};

struct PenaltyOptions
{
    PenaltyPolicyKind policy = PenaltyPolicyKind::ResidualBalancing; // default until Phase 4b (plan O1)
    double sigma0 = 100.0;                                           // scaled units
    double sigma_min = -1.0;                                         // < 0: the policy's own bounds
    double sigma_max = -1.0;                                         // (legacy_adaptive 1e-6..1e6, others 1e-3..1e3)
    double factor = 2.0;                                             // multiplicative step (today's working tree: 1.2)
    StepSchedule step_schedule = StepSchedule::Constant;
    double win_ratio = 1.35;
    bool guard = true;             // winner above max(1, win_ratio * other); false: above win_ratio * other
    double residual_weight = 1.0;  // weight on eta_p in the vote (today's ratioconst)
    // check schedule: (up to iteration, period) pairs; the last entry holds for every later iteration
    std::vector<std::pair<int, int>> schedule = {{200, 10}, {1000, 25}, {5000, 50}, {10000, 200}, {std::numeric_limits<int>::max(), 1000}};
    int freeze_iteration = -1;     // < 0: the policy's own (off for residual_balancing, 1e4 for balance_freeze)
    int pingpong_window = 8;       // balance_freeze only
    int pingpong_net = 2;
    int pingpong_min_iter = 1000;
    FreezeValue freeze_value = FreezeValue::BandCentre;
    double freeze_bias = 1.0;
    // the sGS-phase rule of the adaptive policies (today's --sig-update-threshold, --sig-update-stage-1/2, --sigscale)
    int sgs_update_threshold = 500;
    int sgs_update_stage_1 = 50;
    int sgs_update_stage_2 = 100;
    double sgs_factor = 2.0;
};

struct ScalingOptions
{
    bool enabled = true; // Milestone 1: must stay true (the ported engine always scales)
};

struct LinsysOptions
{
    LinsysBackend backend = LinsysBackend::Auto; // Milestone 1: auto == cholmod
    double regularization = 1e-15;
    bool presolve = false; // off in the bitwise phases (Phase 7 turns it on)
    double memory_fraction = 0.25;
    bool autotune = false;
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
    int small_max = 32;    // fixed (SMALL_MAT_LIMIT) until the tier limits become options
    int medium_max = 1000; // fixed (MEDIUM_MAT_LIMIT)
    int medium_batch_min = 2;
    int medium_streams = 15; // must be >= 1 (B17)
    double jacobi_tol = 1e-16;   // fixed (BatchEigParameter)
    int jacobi_max_sweeps = 15;  // fixed (BatchEigParameter)
    double large_memory_budget = 0.9; // fraction of free memory (Phase 6b)
    WarmupKind warmup = WarmupKind::None; // composite_fp16 once the Phase 6a benchmark confirms the threshold
    FinalProductKind large_final_product = FinalProductKind::Syrk;            // Phase 6b: SYRK on the smaller eigen-side (gemm_compat: main's product)
    BatchedProductKind batched_final_product = BatchedProductKind::ScaledGemm; // Phase 6b: Q scaled in place, one strided GEMM (gemm_compat: main's)
    int medium_max_batch = 0; // blocks per batched EVD call (0: the whole size group)
    int warmup_min_size = 5000;
    double warmup_switch_tol = 1e-2;
    int warmup_switch_iter = 5000;
    bool lowrank_enabled = true;
    int lowrank_start_iter = 2; // blocks without a warm-up; warm-up blocks: the end of the warm-up
    int lowrank_reevaluate = 100;
    double lowrank_ratio = 0.03;      // low-rank threshold as a fraction of n
    double lowrank_relaxation = 1.2;  // eigenpairs requested: ceil(relaxation * rank)
    int lowrank_max_iter = 20;        // LOBPCG iterations
    double lowrank_tol = 1e-5;        // LOBPCG tolerance (absolute; DECISIONS 45)
    double lowrank_rank_rtol = 1e-8;  // rank tolerance relative to max |lambda|
    double lowrank_rank_stop_factor = 1e-3; // rank tolerance floor as a fraction of the stopping tolerance
    double composite_margin = 1.05;   // safety margin on the Lanczos estimate
    int composite_lanczos_steps = 20; // Lanczos steps of the norm estimate
    double composite_norm_slack = 1e-2; // norm test slack of the composite result (B22)
};

struct SolverOptions
{
    AlgorithmOptions algorithm;
    PenaltyOptions penalty;
    ScalingOptions scaling;
    LinsysOptions linsys;
    CertificateOptions certificate;
    ValidationOptions validation;
    IoOptions io;
    OutputOptions output;
    PsdOptions psd;
    std::uint64_t seed = 0;
    Determinism determinism = Determinism::Strict;
    int gpu_device = 0;

    // Checks every option (ranges, combinations, Milestone-1 limits); throws std::invalid_argument with the option
    // path and the reason.
    void validate() const;
};

// Names of the enumerations (lower snake_case); parse_* throws std::invalid_argument on an unknown name.
const char *name_of(SchemeKind v);
const char *name_of(PenaltyPolicyKind v);
const char *name_of(StepSchedule v);
const char *name_of(FreezeValue v);
const char *name_of(LinsysBackend v);
const char *name_of(Determinism v);
const char *name_of(FinalProductKind v);
const char *name_of(BatchedProductKind v);
const char *name_of(WarmupKind v);
const char *name_of(IoFormat v);
const char *name_of(Sense v);
SchemeKind parse_scheme(const std::string &s);
PenaltyPolicyKind parse_penalty_policy(const std::string &s);
StepSchedule parse_step_schedule(const std::string &s);
FreezeValue parse_freeze_value(const std::string &s);
LinsysBackend parse_linsys_backend(const std::string &s);
Determinism parse_determinism(const std::string &s);
WarmupKind parse_warmup(const std::string &s);
FinalProductKind parse_final_product(const std::string &s);
BatchedProductKind parse_batched_product(const std::string &s);
IoFormat parse_io_format(const std::string &s);
Sense parse_sense(const std::string &s);

/// @brief One row of the option table: path, CLI flag, aliases (today's flags), type, default, help.
struct OptionSpec
{
    std::string path;                 // e.g. "penalty.policy"
    std::string flag;                 // e.g. "--policy"
    std::vector<std::string> aliases; // e.g. {"--sigma-policy"} (accepted for one release)
    std::string type;                 // "int", "double", "bool", "string", "enum", "int_list", "double_list", "schedule"
    std::string values;               // allowed values of an enum, "" otherwise
    std::string default_value;        // as printed in the documentation
    std::string help;
    std::string phase;                // "" (honoured now) or the phase that implements it ("2b", "4b", "6a", "7")
};

/// @brief The option table (single source of truth for the CLI, MATLAB and the documentation).
const std::vector<OptionSpec> &option_table();

/// @brief Sets the option `path` from its textual value; throws std::invalid_argument for an unknown path or a value
/// that does not parse. Enum values are their names, booleans 0/1/true/false, lists comma-separated, the schedule
/// "upto:period,upto:period,...".
void set_option(SolverOptions &options, const std::string &path, const std::string &value);

/// @brief The textual value of the option `path` (as set_option accepts it); throws on an unknown path.
std::string get_option(const SolverOptions &options, const std::string &path);

/// @brief Looks up a CLI flag (or alias) in the table; nullptr if unknown.
const OptionSpec *find_flag(const std::string &flag);

/// @brief The option table as Markdown (docs/OPTIONS.md) and as the CLI help text.
std::string option_table_markdown();
std::string option_table_help();

/// @brief All options as a JSON object (for the summary).
std::string options_json(const SolverOptions &options);

} // namespace cuadmm

#endif // CUADMM_OPTIONS_H
