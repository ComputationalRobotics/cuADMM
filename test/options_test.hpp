/*

    options_test.hpp  (CPU only)

    SolverOptions and the option table: every entry round-trips through set_option / get_option, defaults match the
    table, unknown names and values are errors (the options removed in 2.1 among them), validate() rejects what it
    must, the documentation generators produce every option.

*/

#include <gtest/gtest.h>

#include <cctype>

#include <cmath>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>

#include "cuadmm/options.h"

using namespace cuadmm;

TEST(Options, TableIsConsistentAndRoundTrips)
{
    SolverOptions o;
    std::set<std::string> paths, flags;
    for (const OptionSpec &s : option_table())
    {
        SCOPED_TRACE(s.path);
        EXPECT_TRUE(paths.insert(s.path).second) << "duplicate path";
        EXPECT_TRUE(flags.insert(s.flag).second) << "duplicate flag";
        EXPECT_EQ(find_flag(s.flag), &s);
        // the default of the table is the default of the struct (numbers compared as numbers: 1e-4 vs 0.0001)
        const std::string v = get_option(o, s.path);
        if ((s.type == "double" || s.type == "int") && v != s.default_value)
        {
            char *end_v = nullptr, *end_d = nullptr;
            const double dv = std::strtod(v.c_str(), &end_v), dd = std::strtod(s.default_value.c_str(), &end_d);
            EXPECT_TRUE(*end_v == '\0' && *end_d == '\0' && dv == dd) << "default of " << s.path << ": " << v << " vs " << s.default_value;
        }
        else
            EXPECT_EQ(v, s.default_value) << "default of " << s.path;
        // setting the current value back is a no-op
        SolverOptions copy = o;
        ASSERT_NO_THROW(set_option(copy, s.path, v));
        EXPECT_EQ(get_option(copy, s.path), v);
        EXPECT_FALSE(s.help.empty());
    }
    EXPECT_EQ(find_flag("--no-such-flag"), nullptr);
    EXPECT_THROW(set_option(o, "penalty.nonexistent", "1"), std::invalid_argument);
    EXPECT_THROW(get_option(o, "nonexistent"), std::invalid_argument);
    // the options of 2.0 removed in 2.1, and the command-line flags of 1.x
    for (const char *path : {"algorithm.scheme", "algorithm.sgs_tau", "algorithm.scheme_switch_iteration", "algorithm.sgs_iterations",
                             "algorithm.sgs_tau_reduction", "penalty.step_schedule", "penalty.win_ratio", "penalty.guard", "penalty.residual_weight",
                             "penalty.schedule", "penalty.pingpong_window", "penalty.pingpong_net", "penalty.pingpong_min_iter",
                             "penalty.freeze_value", "penalty.freeze_bias", "penalty.sgs_update_threshold", "penalty.sgs_update_stage_1",
                             "penalty.sgs_update_stage_2", "penalty.sgs_factor", "scaling.enabled", "linsys.autotune", "psd.jacobi_tol",
                             "psd.jacobi_max_sweeps", "psd.large.final_product", "psd.batched.final_product", "determinism"})
        EXPECT_THROW(set_option(o, path, "1"), std::invalid_argument) << path;
    // the 19 flags of 1.x that 2.0 accepted as aliases
    for (const char *flag : {"--algorithm", "--checkpoint-iters", "--lobpcg", "--penalty-factor", "--save-final-dir", "--save-solution",
                             "--sig", "--sig-update-stage-1", "--sig-update-stage-2", "--sig-update-threshold", "--sigma-policy",
                             "--sigscale", "--streams", "--switch-admm", "--switch-proj-iter", "--switch-proj-tol",
                             "--validate-interval", "--validate-threads", "--validate-tol"})
        EXPECT_EQ(find_flag(flag), nullptr) << flag;
}

TEST(Options, ValuesParseAndPrint)
{
    SolverOptions o;
    set_option(o, "penalty.policy", "fixed");
    EXPECT_EQ(o.penalty.policy, PenaltyPolicyKind::Fixed);
    set_option(o, "penalty.policy", "balance_freeze");
    EXPECT_EQ(o.penalty.policy, PenaltyPolicyKind::BalanceFreeze);
    for (const char *removed : {"legacy_adaptive", "residual_balancing", "fixed_sgs_then_legacy_admm", "adaptive2"})
        EXPECT_THROW(set_option(o, "penalty.policy", removed), std::invalid_argument) << removed;
    set_option(o, "penalty.freeze_iteration", "100000");
    EXPECT_EQ(o.penalty.freeze_iteration, 100000);
    EXPECT_EQ(get_option(o, "penalty.freeze_iteration"), "100000");
    set_option(o, "penalty.factor", "1.2");
    EXPECT_EQ(get_option(o, "penalty.factor"), "1.2");
    set_option(o, "penalty.sigma0", "1e-4");
    EXPECT_EQ(o.penalty.sigma0, 1e-4);
    EXPECT_EQ(std::strtod(get_option(o, "penalty.sigma0").c_str(), nullptr), 1e-4);
    set_option(o, "algorithm.tau", "1.618");
    EXPECT_EQ(get_option(o, "algorithm.tau"), "1.618"); // shortest round-trip, not 1.6180000000000001
    EXPECT_THROW(set_option(o, "algorithm.max_iter", "1e6"), std::invalid_argument); // strict integers (today's CLI)
    EXPECT_THROW(set_option(o, "algorithm.max_iter", "12abc"), std::invalid_argument);
    set_option(o, "algorithm.max_iter", "400");
    EXPECT_EQ(o.algorithm.max_iter, 400);
    set_option(o, "algorithm.time_limit", "inf");
    EXPECT_TRUE(std::isinf(o.algorithm.time_limit));
    set_option(o, "validation.checkpoint_iterations", "150,300");
    EXPECT_EQ(o.validation.checkpoint_iterations, (std::vector<int>{150, 300}));
    EXPECT_EQ(get_option(o, "validation.checkpoint_iterations"), "150,300");
    EXPECT_THROW(set_option(o, "validation.checkpoint_iterations", "1e4"), std::invalid_argument);
    set_option(o, "psd.lowrank.enabled", "0");
    EXPECT_FALSE(o.psd.lowrank_enabled);
    // siopt-rebuttal branch: the FP32 warm-up, the reject policy, the dumps and the time-based validation cadence
    set_option(o, "psd.large.warmup", "composite_fp32");
    EXPECT_EQ(o.psd.warmup, WarmupKind::CompositeFp32);
    EXPECT_EQ(get_option(o, "psd.large.warmup"), "composite_fp32");
    EXPECT_THROW(set_option(o, "psd.large.warmup", "composite_fp64"), std::invalid_argument);
    set_option(o, "psd.composite.on_reject", "keep");
    EXPECT_EQ(o.psd.composite_on_reject, CompositeOnReject::Keep);
    EXPECT_EQ(get_option(o, "psd.composite.on_reject"), "keep");
    set_option(o, "psd.composite.dump_iterations", "10,500");
    EXPECT_EQ(o.psd.composite_dump_iterations, (std::vector<int>{10, 500}));
    set_option(o, "validation.time_interval", "60");
    EXPECT_EQ(o.validation.time_interval, 60.0);
    EXPECT_EQ(get_option(o, "validation.time_interval"), "60");
    set_option(o, "validation.target_tols", "1e-2,1e-3");
    ASSERT_EQ(o.validation.target_tols.size(), 2u);
    EXPECT_DOUBLE_EQ(o.validation.target_tols[1], 1e-3);
    // the eigenvalues of the external validation: on the device by default, LAPACK on request
    EXPECT_EQ(SolverOptions().validation.device, ValidationDevice::Gpu);
    EXPECT_EQ(get_option(SolverOptions(), "validation.device"), "gpu");
    set_option(o, "validation.device", "cpu");
    EXPECT_EQ(o.validation.device, ValidationDevice::Cpu);
    EXPECT_EQ(get_option(o, "validation.device"), "cpu");
    EXPECT_THROW(set_option(o, "validation.device", "fpga"), std::invalid_argument);
    set_option(o, "io.sense", "max");
    ASSERT_TRUE(o.io.sense.has_value());
    EXPECT_EQ(*o.io.sense, Sense::Maximize);
    set_option(o, "io.sense", "minimize");
    EXPECT_EQ(*o.io.sense, Sense::Minimize);
    set_option(o, "seed", "42");
    EXPECT_EQ(o.seed, 42u);
    EXPECT_THROW(set_option(o, "seed", "-1"), std::invalid_argument);
}

// A minimal JSON validator (objects, arrays, strings with escapes, numbers, true/false/null) for the summary tests.
namespace options_test
{
inline bool skip_ws(const std::string &s, std::size_t &i)
{
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r'))
        i++;
    return i < s.size();
}
inline bool parse_value(const std::string &s, std::size_t &i);
inline bool parse_string(const std::string &s, std::size_t &i)
{
    if (i >= s.size() || s[i] != '"')
        return false;
    for (i++; i < s.size(); i++)
    {
        if (s[i] == '\\')
        {
            i++;
            continue;
        }
        if (s[i] == '"')
        {
            i++;
            return true;
        }
    }
    return false;
}
inline bool parse_number(const std::string &s, std::size_t &i)
{
    const std::size_t start = i;
    if (i < s.size() && s[i] == '-')
        i++;
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i])))
        return false;
    while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-'))
        i++;
    return i > start;
}
inline bool parse_value(const std::string &s, std::size_t &i)
{
    if (!skip_ws(s, i))
        return false;
    if (s[i] == '{')
    {
        i++;
        if (skip_ws(s, i) && s[i] == '}')
        {
            i++;
            return true;
        }
        while (true)
        {
            if (!skip_ws(s, i) || !parse_string(s, i) || !skip_ws(s, i) || s[i] != ':')
                return false;
            i++;
            if (!parse_value(s, i) || !skip_ws(s, i))
                return false;
            if (s[i] == ',')
            {
                i++;
                continue;
            }
            if (s[i] == '}')
            {
                i++;
                return true;
            }
            return false;
        }
    }
    if (s[i] == '[')
    {
        i++;
        if (skip_ws(s, i) && s[i] == ']')
        {
            i++;
            return true;
        }
        while (true)
        {
            if (!parse_value(s, i) || !skip_ws(s, i))
                return false;
            if (s[i] == ',')
            {
                i++;
                continue;
            }
            if (s[i] == ']')
            {
                i++;
                return true;
            }
            return false;
        }
    }
    if (s[i] == '"')
        return parse_string(s, i);
    if (s.compare(i, 4, "true") == 0 || s.compare(i, 4, "null") == 0)
    {
        i += 4;
        return true;
    }
    if (s.compare(i, 5, "false") == 0)
    {
        i += 5;
        return true;
    }
    return parse_number(s, i);
}
inline bool is_valid_json(const std::string &s)
{
    std::size_t i = 0;
    return parse_value(s, i) && !skip_ws(s, i);
}
} // namespace options_test

TEST(Options, OptionsJsonIsValidJson)
{
    using options_test::is_valid_json;
    EXPECT_TRUE(is_valid_json("{\"a\": 1, \"b\": [1.5e-3, \"x\", true, null], \"c\": {}}"));
    EXPECT_FALSE(is_valid_json("{\"a\": tau}"));
    EXPECT_FALSE(is_valid_json("{\"a\": 1,}"));
    SolverOptions o; // the defaults carry the symbolic values "inf", "none", "algorithm.tol", ...
    const std::string js = options_json(o);
    EXPECT_TRUE(is_valid_json(js)) << js;
    EXPECT_NE(js.find("\"algorithm.time_limit\": \"inf\""), std::string::npos) << js;
    EXPECT_NE(js.find("\"validation.tol\": \"algorithm.tol\""), std::string::npos) << js;
    EXPECT_NE(js.find("\"penalty.sigma_min\": 0.001"), std::string::npos) << js;
    EXPECT_NE(js.find("\"penalty.freeze_iteration\": 10000"), std::string::npos) << js;
    EXPECT_NE(js.find("\"algorithm.tau\": 1.618"), std::string::npos) << js;
    EXPECT_NE(js.find("\"psd.lowrank.enabled\": true"), std::string::npos) << js;
    set_option(o, "penalty.factor", "1.2");
    set_option(o, "algorithm.time_limit", "600");
    set_option(o, "validation.checkpoint_iterations", "10,20");
    set_option(o, "output.tag", "a \"quoted\" tag");
    const std::string js2 = options_json(o);
    EXPECT_TRUE(is_valid_json(js2)) << js2;
    EXPECT_NE(js2.find("\"penalty.factor\": 1.2"), std::string::npos) << js2;
    EXPECT_NE(js2.find("\"algorithm.time_limit\": 600"), std::string::npos) << js2;
}

TEST(Options, ValidateRejectsInvalidValues)
{
    auto rejects = [](const std::string &path, const std::string &value)
    {
        SolverOptions o;
        set_option(o, path, value);
        EXPECT_THROW(o.validate(), std::invalid_argument) << path << " = " << value;
    };
    SolverOptions defaults;
    EXPECT_NO_THROW(defaults.validate());
    rejects("algorithm.tau", "2");
    rejects("algorithm.tau", "0");
    rejects("algorithm.tol", "0");
    rejects("algorithm.tol", "-1");
    rejects("algorithm.max_iter", "2147483647");
    rejects("penalty.sigma0", "0");
    rejects("penalty.factor", "0");
    rejects("penalty.factor", "1");
    rejects("penalty.sigma_min", "0");
    rejects("penalty.sigma_max", "1e-4"); // below sigma_min
    rejects("penalty.freeze_iteration", "-1");
    rejects("psd.medium_streams", "0");
    rejects("validation.tol", "1e-4");  // needs an interval
    rejects("validation.checkpoint_iterations", "100");
    rejects("validation.strict_dimacs_tol", "1e-6");
    rejects("linsys.regularization", "1"); // out of [0, 1e-3]
    rejects("psd.small_max", "64");        // the small tier holds n <= 32
    rejects("psd.medium_max", "32");       // above psd.small_max
    rejects("psd.medium.max_batch", "-1");
    rejects("psd.lowrank.ratio", "0");
    rejects("psd.composite.margin", "0.5");
    rejects("psd.lowrank.start_iter", "1");
    rejects("validation.time_interval", "-1");
    rejects("validation.target_tols", "0,1e-3");
    rejects("psd.composite.dump_iterations", "0");
    rejects("psd.composite.dump_iterations", "100"); // needs a dump directory
    {
        SolverOptions o;
        set_option(o, "penalty.sigma_min", "10");
        set_option(o, "penalty.sigma_max", "1");
        EXPECT_THROW(o.validate(), std::invalid_argument);
    }
    {
        SolverOptions o;
        set_option(o, "validation.interval", "100");
        set_option(o, "validation.tol", "1e-4");
        set_option(o, "validation.checkpoint_iterations", "100");
        EXPECT_NO_THROW(o.validate());
    }
    {
        // the time-based cadence alone enables the validated stopping
        SolverOptions o;
        set_option(o, "validation.time_interval", "30");
        set_option(o, "validation.tol", "1e-4");
        EXPECT_NO_THROW(o.validate());
    }
}

TEST(Options, DocumentationCoversEveryOption)
{
    const std::string md = option_table_markdown(), help = option_table_help(), json = options_json(SolverOptions());
    for (const OptionSpec &s : option_table())
    {
        EXPECT_NE(md.find("`" + s.path + "`"), std::string::npos) << s.path;
        EXPECT_NE(help.find(s.flag), std::string::npos) << s.flag;
        EXPECT_NE(json.find("\"" + s.path + "\": "), std::string::npos) << s.path;
    }
    EXPECT_NE(json.find("\"penalty.policy\": \"balance_freeze\""), std::string::npos);
    EXPECT_NE(json.find("\"algorithm.tau\": 1.618"), std::string::npos);
    EXPECT_NE(json.find("\"psd.lowrank.enabled\": true"), std::string::npos);
}
