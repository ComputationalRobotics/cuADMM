/*

    solution_writers.cc

    Solutions in the convention of the input (plan Section 12, "Output"), from the returned iterate X, y, S in
    cuADMM's svec layout:
      SDPA:   x = -y, X_SDPA = S, Y_SDPA = X, in the sparse initial-point layout (x on the first line, then
              "1 blk i j v" for X_SDPA and "2 blk i j v" for Y_SDPA, upper triangles), plus a JSON sidecar with
              the objective values in SDPA's convention;
      SeDuMi: x, y, z as full vectors (PSD blocks as column-major n^2);
      SDPT3:  cell arrays X, Z (full matrices), y, obj = [pobj; dobj];
      MOSEK:  res.sol.itr with barx, bars (lower triangles, column by column), xx, slx, sux, y, slc, suc, pobjval,
              dobjval, so that SPOT's recover_mosek_sol_blk works unchanged. For a maximization MOSEK's dual
              variables have the opposite sign (y = -y_cuADMM, bars = -S) and the slacks are non-positive.
    Objective values are in the file's sense, offset included (SolveResult::pobj / dobj).

*/

#include "io/readers.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "io/svec.h"
#include "report/writers.h"

#ifdef CUADMM_HAVE_MATIO
#include "io/mat_common.h"
#endif

namespace cuadmm
{
namespace io
{
namespace detail
{

namespace
{

// the output units of a solution in a layout that follows the file's blocks: the original blocks when the cone has
// one (emitted once, in order), else the cone itself (blocks the reader created)
struct OutputUnit
{
    char kind;
    std::int64_t size;
    std::vector<double> X, S; // full n*n for 's', n otherwise
};

std::vector<OutputUnit> output_units(const ProblemData &problem, const SolveResult &result)
{
    std::vector<OriginalBlockValues> X = expand_to_original_blocks(problem, result.best.X);
    std::vector<OriginalBlockValues> S = expand_to_original_blocks(problem, result.best.S);
    const std::vector<std::int64_t> off = cone_offsets(problem.cones);
    std::vector<OutputUnit> units;
    std::vector<char> emitted(X.size(), 0);
    for (std::size_t b = 0; b < problem.cones.size(); b++)
    {
        const int o = b < problem.source.original_block_of.size() ? problem.source.original_block_of[b] : -1;
        if (o >= 0)
        {
            if (emitted[static_cast<std::size_t>(o)])
                continue;
            emitted[static_cast<std::size_t>(o)] = 1;
            OutputUnit u;
            u.kind = X[static_cast<std::size_t>(o)].kind;
            u.size = X[static_cast<std::size_t>(o)].size;
            u.X = std::move(X[static_cast<std::size_t>(o)].values);
            u.S = std::move(S[static_cast<std::size_t>(o)].values);
            units.push_back(std::move(u));
        }
        else
        {
            OutputUnit u;
            u.kind = cone_code(problem.cones[b].kind);
            u.size = problem.cones[b].size;
            u.X.assign(result.best.X.begin() + off[b], result.best.X.begin() + off[b + 1]);
            u.S.assign(result.best.S.begin() + off[b], result.best.S.begin() + off[b + 1]);
            units.push_back(std::move(u));
        }
    }
    return units;
}

void check_result_sizes(const ProblemData &problem, const SolveResult &result)
{
    if (static_cast<std::int64_t>(result.best.X.size()) != problem.vec_len() || static_cast<std::int64_t>(result.best.S.size()) != problem.vec_len() ||
        static_cast<std::int64_t>(result.best.y.size()) != problem.num_constraints())
        throw std::invalid_argument("write_solution: the result does not match the problem sizes");
}

std::ofstream open_out(const std::string &path)
{
    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("could not open " + path + " for writing");
    return f;
}

} // namespace

void write_solution_sdpa(const std::string &path, const ProblemData &problem, const SolveResult &result)
{
    check_result_sizes(problem, result);
    const std::vector<OutputUnit> units = output_units(problem, result);
    std::ofstream f = open_out(path);
    // x_SDPA = -y
    for (std::size_t i = 0; i < result.best.y.size(); i++)
        f << (i ? " " : "") << format_double(-result.best.y[i]);
    f << "\n";
    // X_SDPA = S (1), Y_SDPA = X (2); upper triangles, nonzeros
    for (int which = 1; which <= 2; which++)
        for (std::size_t b = 0; b < units.size(); b++)
        {
            const OutputUnit &u = units[b];
            const std::vector<double> &v = which == 1 ? u.S : u.X;
            if (u.kind == 's')
            {
                for (std::int64_t l = 0; l < u.size; l++)
                    for (std::int64_t k = 0; k <= l; k++)
                    {
                        const double x = v[static_cast<std::size_t>(k + l * u.size)];
                        if (x != 0.0)
                            f << which << " " << b + 1 << " " << k + 1 << " " << l + 1 << " " << format_double(x) << "\n";
                    }
            }
            else
                for (std::int64_t k = 0; k < u.size; k++)
                    if (v[static_cast<std::size_t>(k)] != 0.0)
                        f << which << " " << b + 1 << " " << k + 1 << " " << k + 1 << " " << format_double(v[static_cast<std::size_t>(k)]) << "\n";
        }
    if (!f.good())
        throw std::runtime_error("could not write " + path);
    // the objective values in SDPA's convention: primal c^T x = -b^T y, dual <F_0, Y> = -<C, X>
    std::string json_path = path;
    if (ends_with(lower(json_path), ".ini-s"))
        json_path = json_path.substr(0, json_path.size() - 6);
    json_path += ".json";
    std::ofstream j = open_out(json_path);
    j << "{\"format\": \"sdpa\", \"objValPrimal\": " << report::json_number(result.dobj) << ", \"objValDual\": " << report::json_number(result.pobj)
      << ", \"status\": " << report::json_string(name_of(result.status)) << ", \"iterations\": " << result.iterations
      << ", \"eta_p\": " << report::json_number(result.eta_p) << ", \"eta_d\": " << report::json_number(result.eta_d)
      << ", \"eta_g\": " << report::json_number(result.eta_g) << ", \"layout\": \"x = -y_cuADMM; 1: X_SDPA = S_cuADMM, 2: Y_SDPA = X_cuADMM (upper triangles)\"}\n";
    j.close();
    if (!j.good())
        throw std::runtime_error("could not write " + json_path);
}

#ifdef CUADMM_HAVE_MATIO

namespace
{
bool write_v73(const ProblemData &problem) { return problem.source.mat_version == "7.3"; }
}

void write_solution_sedumi(const std::string &path, const ProblemData &problem, const SolveResult &result)
{
    check_result_sizes(problem, result);
    const std::vector<OutputUnit> units = output_units(problem, result);
    std::vector<double> x, z;
    for (const OutputUnit &u : units)
    {
        x.insert(x.end(), u.X.begin(), u.X.end());
        z.insert(z.end(), u.S.begin(), u.S.end());
    }
    MatFile file = MatFile::create(path, write_v73(problem));
    write_variable(file, make_double_matrix("x", static_cast<std::int64_t>(x.size()), 1, x.data()));
    write_variable(file, make_double_matrix("y", static_cast<std::int64_t>(result.best.y.size()), 1, result.best.y.data()));
    write_variable(file, make_double_matrix("z", static_cast<std::int64_t>(z.size()), 1, z.data()));
    write_variable(file, make_double_matrix("pobj", 1, 1, &result.pobj));
    write_variable(file, make_double_matrix("dobj", 1, 1, &result.dobj));
}

void write_solution_sdpt3(const std::string &path, const ProblemData &problem, const SolveResult &result)
{
    check_result_sizes(problem, result);
    const std::vector<OutputUnit> units = output_units(problem, result);
    std::vector<matvar_t *> X_cells, Z_cells;
    for (const OutputUnit &u : units)
    {
        const std::int64_t rows = u.size, cols = u.kind == 's' ? u.size : 1;
        X_cells.push_back(make_double_matrix("X", rows, cols, u.X.data()));
        Z_cells.push_back(make_double_matrix("Z", rows, cols, u.S.data()));
    }
    MatFile file = MatFile::create(path, write_v73(problem));
    write_variable(file, make_cell("X", static_cast<std::int64_t>(units.size()), 1, X_cells));
    write_variable(file, make_double_matrix("y", static_cast<std::int64_t>(result.best.y.size()), 1, result.best.y.data()));
    write_variable(file, make_cell("Z", static_cast<std::int64_t>(units.size()), 1, Z_cells));
    const double obj[2] = {result.pobj, result.dobj};
    write_variable(file, make_double_matrix("obj", 2, 1, obj));
}

void write_solution_mosek(const std::string &path, const ProblemData &problem, const SolveResult &result)
{
    check_result_sizes(problem, result);
    const SourceInfo &s = problem.source;
    const double sgn = problem.sense == Sense::Maximize ? -1.0 : 1.0; // MOSEK's dual variables for a maximization
    // scalar variables: the file's mapping, or (other inputs) every entry of the non-PSD original blocks in order
    std::vector<std::int64_t> var_position = s.mosek_var_position;
    std::vector<double> var_shift = s.mosek_var_shift, var_sign = s.mosek_var_sign;
    std::vector<std::int64_t> row_position = s.mosek_row_position;
    const std::vector<std::int64_t> off = cone_offsets(problem.cones);
    if (var_position.empty() && row_position.empty() && s.format != "mosek")
    {
        for (std::size_t b = 0; b < problem.cones.size(); b++)
            if (problem.cones[b].kind != ConeKind::Psd &&
                (b >= s.original_block_of.size() || s.original_block_of[b] < 0 || s.original_block_kinds[static_cast<std::size_t>(s.original_block_of[b])] != 's'))
                for (std::int64_t k = 0; k < problem.cones[b].size; k++)
                {
                    var_position.push_back(off[b] + k);
                    var_shift.push_back(0.0);
                    var_sign.push_back(1.0);
                }
        for (std::int64_t i = 0; i < problem.num_constraints(); i++)
            row_position.push_back(i);
    }
    const std::size_t n = var_position.size();
    const std::size_t m = row_position.size();
    // MOSEK's duals: y = sgn * y_cuADMM per file row (0 for a dropped row); slc - suc = y
    std::vector<double> y(m, 0.0), slc(m, 0.0), suc(m, 0.0);
    auto split = [&](double t, double &lower, double &upper)
    {
        // lower - upper = t with both non-negative (minimization) or both non-positive (maximization)
        if (sgn > 0)
        {
            lower = std::max(t, 0.0);
            upper = std::max(-t, 0.0);
        }
        else
        {
            lower = std::min(t, 0.0);
            upper = std::min(-t, 0.0);
        }
    };
    for (std::size_t i = 0; i < m; i++)
    {
        if (row_position[i] < 0)
            continue;
        y[i] = sgn * result.best.y[static_cast<std::size_t>(row_position[i])];
        split(y[i], slc[i], suc[i]);
    }
    // the scalar variables: xx from the mapping; slx - sux = c_j - a_j' y (MOSEK's dual constraint), from the dual
    // slacks of the columns the reader created (a ranged variable has two: x' and its bound slack s, with
    // c_j - a_j' y = S_x' - S_s), or from the data for an eliminated fixed variable
    std::vector<double> xx(n, 0.0), slx(n, 0.0), sux(n, 0.0);
    const bool have_slack = s.mosek_var_slack.size() == n, have_fixed = s.mosek_var_cost.size() == n && s.mosek_fixed_colptr.size() == n + 1;
    for (std::size_t j = 0; j < n; j++)
    {
        if (var_position[j] < 0)
        {
            xx[j] = var_shift[j];
            if (!have_fixed)
                continue;
            double t = s.mosek_var_cost[j];
            for (std::int64_t k = s.mosek_fixed_colptr[j]; k < s.mosek_fixed_colptr[j + 1]; k++)
                t -= s.mosek_fixed_val[static_cast<std::size_t>(k)] * y[static_cast<std::size_t>(s.mosek_fixed_rowind[static_cast<std::size_t>(k)])];
            split(t, slx[j], sux[j]);
            continue;
        }
        xx[j] = var_shift[j] + var_sign[j] * result.best.X[static_cast<std::size_t>(var_position[j])];
        double t = sgn * var_sign[j] * result.best.S[static_cast<std::size_t>(var_position[j])];
        if (have_slack && s.mosek_var_slack[j] >= 0)
            t -= sgn * result.best.S[static_cast<std::size_t>(s.mosek_var_slack[j])]; // minus the dual of the bound slack s
        // split by sign: at an optimum at most one of the two multipliers is nonzero, so this is the exact pair
        // (S_x', S_s) there, and the sign convention holds at every iterate
        split(t, slx[j], sux[j]);
    }
    // bar variables: the original 's' blocks, lower triangles column by column
    std::vector<OriginalBlockValues> X = expand_to_original_blocks(problem, result.best.X);
    std::vector<OriginalBlockValues> S = expand_to_original_blocks(problem, result.best.S);
    std::vector<double> barx, bars;
    for (std::size_t o = 0; o < X.size(); o++)
    {
        if (X[o].kind != 's')
            continue;
        const std::int64_t d = X[o].size;
        for (std::int64_t c = 0; c < d; c++)
            for (std::int64_t r = c; r < d; r++)
            {
                barx.push_back(X[o].values[static_cast<std::size_t>(r + c * d)]);
                bars.push_back(sgn * S[o].values[static_cast<std::size_t>(r + c * d)]);
            }
    }
    const char *solsta = result.converged ? "OPTIMAL" : "UNKNOWN";
    const char *prosta = result.converged ? "PRIMAL_AND_DUAL_FEASIBLE" : "UNKNOWN";
    std::vector<std::pair<std::string, matvar_t *>> itr_fields;
    itr_fields.emplace_back("prosta", make_string("prosta", prosta));
    itr_fields.emplace_back("solsta", make_string("solsta", solsta));
    itr_fields.emplace_back("xx", make_double_matrix("xx", static_cast<std::int64_t>(n), 1, xx.data()));
    itr_fields.emplace_back("slx", make_double_matrix("slx", static_cast<std::int64_t>(n), 1, slx.data()));
    itr_fields.emplace_back("sux", make_double_matrix("sux", static_cast<std::int64_t>(n), 1, sux.data()));
    itr_fields.emplace_back("y", make_double_matrix("y", static_cast<std::int64_t>(m), 1, y.data()));
    itr_fields.emplace_back("slc", make_double_matrix("slc", static_cast<std::int64_t>(m), 1, slc.data()));
    itr_fields.emplace_back("suc", make_double_matrix("suc", static_cast<std::int64_t>(m), 1, suc.data()));
    itr_fields.emplace_back("barx", make_double_matrix("barx", static_cast<std::int64_t>(barx.size()), 1, barx.data()));
    itr_fields.emplace_back("bars", make_double_matrix("bars", static_cast<std::int64_t>(bars.size()), 1, bars.data()));
    itr_fields.emplace_back("pobjval", make_double_matrix("pobjval", 1, 1, &result.pobj));
    itr_fields.emplace_back("dobjval", make_double_matrix("dobjval", 1, 1, &result.dobj));
    matvar_t *itr = make_struct("itr", itr_fields);
    std::vector<std::pair<std::string, matvar_t *>> sol_fields;
    sol_fields.emplace_back("itr", itr);
    matvar_t *sol = make_struct("sol", sol_fields);
    std::vector<std::pair<std::string, matvar_t *>> res_fields;
    res_fields.emplace_back("sol", sol);
    matvar_t *res = make_struct("res", res_fields);
    MatFile file = MatFile::create(path, write_v73(problem));
    write_variable(file, res);
}

#else

namespace
{
[[noreturn]] void no_matio(const std::string &path)
{
    throw std::invalid_argument(path + ": writing .mat solutions needs matio, which is not compiled into this build (CUADMM_WITH_MATIO)");
}
} // namespace
void write_solution_sedumi(const std::string &path, const ProblemData &, const SolveResult &) { no_matio(path); }
void write_solution_sdpt3(const std::string &path, const ProblemData &, const SolveResult &) { no_matio(path); }
void write_solution_mosek(const std::string &path, const ProblemData &, const SolveResult &) { no_matio(path); }

#endif

} // namespace detail

void write_solution(const std::string &path, const ProblemData &problem, const SolveResult &result)
{
    write_solution(path, problem, result, "");
}

void write_solution(const std::string &path, const ProblemData &problem, const SolveResult &result, const std::string &format)
{
    const std::string fmt = format.empty() ? problem.source.format : format;
    if (fmt == "txt" || fmt.empty())
        detail::write_solution_txt(path, problem, result);
    else if (fmt == "h5")
        detail::write_solution_h5(detail::solution_file_path(path, {".h5", ".hdf5"}, "solution.h5"), problem, result);
    else if (fmt == "sdpa")
        detail::write_solution_sdpa(detail::solution_file_path(path, {".ini-s"}, "solution.ini-s"), problem, result);
    else if (fmt == "sedumi")
        detail::write_solution_sedumi(detail::solution_file_path(path, {".mat"}, "solution.mat"), problem, result);
    else if (fmt == "sdpt3")
        detail::write_solution_sdpt3(detail::solution_file_path(path, {".mat"}, "solution.mat"), problem, result);
    else if (fmt == "mosek")
        detail::write_solution_mosek(detail::solution_file_path(path, {".mat"}, "solution.mat"), problem, result);
    else
        throw std::invalid_argument("unknown solution format '" + fmt + "' (txt, h5, sdpa, sedumi, sdpt3, mosek)");
}

} // namespace io
} // namespace cuadmm
