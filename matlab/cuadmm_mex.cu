/*

    cuadmm_mex.cu

    MEX gateway of cuADMM 2.0 on SolverOptions (plan Phase 2, decision Q6):

        [X, y, S, info] = cuadmm_mex(At, b, C, K, opts)

    At (vec_len x m, sparse), b (m x 1, sparse), C (vec_len x 1, sparse) in svec form (as today); K describes the
    cones in input order, either SeDuMi-style with fields f (free entries), l (nonnegative entries), s (vector of PSD
    sizes) taken in the order f, l, s, or with fields types (char row/column of 's', 'l', 'u') and sizes (vector) for an
    arbitrary order. opts is a struct whose nested fields are the option paths of the option table
    (opts.penalty.policy = 'fixed', opts.algorithm.max_iter = 1000, ...; strings or numbers); unknown fields are
    errors. A warm start comes from opts.warm_start.X / y / S (dense vectors, original units).

    info is a struct: iterations, status, stop_reason, pobj, dobj, eta_p, eta_d, eta_g, cone_violation_X,
    cone_violation_S, sigma0, sigma_final, sigma_changes, time_s, setup_time_s, solve_time_s, history (matrix with the
    columns iter, time_s, pobj, dobj, eta_p, eta_d, eta_g, sigma, tau), sigma_log (matrix iter, old, new),
    certificate (struct dual_block_min, dual_cone_coef), penalty_policy, build_commit.

    The positional signature of main, cuadmm_MATLAB(eig_stream_num_per_gpu, max_iter, stop_tol, At, b, C, blk_types,
    blk_sizes, X0, y0, S0, sig, ...), is served by the MATLAB shim +cuadmm/solve.m for one release.

    Not yet built or tested against MATLAB (no MATLAB on the development machines); the source compiles against
    stub declarations of the MEX API (nvcc -c with -Wall, 2026-10-23).

*/

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mex.h"
#include "matrix.h"

#include "cuadmm/cuadmm.h"
#include "cuadmm/version.h"

namespace
{

class InputError : public std::runtime_error
{
public:
    explicit InputError(const std::string &msg) : std::runtime_error(msg) {}
};

void require_sparse_double(const mxArray *arr, const char *name, bool column)
{
    if (!mxIsDouble(arr) || mxIsComplex(arr) || !mxIsSparse(arr) || (column && mxGetN(arr) != 1))
        throw InputError(std::string(name) + " must be a real sparse double " + (column ? "column vector" : "matrix"));
}

// MATLAB's sparse arrays are CSC with size_t indices and 0-based values
void read_sparse_matrix(const mxArray *arr, cuadmm::CscMatrix &M)
{
    M.rows = static_cast<std::int64_t>(mxGetM(arr));
    M.cols = static_cast<std::int64_t>(mxGetN(arr));
    const mwIndex *jc = mxGetJc(arr), *ir = mxGetIr(arr);
    const double *pr = mxGetPr(arr);
    M.colptr.assign(jc, jc + M.cols + 1);
    const std::int64_t nnz = M.colptr.back();
    M.rowind.assign(ir, ir + nnz);
    M.val.assign(pr, pr + nnz);
}

void read_sparse_vector(const mxArray *arr, cuadmm::SparseVector &v)
{
    v.size = static_cast<std::int64_t>(mxGetM(arr));
    const mwIndex *jc = mxGetJc(arr), *ir = mxGetIr(arr);
    const double *pr = mxGetPr(arr);
    const std::int64_t nnz = jc[1];
    v.index.assign(ir, ir + nnz);
    v.val.assign(pr, pr + nnz);
}

std::vector<double> read_dense_vector(const mxArray *arr, const char *name)
{
    if (!mxIsDouble(arr) || mxIsComplex(arr) || mxIsSparse(arr))
        throw InputError(std::string(name) + " must be a real dense double vector");
    const double *pr = mxGetPr(arr);
    return std::vector<double>(pr, pr + mxGetNumberOfElements(arr));
}

std::string read_string(const mxArray *arr, const std::string &what)
{
    if (mxIsChar(arr))
    {
        char *c = mxArrayToString(arr);
        std::string s(c ? c : "");
        mxFree(c);
        return s;
    }
    if (mxIsLogicalScalar(arr))
        return mxIsLogicalScalarTrue(arr) ? "1" : "0";
    if (mxIsNumeric(arr) && mxGetNumberOfElements(arr) == 1)
    {
        const double v = mxGetScalar(arr);
        if (!std::isfinite(v))
            return v > 0 ? "inf" : "-inf";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", v);
        return buf;
    }
    if (mxIsNumeric(arr))
    {
        // a numeric vector: comma-separated list
        std::string out;
        const double *pr = mxGetPr(arr);
        for (std::size_t i = 0; i < mxGetNumberOfElements(arr); i++)
        {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g", pr[i]);
            out += (i ? "," : "") + std::string(buf);
        }
        return out;
    }
    throw InputError(what + " must be a string, a number, a logical or a numeric vector");
}

// cones from K
void read_cones(const mxArray *K, std::vector<cuadmm::ConeSpec> &cones)
{
    if (!mxIsStruct(K))
        throw InputError("K must be a struct (fields f, l, s, or types and sizes)");
    const mxArray *types = mxGetField(K, 0, "types"), *sizes = mxGetField(K, 0, "sizes");
    if (types || sizes)
    {
        if (!types || !sizes || !mxIsChar(types))
            throw InputError("K.types (char) and K.sizes (numeric) must both be given");
        char *t = mxArrayToString(types);
        const std::string codes(t ? t : "");
        mxFree(t);
        const std::vector<double> n = read_dense_vector(sizes, "K.sizes");
        if (codes.size() != n.size())
            throw InputError("K.types and K.sizes must have the same length");
        for (std::size_t i = 0; i < n.size(); i++)
        {
            cuadmm::ConeSpec c;
            c.kind = cuadmm::cone_kind_of(codes[i]);
            c.size = static_cast<std::int64_t>(n[i]);
            cones.push_back(c);
        }
        return;
    }
    auto count_field = [&](const char *name) -> std::int64_t
    {
        const mxArray *f = mxGetField(K, 0, name);
        if (!f || mxIsEmpty(f))
            return 0;
        return static_cast<std::int64_t>(mxGetScalar(f));
    };
    const std::int64_t f = count_field("f"), l = count_field("l");
    if (f > 0)
        cones.push_back({cuadmm::ConeKind::Free, f, 0.0});
    if (l > 0)
        cones.push_back({cuadmm::ConeKind::Nonnegative, l, 0.0});
    const mxArray *q = mxGetField(K, 0, "q"), *r = mxGetField(K, 0, "r");
    if ((q && !mxIsEmpty(q)) || (r && !mxIsEmpty(r)))
        throw InputError("second-order cones (K.q, K.r) come with plan Phase 9");
    const mxArray *s = mxGetField(K, 0, "s");
    if (s && !mxIsEmpty(s))
        for (double n : read_dense_vector(s, "K.s"))
            cones.push_back({cuadmm::ConeKind::Psd, static_cast<std::int64_t>(n), 0.0});
}

// opts: nested structs whose leaves are option paths
void read_options(const mxArray *opts, const std::string &prefix, cuadmm::SolverOptions &options, cuadmm::WarmStart &warm, bool &has_warm)
{
    if (!mxIsStruct(opts))
        throw InputError("opts" + (prefix.empty() ? "" : "." + prefix) + " must be a struct");
    const int n = mxGetNumberOfFields(opts);
    for (int i = 0; i < n; i++)
    {
        const std::string name = mxGetFieldNameByNumber(opts, i);
        const mxArray *value = mxGetFieldByNumber(opts, 0, i);
        const std::string path = prefix.empty() ? name : prefix + "." + name;
        if (path == "warm_start")
        {
            if (!mxIsStruct(value))
                throw InputError("opts.warm_start must be a struct with fields X, y, S (and optionally sigma)");
            const mxArray *X = mxGetField(value, 0, "X"), *y = mxGetField(value, 0, "y"), *S = mxGetField(value, 0, "S"), *sg = mxGetField(value, 0, "sigma");
            if (!X || !y || !S)
                throw InputError("opts.warm_start needs X, y and S");
            warm.X = read_dense_vector(X, "warm_start.X");
            warm.y = read_dense_vector(y, "warm_start.y");
            warm.S = read_dense_vector(S, "warm_start.S");
            if (sg)
                warm.sigma = mxGetScalar(sg);
            has_warm = true;
            continue;
        }
        if (mxIsStruct(value))
        {
            read_options(value, path, options, warm, has_warm);
            continue;
        }
        cuadmm::set_option(options, path, read_string(value, "opts." + path)); // unknown paths are errors
    }
}

mxArray *dense_vector(const std::vector<double> &v)
{
    mxArray *out = mxCreateDoubleMatrix(static_cast<mwSize>(v.size()), 1, mxREAL);
    std::memcpy(mxGetPr(out), v.data(), sizeof(double) * v.size());
    return out;
}

void set_field(mxArray *s, const char *name, mxArray *value)
{
    mxAddField(s, name);
    mxSetField(s, 0, name, value);
}

void cuadmm_mex(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 4 || nrhs > 5)
        throw InputError("usage: [X, y, S, info] = cuadmm_mex(At, b, C, K, opts)");
    if (nlhs > 4)
        throw InputError("cuadmm_mex returns at most 4 outputs");
    require_sparse_double(prhs[0], "At", false);
    require_sparse_double(prhs[1], "b", true);
    require_sparse_double(prhs[2], "C", true);

    cuadmm::ProblemData problem;
    read_sparse_matrix(prhs[0], problem.At);
    read_sparse_vector(prhs[1], problem.b);
    read_sparse_vector(prhs[2], problem.C);
    read_cones(prhs[3], problem.cones);
    problem.source.format = "matlab";
    problem.validate(); // sizes against the cones, indices in range

    cuadmm::SolverOptions options;
    cuadmm::WarmStart warm;
    bool has_warm = false;
    if (nrhs == 5)
        read_options(prhs[4], "", options, warm, has_warm);
    options.validate();

    cuadmm::Solver solver(options);
    solver.setup(problem);
    const cuadmm::SolveResult r = solver.solve(has_warm ? &warm : nullptr);

    if (nlhs >= 1 || true)
        plhs[0] = dense_vector(r.best.X);
    if (nlhs >= 2)
        plhs[1] = dense_vector(r.best.y);
    if (nlhs >= 3)
        plhs[2] = dense_vector(r.best.S);
    if (nlhs >= 4)
    {
        mxArray *info = mxCreateStructMatrix(1, 1, 0, nullptr);
        set_field(info, "iterations", mxCreateDoubleScalar(r.iterations));
        set_field(info, "status", mxCreateString(cuadmm::name_of(r.status)));
        set_field(info, "stop_reason", mxCreateString(r.stop_reason.c_str()));
        set_field(info, "converged", mxCreateLogicalScalar(r.converged));
        set_field(info, "pobj", mxCreateDoubleScalar(r.pobj));
        set_field(info, "dobj", mxCreateDoubleScalar(r.dobj));
        set_field(info, "eta_p", mxCreateDoubleScalar(r.eta_p));
        set_field(info, "eta_d", mxCreateDoubleScalar(r.eta_d));
        set_field(info, "eta_g", mxCreateDoubleScalar(r.eta_g));
        set_field(info, "cone_violation_X", mxCreateDoubleScalar(r.cone_violation_X));
        set_field(info, "cone_violation_S", mxCreateDoubleScalar(r.cone_violation_S));
        set_field(info, "sigma0", mxCreateDoubleScalar(r.sigma0));
        set_field(info, "sigma_final", mxCreateDoubleScalar(r.sigma_final));
        set_field(info, "sigma_changes", mxCreateDoubleScalar(r.sigma_changes));
        set_field(info, "penalty_policy", mxCreateString(r.penalty_policy.c_str()));
        set_field(info, "time_s", mxCreateDoubleScalar(r.total_time_s));
        set_field(info, "setup_time_s", mxCreateDoubleScalar(r.setup_time_s));
        set_field(info, "solve_time_s", mxCreateDoubleScalar(r.solve_time_s));
        set_field(info, "y_solves", mxCreateDoubleScalar(static_cast<double>(r.y_solves)));
        mxArray *history = mxCreateDoubleMatrix(static_cast<mwSize>(r.history.size()), 9, mxREAL);
        double *h = mxGetPr(history);
        const std::size_t n = r.history.size();
        for (std::size_t k = 0; k < n; k++)
        {
            const cuadmm::HistoryRow &row = r.history[k];
            const double cols[9] = {double(row.iteration), row.time_s, row.pobj, row.dobj, row.eta_p, row.eta_d, row.eta_g, row.sigma, row.tau};
            for (int c = 0; c < 9; c++)
                h[c * n + k] = cols[c];
        }
        set_field(info, "history", history);
        mxArray *slog = mxCreateDoubleMatrix(static_cast<mwSize>(r.sigma_log.size()), 3, mxREAL);
        double *sl = mxGetPr(slog);
        for (std::size_t k = 0; k < r.sigma_log.size(); k++)
        {
            sl[k] = r.sigma_log[k].iteration;
            sl[r.sigma_log.size() + k] = r.sigma_log[k].old_sigma;
            sl[2 * r.sigma_log.size() + k] = r.sigma_log[k].new_sigma;
        }
        set_field(info, "sigma_log", slog);
        mxArray *cert = mxCreateStructMatrix(1, 1, 0, nullptr);
        set_field(cert, "dual_block_min", dense_vector(r.dual_block_min));
        set_field(cert, "dual_cone_coef", dense_vector(r.dual_cone_coef));
        set_field(info, "certificate", cert);
        set_field(info, "build_commit", mxCreateString(CUADMM_BUILD_COMMIT));
        plhs[3] = info;
    }
}

} // namespace

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // mexErrMsgIdAndTxt does not unwind the C++ stack: every C++ object must be gone before it is called
    const char *err_id = "cuADMM:solverError";
    char err_msg[2048];
    try
    {
        cuadmm_mex(nlhs, plhs, nrhs, prhs);
        return;
    }
    catch (const InputError &e)
    {
        err_id = "cuADMM:invalidInput";
        std::snprintf(err_msg, sizeof(err_msg), "%s", e.what());
    }
    catch (const std::invalid_argument &e)
    {
        err_id = "cuADMM:invalidInput";
        std::snprintf(err_msg, sizeof(err_msg), "%s", e.what());
    }
    catch (const std::exception &e)
    {
        std::snprintf(err_msg, sizeof(err_msg), "%s", e.what());
    }
    catch (...)
    {
        std::snprintf(err_msg, sizeof(err_msg), "unknown C++ exception");
    }
    mexErrMsgIdAndTxt(err_id, "%s", err_msg);
}
