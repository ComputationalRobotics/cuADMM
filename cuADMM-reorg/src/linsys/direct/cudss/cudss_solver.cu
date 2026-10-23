/*

    linsys/direct/cudss/cudss_solver.cu

*/

#include "linsys/direct/cudss/cudss_solver.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <cudss.h>
#include <library_types.h>

#include "core/check.h"

namespace cuadmm
{
namespace linsys
{

namespace
{

const char *cudss_status_string(cudssStatus_t s)
{
    switch (s)
    {
    case CUDSS_STATUS_SUCCESS:
        return "CUDSS_STATUS_SUCCESS";
    case CUDSS_STATUS_NOT_INITIALIZED:
        return "CUDSS_STATUS_NOT_INITIALIZED";
    case CUDSS_STATUS_ALLOC_FAILED:
        return "CUDSS_STATUS_ALLOC_FAILED";
    case CUDSS_STATUS_INVALID_VALUE:
        return "CUDSS_STATUS_INVALID_VALUE";
    case CUDSS_STATUS_NOT_SUPPORTED:
        return "CUDSS_STATUS_NOT_SUPPORTED";
    case CUDSS_STATUS_EXECUTION_FAILED:
        return "CUDSS_STATUS_EXECUTION_FAILED";
    case CUDSS_STATUS_INTERNAL_ERROR:
        return "CUDSS_STATUS_INTERNAL_ERROR";
    case CUDSS_STATUS_IR_FAILED:
        return "CUDSS_STATUS_IR_FAILED";
    default:
        return "CUDSS_STATUS_?";
    }
}

#define CUADMM_CHECK_CUDSS(call)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        const cudssStatus_t cuadmm_status_ = (call);                                                                   \
        if (cuadmm_status_ != CUDSS_STATUS_SUCCESS)                                                                    \
            throw ::cuadmm::Error(std::string("cuDSS error ") + cudss_status_string(cuadmm_status_) + " (" +            \
                                      std::to_string(int(cuadmm_status_)) + ") in " #call,                             \
                                  __FILE__, __LINE__);                                                                 \
    } while (0)

double seconds_since(const std::chrono::steady_clock::time_point &t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::string version_string(int major, int minor, int patch)
{
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

void runtime_version_numbers(int &major, int &minor, int &patch)
{
    major = minor = patch = -1;
    CUADMM_CHECK_CUDSS(cudssGetProperty(MAJOR_VERSION, &major));
    CUADMM_CHECK_CUDSS(cudssGetProperty(MINOR_VERSION, &minor));
    CUADMM_CHECK_CUDSS(cudssGetProperty(PATCH_LEVEL, &patch));
}

inline cudssHandle_t H(cudssContext *p) { return reinterpret_cast<cudssHandle_t>(p); }
inline cudssConfig_t C(cudssConfig *p) { return reinterpret_cast<cudssConfig_t>(p); }
inline cudssData_t D(cudssData *p) { return reinterpret_cast<cudssData_t>(p); }
inline cudssMatrix_t M(cudssMatrix *p) { return reinterpret_cast<cudssMatrix_t>(p); }

} // namespace

std::string CudssSolver::runtime_version()
{
    int major, minor, patch;
    runtime_version_numbers(major, minor, patch);
    return version_string(major, minor, patch);
}

std::string CudssSolver::header_version()
{
    return version_string(CUDSS_VERSION_MAJOR, CUDSS_VERSION_MINOR, CUDSS_VERSION_PATCH);
}

CudssSolver::CudssSolver(bool determinism_strict) : CudssSolver(Options{determinism_strict, true, true}) {}

CudssSolver::CudssSolver(const Options &options) : options_(options)
{
    int major, minor, patch;
    runtime_version_numbers(major, minor, patch);
    if (major != CUDSS_VERSION_MAJOR)
        throw std::runtime_error("CudssSolver: the cuDSS runtime " + version_string(major, minor, patch) +
                                 " does not match the header " + header_version() + " the solver was built with");
    name_ = "cudss " + version_string(major, minor, patch);
    cudssHandle_t handle = nullptr;
    CUADMM_CHECK_CUDSS(cudssCreate(&handle));
    handle_ = reinterpret_cast<cudssContext *>(handle);
    cudssConfig_t config = nullptr;
    CUADMM_CHECK_CUDSS(cudssConfigCreate(&config));
    config_ = reinterpret_cast<cudssConfig *>(config);
    if (options_.determinism_strict)
    {
        int one = 1;
        CUADMM_CHECK_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_DETERMINISTIC_MODE, &one, sizeof(int)));
        int value = 0;
        std::size_t written = 0;
        CUADMM_CHECK_CUDSS(cudssConfigGet(config, CUDSS_CONFIG_DETERMINISTIC_MODE, &value, sizeof(int), &written));
        deterministic_ = written == sizeof(int) && value == 1;
        if (!deterministic_)
            throw std::runtime_error("CudssSolver: CUDSS_CONFIG_DETERMINISTIC_MODE could not be enabled (read back " +
                                     std::to_string(value) + ")");
    }
}

CudssSolver::~CudssSolver()
{
    destroy_objects();
    if (config_ != nullptr)
    {
        const cudssStatus_t s = cudssConfigDestroy(C(config_));
        if (s != CUDSS_STATUS_SUCCESS)
            std::fprintf(stderr, "cuDSS error %s in cudssConfigDestroy\n", cudss_status_string(s));
        config_ = nullptr;
    }
    if (handle_ != nullptr)
    {
        const cudssStatus_t s = cudssDestroy(H(handle_));
        if (s != CUDSS_STATUS_SUCCESS)
            std::fprintf(stderr, "cuDSS error %s in cudssDestroy\n", cudss_status_string(s));
        handle_ = nullptr;
    }
}

Capabilities CudssSolver::capabilities() const
{
    Capabilities c;
    c.exact = true;
    c.device_solves = true;
    c.saddle_point = false;
    c.sigma_dependent_ok = false;
    return c;
}

void CudssSolver::destroy_objects()
{
    auto destroy_matrix = [](cudssMatrix *&p) {
        if (p != nullptr)
        {
            const cudssStatus_t s = cudssMatrixDestroy(M(p));
            if (s != CUDSS_STATUS_SUCCESS)
                std::fprintf(stderr, "cuDSS error %s in cudssMatrixDestroy\n", cudss_status_string(s));
            p = nullptr;
        }
    };
    destroy_matrix(matrix_);
    destroy_matrix(x_);
    destroy_matrix(b_);
    if (data_ != nullptr)
    {
        const cudssStatus_t s = cudssDataDestroy(H(handle_), D(data_));
        if (s != CUDSS_STATUS_SUCCESS)
            std::fprintf(stderr, "cuDSS error %s in cudssDataDestroy\n", cudss_status_string(s));
        data_ = nullptr;
    }
    analyzed_ = false;
    factored_ = false;
}

// the data object and the three matrix objects over the device arrays already uploaded
void CudssSolver::create_objects(bool indefinite)
{
    destroy_objects();
    cudssData_t data = nullptr;
    CUADMM_CHECK_CUDSS(cudssDataCreate(H(handle_), &data));
    data_ = reinterpret_cast<cudssData *>(data);
    cudssMatrix_t matrix = nullptr;
    CUADMM_CHECK_CUDSS(cudssMatrixCreateCsr(&matrix, m_, m_, nnz_, rowptr_.data(), nullptr, colind_.data(), val_.data(), CUDA_R_32I,
                                            CUDA_R_32I, CUDA_R_64F, indefinite ? CUDSS_MTYPE_SYMMETRIC : CUDSS_MTYPE_SPD,
                                            CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO));
    matrix_ = reinterpret_cast<cudssMatrix *>(matrix);
    cudssMatrix_t x = nullptr, b = nullptr;
    CUADMM_CHECK_CUDSS(cudssMatrixCreateDn(&x, m_, 1, m_, xbuf_.data(), CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    x_ = reinterpret_cast<cudssMatrix *>(x);
    CUADMM_CHECK_CUDSS(cudssMatrixCreateDn(&b, m_, 1, m_, bbuf_.data(), CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    b_ = reinterpret_cast<cudssMatrix *>(b);
    indefinite_ = indefinite;
}

void CudssSolver::run_analysis()
{
    CUADMM_CHECK_CUDSS(cudssSetStream(H(handle_), nullptr));
    CUADMM_CHECK_CUDSS(cudssExecute(H(handle_), CUDSS_PHASE_ANALYSIS, C(config_), D(data_), M(matrix_), M(x_), M(b_)));
    CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
    // nnz(L) and the memory estimates of the analysis (int64_t[16]: [0] permanent device, [1] peak device, [2] permanent
    // host, [3] peak host bytes); both queries are informational
    std::size_t written = 0;
    std::int64_t lu_nnz = 0;
    // CUDSS_DATA_LU_NNZ as cuDSS reports it (1.0: 1636 on a dense 50 x 50 system whose factor has 1275 entries, and
    // about 2 nnz(L) on a nearly dense 2000 x 2000 one, so it is not nnz(L) in the CHOLMOD sense; informational)
    if (cudssDataGet(H(handle_), D(data_), CUDSS_DATA_LU_NNZ, &lu_nnz, sizeof(lu_nnz), &written) == CUDSS_STATUS_SUCCESS && written > 0)
        symbolic_.nnz_L = double(lu_nnz);
    std::int64_t estimates[16] = {0};
    written = 0;
    if (cudssDataGet(H(handle_), D(data_), CUDSS_DATA_MEMORY_ESTIMATES, estimates, sizeof(estimates), &written) == CUDSS_STATUS_SUCCESS &&
        written >= 4 * sizeof(std::int64_t))
    {
        symbolic_.device_bytes = std::size_t(std::max<std::int64_t>(estimates[1], estimates[0]));
        symbolic_.host_bytes = std::size_t(std::max<std::int64_t>(estimates[3], estimates[2]));
    }
}

SymbolicInfo CudssSolver::analyze(const NormalSystem &sys)
{
    const auto t0 = std::chrono::steady_clock::now();
    destroy_objects();
    m_ = sys.m();
    symbolic_ = SymbolicInfo();
    factor_ = FactorInfo();
    diag_.clear();
    device_bytes_ = host_bytes_ = 0;

    // the lower triangle in CSR (rows i, columns j <= i): the transpose of the CSC lower triangle, a counting sort
    const HostCscLower &L = sys.lower();
    nnz_ = L.nnz();
    if (nnz_ > std::int64_t(INT_MAX))
        throw std::runtime_error("CudssSolver: the lower triangle has more than INT_MAX nonzeros (32-bit indices)");
    std::vector<int> rowptr(std::size_t(m_) + 1, 0), colind(static_cast<std::size_t>(nnz_));
    std::vector<double> val(static_cast<std::size_t>(nnz_));
    for (std::int64_t k = 0; k < nnz_; k++)
        rowptr[std::size_t(L.rowind[std::size_t(k)]) + 1]++;
    for (int i = 0; i < m_; i++)
        rowptr[std::size_t(i) + 1] += rowptr[std::size_t(i)];
    std::vector<int> next(rowptr.begin(), rowptr.end() - 1);
    for (int j = 0; j < L.n; j++)
        for (int k = L.colptr[std::size_t(j)]; k < L.colptr[std::size_t(j) + 1]; k++)
        {
            const int i = L.rowind[std::size_t(k)];
            const int dst = next[std::size_t(i)]++;
            colind[std::size_t(dst)] = j; // increasing in j within a row
            val[std::size_t(dst)] = L.val[std::size_t(k)];
        }
    rowptr_.allocate(rowptr.size());
    core::to_device(rowptr_.span(), rowptr);
    colind_.allocate(colind.size());
    val_.allocate(val.size());
    if (nnz_ > 0)
    {
        core::to_device(colind_.span(), colind);
        core::to_device(val_.span(), val);
    }
    xbuf_.allocate(std::size_t(std::max(m_, 1)));
    bbuf_.allocate(std::size_t(std::max(m_, 1)));
    xbuf_.zero();
    bbuf_.zero();

    create_objects(!options_.spd_first);
    run_analysis();
    symbolic_.ordering = std::string("cudss_default (runtime ") + name_.substr(6) + ", header " + header_version() + ")";
    symbolic_.seconds = seconds_since(t0);
    analyzed_ = true;
    return symbolic_;
}

// the factorization; returns true on success, else the info and status strings for the message
bool CudssSolver::run_factorization(int *info_out, std::string *status_out)
{
    CUADMM_CHECK_CUDSS(cudssSetStream(H(handle_), nullptr));
    const cudssStatus_t status = cudssExecute(H(handle_), CUDSS_PHASE_FACTORIZATION, C(config_), D(data_), M(matrix_), M(x_), M(b_));
    CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
    int info = 0;
    std::size_t written = 0;
    const cudssStatus_t s2 = cudssDataGet(H(handle_), D(data_), CUDSS_DATA_INFO, &info, sizeof(int), &written);
    if (info_out)
        *info_out = info;
    if (status_out)
        *status_out = std::string(cudss_status_string(status)) + (s2 == CUDSS_STATUS_SUCCESS ? ", info " + std::to_string(info) : ", info unavailable");
    return status == CUDSS_STATUS_SUCCESS && s2 == CUDSS_STATUS_SUCCESS && info == 0;
}

void CudssSolver::read_factor_statistics()
{
    factor_.min_pivot = factor_.max_pivot = 0.0;
    factor_.tiny_pivots = factor_.nonpositive_pivots = 0;
    diag_.clear();
    if (m_ == 0)
        return;
    // CUDSS_DATA_DIAG: the diagonal of the factor, n values in device memory
    core::DeviceBuffer<double> d_diag(static_cast<std::size_t>(m_));
    std::size_t written = 0;
    if (cudssDataGet(H(handle_), D(data_), CUDSS_DATA_DIAG, d_diag.data(), sizeof(double) * std::size_t(m_), &written) == CUDSS_STATUS_SUCCESS)
    {
        CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
        diag_ = core::to_host(d_diag.data(), std::size_t(m_));
        // SPD: the diagonal is L_jj (checked against the CHOLMOD log det in linsys_test.hpp), so the pivots of the
        // FactorInfo convention are L_jj^2; symmetric indefinite: D_jj of the matrix cuDSS factors (scaled by its
        // matching, with the tiny pivots perturbed to +-CUDSS_CONFIG_PIVOT_EPSILON), so the counts are relative
        std::vector<double> pivots(diag_);
        if (!indefinite_)
            for (double &d : pivots)
                d = d * d;
        double max_abs = 0.0;
        factor_.min_pivot = pivots[0];
        factor_.max_pivot = pivots[0];
        for (double d : pivots)
        {
            factor_.min_pivot = std::min(factor_.min_pivot, d);
            factor_.max_pivot = std::max(factor_.max_pivot, d);
            max_abs = std::max(max_abs, std::abs(d));
            if (d <= 0.0)
                factor_.nonpositive_pivots++;
        }
        for (double d : pivots)
            if (std::abs(d) < 1e-12 * max_abs)
                factor_.tiny_pivots++;
    }
    // CUDSS_DATA_INERTIA: the numbers of positive and negative pivots (two integers; int32 or int64 by sizeWritten)
    unsigned char inertia[16] = {0};
    written = 0;
    if (cudssDataGet(H(handle_), D(data_), CUDSS_DATA_INERTIA, inertia, sizeof(inertia), &written) == CUDSS_STATUS_SUCCESS)
    {
        std::int64_t negative = -1;
        if (written == 2 * sizeof(std::int64_t))
            negative = reinterpret_cast<const std::int64_t *>(inertia)[1];
        else if (written == 2 * sizeof(int))
            negative = reinterpret_cast<const int *>(inertia)[1];
        if (negative >= 0 && diag_.empty())
            factor_.nonpositive_pivots = int(negative);
    }
}

FactorInfo CudssSolver::factorize(const NormalSystem &sys)
{
    const auto t0 = std::chrono::steady_clock::now();
    if (!analyzed_ || m_ != sys.m())
        analyze(sys);
    factor_ = FactorInfo();
    int info = 0;
    std::string status;
    bool ok = run_factorization(&info, &status);
    if (!ok && !indefinite_ && options_.allow_indefinite_fallback)
    {
        // the SPD factorization failed (a non-positive pivot: A does not have full row rank, or eps is too small for it):
        // re-analyze and factor as a symmetric indefinite matrix (LDL^T)
        const std::string spd_status = status;
        create_objects(true);
        run_analysis();
        ok = run_factorization(&info, &status);
        if (!ok)
            throw std::runtime_error("CudssSolver: the SPD factorization failed (" + spd_status + ") and so did the symmetric "
                                     "indefinite one (" + status + "): A A^T + eps I could not be factored");
        factor_.nonpositive_pivots = 1; // overwritten by the statistics below when available
    }
    else if (!ok)
        throw std::runtime_error("CudssSolver: the " + std::string(indefinite_ ? "symmetric indefinite" : "SPD") +
                                 " factorization failed (" + status + ")");
    factor_.kind = indefinite_ ? "cudss_ldlt" : "cudss_llt";
    factor_.supernodal = false;
    read_factor_statistics();
    factored_ = true;
    device_bytes_ = symbolic_.device_bytes + sizeof(int) * (rowptr_.size() + colind_.size()) + sizeof(double) * (val_.size() + xbuf_.size() + bbuf_.size());
    host_bytes_ = symbolic_.host_bytes;
    factor_.seconds = seconds_since(t0);
    return factor_;
}

SolveReport CudssSolver::solve(const double *rhs, double *y, int m, const SolveRequest &req, cudaStream_t stream)
{
    (void)req; // exact solve
    if (!factored_)
        throw std::logic_error("CudssSolver::solve called before factorize()");
    if (m != m_)
        throw std::invalid_argument("CudssSolver::solve: the right-hand side has " + std::to_string(m) + " entries, the system " +
                                    std::to_string(m_));
    if (rhs == nullptr || y == nullptr)
        throw std::invalid_argument("CudssSolver::solve: null device pointer");
    CUADMM_CHECK_CUDSS(cudssMatrixSetValues(M(b_), rhs));
    CUADMM_CHECK_CUDSS(cudssMatrixSetValues(M(x_), y));
    CUADMM_CHECK_CUDSS(cudssSetStream(H(handle_), stream));
    CUADMM_CHECK_CUDSS(cudssExecute(H(handle_), CUDSS_PHASE_SOLVE, C(config_), D(data_), M(matrix_), M(x_), M(b_)));
    SolveReport report;
    report.iterations = 0;
    report.rel_residual = 0.0;
    report.converged = true;
    return report;
}

std::string CudssSolver::description() const
{
    char buf[320];
    std::snprintf(buf, sizeof(buf), "%s (header %s), deterministic mode %s, m = %d, nnz(lower) = %lld, nnz(L) = %.3e, %s, "
                  "device estimate %zu bytes, host estimate %zu bytes",
                  name_.c_str(), header_version().c_str(), deterministic_ ? "on" : "off", m_, static_cast<long long>(nnz_), symbolic_.nnz_L,
                  factored_ ? factor_.kind.c_str() : (analyzed_ ? "analyzed" : "not analyzed"), symbolic_.device_bytes, symbolic_.host_bytes);
    return buf;
}

} // namespace linsys
} // namespace cuadmm
