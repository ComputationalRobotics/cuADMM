/*

    linsys/direct/cudss/cudss_solver.h

    CudssSolver (plan Section 11.4): the sparse direct backend of the y-step on NVIDIA cuDSS (optional build,
    CUADMM_WITH_CUDSS, compile definition CUADMM_HAVE_CUDSS; cuDSS 1.0, the PyPI wheel nvidia-cudss-cu12). Analysis
    (reordering and symbolic factorization), numerical factorization and solves run on the device; the lower triangle
    of A A^T + eps I is given in CSR with 32-bit indices and CUDSS_MVIEW_LOWER.

    Matrix types: CUDSS_MTYPE_SPD (Cholesky) first; when the SPD factorization fails or reports a non-positive pivot
    (CUDSS_DATA_INFO != 0), the matrix is re-analyzed and factored as CUDSS_MTYPE_SYMMETRIC (LDL^T, symmetric
    indefinite), which tolerates the tiny and negative pivots of a rank-deficient A with eps = 1e-15 (the SPOT moment
    relaxations). Options::spd_first = false selects LDL^T directly.

    Determinism (plan Section 11.4): cuDSS 1.0 has the configuration parameter CUDSS_CONFIG_DETERMINISTIC_MODE
    (cudss_data_types.h; an int, 0 by default); the solver sets it to 1 when constructed with
    determinism_strict = true, before the analysis. The runtime version (cudssGetProperty) is checked against the
    header at construction (same major version required) and printed by name() and in SymbolicInfo::ordering.

    FactorInfo: kind "cudss_llt" or "cudss_ldlt"; min/max_pivot and the counts come from CUDSS_DATA_DIAG: for the SPD
    kind the diagonal is L_jj (established against the CHOLMOD log det in linsys_test.hpp) and the pivots reported
    are L_jj^2, as for CHOLMOD; for the symmetric indefinite kind it is D_jj of the matrix cuDSS factors (scaled by
    its matching, the tiny pivots perturbed to +-CUDSS_CONFIG_PIVOT_EPSILON = 1e-13), so only the relative counts
    (tiny, nonpositive) are comparable. SymbolicInfo::nnz_L is CUDSS_DATA_LU_NNZ as reported (not nnz(L) in the
    CHOLMOD sense, see cudss_solver.cu) and device_bytes/host_bytes the peak estimates of CUDSS_DATA_MEMORY_ESTIMATES.
    The first analysis with a cold kernel cache takes a few seconds (cuDSS compiles its kernels with NVRTC and caches
    them: 4.0 s on the first run of the tests on the A100 MIG, 0.1 s on the next run).

    This header does not include cudss.h (the cuDSS handles are forward-declared), so the tests and the facade can
    include it with the compile definition alone.

*/

#ifndef CUADMM_LINSYS_DIRECT_CUDSS_CUDSS_SOLVER_H
#define CUADMM_LINSYS_DIRECT_CUDSS_CUDSS_SOLVER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/device_buffer.h"
#include "linsys/normal_equation_solver.h"

// the opaque cuDSS objects (cudss_data_types.h: typedef struct cudssContext *cudssHandle_t; ...)
struct cudssContext;
struct cudssConfig;
struct cudssData;
struct cudssMatrix;

namespace cuadmm
{
namespace linsys
{

class CudssSolver final : public NormalEquationSolver
{
public:
    struct Options
    {
        bool determinism_strict = false;      // CUDSS_CONFIG_DETERMINISTIC_MODE = 1
        bool spd_first = true;                // Cholesky (CUDSS_MTYPE_SPD) first; false: LDL^T (CUDSS_MTYPE_SYMMETRIC) directly
        bool allow_indefinite_fallback = true; // fall back to LDL^T when the SPD factorization fails
    };

    explicit CudssSolver(bool determinism_strict = false);
    explicit CudssSolver(const Options &options);
    ~CudssSolver() override;
    CudssSolver(const CudssSolver &) = delete;
    CudssSolver &operator=(const CudssSolver &) = delete;

    /// "cudss <runtime version>", e.g. "cudss 1.0.0".
    std::string_view name() const override { return name_; }
    Capabilities capabilities() const override;
    SymbolicInfo analyze(const NormalSystem &sys) override;
    FactorInfo factorize(const NormalSystem &sys) override;
    SolveReport solve(const double *rhs, double *y, int m, const SolveRequest &req, cudaStream_t stream) override;
    std::size_t device_bytes() const override { return device_bytes_; }
    std::size_t host_bytes() const override { return host_bytes_; }

    /// The runtime version as "major.minor.patch" and the version of the header the library was built with.
    static std::string runtime_version();
    static std::string header_version();
    /// The deterministic mode is on (set successfully at construction).
    bool deterministic() const { return deterministic_; }
    /// The symmetric indefinite (LDL^T) factorization is in use.
    bool indefinite() const { return indefinite_; }
    const FactorInfo &factor_info() const { return factor_; }
    const SymbolicInfo &symbolic_info() const { return symbolic_; }
    /// The diagonal of the factor as reported by cuDSS after factorize() (CUDSS_DATA_DIAG), empty when unavailable.
    const std::vector<double> &factor_diagonal() const { return diag_; }
    /// One line: versions, deterministic mode, m, nnz(L), the kind of factor and the memory estimates.
    std::string description() const;

private:
    void create_objects(bool indefinite);
    void destroy_objects();
    void run_analysis();
    bool run_factorization(int *info_out, std::string *status_out);
    void read_factor_statistics();

    Options options_;
    std::string name_;
    bool deterministic_ = false;

    cudssContext *handle_ = nullptr;
    cudssConfig *config_ = nullptr;
    cudssData *data_ = nullptr;
    cudssMatrix *matrix_ = nullptr;
    cudssMatrix *x_ = nullptr;
    cudssMatrix *b_ = nullptr;

    int m_ = 0;
    std::int64_t nnz_ = 0;
    bool analyzed_ = false, factored_ = false, indefinite_ = false;
    SymbolicInfo symbolic_;
    FactorInfo factor_;
    std::size_t device_bytes_ = 0, host_bytes_ = 0;
    std::vector<double> diag_;

    core::DeviceBuffer<int> rowptr_, colind_; // the lower triangle in CSR on the device
    core::DeviceBuffer<double> val_;
    core::DeviceBuffer<double> xbuf_, bbuf_; // placeholders for the dense x and b objects during analysis
};

} // namespace linsys
} // namespace cuadmm

#endif // CUADMM_LINSYS_DIRECT_CUDSS_CUDSS_SOLVER_H
