/*

    linsys/direct/dense/dense_cusolver_solver.h

    DenseCusolverSolver (plan Section 11.4): the dense direct backend of the y-step. The lower triangle of
    A A^T + eps I is scattered into a dense m x m column-major matrix on the device and factored with cuSOLVER:
    cusolverDnXpotrf (Cholesky, SPD) first and, when that reports a non-positive leading minor, cusolverDnDsytrf
    (Bunch-Kaufman LDL^T, symmetric indefinite) instead; the solves use the matching routine (cusolverDnXpotrs or
    cusolverDnXsytrs) on the stream given to solve(), with the data on the device throughout.

    Useful when m is a few thousand and the factor is dense (plan: the reference GPUs have weak FP64, so the dense
    factorization is not fast, but the per-iteration solves are). analyze() reports the need: 8 m^2 bytes for the
    matrix plus the cuSOLVER workspaces; factorize() refuses, with the numbers, when that exceeds the free memory
    of the device.

    FactorInfo: kind "dense_llt" or "dense_ldlt"; the pivots are L_jj^2 (LL^T) or the eigenvalues of the 1 x 1 and
    2 x 2 diagonal blocks of D (LDL^T); tiny_pivots counts |pivot| < 1e-12 max |pivot|.

*/

#ifndef CUADMM_LINSYS_DIRECT_DENSE_DENSE_CUSOLVER_SOLVER_H
#define CUADMM_LINSYS_DIRECT_DENSE_DENSE_CUSOLVER_SOLVER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <cusolverDn.h>

#include "core/device_buffer.h"
#include "linsys/normal_equation_solver.h"

namespace cuadmm
{
namespace linsys
{

class DenseCusolverSolver final : public NormalEquationSolver
{
public:
    struct Options
    {
        bool force_indefinite = false; // skip the Cholesky attempt and factor with Dsytrf directly (tests, benchmarks)
        double memory_fraction = 1.0;  // the matrix and workspaces may use at most this fraction of the free device memory
    };

    /// With handle == nullptr the solver creates its own cuSOLVER handle (the stream is set at every call).
    DenseCusolverSolver();
    explicit DenseCusolverSolver(cusolverDnHandle_t handle);
    DenseCusolverSolver(cusolverDnHandle_t handle, const Options &options);
    ~DenseCusolverSolver() override;
    DenseCusolverSolver(const DenseCusolverSolver &) = delete;
    DenseCusolverSolver &operator=(const DenseCusolverSolver &) = delete;

    std::string_view name() const override { return "dense_cusolver"; }
    Capabilities capabilities() const override;
    SymbolicInfo analyze(const NormalSystem &sys) override;
    FactorInfo factorize(const NormalSystem &sys) override;
    SolveReport solve(const double *rhs, double *y, int m, const SolveRequest &req, cudaStream_t stream) override;
    std::size_t device_bytes() const override { return device_bytes_; }
    std::size_t host_bytes() const override { return host_bytes_; }

    /// Bytes of the dense m x m matrix alone.
    static std::size_t matrix_bytes(int m) { return std::size_t(8) * std::size_t(m) * std::size_t(m); }
    /// The LDL^T (Dsytrf) factor is in use (Cholesky failed or was skipped).
    bool indefinite() const { return indefinite_; }
    const FactorInfo &factor_info() const { return factor_; }
    const SymbolicInfo &symbolic_info() const { return symbolic_; }
    /// One line: the backend, the cuSOLVER version, m, the kind of factor and the memory it holds.
    std::string description() const;

private:
    void release_factor();
    void scatter_lower(const NormalSystem &sys);
    void pivot_statistics_llt();
    void pivot_statistics_ldlt();

    cusolverDnHandle_t handle_ = nullptr;
    bool own_handle_ = false;
    cusolverDnParams_t params_ = nullptr;
    Options options_;

    int m_ = 0;
    bool analyzed_ = false, factored_ = false, indefinite_ = false;
    SymbolicInfo symbolic_;
    FactorInfo factor_;
    std::size_t device_bytes_ = 0, host_bytes_ = 0;
    std::size_t potrf_ws_device_ = 0, potrf_ws_host_ = 0; // from analyze()
    std::size_t sytrf_ws_bytes_ = 0;
    std::size_t sytrs_ws_device_ = 0, sytrs_ws_host_ = 0;

    core::DeviceBuffer<double> a_;       // the matrix, then its factor (column-major, lower triangle)
    core::DeviceBuffer<unsigned char> work_;
    std::vector<unsigned char> host_work_;
    core::DeviceBuffer<int> ipiv32_;     // Dsytrf pivots
    core::DeviceBuffer<std::int64_t> ipiv64_; // the same for Xsytrs
    core::DeviceBuffer<int> info_;
};

} // namespace linsys
} // namespace cuadmm

#endif // CUADMM_LINSYS_DIRECT_DENSE_DENSE_CUSOLVER_SOLVER_H
