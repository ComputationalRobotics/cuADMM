/*

    linsys_test.hpp

    Phase 7 backends against the CHOLMOD wrapper of the ported engine (cuadmm_legacy/cholesky_cpu.h):
      - NormalSystem::lower() and apply_host() against a dense A A^T + eps I;
      - DenseCusolverSolver and (CUADMM_HAVE_CUDSS) CudssSolver on a random full-rank sparse A with m = 50 (n = 300) and
        with m = 2000 (n = 3000, dense factor): relative residual <= 1e-10 (relative_residual_host) and agreement with
        the CHOLMOD solution within 1e-9 relative, two different right-hand sides, a wrong-size right-hand side rejected;
      - a rank-deficient A (duplicate rows) with eps = 1e-15 through the natural path and the forced LDL^T path;
      - the memory and name reports; the dense solver's refusal when 8 m^2 bytes exceed the free device memory.

    The literal spec "vec_len 300, m 2000" cannot be full rank (rank <= 300), so the m = 2000 case uses n = 3000.

*/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "core/check.h"
#include "core/device_buffer.h"
#include "cuadmm_legacy/check.h"
#include "cuadmm_legacy/cholesky_cpu.h"
#include "linsys/direct/dense/dense_cusolver_solver.h"
#include "linsys/normal_equation_solver.h"
#include "linsys/normal_system.h"
#ifdef CUADMM_HAVE_CUDSS
#include "linsys/direct/cudss/cudss_solver.h"
#endif

namespace linsys_test
{

using cuadmm::linsys::DenseCusolverSolver;
using cuadmm::linsys::FactorInfo;
using cuadmm::linsys::NormalEquationSolver;
using cuadmm::linsys::NormalSystem;
using cuadmm::linsys::SolveRequest;
using cuadmm::linsys::SymbolicInfo;

// a sparse matrix in CSC
struct Csc
{
    int rows = 0, cols = 0;
    std::vector<int> p, i;
    std::vector<double> x;
    int nnz() const { return int(i.size()); }
};

// m x n matrix A in CSC with density `density` (every row gets at least one entry); the last `dup` rows repeat the
// first ones (rank m - dup), as aat_factorization_test::random_matrix
inline Csc random_A(int m, int n, int dup, double density, unsigned seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    Csc A;
    A.rows = m;
    A.cols = n;
    A.p.push_back(0);
    for (int j = 0; j < n; j++)
    {
        std::vector<double> col(std::size_t(m), 0.0);
        for (int r = 0; r < m - dup; r++)
            if (ud(gen) < density || r == j % (m - dup))
                col[std::size_t(r)] = nd(gen);
        for (int r = m - dup; r < m; r++)
            col[std::size_t(r)] = col[std::size_t(r - (m - dup))];
        for (int r = 0; r < m; r++)
            if (col[std::size_t(r)] != 0.0)
            {
                A.i.push_back(r);
                A.x.push_back(col[std::size_t(r)]);
            }
        A.p.push_back(int(A.i.size()));
    }
    return A;
}

inline Csc transpose(const Csc &A)
{
    Csc T;
    T.rows = A.cols;
    T.cols = A.rows;
    T.p.assign(std::size_t(A.rows) + 1, 0);
    for (int k = 0; k < A.nnz(); k++)
        T.p[std::size_t(A.i[std::size_t(k)]) + 1]++;
    for (int r = 0; r < A.rows; r++)
        T.p[std::size_t(r) + 1] += T.p[std::size_t(r)];
    std::vector<int> next(T.p.begin(), T.p.end() - 1);
    T.i.resize(A.i.size());
    T.x.resize(A.x.size());
    for (int j = 0; j < A.cols; j++)
        for (int k = A.p[std::size_t(j)]; k < A.p[std::size_t(j) + 1]; k++)
        {
            const int dst = next[std::size_t(A.i[std::size_t(k)])]++;
            T.i[std::size_t(dst)] = j;
            T.x[std::size_t(dst)] = A.x[std::size_t(k)];
        }
    return T;
}

// A v (transpose = false) or A^T v
inline std::vector<double> multiply(const Csc &A, const std::vector<double> &v, bool transpose)
{
    std::vector<double> r(std::size_t(transpose ? A.cols : A.rows), 0.0);
    for (int j = 0; j < A.cols; j++)
        for (int k = A.p[std::size_t(j)]; k < A.p[std::size_t(j) + 1]; k++)
        {
            if (transpose)
                r[std::size_t(j)] += A.x[std::size_t(k)] * v[std::size_t(A.i[std::size_t(k)])];
            else
                r[std::size_t(A.i[std::size_t(k)])] += A.x[std::size_t(k)] * v[std::size_t(j)];
        }
    return r;
}

inline std::vector<double> random_vector(int n, unsigned seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd;
    std::vector<double> v(static_cast<std::size_t>(n));
    for (double &e : v)
        e = nd(gen);
    return v;
}

inline double norm(const std::vector<double> &v)
{
    double s = 0.0;
    for (double e : v)
        s += e * e;
    return std::sqrt(s);
}

inline double rel_diff(const std::vector<double> &a, const std::vector<double> &b)
{
    std::vector<double> d(a.size());
    for (std::size_t k = 0; k < a.size(); k++)
        d[k] = a[k] - b[k];
    const double nb = norm(b);
    return nb > 0.0 ? norm(d) / nb : norm(d);
}

// the reference: the CHOLMOD wrapper of the engine on A in CSC (m x n), (eps I + A A^T) y = r; the factor is of the
// permuted matrix, so rhs = P r and y = P^T x (as in aat_factorization_test)
struct CholmodReference
{
    CholeskySolverCPU solver;
    int m = 0;
    explicit CholmodReference(const Csc &A, double eps) : m(A.rows)
    {
        solver.get_A(const_cast<int *>(A.p.data()), const_cast<int *>(A.i.data()), const_cast<double *>(A.x.data()), A.rows, A.cols,
                     A.nnz(), false, eps);
        solver.factorize();
    }
    std::vector<double> solve(const std::vector<double> &r)
    {
        const int *perm = static_cast<const int *>(solver.chol_fac_L->Perm);
        for (int k = 0; k < m; k++)
            static_cast<double *>(solver.chol_dn_rhs->x)[k] = r[std::size_t(perm[k])];
        solver.solve();
        std::vector<double> y(static_cast<std::size_t>(m));
        for (int k = 0; k < m; k++)
            y[std::size_t(perm[k])] = static_cast<double *>(solver.chol_dn_res->x)[k];
        return y;
    }
    // log det (eps I + A A^T) = sum log D_jj of the simplicial LDL^T factor
    double log_det() const
    {
        const int *Lp = static_cast<const int *>(solver.chol_fac_L->p);
        const double *Lx = static_cast<const double *>(solver.chol_fac_L->x);
        double s = 0.0;
        for (int j = 0; j < m; j++)
            s += std::log(std::abs(Lx[Lp[j]]));
        return s;
    }
};

struct Stream
{
    cudaStream_t s = nullptr;
    Stream() { CUADMM_CHECK_CUDA(cudaStreamCreate(&s)); }
    ~Stream() { CUADMM_CHECK_CUDA_NOTHROW(cudaStreamDestroy(s)); }
};

struct SolveOutcome
{
    double residual[2] = {0.0, 0.0};  // relative_residual_host of the two right-hand sides
    double agreement[2] = {0.0, 0.0}; // ||y - y_ref|| / ||y_ref||, or of A^T y when `compare_aty`
};

// analyze, factorize, solve two different right-hand sides on a stream, compare with the CHOLMOD reference
inline SolveOutcome run_solver(NormalEquationSolver &solver, const Csc &A, double eps, unsigned seed, bool range_rhs, bool compare_aty,
                               const char *tag)
{
    const Csc At = transpose(A);
    NormalSystem sys(A.rows, A.cols, At.p.data(), At.i.data(), At.x.data(), eps);
    const SymbolicInfo sym = solver.analyze(sys);
    const FactorInfo fac = solver.factorize(sys);
    std::printf("[ %-8s ] %s: m = %d, nnz(lower) = %lld, nnz(L) = %.3e, device %zu B (held %zu B), ordering %s, %.3f s; factor %s in "
                "%.3f s, pivots in [%.3e, %.3e], %d tiny, %d nonpositive\n",
                tag, std::string(solver.name()).c_str(), A.rows, static_cast<long long>(sys.lower().nnz()), sym.nnz_L, sym.device_bytes,
                solver.device_bytes(), sym.ordering.c_str(), sym.seconds, fac.kind.c_str(), fac.seconds, fac.min_pivot, fac.max_pivot,
                fac.tiny_pivots, fac.nonpositive_pivots);
    CholmodReference ref(A, eps);
    Stream stream;
    cuadmm::core::DeviceBuffer<double> d_rhs(std::size_t(A.rows)), d_y(std::size_t(A.rows));
    SolveOutcome out;
    for (int t = 0; t < 2; t++)
    {
        std::vector<double> r = range_rhs ? multiply(A, random_vector(A.cols, seed + 10 * unsigned(t)), false) : random_vector(A.rows, seed + 10 * unsigned(t));
        cuadmm::core::to_device(d_rhs.span(), r);
        d_y.zero();
        const cuadmm::linsys::SolveReport rep = solver.solve(d_rhs.data(), d_y.data(), A.rows, SolveRequest(), stream.s);
        EXPECT_EQ(rep.iterations, 0);
        EXPECT_TRUE(rep.converged);
        out.residual[t] = cuadmm::linsys::relative_residual_host(sys, d_rhs.data(), d_y.data(), stream.s);
        const std::vector<double> y = cuadmm::core::to_host(d_y.data(), std::size_t(A.rows));
        const std::vector<double> y_ref = ref.solve(r);
        out.agreement[t] = compare_aty ? rel_diff(multiply(A, y, true), multiply(A, y_ref, true)) : rel_diff(y, y_ref);
    }
    std::printf("[ %-8s ] residuals %.2e, %.2e; agreement with CHOLMOD %.2e, %.2e%s\n", tag, out.residual[0], out.residual[1], out.agreement[0],
                out.agreement[1], compare_aty ? " (of A^T y)" : "");
    // a wrong-size right-hand side is rejected before any device work
    EXPECT_THROW(solver.solve(d_rhs.data(), d_y.data(), A.rows + 1, SolveRequest(), stream.s), std::invalid_argument);
    return out;
}

// the two full-rank cases of the spec: m = 50 (n = 300, density 0.2) and m = 2000 (n = 3000, density 0.02: the factor is dense)
inline void full_rank_cases(NormalEquationSolver &solver, const char *tag)
{
    {
        const Csc A = random_A(50, 300, 0, 0.2, 101);
        const SolveOutcome out = run_solver(solver, A, 1e-15, 7, false, false, tag);
        for (int t = 0; t < 2; t++)
        {
            EXPECT_LE(out.residual[t], 1e-10);
            EXPECT_LE(out.agreement[t], 1e-9);
        }
    }
    {
        const Csc A = random_A(2000, 3000, 0, 0.02, 102);
        const SolveOutcome out = run_solver(solver, A, 1e-15, 8, false, false, tag);
        for (int t = 0; t < 2; t++)
        {
            EXPECT_LE(out.residual[t], 1e-10);
            EXPECT_LE(out.agreement[t], 1e-9);
        }
    }
}

// rank-deficient A (10 duplicate rows), eps = 1e-15, right-hand sides in range(A): the solves must be accurate on
// A^T y (the null(A^T) part of y is arbitrary and invisible to the ADMM)
inline SolveOutcome rank_deficient_case(NormalEquationSolver &solver, const char *tag)
{
    const Csc A = random_A(60, 300, 10, 0.3, 103);
    const SolveOutcome out = run_solver(solver, A, 1e-15, 9, true, true, tag);
    for (int t = 0; t < 2; t++)
    {
        EXPECT_LE(out.residual[t], 1e-8);
        EXPECT_LE(out.agreement[t], 1e-6);
    }
    return out;
}

} // namespace linsys_test

// ------------------------------------------------------------------------------------------------ NormalSystem

TEST(NormalSystem, LowerTriangleAndApplyMatchDenseProduct)
{
    using namespace linsys_test;
    const int m = 50, n = 300;
    const double eps = 1e-15;
    const Csc A = random_A(m, n, 0, 0.2, 201);
    const Csc At = transpose(A);
    NormalSystem sys(m, n, At.p.data(), At.i.data(), At.x.data(), eps);
    EXPECT_EQ(sys.m(), m);
    EXPECT_EQ(sys.n(), n);
    EXPECT_EQ(sys.nnz_at(), A.nnz());
    // dense S = A A^T + eps I
    std::vector<double> S(std::size_t(m) * m, 0.0);
    for (int j = 0; j < n; j++)
        for (int k = A.p[std::size_t(j)]; k < A.p[std::size_t(j) + 1]; k++)
            for (int l = A.p[std::size_t(j)]; l < A.p[std::size_t(j) + 1]; l++)
                S[std::size_t(A.i[std::size_t(k)]) + std::size_t(m) * std::size_t(A.i[std::size_t(l)])] += A.x[std::size_t(k)] * A.x[std::size_t(l)];
    for (int j = 0; j < m; j++)
        S[std::size_t(j) * (std::size_t(m) + 1)] += eps;
    const cuadmm::linsys::HostCscLower &L = sys.lower();
    ASSERT_EQ(L.n, m);
    ASSERT_EQ(int(L.colptr.size()), m + 1);
    double max_err = 0.0, max_abs = 0.0;
    std::vector<double> seen(std::size_t(m) * m, 0.0);
    for (int j = 0; j < m; j++)
    {
        ASSERT_LT(L.colptr[std::size_t(j)], L.colptr[std::size_t(j) + 1]) << "column " << j << " is empty";
        EXPECT_EQ(L.rowind[std::size_t(L.colptr[std::size_t(j)])], j) << "the diagonal is not first in column " << j;
        for (int k = L.colptr[std::size_t(j)]; k < L.colptr[std::size_t(j) + 1]; k++)
        {
            const int i = L.rowind[std::size_t(k)];
            ASSERT_GE(i, j);
            if (k > L.colptr[std::size_t(j)])
                ASSERT_GT(i, L.rowind[std::size_t(k) - 1]);
            seen[std::size_t(i) + std::size_t(m) * std::size_t(j)] = L.val[std::size_t(k)];
        }
    }
    for (int j = 0; j < m; j++)
        for (int i = j; i < m; i++)
        {
            const double s = S[std::size_t(i) + std::size_t(m) * std::size_t(j)];
            max_err = std::max(max_err, std::abs(seen[std::size_t(i) + std::size_t(m) * std::size_t(j)] - s));
            max_abs = std::max(max_abs, std::abs(s));
        }
    std::printf("[ NORMAL   ] lower triangle: %lld entries, max |difference| %.2e (max |entry| %.2e)\n", static_cast<long long>(L.nnz()), max_err, max_abs);
    EXPECT_LE(max_err, 1e-13 * max_abs);
    // apply_host against the dense product
    const std::vector<double> y = random_vector(m, 202);
    std::vector<double> out(static_cast<std::size_t>(m)), dense(std::size_t(m), 0.0);
    sys.apply_host(y.data(), out.data());
    for (int j = 0; j < m; j++)
        for (int i = 0; i < m; i++)
            dense[std::size_t(i)] += S[std::size_t(i) + std::size_t(m) * std::size_t(j)] * y[std::size_t(j)];
    EXPECT_LE(rel_diff(out, dense), 1e-13);
    // the same object again: cached
    EXPECT_EQ(&sys.lower(), &L);
}

TEST(NormalSystem, RejectsMalformedInput)
{
    using namespace linsys_test;
    const Csc A = random_A(5, 10, 0, 0.5, 203);
    const Csc At = transpose(A);
    EXPECT_THROW(NormalSystem(-1, 10, At.p.data(), At.i.data(), At.x.data()), std::invalid_argument);
    EXPECT_THROW(NormalSystem(5, 10, nullptr, At.i.data(), At.x.data()), std::invalid_argument);
    EXPECT_THROW(NormalSystem(5, 10, At.p.data(), At.i.data(), At.x.data(), -1.0), std::invalid_argument);
}

// ------------------------------------------------------------------------------------------ DenseCusolverSolver

TEST(DenseCusolver, FullRankSmallAndDense2000)
{
    using namespace linsys_test;
    DenseCusolverSolver solver;
    EXPECT_EQ(solver.name(), "dense_cusolver");
    const cuadmm::linsys::Capabilities cap = solver.capabilities();
    EXPECT_TRUE(cap.exact);
    EXPECT_TRUE(cap.device_solves);
    EXPECT_FALSE(cap.saddle_point);
    full_rank_cases(solver, "DENSE");
    EXPECT_FALSE(solver.indefinite());
    EXPECT_EQ(solver.factor_info().kind, "dense_llt");
    EXPECT_EQ(solver.factor_info().tiny_pivots, 0);
    EXPECT_EQ(solver.factor_info().nonpositive_pivots, 0);
    EXPECT_GE(solver.device_bytes(), DenseCusolverSolver::matrix_bytes(2000));
    std::printf("[ DENSE    ] %s\n", solver.description().c_str());
    EXPECT_NE(solver.description().find("cuSOLVER"), std::string::npos);
}

TEST(DenseCusolver, RankDeficientNaturalAndForcedLDLt)
{
    using namespace linsys_test;
    {
        DenseCusolverSolver solver; // Cholesky first; whether it survives the pivots of about eps is up to the rounding
        rank_deficient_case(solver, "DENSE");
        std::printf("[ DENSE    ] natural path on the rank-deficient system: %s (indefinite %d)\n", solver.factor_info().kind.c_str(),
                    int(solver.indefinite()));
    }
    {
        DenseCusolverSolver::Options options;
        options.force_indefinite = true;
        DenseCusolverSolver solver(nullptr, options);
        rank_deficient_case(solver, "DENSE");
        EXPECT_TRUE(solver.indefinite());
        EXPECT_EQ(solver.factor_info().kind, "dense_ldlt");
        // about one pivot of about eps per duplicate row (the 2 x 2 pivots of Bunch-Kaufman may pair them differently)
        EXPECT_GE(solver.factor_info().tiny_pivots, 5);
        EXPECT_LE(solver.factor_info().tiny_pivots, 10);
    }
}

TEST(DenseCusolver, MemoryRefusalAndReports)
{
    using namespace linsys_test;
    // A^T = I of order 200000: A A^T = I is cheap to assemble, the dense matrix would need 320 GB
    const int m = 200000;
    std::vector<int> p(std::size_t(m) + 1), i(static_cast<std::size_t>(m));
    std::vector<double> x(std::size_t(m), 1.0);
    for (int k = 0; k <= m; k++)
        p[std::size_t(k)] = k;
    for (int k = 0; k < m; k++)
        i[std::size_t(k)] = k;
    NormalSystem sys(m, m, p.data(), i.data(), x.data(), 1e-15);
    DenseCusolverSolver solver;
    const SymbolicInfo sym = solver.analyze(sys);
    EXPECT_GE(sym.device_bytes, DenseCusolverSolver::matrix_bytes(m));
    std::size_t free_b = 0, total_b = 0;
    CUADMM_CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
    std::printf("[ DENSE    ] m = %d: analyze reports %zu device bytes (8 m^2 = %zu), %zu free of %zu\n", m, sym.device_bytes,
                DenseCusolverSolver::matrix_bytes(m), free_b, total_b);
    ASSERT_GT(DenseCusolverSolver::matrix_bytes(m), free_b);
    try
    {
        solver.factorize(sys);
        FAIL() << "factorize() must refuse the 320 GB matrix";
    }
    catch (const std::runtime_error &e)
    {
        const std::string what = e.what();
        std::printf("[ DENSE    ] refused: %s\n", what.c_str());
        EXPECT_NE(what.find(std::to_string(DenseCusolverSolver::matrix_bytes(m))), std::string::npos);
        EXPECT_NE(what.find("free"), std::string::npos);
    }
    EXPECT_EQ(solver.device_bytes(), std::size_t(0));
    std::printf("[ DENSE    ] %s\n", solver.description().c_str());
}

// ------------------------------------------------------------------------------------------------- CudssSolver

#ifdef CUADMM_HAVE_CUDSS

using cuadmm::linsys::CudssSolver;

TEST(Cudss, VersionAndName)
{
    const std::string runtime = CudssSolver::runtime_version(), header = CudssSolver::header_version();
    std::printf("[ CUDSS    ] runtime %s, header %s\n", runtime.c_str(), header.c_str());
    EXPECT_FALSE(runtime.empty());
    CudssSolver solver;
    EXPECT_EQ(std::string(solver.name()), "cudss " + runtime);
    EXPECT_FALSE(solver.deterministic());
    const cuadmm::linsys::Capabilities cap = solver.capabilities();
    EXPECT_TRUE(cap.exact);
    EXPECT_TRUE(cap.device_solves);
    EXPECT_FALSE(cap.saddle_point);
    EXPECT_FALSE(cap.sigma_dependent_ok);
    std::printf("[ CUDSS    ] %s\n", solver.description().c_str());
}

TEST(Cudss, FullRankSmallAndDense2000)
{
    using namespace linsys_test;
    CudssSolver solver;
    full_rank_cases(solver, "CUDSS");
    EXPECT_FALSE(solver.indefinite());
    EXPECT_EQ(solver.factor_info().kind, "cudss_llt");
    EXPECT_NE(solver.symbolic_info().ordering.find("runtime"), std::string::npos);
    std::printf("[ CUDSS    ] %s\n", solver.description().c_str());
    // what CUDSS_DATA_DIAG holds: the product of its entries against det(S) from the CHOLMOD LDL^T (information)
    if (!solver.factor_diagonal().empty())
    {
        const Csc A = random_A(2000, 3000, 0, 0.02, 102);
        CholmodReference ref(A, 1e-15);
        double log_prod = 0.0;
        for (double d : solver.factor_diagonal())
            log_prod += std::log(std::abs(d));
        std::printf("[ CUDSS    ] sum log |diag| = %.6f, log det from CHOLMOD = %.6f (half: %.6f): diag is %s\n", log_prod, ref.log_det(),
                    0.5 * ref.log_det(), std::abs(log_prod - ref.log_det()) < 1e-6 * std::abs(ref.log_det()) ? "D of LDL^T (or L_jj^2)"
                    : (std::abs(log_prod - 0.5 * ref.log_det()) < 1e-6 * std::abs(ref.log_det()) ? "L_jj of LL^T" : "neither (unknown)"));
    }
}

TEST(Cudss, RankDeficientNaturalAndForcedLDLt)
{
    using namespace linsys_test;
    {
        CudssSolver solver; // SPD first
        rank_deficient_case(solver, "CUDSS");
        std::printf("[ CUDSS    ] natural path on the rank-deficient system: %s (indefinite %d), %d tiny, %d nonpositive\n",
                    solver.factor_info().kind.c_str(), int(solver.indefinite()), solver.factor_info().tiny_pivots,
                    solver.factor_info().nonpositive_pivots);
    }
    {
        CudssSolver::Options options;
        options.spd_first = false;
        CudssSolver solver(options);
        rank_deficient_case(solver, "CUDSS");
        EXPECT_TRUE(solver.indefinite());
        EXPECT_EQ(solver.factor_info().kind, "cudss_ldlt");
        std::printf("[ CUDSS    ] forced LDL^T: %s\n", solver.description().c_str());
    }
}

TEST(Cudss, DeterministicModeSolvesAndRepeatsBitwise)
{
    using namespace linsys_test;
    CudssSolver solver(true);
    EXPECT_TRUE(solver.deterministic());
    const Csc A = random_A(300, 900, 0, 0.05, 104);
    const SolveOutcome out = run_solver(solver, A, 1e-15, 11, false, false, "CUDSS-D");
    for (int t = 0; t < 2; t++)
    {
        EXPECT_LE(out.residual[t], 1e-10);
        EXPECT_LE(out.agreement[t], 1e-9);
    }
    // the same right-hand side solved twice gives the same bits
    const Csc At = transpose(A);
    NormalSystem sys(A.rows, A.cols, At.p.data(), At.i.data(), At.x.data(), 1e-15);
    solver.factorize(sys);
    Stream stream;
    cuadmm::core::DeviceBuffer<double> d_rhs(std::size_t(A.rows)), d_y(std::size_t(A.rows));
    cuadmm::core::to_device(d_rhs.span(), random_vector(A.rows, 12));
    solver.solve(d_rhs.data(), d_y.data(), A.rows, SolveRequest(), stream.s);
    CUADMM_CHECK_CUDA(cudaStreamSynchronize(stream.s));
    const std::vector<double> y1 = cuadmm::core::to_host(d_y.data(), std::size_t(A.rows));
    d_y.zero();
    solver.solve(d_rhs.data(), d_y.data(), A.rows, SolveRequest(), stream.s);
    CUADMM_CHECK_CUDA(cudaStreamSynchronize(stream.s));
    const std::vector<double> y2 = cuadmm::core::to_host(d_y.data(), std::size_t(A.rows));
    EXPECT_EQ(y1, y2);
    std::printf("[ CUDSS-D  ] %s\n", solver.description().c_str());
}

#endif // CUADMM_HAVE_CUDSS
