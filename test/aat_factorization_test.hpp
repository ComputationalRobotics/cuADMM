/*

    aat_factorization_test.hpp

    CPU tests of CholeskySolverCPU::factorize: the supernodal method for a dense AA^T, the simplicial LDL^T
    fallback when the supernodal LL^T factorization fails, and accurate solves for right-hand sides in range(A)
    when A does not have full row rank (as in the SPOT moment relaxations).

*/

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "cuadmm/cholesky_cpu.h"

namespace aat_factorization_test
{

// m x n matrix A in CSC with density `density`; the last `dup` rows repeat the first ones (rank m - dup)
struct MatrixCSC
{
    int m, n;
    std::vector<int> p, i;
    std::vector<double> x;
};

inline MatrixCSC random_matrix(int m, int n, int dup, double density, unsigned seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    MatrixCSC A{m, n, {0}, {}, {}};
    for (int j = 0; j < n; j++)
    {
        std::vector<double> col(m, 0.0);
        for (int r = 0; r < m - dup; r++)
            if (ud(gen) < density || r == j % (m - dup))
                col[r] = nd(gen);
        for (int r = m - dup; r < m; r++)
            col[r] = col[r - (m - dup)];
        for (int r = 0; r < m; r++)
            if (col[r] != 0.0)
            {
                A.i.push_back(r);
                A.x.push_back(col[r]);
            }
        A.p.push_back(int(A.i.size()));
    }
    return A;
}

// y = A^T v (transpose = true) or A v
inline std::vector<double> multiply(const MatrixCSC &A, const std::vector<double> &v, bool transpose)
{
    std::vector<double> r(transpose ? A.n : A.m, 0.0);
    for (int j = 0; j < A.n; j++)
        for (int k = A.p[j]; k < A.p[j + 1]; k++)
        {
            if (transpose)
                r[j] += A.x[k] * v[A.i[k]];
            else
                r[A.i[k]] += A.x[k] * v[j];
        }
    return r;
}

// Factorizes eps*I + AA^T, solves it for r = A z (in range(A)) and returns
// || (eps*I + AA^T) y - r || / || r ||.
inline double range_solve_residual(const MatrixCSC &A, double eps, CholeskySolverCPU &solver)
{
    solver.get_A(const_cast<int *>(A.p.data()), const_cast<int *>(A.i.data()), const_cast<double *>(A.x.data()),
                 A.m, A.n, int(A.x.size()), false, eps);
    solver.factorize();
    std::mt19937 gen(3);
    std::normal_distribution<double> nd;
    std::vector<double> z(A.n);
    for (double &v : z)
        v = nd(gen);
    const std::vector<double> r = multiply(A, z, false);
    // solve() solves L D L^T x = rhs for the permuted matrix P (eps*I + AA^T) P^T = L D L^T (the solver permutes
    // on the GPU): rhs = P r and y = P^T x
    const int *perm = (const int *)solver.chol_fac_L->Perm;
    for (int k = 0; k < A.m; k++)
        ((double *)solver.chol_dn_rhs->x)[k] = r[perm[k]];
    solver.solve();
    std::vector<double> y(A.m);
    for (int k = 0; k < A.m; k++)
        y[perm[k]] = ((double *)solver.chol_dn_res->x)[k];
    const std::vector<double> AAty = multiply(A, multiply(A, y, true), false);
    double res = 0.0, rn = 0.0;
    for (int k = 0; k < A.m; k++)
    {
        res += std::pow(AAty[k] + eps * y[k] - r[k], 2);
        rn += r[k] * r[k];
    }
    return std::sqrt(res / rn);
}

} // namespace aat_factorization_test

// dense AA^T: CHOLMOD chooses the supernodal method
TEST(AAtFactorization, DenseFullRankUsesSupernodal)
{
    using namespace aat_factorization_test;
    const MatrixCSC A = random_matrix(300, 600, 0, 1.0, 11);
    CholeskySolverCPU solver;
    const double res = range_solve_residual(A, 1e-15, solver);
    EXPECT_TRUE(solver.supernodal_used);
    EXPECT_FALSE(solver.chol_fac_L->is_super); // converted to the simplicial LDL^T form used by solve()
    EXPECT_FALSE(solver.chol_fac_L->is_ll);
    EXPECT_LT(res, 1e-12);
    EXPECT_EQ(solver.tiny_pivots, 0);
    EXPECT_EQ(solver.nonpositive_pivots, 0);
    EXPECT_GT(solver.min_pivot, 0.0);
}

// very sparse AA^T: the simplicial method, as before
TEST(AAtFactorization, SparseUsesSimplicial)
{
    using namespace aat_factorization_test;
    const MatrixCSC A = random_matrix(2000, 3000, 0, 0.0003, 12);
    CholeskySolverCPU solver;
    const double res = range_solve_residual(A, 1e-15, solver);
    EXPECT_FALSE(solver.supernodal_used);
    EXPECT_LT(res, 1e-12);
    EXPECT_EQ(solver.tiny_pivots, 0);
    EXPECT_EQ(solver.nonpositive_pivots, 0);
}

// eps*I + AA^T indefinite (rank-deficient A, eps < 0): the supernodal LL^T factorization fails for sure and the
// simplicial LDL^T one, which accepts negative pivots, is used
TEST(AAtFactorization, FallsBackToSimplicialLDLt)
{
    using namespace aat_factorization_test;
    const MatrixCSC A = random_matrix(300, 600, 20, 1.0, 13);
    CholeskySolverCPU solver;
    double res = 1.0;
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    ASSERT_NO_THROW(res = range_solve_residual(A, -1e-3, solver));
    const std::string output = testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();
    EXPECT_EQ(output.find("CHOLMOD"), std::string::npos) << "the handled failure must not print: " << output;
    EXPECT_FALSE(solver.supernodal_used);
    EXPECT_FALSE(solver.chol_fac_L->is_ll);
    EXPECT_LT(res, 1e-10);
    EXPECT_EQ(solver.nonpositive_pivots, 20); // one per dependent row, about -1e-3
    EXPECT_EQ(solver.tiny_pivots, 0);
}

// rank-deficient A with the solver's eps = 1e-15: whichever method succeeds, solves with right-hand sides in
// range(A) are accurate (the huge null(A^T) part of the solution is invisible in A^T y)
TEST(AAtFactorization, RankDeficientRangeSolves)
{
    using namespace aat_factorization_test;
    for (double density : {1.0, 0.002})
    {
        const MatrixCSC A = random_matrix(density == 1.0 ? 300 : 3000, density == 1.0 ? 600 : 5000, 20, density, 14);
        CholeskySolverCPU solver;
        double res = 1.0;
        ASSERT_NO_THROW(res = range_solve_residual(A, 1e-15, solver));
        printf("[ AAT      ] density %g: supernodal %d, relative residual %.2e, pivots in [%.2e, %.2e], %d tiny, %d nonpositive\n",
               density, int(solver.supernodal_used), res, solver.min_pivot, solver.max_pivot, solver.tiny_pivots,
               solver.nonpositive_pivots);
        EXPECT_LT(res, 1e-8);
        // about one per dependent row: rounding in the elimination can leave a few of them above the threshold
        EXPECT_GE(solver.tiny_pivots, 10);
        EXPECT_LE(solver.tiny_pivots, 20);
    }
}
