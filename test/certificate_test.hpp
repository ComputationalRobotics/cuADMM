/*

    certificate_test.hpp

    Tests of the cone measures computed at termination (errPSD_X / errPSD_S over all blocks) and of the
    certified lower bound of the paper (arXiv 2406.05846, eq. 29c-30):
        <b, y> + sum_b R_b c_b <= p*   for any R_b bounding the blocks of an optimal X,
    where c_b = min(0, lambda_min((C - A^T y)_b)) for PSD blocks.

*/

#ifdef CUADMM_PLAIN_ADMM_ONLY
#define CUADMM_PLAIN_AND_SGS_SWITCHES {0}
#else
#define CUADMM_PLAIN_AND_SGS_SWITCHES {0, 100000}
#endif
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "cuadmm/solver.h"
#include "cuadmm/io.h"
#include "regression_problems.hpp"
#include "solver_setup.hpp"

namespace certificate_test
{

using solver_setup::init_solver;

// svec of the n x n matrix sum_k R_k diag(lambda_k) R_k^T, where the rotations R_k (g x g) act on consecutive
// groups of a random permutation of the indices; its eigenvalues are exactly lambda.
inline void fill_known_spectrum(
    std::mt19937 &gen, int n, int g, const std::vector<double> &lambda, int offset, std::vector<double> &v)
{
    const std::vector<int> perm = regression::random_permutation(gen, n);
    for (int start = 0; start < n; start += g)
    {
        const int gs = std::min(g, n - start);
        const std::vector<double> R = regression::random_orthogonal(gen, gs);
        for (int a = 0; a < gs; a++)
            for (int c = a; c < gs; c++)
            {
                double m = 0.0;
                for (int l = 0; l < gs; l++)
                    m += R[a + size_t(l) * gs] * R[c + size_t(l) * gs] * lambda[start + l];
                v[offset + regression::svec_index(perm[start + a], perm[start + c])] = (a == c ? 1.0 : std::sqrt(2.0)) * m;
            }
    }
}

// Bounds R_b of the blocks of the known optimal X*: trace for 's' blocks, largest (absolute) entry otherwise.
inline std::vector<double> optimal_block_bounds(const regression::RegressionProblem &p)
{
    std::vector<double> bounds;
    int offset = 0;
    for (size_t b = 0; b < p.blk_types.size(); b++)
    {
        const int n = p.blk_sizes[b];
        double r = 0.0;
        if (p.blk_types[b] == 's')
        {
            for (int i = 0; i < n; i++)
                r += p.X_star[offset + regression::svec_index(i, i)];
            offset += n * (n + 1) / 2;
        }
        else
        {
            for (int i = 0; i < n; i++)
                r = std::max(r, std::abs(p.X_star[offset + i]));
            offset += n;
        }
        bounds.push_back(r);
    }
    return bounds;
}

} // namespace certificate_test

// cone_measures on a vector with known spectra: small (batched Jacobi), medium (streamed syevd) and large (syevd)
// PSD blocks, plus 'l' and 'u' blocks, in the input order of the blocks.
TEST(Certificate, ConeMeasuresMatchKnownSpectra)
{
    using namespace certificate_test;
    // {type, size, rank_X, group, interleave}: only the layout matters here
    regression::ProblemSpec spec = {
        {{'s', 8, 4, 0, false}, {'l', 6, 3, 0, false}, {'s', 40, 20, 0, false}, {'u', 5, 0, 0, false},
         {'s', 8, 4, 0, false}, {'s', 1050, 525, 10, true}},
        200, 12, 2001}; // few constraints: A has full row rank
    const regression::RegressionProblem p = regression::generate(spec);
    SDPSolver solver;
    ASSERT_NO_THROW(init_solver(p, solver));
    ASSERT_EQ(solver.blk_info.size(), spec.blocks.size());

    // spectra: a block with a negative eigenvalue, a PSD block, and so on
    std::mt19937 gen(77);
    std::vector<double> v(p.vec_len, 0.0);
    std::vector<double> expected_min, expected_coef;
    int offset = 0;
    for (size_t b = 0; b < spec.blocks.size(); b++)
    {
        const int n = spec.blocks[b].size;
        if (spec.blocks[b].type == 's')
        {
            std::vector<double> lambda(n);
            const double lo = (b == 4) ? 0.5 : -2.0 - 0.1 * b; // block 4 is positive definite
            for (int i = 0; i < n; i++)
                lambda[i] = lo + (3.0 - lo) * regression::uniform01(gen);
            lambda[regression::uniform_index(gen, n)] = lo; // attained minimum
            fill_known_spectrum(gen, n, n > 100 ? 10 : n, lambda, offset, v);
            expected_min.push_back(lo);
            expected_coef.push_back(std::min(0.0, lo));
            offset += n * (n + 1) / 2;
        }
        else if (spec.blocks[b].type == 'l')
        {
            const std::vector<double> e = {0.5, -0.2, 0.1, -0.7, 0.3, 0.0};
            std::copy(e.begin(), e.end(), v.begin() + offset);
            expected_min.push_back(-0.7);
            expected_coef.push_back(-0.9);
            offset += n;
        }
        else
        {
            const std::vector<double> e = {0.1, -0.4, 0.2, 0.0, 0.3};
            std::copy(e.begin(), e.end(), v.begin() + offset);
            expected_min.push_back(0.4);   // largest absolute entry
            expected_coef.push_back(-1.0); // -sum |entry|
            offset += n;
        }
    }
    ASSERT_EQ(offset, p.vec_len);

    DeviceDenseVector<double> d_v;
    d_v.allocate(GPU0, p.vec_len);
    ASSERT_EQ(cudaMemcpy(d_v.vals, v.data(), sizeof(double) * p.vec_len, cudaMemcpyHostToDevice), cudaSuccess);
    std::vector<double> block_min, bound_coef;
    ASSERT_NO_THROW(solver.cone_measures(d_v, block_min, bound_coef));
    ASSERT_EQ(block_min.size(), spec.blocks.size());
    for (size_t b = 0; b < spec.blocks.size(); b++)
    {
        SCOPED_TRACE("block " + std::to_string(b) + " of type " + spec.blocks[b].type);
        EXPECT_NEAR(block_min[b], expected_min[b], 1e-10);
        EXPECT_NEAR(bound_coef[b], expected_coef[b], 1e-10);
    }
}

// The certified lower bound is valid (<= p* for the true bounds of the known optimal X*) and tight after
// convergence, for plain ADMM and for sGS-ADMM, on layouts with 's', 'l' and 'u' blocks.
TEST(Certificate, CertifiedLowerBoundIsValidAndTight)
{
    using namespace certificate_test;
    for (const std::string name : {"PSD", "PSD_Nonneg", "Nonneg_Free_PSD", "PSD_Free_PSD", "SmallBatch_Free", "Medium_Free"})
    {
        const regression::RegressionProblem p = regression::generate(name);
        const std::vector<double> bounds = optimal_block_bounds(p);
        for (int switch_admm : CUADMM_PLAIN_AND_SGS_SWITCHES)
        {
            SCOPED_TRACE(name + (switch_admm > 0 ? ", sGS-ADMM" : ", ADMM"));
            SDPSolver solver;
            ASSERT_NO_THROW(init_solver(p, solver));
            ASSERT_NO_THROW(solver.solve(20000, 1e-6, 500, 50, 100, switch_admm, 0));
            ASSERT_TRUE(solver.converged);
            ASSERT_EQ(solver.dual_cone_coef.size(), p.blk_types.size());

            const double scale = 1.0 + std::abs(p.p_star);
            const double bound = solver.certified_lower_bound(bounds);
            printf("[ CERT     ] %s switch_admm=%d: p* = %.10e, <b,y> = %.10e, certified bound = %.10e, "
                   "(p* - bound)/(1+|p*|) = %.2e, errPSD_X = %.2e\n",
                   name.c_str(), switch_admm, p.p_star, solver.dobj, bound, (p.p_star - bound) / scale, solver.errPSD_X);
            EXPECT_LE(bound, p.p_star + 1e-8 * scale) << "the certified bound must not exceed the optimal value";
            EXPECT_GE(bound, p.p_star - 1e-4 * scale) << "the certified bound should be tight after convergence";
            // larger bounds give smaller (still valid) lower bounds
            std::vector<double> larger = bounds;
            for (double &r : larger)
                r *= 10.0;
            EXPECT_LE(solver.certified_lower_bound(larger), bound + 1e-12 * scale);
            EXPECT_GE(solver.errPSD_X, 0.0);
            EXPECT_GE(solver.errPSD_S, 0.0);
        }
    }
}

// certified_lower_bound needs one bound per block.
TEST(Certificate, RejectsWrongNumberOfBounds)
{
    using namespace certificate_test;
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    SDPSolver solver;
    ASSERT_NO_THROW(init_solver(p, solver));
    ASSERT_NO_THROW(solver.solve(50, 1e-6, 500, 50, 100, 0, 0));
    EXPECT_THROW(solver.certified_lower_bound(std::vector<double>(p.blk_types.size() + 1, 1.0)), std::invalid_argument);
    EXPECT_NO_THROW(solver.certified_lower_bound(1.0));
}

