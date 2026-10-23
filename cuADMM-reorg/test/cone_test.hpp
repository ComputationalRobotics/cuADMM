/*

    cone_test.hpp  (GPU)

    Phase 5, the cone layer (plan Sections 8-9): the registry; the projection properties of the free, nonnegative and
    PSD cones and of the product (idempotence, Moreau decomposition, orthogonality, nonexpansiveness, membership);
    the product's measures against the engine's cone_measures of main (bitwise); the host reference measures against
    the device measures; the free-variable tests of Section 9.3 through the facade (LP with free variables, free only
    with non-unique X_u, free with a large block on the LOBPCG path, a warm start with S_u != 0, sGS-ADMM with free
    blocks, the certificate with per-block bounds and with a missing bound).

*/

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cone/cone_product.h"
#include "cone/cone_registry.h"
#include "cuadmm/cuadmm.h"
#include "cuadmm_legacy/io.h"
#include "cuadmm_legacy/solver.h"
#include "regression_problems.hpp"
#include "solver_setup.hpp"
#include "termination/certificate.h"

namespace cone_test
{
using namespace cuadmm;

inline std::vector<double> random_vector(int n, unsigned seed, double scale = 1.0)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd(0.0, scale);
    std::vector<double> v(n);
    for (double &x : v)
        x = nd(gen);
    return v;
}

inline double dot(const std::vector<double> &a, const std::vector<double> &b)
{
    long double s = 0;
    for (std::size_t i = 0; i < a.size(); i++)
        s += (long double)a[i] * b[i];
    return (double)s;
}

inline double norm(const std::vector<double> &a) { return std::sqrt(dot(a, a)); }

// ProblemData of a generated problem for the facade.
inline ProblemData to_problem_data(const regression::RegressionProblem &p)
{
    std::vector<int> rows = p.At_rows, cols = p.At_cols, col_ptrs;
    std::vector<double> vals = p.At_vals;
    COO_to_CSC(col_ptrs, cols, rows, vals, (int)vals.size(), p.con_num);
    ProblemData d;
    d.At.rows = p.vec_len;
    d.At.cols = p.con_num;
    d.At.colptr.assign(col_ptrs.begin(), col_ptrs.end());
    d.At.rowind.assign(rows.begin(), rows.end());
    d.At.val = vals;
    d.b.size = p.con_num;
    for (int i = 0; i < p.con_num; i++)
        if (p.b[i] != 0.0)
        {
            d.b.index.push_back(i);
            d.b.val.push_back(p.b[i]);
        }
    d.C.size = p.vec_len;
    for (int i = 0; i < p.vec_len; i++)
        if (p.C[i] != 0.0)
        {
            d.C.index.push_back(i);
            d.C.val.push_back(p.C[i]);
        }
    for (std::size_t b = 0; b < p.blk_types.size(); b++)
    {
        ConeSpec c;
        c.kind = cone_kind_of(p.blk_types[b]);
        c.size = p.blk_sizes[b];
        d.cones.push_back(c);
    }
    d.source.format = "generator";
    return d;
}

// An engine on a generated problem and the product cone on it.
struct Fixture
{
    regression::RegressionProblem p;
    SDPSolver solver;
    std::unique_ptr<cone::ConeProduct> cones;
    core::ExecContext ctx;
    explicit Fixture(const std::string &name, bool use_lobpcg = true)
    {
        p = regression::generate(name);
        solver_setup::init_solver(p, solver, use_lobpcg, 1.0);
        cones.reset(new cone::ConeProduct(p.blk_types, p.blk_sizes, cone::ConeBuildContext{p.vec_len, &solver}));
        ctx.stream = nullptr;
        ctx.cublas = solver.cublasH.cublas_handle;
        ctx.cusparse = solver.cusparseH.cusparse_handle;
    }
    std::vector<double> project_dual(const std::vector<double> &z, int k = 1)
    {
        DeviceDenseVector<double> dz, ds;
        dz.allocate(0, p.vec_len);
        ds.allocate(0, p.vec_len);
        CHECK_CUDA(cudaMemcpy(dz.vals, z.data(), sizeof(double) * p.vec_len, H2D));
        CHECK_CUDA(cudaMemset(ds.vals, 0, sizeof(double) * p.vec_len));
        cones->project_dual(core::DeviceSpan<const double>(dz.vals, p.vec_len), core::DeviceSpan<double>(ds.vals, p.vec_len), k, ctx);
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<double> s(p.vec_len);
        CHECK_CUDA(cudaMemcpy(s.data(), ds.vals, sizeof(double) * p.vec_len, D2H));
        return s;
    }
    std::vector<double> project_primal(const std::vector<double> &z)
    {
        DeviceDenseVector<double> dz, dx;
        dz.allocate(0, p.vec_len);
        dx.allocate(0, p.vec_len);
        CHECK_CUDA(cudaMemcpy(dz.vals, z.data(), sizeof(double) * p.vec_len, H2D));
        CHECK_CUDA(cudaMemset(dx.vals, 0, sizeof(double) * p.vec_len));
        cones->project_primal(core::DeviceSpan<const double>(dz.vals, p.vec_len), core::DeviceSpan<double>(dx.vals, p.vec_len), ctx);
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<double> x(p.vec_len);
        CHECK_CUDA(cudaMemcpy(x.data(), dx.vals, sizeof(double) * p.vec_len, D2H));
        return x;
    }
    cone::ProductMeasures measure(const std::vector<double> &v, cone::Side side)
    {
        DeviceDenseVector<double> dv;
        dv.allocate(0, p.vec_len);
        CHECK_CUDA(cudaMemcpy(dv.vals, v.data(), sizeof(double) * p.vec_len, H2D));
        return cones->measure(core::DeviceSpan<const double>(dv.vals, p.vec_len), side, ctx);
    }
};
} // namespace cone_test

using namespace cone_test;

TEST(ConeLayer, RegistryCodesAndRejections)
{
    const cone::ConeRegistry &reg = cone::ConeRegistry::instance();
    EXPECT_TRUE(reg.by_code('s').supported);
    EXPECT_EQ(reg.by_code('s').kind, ConeKind::Psd);
    EXPECT_TRUE(reg.by_code('l').supported);
    EXPECT_TRUE(reg.by_code('u').supported);
    EXPECT_FALSE(reg.by_code('q').supported);
    EXPECT_NE(reg.by_code('q').message.find("Phase 9"), std::string::npos);
    EXPECT_FALSE(reg.by_code('r').supported);
    EXPECT_THROW(reg.by_code('x'), std::invalid_argument);
    std::vector<ConeSpec> specs(3);
    specs[0].kind = ConeKind::Psd;
    specs[0].size = 4;
    specs[1].kind = ConeKind::Nonnegative;
    specs[1].size = 3;
    specs[2].kind = ConeKind::Free;
    specs[2].size = 2;
    const std::vector<cone::BlockSlice> sl = cone::ConeRegistry::slices(specs);
    ASSERT_EQ(sl.size(), 3u);
    EXPECT_EQ(sl[0].length, 10);
    EXPECT_EQ(sl[1].offset, 10);
    EXPECT_EQ(sl[1].length, 3);
    EXPECT_EQ(sl[2].offset, 13);
    specs[1].kind = ConeKind::SecondOrder;
    EXPECT_THROW(reg.build(specs, cone::ConeBuildContext{15, nullptr}), std::invalid_argument);
}

// The projection properties of plan Section 8.3 on the product cone of problems with every cone kind and tier.
static void projection_properties(const std::string &name)
{
    Fixture f(name, false);
    const int n = f.p.vec_len;
    const std::vector<double> z1 = random_vector(n, 7), z2 = random_vector(n, 11, 0.3);
    const std::vector<double> s1 = f.project_dual(z1), s2 = f.project_dual(z2);
    // membership: the projection lies in the dual cone
    const cone::ProductMeasures m = f.measure(s1, cone::Side::Dual);
    for (std::size_t b = 0; b < m.violation.size(); b++)
        EXPECT_LE(m.violation[b], 1e-11 * (1.0 + norm(s1))) << name << " block " << b;
    // idempotence
    const std::vector<double> s11 = f.project_dual(s1);
    double d = 0.0;
    for (int i = 0; i < n; i++)
        d = std::max(d, std::abs(s11[i] - s1[i]));
    EXPECT_LE(d, 1e-11 * (1.0 + norm(s1))) << name;
    // Moreau: z = Pi_K(z) - Pi_K*(-z), with <Pi_K(z), Pi_K*(-z)> = 0
    std::vector<double> mz(n);
    for (int i = 0; i < n; i++)
        mz[i] = -z1[i];
    const std::vector<double> x1 = f.project_primal(z1), sm = f.project_dual(mz);
    double moreau = 0.0;
    for (int i = 0; i < n; i++)
        moreau = std::max(moreau, std::abs(z1[i] - (x1[i] - sm[i])));
    EXPECT_LE(moreau, 1e-11 * (1.0 + norm(z1))) << name;
    EXPECT_LE(std::abs(dot(x1, sm)), 1e-9 * (1.0 + norm(z1) * norm(z1))) << name;
    const cone::ProductMeasures mp = f.measure(x1, cone::Side::Primal);
    for (std::size_t b = 0; b < mp.violation.size(); b++)
        EXPECT_LE(mp.violation[b], 1e-11 * (1.0 + norm(x1))) << name << " block " << b;
    // nonexpansiveness
    std::vector<double> dz(n), ds(n);
    for (int i = 0; i < n; i++)
    {
        dz[i] = z1[i] - z2[i];
        ds[i] = s1[i] - s2[i];
    }
    EXPECT_LE(norm(ds), norm(dz) * (1.0 + 1e-12)) << name;
    // the free entries of the dual projection are exactly zero, the nonnegative ones exactly max(z, 0)
    const std::vector<cone::BlockSlice> &bl = f.cones->blocks();
    for (std::size_t b = 0; b < bl.size(); b++)
        for (std::int64_t i = bl[b].offset; i < bl[b].offset + bl[b].length; i++)
        {
            if (f.p.blk_types[b] == 'u')
                EXPECT_EQ(s1[i], 0.0);
            else if (f.p.blk_types[b] == 'l')
                EXPECT_EQ(s1[i], z1[i] > 0.0 ? z1[i] : 0.0);
        }
}

TEST(ConeLayer, ProjectionPropertiesSmallTier) { projection_properties("Nonneg_Free_PSD"); }
TEST(ConeLayer, ProjectionPropertiesSmallBatch) { projection_properties("SmallBatch_Free"); }
TEST(ConeLayer, ProjectionPropertiesMediumTier) { projection_properties("Medium_Free"); }
TEST(ConeLayer, ProjectionPropertiesLpOnly) { projection_properties("Nonneg_Free"); }
TEST(ConeLayer, ProjectionPropertiesFreeOnly) { projection_properties("Free"); }

// The product's measures are bit for bit the engine's cone_measures of main (the same EVDs and loops).
static void measures_match_engine(const std::string &name)
{
    Fixture f(name, false);
    const std::vector<double> v = random_vector(f.p.vec_len, 3);
    const cone::ProductMeasures m = f.measure(v, cone::Side::Dual);
    DeviceDenseVector<double> dv;
    dv.allocate(0, f.p.vec_len);
    CHECK_CUDA(cudaMemcpy(dv.vals, v.data(), sizeof(double) * f.p.vec_len, H2D));
    std::vector<double> block_min, bound_coef;
    f.solver.cone_measures(dv, block_min, bound_coef);
    ASSERT_EQ(m.block_min.size(), block_min.size());
    for (std::size_t b = 0; b < block_min.size(); b++)
    {
        EXPECT_EQ(m.block_min[b], block_min[b]) << name << " block " << b;
        EXPECT_EQ(m.bound_coef[b], bound_coef[b]) << name << " block " << b;
    }
}

TEST(ConeLayer, MeasuresMatchTheEngineSmall) { measures_match_engine("Nonneg_Free_PSD"); }
TEST(ConeLayer, MeasuresMatchTheEngineMedium) { measures_match_engine("Medium_Free"); }
TEST(ConeLayer, MeasuresMatchTheEngineLarge) { measures_match_engine("Large_LowRankS"); }

TEST(ConeLayer, HostReferenceMatchesDeviceMeasure)
{
    Fixture f("Nonneg_Free_PSD", false);
    const std::vector<double> v = random_vector(f.p.vec_len, 5);
    const cone::ProductMeasures m = f.measure(v, cone::Side::Dual);
    std::vector<double> work;
    std::vector<int> iwork;
    for (const auto &c : f.cones->cones())
    {
        const cone::HostReferenceMeasure &ref = c->host_reference();
        ASSERT_NE(ref.block_min, nullptr);
        for (const cone::BlockSlice &b : c->blocks())
        {
            const double h = ref.block_min(v.data() + b.offset, b.size, work, iwork);
            EXPECT_NEAR(h, m.block_min[b.block], 1e-10 * (1.0 + std::abs(h))) << c->name() << " block " << b.block;
        }
    }
}

// ---- free-variable tests of plan Section 9.3 through the facade ----

static SolveResult solve_generated(const std::string &name, SolverOptions o, const WarmStart *warm = nullptr)
{
    const regression::RegressionProblem p = regression::generate(name);
    const ProblemData d = to_problem_data(p);
    Solver solver(o);
    solver.setup(d);
    return solver.solve(warm);
}

static SolverOptions quiet_options()
{
    SolverOptions o;
    o.output.verbosity = 0;
    o.penalty.policy = PenaltyPolicyKind::ResidualBalancing;
    o.penalty.sigma0 = 100.0;
    o.algorithm.tol = 1e-6;
    o.algorithm.max_iter = 20000;
    return o;
}

TEST(FreeVariables, LpWithFreeVariablesConverges)
{
    // an LP has no strongly convex block: plain ADMM needs about 19,000 iterations here at fixed sigma = 1 (the adaptive
    // rules do not reach 1e-6 within 20,000); the test keeps the fixed sigma and a budget of 30,000
    const regression::RegressionProblem p = regression::generate("Nonneg_Free");
    SolverOptions o = quiet_options();
    o.penalty.policy = PenaltyPolicyKind::Fixed;
    o.penalty.sigma0 = 1.0;
    o.algorithm.max_iter = 30000;
    const SolveResult r = solve_generated("Nonneg_Free", o);
    EXPECT_TRUE(r.converged) << r.stop_reason;
    EXPECT_NEAR(r.pobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
    EXPECT_LE(r.cone_violation_S, 1e-6);
}

TEST(FreeVariables, FreeOnlyWithNonUniqueXConverges)
{
    // "Free": 30 free variables, 15 constraints: X_u is not unique (ker A_u has dimension 15); eta and the validator converge
    const regression::RegressionProblem p = regression::generate("Free");
    SolverOptions o = quiet_options();
    o.validation.interval = 100;
    o.validation.tol = 1e-6;
    const SolveResult r = solve_generated("Free", o);
    EXPECT_TRUE(r.externally_validated || r.converged) << r.stop_reason;
    EXPECT_NEAR(r.dobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
    for (double s : r.best.S)
        EXPECT_EQ(s, 0.0); // S_u = 0 exactly
}

TEST(FreeVariables, WarmStartWithNonzeroDualOnFreeBlock)
{
    // a warm start at the optimum with S_u != 0: the first cone step sets S_u = 0 and the solve converges at once
    const regression::RegressionProblem p = regression::generate("PSD_Free");
    WarmStart w;
    w.X = p.X_star;
    w.y = p.y_star;
    w.S = p.S_star;
    const std::size_t off = p.blk_sizes[0] * (p.blk_sizes[0] + 1) / 2;
    for (std::size_t i = off; i < w.S.size(); i++)
        w.S[i] = 1.0;
    SolverOptions o = quiet_options();
    o.algorithm.tol = 1e-5;
    const SolveResult r = solve_generated("PSD_Free", o, &w);
    EXPECT_TRUE(r.converged) << r.stop_reason;
    EXPECT_LT(r.admm_iterations, 200);
    for (std::size_t i = off; i < r.best.S.size(); i++)
        EXPECT_EQ(r.best.S[i], 0.0);
}

TEST(FreeVariables, SgsAdmmWithFreeBlocks)
{
    const regression::RegressionProblem p = regression::generate("Nonneg_Free_PSD");
    SolverOptions o = quiet_options();
    o.algorithm.scheme = SchemeKind::SgsAdmm;
    o.algorithm.tau = 1.95;
    o.penalty.policy = PenaltyPolicyKind::Fixed;
    o.penalty.sigma0 = 10.0;
    o.algorithm.tol = 1e-5;
    const SolveResult r = solve_generated("Nonneg_Free_PSD", o);
    EXPECT_TRUE(r.converged) << r.stop_reason;
    EXPECT_EQ(r.admm_iterations, 0);
    EXPECT_NEAR(r.pobj, p.p_star, 1e-3 * (1.0 + std::abs(p.p_star)));
}

// free variables with a large PSD block on the LOBPCG path: RegressionSolver.Large_Free (regression_solver_test.hpp)

TEST(Certificate, PerBlockBoundsGiveAValidLowerBoundAndAMissingBoundIsReported)
{
    const regression::RegressionProblem p = regression::generate("Nonneg_Free_PSD");
    // bounds from the optimal solution: trace of the PSD block, max entry of the 'l' block, max |entry| of the free block
    std::vector<double> bounds(p.blk_types.size(), 0.0);
    std::size_t off = 0;
    for (std::size_t b = 0; b < p.blk_types.size(); b++)
    {
        const int n = p.blk_sizes[b];
        if (p.blk_types[b] == 's')
        {
            double tr = 0.0;
            for (int c = 0; c < n; c++)
                tr += p.X_star[off + (std::size_t)c * (c + 1) / 2 + c];
            bounds[b] = 1.5 * tr;
            off += (std::size_t)n * (n + 1) / 2;
        }
        else
        {
            double mx = 0.0;
            for (int i = 0; i < n; i++)
                mx = std::max(mx, std::abs(p.X_star[off + i]));
            bounds[b] = 1.5 * mx + 1e-3;
            off += n;
        }
    }
    SolverOptions o = quiet_options();
    o.certificate.block_bounds = bounds;
    const SolveResult r = solve_generated("Nonneg_Free_PSD", o);
    ASSERT_TRUE(r.converged) << r.stop_reason;
    ASSERT_TRUE(std::isfinite(r.certified_lower_bound));
    EXPECT_TRUE(r.certificate_missing_bounds.empty());
    EXPECT_TRUE(r.certificate_conditional_on_free_bounds); // the free block's bound is in force (B24)
    EXPECT_LE(r.certified_lower_bound, p.p_star + 1e-6 * (1.0 + std::abs(p.p_star)));
    EXPECT_GE(r.certified_lower_bound, p.p_star - 1e-2 * (1.0 + std::abs(p.p_star)));
    // a missing bound (negative entry) makes the certificate unavailable and names the block
    o.certificate.block_bounds[1] = -1.0;
    const SolveResult r2 = solve_generated("Nonneg_Free_PSD", o);
    EXPECT_TRUE(std::isnan(r2.certified_lower_bound));
    ASSERT_EQ(r2.certificate_missing_bounds.size(), 1u);
    EXPECT_EQ(r2.certificate_missing_bounds[0], 1);
    EXPECT_NE(r2.certificate_note.find("unavailable"), std::string::npos);
    // the per-block coefficients are those of the engine (original units), one per block
    EXPECT_EQ(r2.dual_cone_coef.size(), p.blk_types.size());
    // the wrong number of bounds is rejected at setup
    o.certificate.block_bounds.push_back(1.0);
    EXPECT_THROW(solve_generated("Nonneg_Free_PSD", o), std::invalid_argument);
}
