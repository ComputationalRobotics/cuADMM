/*

    psd_strategies_test.hpp

    Phase 6a: the PSD projector classes on the GPU, through PsdCone on an engine set up from a generated problem
    with large blocks (n = 1100 and 1050).

    - FullEvd against the exact projection of a matrix with a prescribed spectrum (built on the host from
      Householder reflections: A = H diag(lambda) H^T, P = H max(lambda, 0) H^T).
    - RankAnalyzer: the modes, the pair counts, the warm start (positive and negative low rank, dense, tiny
      eigenvalues below the rank tolerance), then Lobpcg from that warm start; a rank that grew past k is
      rejected by LOBPCG's certificate and falls back to FullEvd with a new analysis.
    - CompositeFilter: accuracy (relative Frobenius error below 2e-2, as the kernel test), determinism of the
      deterministic Lanczos start (B8: two projections bit for bit), rejection by the norm test with the input
      restored (B22).
    - LargeBlockPolicy inside the cone: the per-block warm-up threshold on two large blocks (a mixed iteration with
      per-block products), the switch, the engine's bookkeeping of it.
    - End to end through the facade: the composite warm-up on Large_LowRankS and Large_Two converges to the
      generator's optimum, is reproducible, and reports the switch iteration; without a warm-up it reports none.

*/

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <fstream>
#include <string>
#include <vector>

#include "cone/cone_product.h"
#include "cone/psd/psd_cone.h"
#include "core/stream_pool.h"
#include "core/workspace_arena.h"
#include "core/device_buffer.h"
#include "cuadmm/solver.h"
#include "cuadmm_legacy/problem.h"
#include "cuadmm_legacy/solver.h"
#include "regression_problems.hpp"

namespace psd_strategies
{
using cuadmm::cone::psd::kModeFull;
using cuadmm::cone::psd::kModeLowRankNegative;
using cuadmm::cone::psd::kModeLowRankPositive;
using cuadmm::cone::psd::LargeStrategy;

const double kSqrt2 = std::sqrt(2.0);

std::string data_dir()
{
    const char *dir = std::getenv("CUADMM_REGRESSION_DATA_DIR");
    return (dir != nullptr && dir[0] != '\0') ? std::string(dir) : std::string("regression_data");
}

/// A symmetric matrix with a prescribed spectrum and its exact projection, column-major dense:
/// A = H diag(lambda) H^T and P = H max(lambda, 0) H^T, H a product of Householder reflections from a fixed generator.
struct Spectral
{
    int n = 0;
    std::vector<double> A, P;
};

void reflect(std::vector<double> &M, int n, const std::vector<double> &v)
{
    // M <- (I - 2 v v^T) M (I - 2 v v^T) with M symmetric: w = M v, a = v^T w, M -= 2 v w^T + 2 w v^T - 4 a v v^T
    std::vector<double> w(n, 0.0);
    for (int c = 0; c < n; c++)
        for (int r = 0; r < n; r++)
            w[r] += M[r + (std::size_t)c * n] * v[c];
    double a = 0.0;
    for (int r = 0; r < n; r++)
        a += v[r] * w[r];
    for (int c = 0; c < n; c++)
        for (int r = 0; r < n; r++)
            M[r + (std::size_t)c * n] += -2.0 * v[r] * w[c] - 2.0 * w[r] * v[c] + 4.0 * a * v[r] * v[c];
}

Spectral make_spectral(int n, const std::vector<double> &lambda, unsigned seed, int reflections = 8)
{
    Spectral s;
    s.n = n;
    s.A.assign((std::size_t)n * n, 0.0);
    s.P.assign((std::size_t)n * n, 0.0);
    for (int i = 0; i < n; i++)
    {
        s.A[i + (std::size_t)i * n] = lambda[i];
        s.P[i + (std::size_t)i * n] = std::max(lambda[i], 0.0);
    }
    std::mt19937 gen(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int r = 0; r < reflections; r++)
    {
        std::vector<double> v(n);
        double norm = 0.0;
        for (int i = 0; i < n; i++)
        {
            v[i] = normal(gen);
            norm += v[i] * v[i];
        }
        norm = std::sqrt(norm);
        for (int i = 0; i < n; i++)
            v[i] /= norm;
        reflect(s.A, n, v);
        reflect(s.P, n, v);
    }
    return s;
}

/// The svec of a dense symmetric matrix (upper triangle by columns, off-diagonal entries times sqrt 2).
std::vector<double> svec(const std::vector<double> &M, int n)
{
    std::vector<double> v((std::size_t)n * (n + 1) / 2);
    for (int c = 0; c < n; c++)
        for (int r = 0; r <= c; r++)
            v[(std::size_t)c * (c + 1) / 2 + r] = (r == c ? 1.0 : kSqrt2) * M[r + (std::size_t)c * n];
    return v;
}

/// A spectrum with a gap around 0: alternating signs, magnitudes uniform in [0.05, 1] (the composite filter's regime).
std::vector<double> gapped_spectrum(int n, unsigned seed)
{
    std::vector<double> lambda(n);
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> u(0.05, 1.0);
    for (int i = 0; i < n; i++)
        lambda[i] = (i % 2 == 0 ? 1.0 : -1.0) * u(g);
    return lambda;
}

/// Spectra: `positive` eigenvalues top, top - 1, ..., the rest uniform in [neg_lo, neg_hi] (positive <= top + 1).
std::vector<double> spectrum(int n, int positive, double top, double neg_lo, double neg_hi, unsigned seed)
{
    std::vector<double> lambda(n);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> u(neg_lo, neg_hi);
    for (int i = 0; i < n; i++)
        lambda[i] = i < positive ? top - i : u(gen);
    return lambda;
}

double relative_error(const std::vector<double> &s, std::size_t offset, const std::vector<double> &ref)
{
    double diff = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < ref.size(); i++)
    {
        diff += (s[offset + i] - ref[i]) * (s[offset + i] - ref[i]);
        norm += ref[i] * ref[i];
    }
    return std::sqrt(diff) / std::sqrt(norm);
}
} // namespace psd_strategies

class PsdStrategies : public ::testing::Test
{
protected:
    // the engine of a generated problem, as the regression harness builds it (TXT round trip through the loader)
    void init(const std::string &name, bool use_lobpcg, ProjectionMethod initial)
    {
        gen = regression::generate(name);
        std::string dir = psd_strategies::data_dir() + "/" + name;
        regression::write_txt(gen, dir);
        Problem problem;
        problem.from_txt(dir);
        ASSERT_EQ(problem.vec_len, gen.vec_len);
        blk_types.clear();
        blk_sizes.clear();
        for (const auto &blk : problem.blk_vals)
        {
            blk_types.push_back(std::get<0>(blk));
            blk_sizes.push_back(std::get<1>(blk));
        }
        engine.init(
            15, problem.vec_len, problem.con_num,
            problem.At_csc_col_ptrs.data(), problem.At_csc_row_ids.data(), problem.At_csc_vals.data(), problem.At_nnz,
            problem.b_indices.data(), problem.b_vals.data(), problem.b_nnz,
            problem.C_indices.data(), problem.C_vals.data(), problem.C_nnz,
            blk_types.data(), blk_sizes.data(), problem.mat_num,
            initial, ProjectionMethod::EIG_FP64,
            nullptr, nullptr, nullptr,
            1.0, use_lobpcg);
        engine.stop_tol = 1e-6;
        vec_len = problem.vec_len;
        z.allocate(vec_len);
        s.allocate(vec_len);
        z.zero();
        s.zero();
    }

    cuadmm::cone::PsdCone &make_cone(const cuadmm::cone::PsdConeConfig &cfg)
    {
        product.reset(new cuadmm::cone::ConeProduct(blk_types, blk_sizes, cuadmm::cone::ConeBuildContext{vec_len, &engine, &cfg, &arena, &pool}));
        arena.commit();
        cuadmm::cone::PsdCone &cone = *static_cast<cuadmm::cone::PsdCone *>(product->cone(cuadmm::ConeKind::Psd));
        cone.bind();
        return cone;
    }

    // z <- the svec of the blocks (one dense matrix per PSD block, in input order)
    void upload(const std::vector<const psd_strategies::Spectral *> &blocks)
    {
        std::vector<double> h;
        for (const psd_strategies::Spectral *b : blocks)
        {
            const std::vector<double> v = psd_strategies::svec(b->A, b->n);
            h.insert(h.end(), v.begin(), v.end());
        }
        ASSERT_EQ((int)h.size(), vec_len);
        CUADMM_CHECK_CUDA(cudaMemcpy(z.data(), h.data(), sizeof(double) * h.size(), cudaMemcpyHostToDevice));
    }

    // the flat index of input block b among the large blocks (the engine orders them by size group)
    int flat_index(int b) const
    {
        const BlockInfo &bi = engine.blk_info[b];
        int flat = bi.same_size_index;
        for (int i = 0; i < bi.size_index; i++)
            flat += engine.sizes.large_mat_nums[i];
        return flat;
    }

    std::vector<double> download(const cuadmm::core::DeviceBuffer<double> &d)
    {
        std::vector<double> h(d.size());
        CUADMM_CHECK_CUDA(cudaMemcpy(h.data(), d.data(), sizeof(double) * h.size(), cudaMemcpyDeviceToHost));
        return h;
    }

    // one cone step at iteration k with the residual eta: s <- Pi(z)
    void project(cuadmm::cone::PsdCone &cone, int k, double eta)
    {
        cone.begin_iteration(k, eta, false);
        cone.project_dual(z.span().as_const(), s.span(), ctx);
        CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
    }

    regression::RegressionProblem gen;
    SDPSolver engine;
    std::vector<char> blk_types;
    std::vector<int> blk_sizes;
    int vec_len = 0;
    cuadmm::core::DeviceBuffer<double> z, s;
    cuadmm::core::ExecContext ctx;
    cuadmm::core::StreamPool pool{cuadmm::core::StreamPoolConfig{}};
    cuadmm::core::WorkspaceArena arena;
    std::unique_ptr<cuadmm::cone::ConeProduct> product;
};

TEST_F(PsdStrategies, FullEvdMatchesTheHostProjection)
{
    using namespace psd_strategies;
    ASSERT_NO_FATAL_FAILURE(init("Large_LowRankS", true, ProjectionMethod::EIG_FP64));
    const int n = 1100;
    const Spectral sp = make_spectral(n, spectrum(n, 550, 600.0, -2.0, -1.0, 11), 101);
    ASSERT_NO_FATAL_FAILURE(upload({&sp}));
    cuadmm::cone::PsdCone &cone = make_cone(cuadmm::cone::PsdConeConfig{});
    project(cone, 1, std::numeric_limits<double>::infinity());
    ASSERT_EQ(cone.last_strategies().size(), 1u);
    EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::FullEvd);
    const std::vector<double> out = download(s);
    EXPECT_LT(relative_error(out, 0, svec(sp.P, n)), 1e-10);
    EXPECT_EQ(engine.lobpcg_calls, 0);
}

TEST_F(PsdStrategies, RankAnalysisPositiveThenLobpcg)
{
    using namespace psd_strategies;
    ASSERT_NO_FATAL_FAILURE(init("Large_LowRankS", true, ProjectionMethod::EIG_FP64));
    const int n = 1100;
    const Spectral sp = make_spectral(n, spectrum(n, 3, 5.0, -2.0, -1.0, 12), 102);
    ASSERT_NO_FATAL_FAILURE(upload({&sp}));
    cuadmm::cone::PsdCone &cone = make_cone(cuadmm::cone::PsdConeConfig{});
    project(cone, 2, 1.0); // the first analysis (psd.lowrank.start_iter = 2)
    EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::FullEvdWithRankAnalysis);
    EXPECT_EQ(engine.large_mode[0], kModeLowRankPositive);
    EXPECT_EQ(engine.large_k[0], 4) << "ceil(1.2 * 3)";
    EXPECT_EQ(engine.cpu_positive_ranks.vals[0], 3);
    EXPECT_EQ(engine.cpu_negative_ranks.vals[0], n - 3);
    EXPECT_NEAR(engine.large_rank_tol[0], 1e-8 * 5.0, 1e-12) << "max(rtol * max|lambda|, 1e-3 * stop_tol)";
    std::vector<double> D(4);
    CUADMM_CHECK_CUDA(cudaMemcpy(D.data(), cone.large_tier().D(0), sizeof(double) * 4, cudaMemcpyDeviceToHost));
    EXPECT_NEAR(D[0], 5.0, 1e-9);
    EXPECT_NEAR(D[1], 4.0, 1e-9);
    EXPECT_NEAR(D[2], 3.0, 1e-9);
    EXPECT_LT(D[3], 0.0) << "the relaxation pair is the largest negative eigenvalue";
    EXPECT_LT(relative_error(download(s), 0, svec(sp.P, n)), 1e-10);

    project(cone, 3, 1.0); // the warm-started LOBPCG
    EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::Lobpcg);
    EXPECT_EQ(engine.lobpcg_calls, 1);
    EXPECT_EQ(engine.lobpcg_fallbacks, 0);
    EXPECT_LT(relative_error(download(s), 0, svec(sp.P, n)), 1e-8);
}

TEST_F(PsdStrategies, RankAnalysisNegativeThenLobpcg)
{
    using namespace psd_strategies;
    ASSERT_NO_FATAL_FAILURE(init("Large_LowRankS", true, ProjectionMethod::EIG_FP64));
    const int n = 1100;
    std::vector<double> lambda = spectrum(n, 3, 5.0, 1.0, 2.0, 13);
    for (int i = 0; i < 3; i++)
        lambda[i] = -lambda[i]; // -5, -4, -3 and the rest in [1, 2]
    const Spectral sp = make_spectral(n, lambda, 103);
    ASSERT_NO_FATAL_FAILURE(upload({&sp}));
    cuadmm::cone::PsdCone &cone = make_cone(cuadmm::cone::PsdConeConfig{});
    project(cone, 2, 1.0);
    EXPECT_EQ(engine.large_mode[0], kModeLowRankNegative);
    EXPECT_EQ(engine.large_k[0], 4);
    EXPECT_EQ(engine.cpu_negative_ranks.vals[0], 3);
    std::vector<double> D(4);
    CUADMM_CHECK_CUDA(cudaMemcpy(D.data(), cone.large_tier().D(0), sizeof(double) * 4, cudaMemcpyDeviceToHost));
    EXPECT_NEAR(D[0], 5.0, 1e-9) << "the eigenvalues of -A, decreasing";
    EXPECT_NEAR(D[1], 4.0, 1e-9);
    EXPECT_NEAR(D[2], 3.0, 1e-9);
    EXPECT_LT(relative_error(download(s), 0, svec(sp.P, n)), 1e-10);
    project(cone, 3, 1.0);
    EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::Lobpcg);
    EXPECT_EQ(engine.lobpcg_calls, 1);
    EXPECT_EQ(engine.lobpcg_fallbacks, 0);
    EXPECT_LT(relative_error(download(s), 0, svec(sp.P, n)), 1e-8);
}

TEST_F(PsdStrategies, DenseSpectrumStaysExactAndTinyEigenvaluesDoNotCount)
{
    using namespace psd_strategies;
    ASSERT_NO_FATAL_FAILURE(init("Large_LowRankS", true, ProjectionMethod::EIG_FP64));
    const int n = 1100;
    const Spectral half = make_spectral(n, spectrum(n, 550, 600.0, -2.0, -1.0, 14), 104);
    ASSERT_NO_FATAL_FAILURE(upload({&half}));
    cuadmm::cone::PsdCone &cone = make_cone(cuadmm::cone::PsdConeConfig{});
    project(cone, 2, 1.0);
    EXPECT_EQ(engine.large_mode[0], kModeFull);
    EXPECT_EQ(engine.large_k[0], 0);
    project(cone, 3, 1.0);
    EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::FullEvd);
    EXPECT_EQ(engine.lobpcg_calls, 0);

    // three eigenvalues of 1e-12 are below tol_r = max(1e-8 * 2, 1e-3 * 1e-6): neither positive nor negative
    std::vector<double> lambda = spectrum(n, 3, 0.0, -2.0, -1.0, 15);
    for (int i = 0; i < 3; i++)
        lambda[i] = 1e-12;
    const Spectral tiny = make_spectral(n, lambda, 105);
    ASSERT_NO_FATAL_FAILURE(upload({&tiny}));
    project(cone, 102, 1.0); // the next analysis
    EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::FullEvdWithRankAnalysis);
    EXPECT_EQ(engine.cpu_positive_ranks.vals[0], 0);
    EXPECT_EQ(engine.cpu_negative_ranks.vals[0], n - 3);
    EXPECT_EQ(engine.large_mode[0], kModeFull);
    EXPECT_NEAR(cone.rank_analyzer().rank_tolerance(-2.0, 1e-12, 1e-6), 2e-8, 1e-15);
    EXPECT_NEAR(cone.rank_analyzer().rank_tolerance(-1e-9, 1e-9, 1e-2), 1e-5, 1e-15) << "the floor from the stopping tolerance";
}

TEST_F(PsdStrategies, LobpcgRejectsAGrownRankAndFallsBackToFullEvd)
{
    using namespace psd_strategies;
    ASSERT_NO_FATAL_FAILURE(init("Large_LowRankS", true, ProjectionMethod::EIG_FP64));
    const int n = 1100;
    const Spectral before = make_spectral(n, spectrum(n, 3, 5.0, -2.0, -1.0, 16), 106);
    const Spectral after = make_spectral(n, spectrum(n, 30, 32.0, -2.0, -1.0, 16), 106); // the same eigenvectors
    ASSERT_NO_FATAL_FAILURE(upload({&before}));
    cuadmm::cone::PsdCone &cone = make_cone(cuadmm::cone::PsdConeConfig{});
    project(cone, 2, 1.0);
    ASSERT_EQ(engine.large_mode[0], kModeLowRankPositive);
    ASSERT_EQ(engine.large_k[0], 4);

    ASSERT_NO_FATAL_FAILURE(upload({&after}));
    project(cone, 3, 1.0);
    EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::Lobpcg);
    EXPECT_EQ(engine.lobpcg_calls, 1);
    EXPECT_EQ(engine.lobpcg_fallbacks, 1) << "4 pairs cannot cover 30 positive eigenvalues: the certificate rejects";
    EXPECT_LT(relative_error(download(s), 0, svec(after.P, n)), 1e-10) << "the fallback is exact";
    EXPECT_EQ(engine.cpu_positive_ranks.vals[0], 30) << "the fallback analyzed the rank again";
    EXPECT_EQ(engine.large_mode[0], kModeLowRankPositive) << "30 < 0.05 * 1100";
    EXPECT_EQ(engine.large_k[0], 36) << "ceil(1.2 * 30) within k_alloc = ceil(1100 * 0.06) = 66";
    project(cone, 4, 1.0);
    EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::Lobpcg);
    EXPECT_EQ(engine.lobpcg_fallbacks, 1);
    EXPECT_LT(relative_error(download(s), 0, svec(after.P, n)), 1e-8);
}

// psd.lowrank.ratio = 0.05 (2026-11) from the options: n = 1100 gets a warm start of ceil(1100 * 0.05 * 1.2) = 66 columns;
// 50 positive eigenvalues are low rank (50 < 55; with 0.03 the bound was 33) and LOBPCG gives the exact projection
// with ceil(1.2 * 50) = 60 pairs; 56 positive eigenvalues keep the full EVD
TEST_F(PsdStrategies, LowRankRatioOfTheOptionsSizesTheWarmStartAndDecidesTheMode)
{
    using namespace psd_strategies;
    ASSERT_NO_FATAL_FAILURE(init("Large_LowRankS", true, ProjectionMethod::EIG_FP64));
    const int n = 1100;
    cuadmm::SolverOptions so;
    EXPECT_EQ(so.psd.lowrank_ratio, 0.05);
    EXPECT_EQ(so.psd.lowrank_relaxation, 1.2);
    EXPECT_EQ(so.psd.lowrank_reevaluate, 100);
    so.psd.warmup = cuadmm::WarmupKind::None; // this test is about the rank analysis (no composite warm-up first)
    cuadmm::cone::PsdCone &cone = make_cone(cuadmm::cone::PsdConeConfig::from_options(so.psd, 0));
    ASSERT_EQ(cone.large_tier().count(), 1);
    EXPECT_EQ(cone.large_tier().k_alloc(0), 66);
    const Spectral fifty = make_spectral(n, spectrum(n, 50, 64.0, -2.0, -1.0, 18), 108);
    ASSERT_NO_FATAL_FAILURE(upload({&fifty}));
    project(cone, 2, 1.0); // the first analysis (blocks without a warm-up start at psd.lowrank.start_iter = 2)
    EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::FullEvdWithRankAnalysis);
    EXPECT_EQ(engine.cpu_positive_ranks.vals[0], 50);
    EXPECT_EQ(engine.large_mode[0], kModeLowRankPositive) << "50 < 0.05 * 1100";
    EXPECT_EQ(engine.large_k[0], 60) << "ceil(1.2 * 50) within k_alloc = 66";
    EXPECT_LT(relative_error(download(s), 0, svec(fifty.P, n)), 1e-10);
    for (int k = 3; k <= 5; k++)
    {
        project(cone, k, 1.0);
        EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::Lobpcg) << k;
        EXPECT_LT(relative_error(download(s), 0, svec(fifty.P, n)), 1e-8) << k;
    }
    EXPECT_EQ(engine.lobpcg_calls, 3);
    EXPECT_EQ(engine.lobpcg_fallbacks, 0);
    const Spectral fifty_six = make_spectral(n, spectrum(n, 56, 64.0, -2.0, -1.0, 18), 109);
    ASSERT_NO_FATAL_FAILURE(upload({&fifty_six}));
    project(cone, 102, 1.0); // the next analysis
    EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::FullEvdWithRankAnalysis);
    EXPECT_EQ(engine.cpu_positive_ranks.vals[0], 56);
    EXPECT_EQ(engine.large_mode[0], kModeFull) << "56 >= 0.05 * 1100";
    EXPECT_LT(relative_error(download(s), 0, svec(fifty_six.P, n)), 1e-10);
}

TEST_F(PsdStrategies, CompositeFilterIsAccurateAndDeterministic)
{
    using namespace psd_strategies;
    ASSERT_NO_FATAL_FAILURE(init("Large_LowRankS", true, ProjectionMethod::COMPOSITE_FP16));
    const int n = 1100;
    const Spectral sp = make_spectral(n, gapped_spectrum(n, 17), 107);
    ASSERT_NO_FATAL_FAILURE(upload({&sp}));

    cuadmm::cone::PsdConeConfig cfg;
    cfg.policy.warmup = true;
    cfg.policy.warmup_min_size = 1000;
    cfg.composite.seed = 7;
    cuadmm::cone::PsdCone &cone = make_cone(cfg);
    EXPECT_TRUE(cone.policy().warm(0));
    project(cone, 1, std::numeric_limits<double>::infinity());
    EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::CompositeFilter);
    EXPECT_EQ(engine.composite_fallbacks, 0);
    const std::vector<double> first = download(s);
    const double err = relative_error(first, 0, svec(sp.P, n));
    EXPECT_LT(err, 2e-2) << "FP16 composite filter";
    EXPECT_GT(err, 1e-12) << "the filter is not exact (the test would not distinguish it from FullEvd)";

    // B8: the same input and seed give the same bits
    project(cone, 1, std::numeric_limits<double>::infinity());
    const std::vector<double> second = download(s);
    std::size_t differences = 0;
    for (std::size_t i = 0; i < first.size(); i++)
        differences += first[i] != second[i];
    EXPECT_EQ(differences, 0u);
    EXPECT_EQ(cuadmm::cone::psd::CompositeFilter::block_seed(7, 0), 7ULL ^ 0x9E3779B97F4A7C15ULL);

    // the warm-up ends at the switch: the exact projection with the rank analysis
    project(cone, 5, 1e-3);
    EXPECT_TRUE(cone.policy().switched());
    EXPECT_EQ(cone.policy().switch_iteration(), 5);
    EXPECT_EQ(engine.switched_proj_method_iter, 5);
    EXPECT_EQ(engine.current_proj_method, ProjectionMethod::EIG_FP64);
    EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::FullEvdWithRankAnalysis);
    EXPECT_LT(relative_error(download(s), 0, svec(sp.P, n)), 1e-10);
}

TEST_F(PsdStrategies, CompositeFilterRejectsByTheNormTestAndRestoresTheInput)
{
    using namespace psd_strategies;
    ASSERT_NO_FATAL_FAILURE(init("Large_LowRankS", true, ProjectionMethod::COMPOSITE_FP16));
    const int n = 1100;
    const Spectral sp = make_spectral(n, gapped_spectrum(n, 18), 108);
    ASSERT_NO_FATAL_FAILURE(upload({&sp}));
    cuadmm::cone::PsdConeConfig cfg;
    cfg.policy.warmup = true;
    cfg.policy.warmup_min_size = 1000;
    cuadmm::cone::PsdCone &cone = make_cone(cfg);

    // the dense input of the block, as the cone step builds it
    cone.large_tier().unpack(0, z.data());
    CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<double> before((std::size_t)n * n);
    CUADMM_CHECK_CUDA(cudaMemcpy(before.data(), cone.large_tier().M(), sizeof(double) * before.size(), cudaMemcpyDeviceToHost));

    cuadmm::cone::psd::CompositeFilterConfig strict;
    strict.norm_slack = -1.0; // every result fails ||P|| <= (1 + slack) ||Z||
    cuadmm::cone::psd::CompositeFilter filter(strict, cone.large_tier());
    EXPECT_FALSE(filter.project(cone.job(0), z.data(), s.data()));
    EXPECT_EQ(engine.composite_fallbacks, 1);
    CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<double> after((std::size_t)n * n);
    CUADMM_CHECK_CUDA(cudaMemcpy(after.data(), cone.large_tier().M(), sizeof(double) * after.size(), cudaMemcpyDeviceToHost));
    std::size_t differences = 0;
    for (std::size_t i = 0; i < before.size(); i++)
        differences += before[i] != after[i];
    EXPECT_EQ(differences, 0u) << "the input is restored for the exact fallback";

    // through the cone the fallback is the exact projection
    project(cone, 1, std::numeric_limits<double>::infinity());
    EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::CompositeFilter);
    EXPECT_LT(relative_error(download(s), 0, svec(sp.P, n)), 2e-2);
}

TEST_F(PsdStrategies, PerBlockWarmupThresholdOnTwoLargeBlocks)
{
    using namespace psd_strategies;
    ASSERT_NO_FATAL_FAILURE(init("Large_Two", true, ProjectionMethod::COMPOSITE_FP16));
    ASSERT_EQ(engine.sizes.large_mat_num, 2);
    const int n0 = 1100, n1 = 1050;
    const Spectral s0 = make_spectral(n0, gapped_spectrum(n0, 19), 109), s1 = make_spectral(n1, spectrum(n1, 3, 5.0, -2.0, -1.0, 20), 110);
    ASSERT_NO_FATAL_FAILURE(upload({&s0, &s1}));
    const std::size_t off1 = (std::size_t)n0 * (n0 + 1) / 2;

    cuadmm::cone::PsdConeConfig cfg;
    cfg.policy.warmup = true;
    cfg.policy.warmup_min_size = 1080; // block 0 (1100) warm, block 1 (1050) exact
    cfg.policy.switch_tol = 1e-2;
    cuadmm::cone::PsdCone &cone = make_cone(cfg);
    const int f0 = flat_index(0), f1 = flat_index(1); // the engine's size-group order, not the input order
    ASSERT_NE(f0, f1);
    EXPECT_EQ(engine.sizes.large_mat_sizes[engine.blk_info[0].size_index], n0);
    EXPECT_TRUE(cone.policy().warm(f0));
    EXPECT_FALSE(cone.policy().warm(f1));

    project(cone, 1, std::numeric_limits<double>::infinity()); // a mixed iteration: per-block products
    ASSERT_EQ(cone.last_strategies().size(), 2u);
    EXPECT_EQ(cone.last_strategies()[f0], LargeStrategy::CompositeFilter);
    EXPECT_EQ(cone.last_strategies()[f1], LargeStrategy::FullEvd);
    std::vector<double> out = download(s);
    EXPECT_LT(relative_error(out, 0, svec(s0.P, n0)), 2e-2);
    EXPECT_LT(relative_error(out, off1, svec(s1.P, n1)), 1e-10);

    project(cone, 2, 1.0);
    EXPECT_EQ(cone.last_strategies()[f0], LargeStrategy::CompositeFilter);
    EXPECT_EQ(cone.last_strategies()[f1], LargeStrategy::FullEvdWithRankAnalysis);
    EXPECT_EQ(engine.large_mode[f1], kModeLowRankPositive);
    EXPECT_EQ(engine.large_mode[f0], kModeFull) << "no analysis of a block in its warm-up";

    project(cone, 3, 1e-3); // the switch
    EXPECT_EQ(cone.policy().switch_iteration(), 3);
    EXPECT_EQ(cone.last_strategies()[f0], LargeStrategy::FullEvdWithRankAnalysis);
    EXPECT_EQ(cone.last_strategies()[f1], LargeStrategy::Lobpcg);
    out = download(s);
    EXPECT_LT(relative_error(out, 0, svec(s0.P, n0)), 1e-10);
    EXPECT_LT(relative_error(out, off1, svec(s1.P, n1)), 1e-8);
    EXPECT_EQ(engine.lobpcg_calls, 1);
    EXPECT_EQ(engine.lobpcg_fallbacks, 0);
    EXPECT_EQ(engine.composite_fallbacks, 0);
}

// end to end through the facade (the helpers of cone_test.hpp: solve_generated, quiet_options)

namespace
{
cuadmm::SolverOptions warmup_options(bool warmup, int min_size)
{
    cuadmm::SolverOptions o = quiet_options();
    o.penalty.sigma0 = 1e-4; // as the regression harness: not solved before the rank analysis
    o.algorithm.max_iter = 5000;
    o.psd.warmup = warmup ? cuadmm::WarmupKind::CompositeFp16 : cuadmm::WarmupKind::None;
    o.psd.warmup_min_size = min_size;
    return o;
}
} // namespace

// The default cutoffs 32 / 1000 and warm-up minimum size 5000 (2026-11-10, as before 2026-11-07) through the facade:
// no block of the tier problem is large, so no warm-up and no LOBPCG run; with the options' cutoffs 32 / 64 and the
// warm-up from 128 (the defaults of 2026-11-07) the 128 block is large with the warm-up (a switch is reported).
// The facade's cutoffs hold during its setup only: afterwards the C++ API of main sees 32 / 1000 again.
TEST(TierCutoffs, DefaultCutoffsKeepTheBlocksUpTo1000MediumThroughTheFacade)
{
    const regression::RegressionProblem p = regression::generate("Tiers_32_33_64_65_128");
    cuadmm::SolverOptions o = quiet_options();
    EXPECT_EQ(o.psd.small_max, 32);
    EXPECT_EQ(o.psd.medium_max, 1000);
    EXPECT_EQ(o.psd.warmup_min_size, 5000);
    o.algorithm.max_iter = 5000;
    const cuadmm::SolveResult d = solve_generated("Tiers_32_33_64_65_128", o);
    EXPECT_TRUE(d.converged) << d.stop_reason;
    EXPECT_NEAR(d.pobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
    EXPECT_EQ(d.warmup_switch_iteration, -1) << "no large block with the default medium cutoff 1000";
    EXPECT_EQ(d.lobpcg_calls, 0);
    o.psd.medium_max = 64;
    o.psd.warmup_min_size = 128;
    const cuadmm::SolveResult r = solve_generated("Tiers_32_33_64_65_128", o);
    EXPECT_TRUE(r.converged) << r.stop_reason;
    EXPECT_NEAR(r.pobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
    EXPECT_GT(r.warmup_switch_iteration, 1) << "the 128 block had the composite warm-up";
    EXPECT_EQ(MatrixSizes::get_size_category(65), MatrixSizeCategory::MEDIUM) << "the facade restored the cutoffs of the C++ API";
    printf("[ RESULT   ] Tiers_32_33_64_65_128: cutoffs 32/1000 (default): %d iterations; 32/64 with the warm-up from 128: %d iterations, switch at %d, LOBPCG calls %d\n",
           d.iterations, r.iterations, r.warmup_switch_iteration, r.lobpcg_calls);
}

namespace
{
// psd.trace lines: the number of key, the string of key ("" when absent)
double trace_field(const std::string &line, const std::string &key)
{
    const std::size_t at = line.find("\"" + key + "\": ");
    return at == std::string::npos ? std::nan("") : std::strtod(line.c_str() + at + key.size() + 4, nullptr);
}
std::string trace_text(const std::string &line, const std::string &key)
{
    const std::string head = "\"" + key + "\": \"";
    const std::size_t at = line.find(head);
    if (at == std::string::npos)
        return "";
    const std::size_t end = line.find('"', at + head.size());
    return line.substr(at + head.size(), end - at - head.size());
}
} // namespace

// psd.trace (2026-11) on the tier problem with the cutoffs 32 / 64 and the warm-up from 128 (options; the defaults
// 1000 / 5000 make no block large): the settings, every PSD block with its tier, one line per rank analysis of
// a large block (scheduled every 100 iterations from its start, or after a rejected LOBPCG result), the warm-up switch,
// and per-block totals (written when the solver is destroyed) that agree with the counters of the solve and account
// for one projection per block and iteration
TEST(PsdTrace, RecordsTiersRankAnalysesAndTotals)
{
    cuadmm::SolverOptions o = quiet_options();
    o.algorithm.max_iter = 5000;
    o.psd.medium_max = 64;       // blocks 65 and 128 are large
    o.psd.warmup_min_size = 128; // the 128 block has the warm-up
    const std::string path = psd_strategies::data_dir() + "/psd_trace_tiers.jsonl";
    o.psd.trace = path;
    const cuadmm::SolveResult r = solve_generated("Tiers_32_33_64_65_128", o); // the solver is destroyed on return
    std::ifstream in(path);
    ASSERT_TRUE(in.good()) << path;
    std::vector<std::string> config, blocks, ranks, switches, totals, ends;
    std::string line;
    while (std::getline(in, line))
    {
        const std::string ev = trace_text(line, "ev");
        (ev == "config" ? config : ev == "block" ? blocks : ev == "rank" ? ranks : ev == "warmup_switch" ? switches : ev == "block_totals" ? totals : ends)
            .push_back(line);
    }
    ASSERT_EQ(config.size(), 1u);
    EXPECT_EQ(trace_field(config[0], "small_max"), 32);
    EXPECT_EQ(trace_field(config[0], "medium_max"), 64);
    EXPECT_EQ(trace_field(config[0], "warmup_min_size"), 128);
    EXPECT_EQ(trace_field(config[0], "lowrank_ratio"), 0.05);
    EXPECT_EQ(trace_field(config[0], "lowrank_relaxation"), 1.2);
    EXPECT_EQ(trace_field(config[0], "lowrank_reevaluate"), 100);
    ASSERT_EQ(blocks.size(), 5u) << "the five PSD blocks (the LP block is not one)";
    const char *tiers[] = {"small", "medium", "medium", "large", "large"};
    for (int b = 0; b < 5; b++)
        EXPECT_EQ(trace_text(blocks[b], "tier"), tiers[b]) << blocks[b];
    EXPECT_EQ(trace_text(blocks[3], "warm"), "") << "booleans are not strings";
    EXPECT_NE(blocks[3].find("\"warm\": false"), std::string::npos) << "65 < 128";
    EXPECT_NE(blocks[4].find("\"warm\": true"), std::string::npos);
    EXPECT_EQ(trace_field(blocks[3], "k_alloc"), 4) << "ceil(65 * 0.05 * 1.2)";
    EXPECT_EQ(trace_field(blocks[4], "k_alloc"), 8) << "ceil(128 * 0.05 * 1.2)";
    ASSERT_EQ(switches.size(), 1u);
    EXPECT_EQ(trace_field(switches[0], "k"), r.warmup_switch_iteration);
    ASSERT_EQ(totals.size(), 2u);
    ASSERT_EQ(ends.size(), 1u);
    double lobpcg = 0, rejected = 0, composite_rejected = 0;
    for (const std::string &t : totals)
    {
        const int input = static_cast<int>(trace_field(t, "block"));
        const bool warm = input == 4;
        const double K = trace_field(t, "last_iteration");
        EXPECT_EQ(K, r.iterations) << t;
        const double start = warm ? r.warmup_switch_iteration : 2;
        EXPECT_EQ(trace_field(t, "first_rank_analysis"), start) << t;
        EXPECT_EQ(trace_field(t, "composite_calls"), warm ? r.warmup_switch_iteration - 1 : 0) << t;
        EXPECT_EQ(trace_field(t, "warmup_exit"), warm ? r.warmup_switch_iteration : -1) << t;
        // one projection per iteration: an accepted composite or LOBPCG result, or an exact EVD
        EXPECT_EQ(trace_field(t, "composite_calls") - trace_field(t, "composite_rejected") + trace_field(t, "lobpcg_accepted") + trace_field(t, "full_evd_calls"), K)
            << t;
        // the scheduled analyses every 100 iterations from the start, and one per rejected LOBPCG result
        EXPECT_EQ(trace_field(t, "rank_analyses"), std::floor((K - start) / 100) + 1 + trace_field(t, "lobpcg_rejected")) << t;
        std::size_t lines = 0;
        for (const std::string &rk : ranks)
            if (static_cast<int>(trace_field(rk, "block")) == input)
                lines++;
        EXPECT_EQ(static_cast<double>(lines), trace_field(t, "rank_analyses")) << t;
        lobpcg += trace_field(t, "lobpcg_calls");
        rejected += trace_field(t, "lobpcg_rejected");
        composite_rejected += trace_field(t, "composite_rejected");
    }
    EXPECT_EQ(lobpcg, r.lobpcg_calls);
    EXPECT_EQ(rejected, r.lobpcg_fallbacks);
    EXPECT_EQ(composite_rejected, r.composite_fallbacks);
    for (const std::string &rk : ranks)
    {
        EXPECT_GE(trace_field(rk, "pos"), 0) << rk;
        EXPECT_GE(trace_field(rk, "rank_tol"), 0) << rk;
        const std::string cause = trace_text(rk, "cause");
        EXPECT_TRUE(cause == "scheduled" || cause == "lobpcg_rejected") << rk;
    }
    printf("[ RESULT   ] psd.trace of Tiers_32_33_64_65_128: %zu rank analyses, switch at %d, LOBPCG calls %d (%d rejected)\n", ranks.size(),
           r.warmup_switch_iteration, r.lobpcg_calls, r.lobpcg_fallbacks);
}

TEST(WarmUp, CompositeWarmupConvergesAndReportsTheSwitch)
{
    const regression::RegressionProblem p = regression::generate("Large_LowRankS");
    const cuadmm::SolveResult r = solve_generated("Large_LowRankS", warmup_options(true, 1000));
    EXPECT_TRUE(r.converged) << r.stop_reason;
    EXPECT_NEAR(r.pobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
    EXPECT_NEAR(r.dobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
    EXPECT_GT(r.warmup_switch_iteration, 1) << "the warm-up ended";
    EXPECT_LT(r.warmup_switch_iteration, r.iterations);
    printf("[ RESULT   ] Large_LowRankS with the composite warm-up: %d iterations, switch at %d, composite fallbacks %d, LOBPCG calls %d\n",
           r.iterations, r.warmup_switch_iteration, r.composite_fallbacks, r.lobpcg_calls);
}

TEST(WarmUp, CompositeWarmupIsReproducible)
{
    const cuadmm::SolveResult a = solve_generated("Large_LowRankS", warmup_options(true, 1000));
    const cuadmm::SolveResult b = solve_generated("Large_LowRankS", warmup_options(true, 1000));
    EXPECT_EQ(a.iterations, b.iterations);
    EXPECT_EQ(a.pobj, b.pobj);
    EXPECT_EQ(a.dobj, b.dobj);
    EXPECT_EQ(a.warmup_switch_iteration, b.warmup_switch_iteration);
    EXPECT_EQ(a.composite_fallbacks, b.composite_fallbacks);
}

TEST(WarmUp, TwoBlocksWithThePerBlockThreshold)
{
    const regression::RegressionProblem p = regression::generate("Large_Two");
    const cuadmm::SolveResult r = solve_generated("Large_Two", warmup_options(true, 1080));
    EXPECT_TRUE(r.converged) << r.stop_reason;
    EXPECT_NEAR(r.pobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
    EXPECT_GT(r.warmup_switch_iteration, 1);
}

TEST(WarmUp, WithoutWarmupNoSwitchIsReported)
{
    const regression::RegressionProblem p = regression::generate("Large_LowRankS");
    const cuadmm::SolveResult r = solve_generated("Large_LowRankS", warmup_options(false, 1000));
    EXPECT_TRUE(r.converged) << r.stop_reason;
    EXPECT_NEAR(r.pobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
    EXPECT_EQ(r.warmup_switch_iteration, -1);
    EXPECT_EQ(r.composite_fallbacks, 0);
}
