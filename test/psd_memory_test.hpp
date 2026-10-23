/*

    psd_memory_test.hpp

    Phase 6b, the PSD memory model: the svec <-> dense conversions and the column scalings of the tiers; the SYRK
    product of FullEvd and of Lobpcg against the host projection and against the gemm_compat product; the
    memory plan of the facade (the measured peak within 5% of the plan, the budget check); the chunked batched EVD
    and the single-stream mode reproducing the whole-group / many-stream runs; the compat products within
    rounding of the defaults end to end. Uses the PsdStrategies fixture of psd_strategies_test.hpp (the engine of
    a generated problem, the cone on the arena) and the facade helpers of cone_test.hpp.

*/

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cone/psd/psd_kernels.h"
#include "cuadmm_legacy/kernels.h"

namespace psd_memory
{
std::vector<double> uniform_vector(std::size_t n, unsigned seed)
{
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> v(n);
    for (double &x : v)
        x = u(g);
    return v;
}
} // namespace psd_memory

TEST(PsdKernels, SvecDenseRoundTripsAndBatchedTableAgree)
{
    using namespace psd_memory;
    using namespace cuadmm::cone::psd;
    const std::vector<int> sizes = {5, 12, 1, 7};
    std::vector<BlockRef> refs;
    long long svec_total = 0, mat_total = 0;
    for (int n : sizes)
    {
        refs.push_back(BlockRef{svec_total, n, mat_total, 0});
        svec_total += (long long)n * (n + 1) / 2;
        mat_total += (long long)n * n;
    }
    const std::vector<double> z = psd_memory::uniform_vector(svec_total, 5);
    cuadmm::core::DeviceBuffer<double> dz(svec_total), ds(svec_total), mats(mat_total), M(12 * 12), s1(svec_total);
    CUADMM_CHECK_CUDA(cudaMemcpy(dz.data(), z.data(), sizeof(double) * svec_total, cudaMemcpyHostToDevice));

    // per block: unpack, then pack from the same matrix (the upper triangle): the off-diagonal entries go through
    // z / sqrt 2 * sqrt 2 (one rounding each way), the diagonal is exact
    std::vector<double> round_trip(svec_total, 0.0);
    for (const BlockRef &r : refs)
    {
        svec_to_dense(dz.data() + r.svec_offset, r.n, M.data(), nullptr);
        dense_to_svec(M.data(), r.n, s1.data() + r.svec_offset, nullptr, nullptr);
        // the dense matrix is symmetric with both triangles filled
        std::vector<double> h((std::size_t)r.n * r.n);
        CUADMM_CHECK_CUDA(cudaMemcpy(h.data(), M.data(), sizeof(double) * h.size(), cudaMemcpyDeviceToHost));
        for (int c = 0; c < r.n; c++)
            for (int rr = 0; rr <= c; rr++)
            {
                const double zz = z[r.svec_offset + (std::size_t)c * (c + 1) / 2 + rr];
                const double expect = rr == c ? zz : zz * SQRT2INV; // main's constant (one ulp below the nearest double of 1 / sqrt 2)
                EXPECT_EQ(h[rr + (std::size_t)c * r.n], expect);
                EXPECT_EQ(h[c + (std::size_t)rr * r.n], expect);
            }
    }
    CUADMM_CHECK_CUDA(cudaMemcpy(round_trip.data(), s1.data(), sizeof(double) * svec_total, cudaMemcpyDeviceToHost));
    for (long long i = 0; i < svec_total; i++)
        EXPECT_NEAR(round_trip[i], z[i], 3e-16 * std::abs(z[i]) + 1e-300) << i;

    // the batched table gives the same bits as the per-block kernels
    BlockTable table(refs);
    EXPECT_EQ(table.count(), 4);
    EXPECT_EQ(table.total(), svec_total);
    svec_to_dense_batch(table, dz.data(), mats.data(), nullptr);
    dense_to_svec_batch(table, mats.data(), ds.data(), nullptr);
    std::vector<double> batched(svec_total);
    CUADMM_CHECK_CUDA(cudaMemcpy(batched.data(), ds.data(), sizeof(double) * svec_total, cudaMemcpyDeviceToHost));
    for (long long i = 0; i < svec_total; i++)
        EXPECT_EQ(batched[i], round_trip[i]) << i;

    // packing with the input slice added: s = z + svec(P)
    const BlockRef &r = refs[1];
    svec_to_dense(dz.data() + r.svec_offset, r.n, M.data(), nullptr);
    dense_to_svec(M.data(), r.n, s1.data() + r.svec_offset, dz.data() + r.svec_offset, nullptr);
    std::vector<double> added((std::size_t)r.n * (r.n + 1) / 2);
    CUADMM_CHECK_CUDA(cudaMemcpy(added.data(), s1.data() + r.svec_offset, sizeof(double) * added.size(), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < added.size(); i++)
        EXPECT_EQ(added[i], z[r.svec_offset + i] + round_trip[r.svec_offset + i]);
}

TEST(PsdKernels, ColumnScalingsMatchTheHost)
{
    using namespace psd_memory;
    using namespace cuadmm::cone::psd;
    const int n = 9, count = 3;
    const std::vector<double> Q = psd_memory::uniform_vector((std::size_t)n * n * count, 6), w = psd_memory::uniform_vector((std::size_t)n * count, 7);
    cuadmm::core::DeviceBuffer<double> dQ(Q.size()), dB(Q.size()), dw(w.size());
    CUADMM_CHECK_CUDA(cudaMemcpy(dQ.data(), Q.data(), sizeof(double) * Q.size(), cudaMemcpyHostToDevice));
    CUADMM_CHECK_CUDA(cudaMemcpy(dw.data(), w.data(), sizeof(double) * w.size(), cudaMemcpyHostToDevice));
    std::vector<double> B(Q.size());
    for (int mode : {SCALE_RELU, SCALE_SQRT_RELU, SCALE_SQRT_NEG})
    {
        scale_columns_batch(dQ.data(), dB.data(), n, count, dw.data(), mode, nullptr);
        CUADMM_CHECK_CUDA(cudaMemcpy(B.data(), dB.data(), sizeof(double) * B.size(), cudaMemcpyDeviceToHost));
        for (int k = 0; k < count; k++)
            for (int j = 0; j < n; j++)
            {
                const double wj = w[(std::size_t)k * n + j];
                const double f = mode == SCALE_RELU ? std::max(wj, 0.0) : (mode == SCALE_SQRT_RELU ? std::sqrt(std::max(wj, 0.0)) : std::sqrt(std::max(-wj, 0.0)));
                for (int i = 0; i < n; i++)
                {
                    const std::size_t e = (std::size_t)k * n * n + (std::size_t)j * n + i;
                    EXPECT_EQ(B[e], Q[e] * f) << "mode " << mode;
                }
            }
    }
    // the single-matrix variant on a column range, in place
    cuadmm::core::DeviceBuffer<double> dQ1((std::size_t)n * n);
    CUADMM_CHECK_CUDA(cudaMemcpy(dQ1.data(), Q.data(), sizeof(double) * n * n, cudaMemcpyHostToDevice));
    scale_columns(dQ1.data(), dQ1.data(), n, 4, 3, dw.data(), SCALE_SQRT_RELU, nullptr);
    std::vector<double> Q1((std::size_t)n * n);
    CUADMM_CHECK_CUDA(cudaMemcpy(Q1.data(), dQ1.data(), sizeof(double) * Q1.size(), cudaMemcpyDeviceToHost));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
        {
            const std::size_t e = (std::size_t)j * n + i;
            const double f = (j >= 4 && j < 7) ? std::sqrt(std::max(w[j], 0.0)) : 1.0;
            EXPECT_EQ(Q1[e], Q[e] * f);
        }
}

class PsdMemory : public PsdStrategies
{
};

TEST_F(PsdMemory, FullEvdSyrkMatchesTheHostOnBothEigenSides)
{
    using namespace psd_strategies;
    ASSERT_NO_FATAL_FAILURE(init("Large_LowRankS", true, ProjectionMethod::EIG_FP64));
    const int n = 1100;
    // r_+ < r_-: the SYRK of the positive side; then r_+ > r_-: the negative side with the input added back
    const Spectral few_positive = make_spectral(n, spectrum(n, 300, 400.0, -2.0, -1.0, 31), 301);
    const Spectral few_negative = make_spectral(n, spectrum(n, 800, 900.0, -2.0, -1.0, 32), 302);
    cuadmm::cone::PsdCone &cone = make_cone(cuadmm::cone::PsdConeConfig{});
    EXPECT_FALSE(cone.config().gemm_compat);
    for (const Spectral *sp : {&few_positive, &few_negative})
    {
        ASSERT_NO_FATAL_FAILURE(upload({sp}));
        project(cone, 1, std::numeric_limits<double>::infinity());
        EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::FullEvd);
        EXPECT_LT(relative_error(download(s), 0, svec(sp->P, n)), 1e-10);
    }
    // the memory text names the tiers
    EXPECT_NE(cone.memory_text().find("large tier: 1 block(s), n_max 1100"), std::string::npos) << cone.memory_text();
}

TEST_F(PsdMemory, CompatProductsReproduceTheMilestone1ArithmeticWithinRounding)
{
    using namespace psd_strategies;
    ASSERT_NO_FATAL_FAILURE(init("Large_LowRankS", true, ProjectionMethod::EIG_FP64));
    const int n = 1100;
    const Spectral sp = make_spectral(n, spectrum(n, 300, 400.0, -2.0, -1.0, 33), 303);
    ASSERT_NO_FATAL_FAILURE(upload({&sp}));
    cuadmm::cone::PsdConeConfig cfg;
    cfg.gemm_compat = true;
    cfg.batched_gemm_compat = true;
    cuadmm::cone::PsdCone &cone = make_cone(cfg);
    EXPECT_TRUE(cone.large_tier().config().gemm_compat);
    project(cone, 1, std::numeric_limits<double>::infinity());
    const std::vector<double> compat = download(s);
    EXPECT_LT(relative_error(compat, 0, svec(sp.P, n)), 1e-10);
}

TEST_F(PsdMemory, LobpcgSyrkOnBothModes)
{
    using namespace psd_strategies;
    ASSERT_NO_FATAL_FAILURE(init("Large_LowRankS", true, ProjectionMethod::EIG_FP64));
    const int n = 1100;
    const Spectral positive = make_spectral(n, spectrum(n, 3, 5.0, -2.0, -1.0, 34), 304);
    std::vector<double> lambda = spectrum(n, 3, 5.0, 1.0, 2.0, 35);
    for (int i = 0; i < 3; i++)
        lambda[i] = -lambda[i];
    const Spectral negative = make_spectral(n, lambda, 305);
    cuadmm::cone::PsdCone &cone = make_cone(cuadmm::cone::PsdConeConfig{});
    int expected_calls = 0;
    for (const Spectral *sp : {&positive, &negative})
    {
        ASSERT_NO_FATAL_FAILURE(upload({sp}));
        project(cone, 2, 1.0); // the analysis
        ASSERT_NE(engine.large_mode[0], cuadmm::cone::psd::kModeFull);
        project(cone, 3, 1.0); // LOBPCG, SYRK product (the negative mode adds the input back while packing)
        EXPECT_EQ(cone.last_strategies()[0], LargeStrategy::Lobpcg);
        EXPECT_EQ(engine.lobpcg_calls, ++expected_calls);
        EXPECT_EQ(engine.lobpcg_fallbacks, 0);
        EXPECT_LT(relative_error(download(s), 0, svec(sp->P, n)), 1e-8);
        project(cone, 102, 1.0); // the next analysis resets the mode for the other spectrum
    }
}

// end to end through the facade (the helpers of cone_test.hpp)

TEST(PsdMemoryPlan, PeakWithinFivePercentOfThePlanAndTheBudgetIsChecked)
{
    cuadmm::SolverOptions o = quiet_options();
    o.algorithm.max_iter = 200;
    o.algorithm.tol = 1e-300; // 200 iterations, whatever the convergence
    const cuadmm::SolveResult r = solve_generated("Large_LowRankS", o);
    EXPECT_GT(r.memory_plan_bytes, 0u);
    EXPECT_GE(r.peak_gpu_memory_bytes, r.memory_plan_bytes * 0.99) << "the peak is measured at the end of the solve";
    EXPECT_LE(r.peak_gpu_memory_bytes, r.memory_plan_bytes * 1.05) << "nothing is allocated inside the loop (B7, B19, B20)";
    printf("[ MEMORY   ] Large_LowRankS: plan %.1f MiB, peak %.1f MiB (%.2f%% above the plan)\n", r.memory_plan_bytes / 1048576.0,
           r.peak_gpu_memory_bytes / 1048576.0, 100.0 * ((double)r.peak_gpu_memory_bytes / r.memory_plan_bytes - 1.0));

    // the budget check at setup: a tiny fraction of the free memory cannot hold the workspaces
    cuadmm::SolverOptions tight = quiet_options();
    tight.psd.large_memory_budget = 1e-9;
    const regression::RegressionProblem p = regression::generate("Large_LowRankS");
    const cuadmm::ProblemData d = to_problem_data(p);
    cuadmm::Solver solver(tight);
    EXPECT_THROW(solver.setup(d), std::runtime_error);
}

TEST(PsdMemoryPlan, ChunkedBatchedEvdAndSingleStreamReproduceTheDefaults)
{
    // six 40 x 40 blocks and three 50 x 50 blocks (two batched medium groups), a small block and a free block
    const regression::ProblemSpec spec = {
        {{'s', 40, 20, 0, false}, {'s', 50, 25, 0, false}, {'s', 40, 20, 0, false}, {'u', 30, 0, 0, false},
         {'s', 40, 20, 0, false}, {'s', 50, 25, 0, false}, {'s', 40, 20, 0, false}, {'s', 12, 6, 0, false},
         {'s', 40, 20, 0, false}, {'s', 50, 25, 0, false}, {'s', 40, 20, 0, false}},
        0, 12, 3002};
    const regression::RegressionProblem p = regression::generate(spec);
    const cuadmm::ProblemData d = to_problem_data(p);
    auto run = [&d](cuadmm::SolverOptions o)
    {
        cuadmm::Solver solver(o);
        solver.setup(d);
        return solver.solve();
    };
    cuadmm::SolverOptions base = quiet_options();
    base.algorithm.max_iter = 300;
    base.algorithm.tol = 1e-300;
    const cuadmm::SolveResult reference = run(base);
    EXPECT_EQ(reference.iterations, 300);

    cuadmm::SolverOptions chunked = base;
    chunked.psd.medium_max_batch = 2; // the batched groups in chunks of 2 (3 chunks of the six 40s, 2 of the three 50s)
    const cuadmm::SolveResult c = run(chunked);
    EXPECT_EQ(c.iterations, reference.iterations);
    // cusolverDnXsyevBatched on a chunk is within rounding of the whole group's call (DECISIONS 52), not bit for bit
    EXPECT_NEAR(c.pobj, reference.pobj, 1e-9 * (1.0 + std::abs(reference.pobj)));
    EXPECT_NEAR(c.dobj, reference.dobj, 1e-9 * (1.0 + std::abs(reference.dobj)));
    EXPECT_NE(c.pobj, reference.pobj) << "(informational) the chunked call rounds differently";

    cuadmm::SolverOptions streamed = base;
    streamed.psd.medium_batch_min = 0; // every medium block through the streamed EVD
    streamed.psd.medium_streams = 1;   // on one stream
    const cuadmm::SolveResult s1 = run(streamed);
    cuadmm::SolverOptions many = streamed;
    many.psd.medium_streams = 7;
    const cuadmm::SolveResult s7 = run(many);
    EXPECT_EQ(s1.iterations, s7.iterations);
    EXPECT_EQ(s1.pobj, s7.pobj) << "the stream assignment does not change the bits";
    EXPECT_EQ(s1.dobj, s7.dobj);
    // the streamed and the batched EVD agree within rounding (plan Section 10.5: both accurate to machine precision)
    EXPECT_NEAR(s1.pobj, reference.pobj, 1e-8 * (1.0 + std::abs(reference.pobj)));
}

TEST(PsdMemoryPlan, CompatProductsAgreeWithTheDefaultsWithinRounding)
{
    const regression::RegressionProblem p = regression::generate("Large_LowRankS");
    cuadmm::SolverOptions o = quiet_options();
    o.penalty.sigma0 = 1e-4;
    o.algorithm.max_iter = 5000;
    const cuadmm::SolveResult a = solve_generated("Large_LowRankS", o);
    cuadmm::SolverOptions compat = o;
    compat.psd.large_final_product = cuadmm::FinalProductKind::GemmCompat;
    compat.psd.batched_final_product = cuadmm::BatchedProductKind::GemmCompat;
    const cuadmm::SolveResult b = solve_generated("Large_LowRankS", compat);
    EXPECT_TRUE(a.converged) << a.stop_reason;
    EXPECT_TRUE(b.converged) << b.stop_reason;
    EXPECT_NEAR(a.pobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
    EXPECT_NEAR(b.pobj, p.p_star, 1e-4 * (1.0 + std::abs(p.p_star)));
    EXPECT_NEAR(a.pobj, b.pobj, 1e-6 * (1.0 + std::abs(b.pobj)));
    EXPECT_LE(std::abs(a.iterations - b.iterations), 2) << "the products differ by rounding only";
    printf("[ PRODUCTS ] Large_LowRankS: syrk/scaled %d iterations, gemm_compat %d iterations, |pobj diff| = %.2e\n", a.iterations, b.iterations,
           std::abs(a.pobj - b.pobj));
}
