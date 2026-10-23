/*

    core_runtime_test.hpp

    WorkspaceArena on the device (commit, spans, release, recommit) and StreamPool (the roles, the non-blocking
    medium streams with their own handles, cross-stream waits, single_stream), Phase 6b.

*/

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "core/check.h"
#include "core/stream_pool.h"
#include "core/workspace_arena.h"

TEST(CoreRuntime, ArenaCommitsOnceAndHandsOutDistinctSpans)
{
    cuadmm::core::WorkspaceArena a;
    const int h0 = a.request("doubles", 1000 * sizeof(double));
    const int h1 = a.request("ints", 10 * sizeof(int), 64);
    const int h2 = a.request_shared("shared-a", 4096);
    const int h3 = a.request_shared("shared-b", 2048);
    a.commit();
    EXPECT_TRUE(a.committed());
    EXPECT_THROW(a.request("late", 1), std::logic_error);
    EXPECT_THROW(a.commit(), std::logic_error);
    cuadmm::core::DeviceSpan<double> d = a.span<double>(h0);
    cuadmm::core::DeviceSpan<int> i = a.span<int>(h1);
    EXPECT_EQ(d.size, 1000u);
    EXPECT_EQ(i.size, 10u);
    EXPECT_EQ(a.data(h2), a.data(h3)) << "the shared area";
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.data(h1)) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.data(h2)) % 256, 0u);
    EXPECT_GE(reinterpret_cast<const unsigned char *>(a.data(h1)), reinterpret_cast<const unsigned char *>(a.data(h0)) + 1000 * sizeof(double));

    // the regions are usable device memory
    CUADMM_CHECK_CUDA(cudaMemset(d.data, 0, sizeof(double) * d.size));
    std::vector<double> h(d.size, 1.5);
    CUADMM_CHECK_CUDA(cudaMemcpy(d.data, h.data(), sizeof(double) * d.size, cudaMemcpyHostToDevice));
    std::vector<double> back(d.size, 0.0);
    CUADMM_CHECK_CUDA(cudaMemcpy(back.data(), d.data, sizeof(double) * d.size, cudaMemcpyDeviceToHost));
    EXPECT_EQ(back[0], 1.5);
    EXPECT_EQ(back[999], 1.5);

    // release and recommit with one more region
    a.release();
    EXPECT_FALSE(a.committed());
    EXPECT_THROW(a.data(h0), std::logic_error);
    const int h4 = a.request("more", 128);
    a.commit();
    EXPECT_NE(a.data(h4), nullptr);
    EXPECT_EQ(a.planned_bytes(), a.offset(h2) + 4096);
}

TEST(CoreRuntime, StreamPoolRolesHandlesAndWaits)
{
    cuadmm::core::StreamPoolConfig c;
    c.medium_streams = 3;
    cuadmm::core::StreamPool pool(c);
    EXPECT_EQ(pool.medium_count(), 3);
    // Milestone 2: main, small, large and copy are the legacy default stream
    EXPECT_EQ(pool.stream(cuadmm::core::StreamRole::Main), nullptr);
    EXPECT_EQ(pool.stream(cuadmm::core::StreamRole::Small), nullptr);
    EXPECT_EQ(pool.stream(cuadmm::core::StreamRole::Large), nullptr);
    EXPECT_EQ(pool.stream(cuadmm::core::StreamRole::Copy), nullptr);
    std::set<cudaStream_t> medium;
    for (int i = 0; i < 3; i++)
    {
        cudaStream_t s = pool.stream(cuadmm::core::StreamRole::Medium, i);
        ASSERT_NE(s, nullptr);
        unsigned flags = 0;
        CUADMM_CHECK_CUDA(cudaStreamGetFlags(s, &flags));
        EXPECT_EQ(flags & cudaStreamNonBlocking, (unsigned)cudaStreamNonBlocking);
        medium.insert(s);
        cuadmm::core::ExecContext ctx = pool.context(cuadmm::core::StreamRole::Medium, i);
        cudaStream_t bound = nullptr;
        CUADMM_CHECK_CUBLAS(cublasGetStream(ctx.cublas, &bound));
        EXPECT_EQ(bound, s) << "the cuBLAS handle is bound to its stream";
        CUADMM_CHECK_CUSOLVER(cusolverDnGetStream(ctx.cusolver, &bound));
        EXPECT_EQ(bound, s) << "the cuSOLVER handle is bound to its stream";
        EXPECT_EQ(ctx.cusparse, nullptr);
    }
    EXPECT_EQ(medium.size(), 3u) << "distinct medium streams";
    EXPECT_EQ(pool.stream(cuadmm::core::StreamRole::Medium, 4), pool.stream(cuadmm::core::StreamRole::Medium, 1)) << "modulo the count";
    EXPECT_THROW(pool.stream(cuadmm::core::StreamRole::Medium, -1), std::out_of_range);
    cublasMath_t mode;
    CUADMM_CHECK_CUBLAS(cublasGetMathMode(pool.context(cuadmm::core::StreamRole::Main).cublas, &mode));
    EXPECT_EQ(mode, CUBLAS_DEFAULT_MATH) << "B1: the math mode is explicit";

    // a cross-stream dependency: work on medium 0, then the main stream reads its result after wait()
    cuadmm::core::DeviceBuffer<int> buf(1 << 20);
    CUADMM_CHECK_CUDA(cudaMemsetAsync(buf.data(), 0, sizeof(int) * buf.size(), pool.stream(cuadmm::core::StreamRole::Medium, 0)));
    CUADMM_CHECK_CUDA(cudaMemsetAsync(buf.data(), 0x11, sizeof(int) * buf.size(), pool.stream(cuadmm::core::StreamRole::Medium, 0)));
    pool.wait(cuadmm::core::StreamRole::Main, 0, cuadmm::core::StreamRole::Medium, 0);
    std::vector<int> h(buf.size());
    CUADMM_CHECK_CUDA(cudaMemcpyAsync(h.data(), buf.data(), sizeof(int) * h.size(), cudaMemcpyDeviceToHost, pool.stream(cuadmm::core::StreamRole::Main)));
    pool.synchronize(cuadmm::core::StreamRole::Main);
    EXPECT_EQ(h.front(), 0x11111111);
    EXPECT_EQ(h.back(), 0x11111111);
    pool.synchronize_all();

    cuadmm::core::StreamPoolConfig bad;
    bad.medium_streams = 0;
    EXPECT_THROW(cuadmm::core::StreamPool p(bad), std::invalid_argument) << "B17";
}

TEST(CoreRuntime, StreamPoolSingleStreamMode)
{
    cuadmm::core::StreamPoolConfig c;
    c.medium_streams = 4;
    c.single_stream = true;
    cuadmm::core::StreamPool pool(c);
    EXPECT_TRUE(pool.single_stream());
    EXPECT_EQ(pool.medium_count(), 0);
    for (int i = 0; i < 6; i++)
        EXPECT_EQ(pool.stream(cuadmm::core::StreamRole::Medium, i), pool.stream(cuadmm::core::StreamRole::Main));
    EXPECT_EQ(pool.context(cuadmm::core::StreamRole::Medium, 2).cublas, pool.context(cuadmm::core::StreamRole::Main).cublas);
    pool.wait(cuadmm::core::StreamRole::Main, 0, cuadmm::core::StreamRole::Medium, 1); // the same queue: a no-op
    pool.synchronize_all();
}
