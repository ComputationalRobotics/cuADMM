/*

    workspace_arena_test.hpp

    The layout of WorkspaceArena (plan Section 14, Phase 6b) on the host: private regions in request order with
    their alignment, the shared area after them sized to the largest shared request, the plan text, and the
    argument and state checks. The allocation itself is tested on the GPU (core_runtime_test.hpp).

*/

#include <gtest/gtest.h>

#include <stdexcept>

#include "core/workspace_arena.h"

TEST(WorkspaceArena, LayoutAlignsPrivateRegionsAndSharesTheSharedArea)
{
    cuadmm::core::WorkspaceArena a;
    const int h0 = a.request("first", 100);           // [0, 100)
    const int h1 = a.request("second", 300);          // aligned to 256: [256, 556)
    const int h2 = a.request_shared("evd", 1000);     // the shared area
    const int h3 = a.request_shared("lobpcg", 500);   // the same area
    const int h4 = a.request("third", 10, 64);        // aligned to 64 after 556: [576, 586)
    EXPECT_EQ(a.offset(h0), 0u);
    EXPECT_EQ(a.offset(h1), 256u);
    EXPECT_EQ(a.offset(h4), 576u);
    EXPECT_EQ(a.offset(h2), 768u) << "the shared area starts after the private regions, aligned to 256";
    EXPECT_EQ(a.offset(h3), 768u);
    EXPECT_EQ(a.bytes(h2), 1000u);
    EXPECT_EQ(a.bytes(h3), 500u);
    EXPECT_EQ(a.shared_bytes(), 1000u);
    EXPECT_EQ(a.planned_bytes(), 1768u);
    EXPECT_FALSE(a.committed());
    EXPECT_NE(a.plan_text().find("evd: 1000 bytes at 768 (shared area)"), std::string::npos) << a.plan_text();
    EXPECT_NE(a.plan_text().find("total: 1768 bytes"), std::string::npos) << a.plan_text();
    EXPECT_EQ(a.entries().size(), 5u);
}

TEST(WorkspaceArena, EmptyAndPrivateOnlyLayouts)
{
    cuadmm::core::WorkspaceArena a;
    EXPECT_EQ(a.planned_bytes(), 0u);
    EXPECT_EQ(a.shared_bytes(), 0u);
    const int h = a.request("only", 1);
    EXPECT_EQ(a.planned_bytes(), 1u) << "no shared area: the total ends with the last private region";
    EXPECT_EQ(a.offset(h), 0u);
    cuadmm::core::WorkspaceArena b;
    b.request_shared("x", 0);
    EXPECT_EQ(b.planned_bytes(), 0u) << "an empty shared request adds nothing";
}

TEST(WorkspaceArena, ArgumentAndStateChecks)
{
    cuadmm::core::WorkspaceArena a;
    EXPECT_THROW(a.request("bad", 10, 100), std::invalid_argument) << "alignment must be a power of two";
    EXPECT_THROW(a.request("bad", 10, 0), std::invalid_argument);
    const int h = a.request("ok", 10);
    EXPECT_THROW(a.data(h), std::logic_error) << "no pointer before commit";
    EXPECT_THROW(a.bytes(h + 1), std::out_of_range);
    EXPECT_THROW(a.offset(-1), std::out_of_range);
}
