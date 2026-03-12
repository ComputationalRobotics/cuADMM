#include <gtest/gtest.h>
#include <deque>
#include <sstream>

// Adjust include path to where you keep the class:
#include "cuadmm/monitors.h"

TEST(SlidingDiffQueue, PushAndDiffBasic)
{
    SlidingDiffQueue<int> q(3);
    q.push(1);
    q.push(2);
    q.push(3);
    q.push(4); // evicts 1

    std::deque<int> expected_q1{2, 3, 4};
    std::deque<int> expected_dq{1, 1};

    EXPECT_EQ(q.q1, expected_q1);
    EXPECT_EQ(q.dq, expected_dq);

    EXPECT_EQ(q.get_size(), 3u);
    EXPECT_EQ(q.get_capacity(), 3u);
    EXPECT_TRUE(q.if_full());
    EXPECT_FALSE(q.if_empty());
}

TEST(SlidingDiffQueue, Predicates)
{
    SlidingDiffQueue<int> q(3);
    q.push(2);
    q.push(3);
    q.push(4);

    // data = [2,3,4], diffs = [1,1]
    EXPECT_TRUE(q.data_all_greater(1));
    EXPECT_FALSE(q.data_all_smaller(0));

    EXPECT_TRUE(q.diff_all_greater(0));
    EXPECT_FALSE(q.diff_all_smaller(0));
}

TEST(SlidingDiffQueue, ResetClearsBoth)
{
    SlidingDiffQueue<int> q(3);
    q.push(5);
    q.push(6);
    q.push(7);

    q.reset();
    EXPECT_TRUE(q.q1.empty());
    EXPECT_TRUE(q.dq.empty());
    EXPECT_EQ(q.get_size(), 0u);
    EXPECT_TRUE(q.if_empty());
    EXPECT_FALSE(q.if_full());
}

TEST(SlidingDiffQueue, CapacityOneKeepsNoDiffs)
{
    SlidingDiffQueue<int> q(1);
    q.push(5);
    // data=[5], diffs=[]
    EXPECT_TRUE(q.dq.empty());

    q.push(6);
    // transient diff created, then evicted with front element; end state diffs=[]
    EXPECT_TRUE(q.dq.empty());

    std::deque<int> expected_q1{6};
    EXPECT_EQ(q.q1, expected_q1);
}

TEST(SlidingDiffQueue, PrintHelpers)
{
    SlidingDiffQueue<int> q(3);
    q.push(2);
    q.push(3);
    q.push(4);

    std::ostringstream os1, os2, os3;
    q.print_q1(os1, ",");   // expect "2,3,4"
    q.print_dq(os2, " ");   // expect "1 1"

    EXPECT_EQ(os1.str(), "2,3,4");
    EXPECT_EQ(os2.str(), "1 1");

    q.reset();
    q.print_q1(os3);
    EXPECT_EQ(os3.str(), "(empty)");
}
