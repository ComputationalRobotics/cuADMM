/*

    problem_loader_test.hpp

    CPU tests of Problem::from_txt: sizes come from blk.txt / con_num.txt, indices outside the problem are
    rejected (they would be read out of bounds on the GPU), and warm-start vectors must have the right lengths.

*/

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "cuadmm/problem.h"

namespace problem_loader_test
{

// Writes a small problem (blocks 's 2' and 'l 1', vec_len 4, 2 constraints) to dir, with optional overrides.
inline std::string write_problem(
    const std::string &name,
    const std::string &blk = "s 2\nl 1\n",
    const std::string &At = "0 0 1.0\n2 0 1.0\n3 1 1.0\n",
    const std::string &b = "0 0 2.0\n1 0 1.0\n",
    const std::string &C = "0 0 1.0\n3 0 1.0\n")
{
    const std::string dir = "problem_loader_data/" + name;
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/blk.txt") << blk;
    std::ofstream(dir + "/At.txt") << At;
    std::ofstream(dir + "/b.txt") << b;
    std::ofstream(dir + "/C.txt") << C;
    std::ofstream(dir + "/con_num.txt") << "2\n";
    return dir;
}

} // namespace problem_loader_test

TEST(ProblemLoader, SizesFromBlkAndConNum)
{
    std::string dir = problem_loader_test::write_problem("valid");
    Problem p;
    ASSERT_NO_THROW(p.from_txt(dir));
    EXPECT_EQ(p.vec_len, 4);
    EXPECT_EQ(p.con_num, 2);
    EXPECT_EQ(p.At_nnz, 3);
    EXPECT_TRUE(p.X_vals.empty());
}

TEST(ProblemLoader, RejectsOutOfRangeIndices)
{
    using problem_loader_test::write_problem;
    {
        std::string dir = write_problem("svec_index", "s 2\nl 1\n", "0 0 1.0\n4 1 1.0\n"); // svec index 4 >= vec_len
        Problem p;
        EXPECT_THROW(p.from_txt(dir), std::invalid_argument);
    }
    {
        std::string dir = write_problem("C_index", "s 2\nl 1\n", "0 0 1.0\n3 1 1.0\n", "0 0 2.0\n", "4 0 1.0\n");
        Problem p;
        EXPECT_THROW(p.from_txt(dir), std::invalid_argument);
    }
    {
        std::string dir = write_problem("b_index", "s 2\nl 1\n", "0 0 1.0\n3 1 1.0\n", "2 0 2.0\n");
        Problem p;
        EXPECT_THROW(p.from_txt(dir), std::invalid_argument);
    }
    {
        std::string dir = write_problem("con_index", "s 2\nl 1\n", "0 0 1.0\n3 2 1.0\n"); // constraint 2 >= con_num
        Problem p;
        EXPECT_THROW(p.from_txt(dir), std::invalid_argument);
    }
    {
        std::string dir = write_problem("block_type", "s 2\nq 1\n");
        Problem p;
        EXPECT_THROW(p.from_txt(dir), std::invalid_argument);
    }
}

// The mater-6 situation: a block written as a bare size is read as a PSD block, making vec_len larger than
// the largest index; this is legal (unused trailing entries) and only warns.
TEST(ProblemLoader, UnusedTrailingEntriesOnlyWarn)
{
    std::string dir = problem_loader_test::write_problem("trailing", "s 2\nl 2\n");
    Problem p;
    ASSERT_NO_THROW(p.from_txt(dir));
    EXPECT_EQ(p.vec_len, 5);
}

TEST(ProblemLoader, WarmStartLengthsAreChecked)
{
    std::string dir = problem_loader_test::write_problem("warm");
    std::ofstream(dir + "/X.txt") << "1\n0\n1\n0.5\n";
    std::ofstream(dir + "/y.txt") << "0.1\n0.2\n";
    std::ofstream(dir + "/S.txt") << "0\n0\n0\n0\n";
    {
        Problem p;
        ASSERT_NO_THROW(p.from_txt(dir, true));
        EXPECT_EQ(p.X_vals.size(), 4u);
        EXPECT_EQ(p.y_vals.size(), 2u);
        EXPECT_EQ(p.S_vals.size(), 4u);
    }
    std::ofstream(dir + "/S.txt") << "0\n0\n0\n"; // too short: init would copy vec_len entries from it
    {
        Problem p;
        EXPECT_THROW(p.from_txt(dir, true), std::invalid_argument);
    }
}
