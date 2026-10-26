/*

    io_simplify_test.hpp  (CPU only)

    The exact simplifications (1x1 and diagonal-only PSD blocks -> l), their report, the mapping of solutions back
    to the file's blocks (expand_to_original_blocks), the TXT problem writer and the TXT solution writer.

*/

#include <gtest/gtest.h>

#include "io/readers.h"
#include "io_test_support.hpp"

namespace
{
using namespace io_test;

// blocks: s 1 | s 3 with off-diagonal data | s 2 diagonal-only | l 2 ; m = 2
ProblemData simplifiable()
{
    ProblemData p;
    p.cones = {{ConeKind::Psd, 1, 0.0}, {ConeKind::Psd, 3, 0.0}, {ConeKind::Psd, 2, 0.0}, {ConeKind::Nonnegative, 2, 0.0}};
    // offsets: 0 | 1..6 | 7..9 | 10..11 ; vec_len 12
    p.At.rows = 12;
    p.At.cols = 2;
    p.At.colptr = {0, 3, 6};
    p.At.rowind = {0, 2, 7, 1, 9, 11}; // (block1 (0,1) off-diagonal = index 2), block2 diagonal (0,0)=7 and (1,1)=9
    p.At.val = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    p.b.size = 2;
    p.b.index = {0, 1};
    p.b.val = {1.0, 2.0};
    p.C.size = 12;
    p.C.index = {0, 6, 9, 10};
    p.C.val = {1.0, 1.0, 1.0, 1.0};
    p.source.format = "test";
    cuadmm::io::detail::set_identity_original_blocks(p);
    cuadmm::io::detail::set_dimacs_kinds_from_original(p);
    p.validate();
    return p;
}
} // namespace

TEST(IoSimplify, DiagonalOnlyAndOneByOneBlocksBecomeNonnegative)
{
    ProblemData p = simplifiable();
    cuadmm::io::detail::apply_simplifications(p, true);
    ASSERT_NO_THROW(p.validate());
    ASSERT_EQ(p.cones.size(), 4u);
    EXPECT_EQ(p.cones[0].kind, ConeKind::Nonnegative);
    EXPECT_EQ(p.cones[1].kind, ConeKind::Psd);
    EXPECT_EQ(p.cones[2].kind, ConeKind::Nonnegative);
    EXPECT_EQ(p.cones[2].size, 2);
    EXPECT_EQ(p.cones[3].kind, ConeKind::Nonnegative);
    EXPECT_EQ(p.vec_len(), 1 + 6 + 2 + 2);
    // indices remapped: 7 -> 7 (diag 0), 9 -> 8 (diag 1), 10 -> 9, 11 -> 10
    EXPECT_EQ(p.At.rowind, (std::vector<std::int64_t>{0, 2, 7, 1, 8, 10}));
    EXPECT_EQ(p.At.val, (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
    EXPECT_EQ(p.C.index, (std::vector<std::int64_t>{0, 6, 8, 9}));
    EXPECT_EQ(p.source.dimacs_kinds, (std::vector<char>{'s', 's', 's', 'l'})); // unchanged: the DIMACS kinds of the file
    EXPECT_EQ(p.source.original_block_of, (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(p.source.original_block_kinds, (std::vector<char>{'s', 's', 's', 'l'}));
    EXPECT_EQ(p.source.report.size(), 2u);
    EXPECT_NE(p.source.report[0].find("1x1"), std::string::npos);
    EXPECT_NE(p.source.report[1].find("diagonal only"), std::string::npos);
    // off: nothing changes
    ProblemData q = simplifiable();
    cuadmm::io::detail::apply_simplifications(q, false);
    EXPECT_EQ(q.vec_len(), 12);
    EXPECT_EQ(q.cones[0].kind, ConeKind::Psd);
    EXPECT_TRUE(q.source.report.empty());
}

TEST(IoSimplify, SolutionsMapBackToTheOriginalBlocks)
{
    ProblemData p = simplifiable();
    cuadmm::io::detail::apply_simplifications(p, true);
    std::vector<double> X(static_cast<std::size_t>(p.vec_len()));
    for (std::size_t i = 0; i < X.size(); i++)
        X[i] = static_cast<double>(i) + 1.0;
    const std::vector<cuadmm::io::detail::OriginalBlockValues> blocks = cuadmm::io::detail::expand_to_original_blocks(p, X);
    ASSERT_EQ(blocks.size(), 4u);
    EXPECT_EQ(blocks[0].kind, 's');
    EXPECT_EQ(blocks[0].values, (std::vector<double>{1.0}));
    EXPECT_EQ(blocks[1].size, 3);
    ASSERT_EQ(blocks[1].values.size(), 9u);
    // svec entries 2..7 -> (0,0)=2, (0,1)=3/sqrt2, (1,1)=4, (0,2)=5/sqrt2, (1,2)=6/sqrt2, (2,2)=7 ; column-major full matrix
    EXPECT_EQ(blocks[1].values[0], 2.0);
    EXPECT_EQ(blocks[1].values[3], 3.0 * cuadmm::io::detail::kHalfSqrt2); // (0,1)
    EXPECT_EQ(blocks[1].values[1], 3.0 * cuadmm::io::detail::kHalfSqrt2); // (1,0)
    EXPECT_EQ(blocks[1].values[4], 4.0);
    EXPECT_EQ(blocks[1].values[8], 7.0);
    EXPECT_EQ(blocks[2].kind, 's'); // simplified block: diagonal matrix
    EXPECT_EQ(blocks[2].values, (std::vector<double>{8.0, 0.0, 0.0, 9.0}));
    EXPECT_EQ(blocks[3].kind, 'l');
    EXPECT_EQ(blocks[3].values, (std::vector<double>{10.0, 11.0}));
}

TEST(IoSimplify, TxtWriterRoundTrip)
{
    const ProblemData p = small_problem();
    const std::string dir = temp_dir("txt_writer/small");
    io::write_problem_txt(dir, p);
    const ProblemData q = io::read_txt(dir);
    expect_bitwise(p, q, "TXT round trip");
    EXPECT_EQ(q.source.dimacs_kinds, p.source.dimacs_kinds);
    // the legacy TXT reader never simplifies and ignores io.sense = min (TXT is a minimization), rejects max
    IoOptions o;
    o.simplify = true;
    EXPECT_NO_THROW(io::read_problem(dir, o));
    o.sense = Sense::Maximize;
    EXPECT_THROW(io::read_problem(dir, o), std::invalid_argument);
}

TEST(IoSimplify, TxtSolutionWriterAndFormatDispatch)
{
    const ProblemData p = small_problem();
    const SolveResult r = fake_result(p);
    const std::string dir = temp_dir("txt_writer/solution");
    ProblemData t = p;
    t.source.format = "txt";
    io::write_solution(dir, t, r);
    const WarmStart w = io::read_warm_start(dir, p);
    EXPECT_EQ(w.X, r.best.X);
    EXPECT_EQ(w.y, r.best.y);
    EXPECT_EQ(w.S, r.best.S);
    EXPECT_TRUE(std::filesystem::exists(dir + "/certificate.txt"));
    ProblemData s = p;
    s.source.format = "sdpa";
    io::write_solution(temp_dir("txt_writer/forced"), s, r, "txt"); // io.save_format forces the TXT convention
    EXPECT_TRUE(std::filesystem::exists(temp_dir("txt_writer/forced") + "/X.txt"));
    // --save-solution <dir> with an SDPA input names a directory that may not exist yet: the writer creates it, as
    // the TXT writer does (2.0.0 failed with "could not open .../solution.ini-s for writing")
    const std::string fresh = temp_dir("txt_writer") + "/fresh_sdpa_dir";
    std::filesystem::remove_all(fresh);
    ASSERT_FALSE(std::filesystem::exists(fresh));
    EXPECT_NO_THROW(io::write_solution(fresh, s, r, "sdpa"));
    EXPECT_TRUE(std::filesystem::exists(fresh + "/solution.ini-s"));
    EXPECT_THROW(io::write_solution(dir, p, r, "nonsense"), std::invalid_argument);
    const std::vector<std::string> formats = io::available_formats();
    EXPECT_NE(std::find(formats.begin(), formats.end(), "txt"), formats.end());
    EXPECT_NE(std::find(formats.begin(), formats.end(), "sdpa"), formats.end());
    EXPECT_THROW(io::read_problem("io_test_tmp/does_not_exist.xyz", IoOptions()), std::runtime_error);
    write_file(temp_dir("detect") + "/unknown.xyz", "hello");
    EXPECT_THROW(io::read_problem(temp_dir("detect") + "/unknown.xyz", IoOptions()), std::invalid_argument);
    write_file(temp_dir("detect") + "/problem.cbf", "VER\n3\n");
    try
    {
        io::read_problem(temp_dir("detect") + "/problem.cbf", IoOptions());
        ADD_FAILURE() << "a .cbf file was accepted";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_NE(std::string(e.what()).find("CBF"), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find("Phase 9"), std::string::npos) << e.what();
    }
    // the TXT convention names the layout of X.txt / S.txt (the solver's cones) in solution_blocks.txt
    std::ifstream blocks(dir + "/solution_blocks.txt");
    std::string line;
    ASSERT_TRUE(std::getline(blocks, line));
    EXPECT_EQ(line, "s 2");
}

TEST(IoSimplify, SdpaSolutionWriterLayout)
{
    const ProblemData p = small_problem();
    const SolveResult r = fake_result(p);
    const std::string file = temp_dir("sdpa_solution") + "/sol.ini-s";
    io::write_solution(file, p, r, "sdpa");
    std::ifstream f(file);
    std::string line;
    ASSERT_TRUE(std::getline(f, line));
    EXPECT_EQ(line, "0.75 -0.25 -1.25"); // x = -y
    std::vector<std::string> lines;
    while (std::getline(f, line))
        lines.push_back(line);
    // X_SDPA = S: block 1 upper triangle (3 entries), block 2 (2 diagonal), block 3 (1); then Y_SDPA = X
    ASSERT_GE(lines.size(), 12u);
    EXPECT_EQ(lines[0], "1 1 1 1 1");                   // S[0] = 1
    EXPECT_EQ(lines[1], "1 1 1 2 0.6187184335382291");  // S[1] = 0.875 -> 0.875/sqrt2
    EXPECT_EQ(lines[6], "2 1 1 1 0.25");                // X[0]
    EXPECT_TRUE(std::filesystem::exists(temp_dir("sdpa_solution") + "/sol.json"));
}
