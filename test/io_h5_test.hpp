/*

    io_h5_test.hpp  (CPU only)

    The native HDF5 format: problem round trips (plain and compressed, with the metadata), the solution group,
    the warm start group, malformed files, detection, and the message of a build without HDF5.

*/

#include <gtest/gtest.h>

#include <stdexcept>

#include "io/readers.h"
#include "io_test_support.hpp"

namespace
{
using namespace io_test;
}

TEST(IoH5, ProblemRoundTripIsIdentical)
{
#ifndef CUADMM_HAVE_HDF5
    const std::string path = temp_dir("h5") + "/nohdf5.h5";
    EXPECT_THROW(io::write_problem(path, small_problem()), std::invalid_argument);
    EXPECT_FALSE(io::format_available("h5"));
    GTEST_SKIP() << "HDF5 not compiled in";
#else
    EXPECT_TRUE(io::format_available("h5"));
    ProblemData p = small_problem();
    p.sense = Sense::Maximize;
    p.objective_offset = -2.5;
    p.source.path = "io_test_tmp/h5/source_of_small.txt";
    write_file(p.source.path, "the converted file");
    for (int compression : {0, 4})
    {
        const std::string path = temp_dir("h5") + "/small_" + std::to_string(compression) + ".h5";
        io::write_problem(path, p, compression);
        IoOptions o;
        o.simplify = false;
        const ProblemData q = io::read_problem(path, o);
        expect_bitwise(p, q, "h5 round trip, compression " + std::to_string(compression));
        EXPECT_EQ(q.source.format, "h5");
        EXPECT_EQ(q.source.path, path);
        EXPECT_EQ(q.source.dimacs_kinds, p.source.dimacs_kinds);
        EXPECT_EQ(q.source.original_block_of, p.source.original_block_of);
        EXPECT_EQ(q.source.original_block_sizes, p.source.original_block_sizes);
        EXPECT_EQ(q.source.original_block_kinds, p.source.original_block_kinds);
        ASSERT_GE(q.source.report.size(), 2u);
        EXPECT_NE(q.source.report[0].find("native HDF5"), std::string::npos);
        EXPECT_NE(q.source.report[0].find("converted from test"), std::string::npos);
        EXPECT_NE(q.source.report[1].find("hand-made problem"), std::string::npos); // the stored conversion report
        // a second round trip through the same file keeps everything
        const std::string path2 = temp_dir("h5") + "/small_again.h5";
        io::write_problem(path2, q, 0);
        expect_bitwise(q, io::read_problem(path2, o), "h5 second round trip");
    }
    // the diagonal-only block of a stored problem is simplified unless io.simplify is off
    ProblemData d = small_problem();
    d.At.rowind = {0, 2, 3, 3, 5, 2, 4}; // the s 2 block now touches (0,0) and (1,1) only (columns stay sorted)
    d.C.index = {0, 3, 5};
    d.C.val = {1.0, 0.25, -1.0};
    d.validate();
    const std::string path = temp_dir("h5") + "/diag.h5";
    io::write_problem(path, d);
    EXPECT_EQ(io::read_problem(path, IoOptions()).cones[0].kind, ConeKind::Nonnegative);
    IoOptions raw;
    raw.simplify = false;
    EXPECT_EQ(io::read_problem(path, raw).cones[0].kind, ConeKind::Psd);
    EXPECT_THROW(io::write_problem(path, d, 10), std::invalid_argument);
#endif
}

TEST(IoH5, SolutionAndWarmStartGroups)
{
#ifndef CUADMM_HAVE_HDF5
    GTEST_SKIP() << "HDF5 not compiled in";
#else
    ProblemData p = small_problem();
    const std::string path = temp_dir("h5") + "/with_solution.h5";
    io::write_problem(path, p);
    p = io::read_problem(path, IoOptions());
    const SolveResult r = fake_result(p);
    io::write_solution(path, p, r); // the input's convention: the group /solution of the file
    std::vector<double> X, y, S;
    double pobj = 0, dobj = 0;
    cuadmm::io::detail::read_h5_solution(path, X, y, S, &pobj, &dobj);
    EXPECT_EQ(X, r.best.X);
    EXPECT_EQ(y, r.best.y);
    EXPECT_EQ(S, r.best.S);
    EXPECT_EQ(pobj, r.pobj);
    EXPECT_EQ(dobj, r.dobj);
    io::write_solution(path, p, r); // replaced, not duplicated
    cuadmm::io::detail::read_h5_solution(path, X, y, S, nullptr, nullptr);
    EXPECT_EQ(X, r.best.X);
    // a directory target gets solution.h5
    const std::string dir = temp_dir("h5/soldir");
    io::write_solution(dir, p, r);
    EXPECT_TRUE(std::filesystem::exists(dir + "/solution.h5"));
    // the problem is still readable after the solution was added
    expect_bitwise(p, io::read_problem(path, IoOptions()), "problem after /solution");
    // warm start
    WarmStart w;
    w.X = r.best.X;
    w.y = r.best.y;
    w.S = r.best.S;
    w.sigma = 3.5;
    cuadmm::io::detail::write_h5_warm_start(path, w);
    const WarmStart v = io::read_warm_start(path, p);
    EXPECT_EQ(v.X, w.X);
    EXPECT_EQ(v.y, w.y);
    EXPECT_EQ(v.S, w.S);
    EXPECT_EQ(v.sigma, 3.5);
    ProblemData other = p;
    other.At.cols = 4;
    other.At.colptr.push_back(other.At.colptr.back());
    other.b.size = 4;
    EXPECT_THROW(io::read_warm_start(path, other), std::invalid_argument);
    EXPECT_THROW(io::read_warm_start(temp_dir("h5") + "/with_solution.xyz", p), std::invalid_argument);
#endif
}

TEST(IoH5, MalformedFilesAreRejected)
{
#ifndef CUADMM_HAVE_HDF5
    GTEST_SKIP() << "HDF5 not compiled in";
#else
    const std::string dir = temp_dir("h5");
    write_file(dir + "/not_hdf5.h5", "plain text");
    EXPECT_THROW(io::read_problem(dir + "/not_hdf5.h5", IoOptions()), std::invalid_argument);
    IoOptions raw;
    raw.simplify = false;
    {
        ProblemData p = small_problem();
        std::swap(p.At.rowind[0], p.At.rowind[1]); // unsorted rows: the writer validates
        EXPECT_THROW(io::write_problem(dir + "/unsorted.h5", p), std::invalid_argument);
    }
    {
        ProblemData p = small_problem();
        p.At.val[0] = std::numeric_limits<double>::infinity(); // validate() does not check finiteness: the reader does
        io::write_problem(dir + "/inf.h5", p);
        EXPECT_THROW(io::read_problem(dir + "/inf.h5", raw), std::invalid_argument);
    }
    // io.sense must agree with the stored sense
    io::write_problem(dir + "/sense.h5", small_problem());
    IoOptions o;
    o.sense = Sense::Maximize;
    EXPECT_THROW(io::read_problem(dir + "/sense.h5", o), std::invalid_argument);
    o.sense = Sense::Minimize;
    EXPECT_NO_THROW(io::read_problem(dir + "/sense.h5", o));
    // detection by signature without the extension
    std::filesystem::copy_file(dir + "/sense.h5", dir + "/sense.noext", std::filesystem::copy_options::overwrite_existing);
    EXPECT_NO_THROW(io::read_problem(dir + "/sense.noext", IoOptions()));
    IoOptions forced;
    forced.format = IoFormat::Sdpa;
    EXPECT_THROW(io::read_problem(dir + "/sense.h5", forced), std::invalid_argument); // io.format overrides the detection
#endif
}

// files written by h5py (tools/io/make_h5_testfiles.py): string attributes as variable-length UTF-8 (h5py's
// default for str), fixed-length ASCII (MATLAB-like), variable-length ASCII, fixed-length UTF-8; int32 datasets
TEST(IoH5, FilesWrittenByH5pyAndMatlabStyle)
{
#ifndef CUADMM_HAVE_HDF5
    GTEST_SKIP() << "HDF5 not compiled in";
#else
    for (const char *name : {"h5_vlen_utf8.h5", "h5_fixed_ascii.h5", "h5_vlen_ascii.h5", "h5_fixed_utf8.h5", "h5_int32.h5"})
    {
        const std::string path = std::string(CUADMM_TEST_DATA_DIR) + "/" + name;
        SCOPED_TRACE(path);
        ASSERT_TRUE(exists(path));
        IoOptions raw;
        raw.simplify = false;
        const ProblemData p = io::read_problem(path, raw);
        EXPECT_EQ(cones_string(p.cones), "s2 l2 ");
        EXPECT_EQ(p.sense, Sense::Minimize);
        EXPECT_EQ(p.objective_offset, 0.5);
        EXPECT_EQ(p.At.colptr, (std::vector<std::int64_t>{0, 2, 4}));
        EXPECT_EQ(p.At.rowind, (std::vector<std::int64_t>{0, 3, 2, 4}));
        EXPECT_EQ(p.At.val, (std::vector<double>{1.0, 2.0, 3.0, 4.0}));
        EXPECT_EQ(p.b.index, (std::vector<std::int64_t>{0, 1}));
        EXPECT_EQ(p.b.val, (std::vector<double>{1.0, 2.0}));
        EXPECT_EQ(p.C.index, (std::vector<std::int64_t>{0, 3}));
        EXPECT_EQ(p.source.dimacs_kinds, (std::vector<char>{'s', 'l'}));
        EXPECT_EQ(p.source.h5_block_kinds, (std::vector<char>{'s', 'l'}));
        bool stored = false;
        for (const std::string &line : p.source.report)
            stored = stored || line.find("written by h5py") != std::string::npos;
        EXPECT_TRUE(stored) << "the stored /meta/report was not read";
        const ProblemData q = io::read_problem(path, IoOptions()); // the s 2 block carries diagonal data only
        EXPECT_EQ(cones_string(q.cones), "l2 l2 ");
    }
#endif
}

TEST(IoH5, MalformedIndicesAreRejectedBeforeTheSimplification)
{
#ifndef CUADMM_HAVE_HDF5
    GTEST_SKIP() << "HDF5 not compiled in";
#else
    for (const char *name : {"h5_bad_oob.h5", "h5_bad_oob_small.h5", "h5_bad_neg.h5", "h5_bad_C_oob.h5"})
    {
        const std::string path = std::string(CUADMM_TEST_DATA_DIR) + "/" + name;
        SCOPED_TRACE(path);
        ASSERT_TRUE(exists(path));
        for (bool simplify : {true, false})
        {
            IoOptions o;
            o.simplify = simplify;
            try
            {
                io::read_problem(path, o);
                ADD_FAILURE() << "accepted with simplify = " << simplify;
            }
            catch (const std::invalid_argument &e)
            {
                EXPECT_NE(std::string(e.what()).find("outside [0, "), std::string::npos) << e.what();
            }
        }
    }
#endif
}

TEST(IoH5, MissingSenseAttributeNeedsIoSense)
{
#ifndef CUADMM_HAVE_HDF5
    GTEST_SKIP() << "HDF5 not compiled in";
#else
    const std::string path = std::string(CUADMM_TEST_DATA_DIR) + "/h5_no_sense.h5";
    ASSERT_TRUE(exists(path));
    try
    {
        io::read_problem(path, IoOptions());
        ADD_FAILURE() << "a file without a sense attribute was accepted";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_NE(std::string(e.what()).find("io.sense"), std::string::npos) << e.what();
    }
    IoOptions o;
    o.sense = Sense::Maximize;
    const ProblemData p = io::read_problem(path, o);
    EXPECT_EQ(p.sense, Sense::Maximize);
    bool noted = false;
    for (const std::string &line : p.source.report)
        noted = noted || line.find("taken from io.sense") != std::string::npos;
    EXPECT_TRUE(noted);
#endif
}

// the solution of a simplified problem is written in the layout of the file's blocks, and /start is accepted in it
TEST(IoH5, SolutionAndWarmStartInTheFileBlockLayout)
{
#ifndef CUADMM_HAVE_HDF5
    GTEST_SKIP() << "HDF5 not compiled in";
#else
    ProblemData d = small_problem(); // blocks s 2, l 2, u 1 (vec_len 6)
    d.At.rowind = {0, 2, 3, 3, 5, 2, 4};
    d.C.index = {0, 3, 5};
    d.C.val = {1.0, 0.25, -1.0};
    d.validate();
    const std::string path = temp_dir("h5") + "/layout.h5";
    io::write_problem(path, d);
    const ProblemData p = io::read_problem(path, IoOptions()); // the s 2 block becomes l 2: vec_len 5
    ASSERT_EQ(p.vec_len(), 5);
    EXPECT_EQ(p.source.h5_block_kinds, (std::vector<char>{'s', 'l', 'u'}));
    SolveResult r = fake_result(p);
    r.best.X = {1.0, 2.0, 3.0, 4.0, 5.0};
    r.best.S = {-1.0, -2.0, -3.0, -4.0, -5.0};
    io::write_solution(path, p, r);
    std::vector<double> X, y, S;
    cuadmm::io::detail::read_h5_solution(path, X, y, S, nullptr, nullptr);
    EXPECT_EQ(X, (std::vector<double>{1.0, 0.0, 2.0, 3.0, 4.0, 5.0})); // the file's s 2 block: diagonal (0,0), (1,1), zero off it
    EXPECT_EQ(S, (std::vector<double>{-1.0, 0.0, -2.0, -3.0, -4.0, -5.0}));
    EXPECT_EQ(y, r.best.y);
    std::vector<char> kinds;
    std::vector<std::int64_t> sizes;
    cuadmm::io::detail::read_h5_solution_blocks(path, kinds, sizes);
    EXPECT_EQ(kinds, (std::vector<char>{'s', 'l', 'u'}));
    EXPECT_EQ(sizes, (std::vector<std::int64_t>{2, 2, 1}));
    // a warm start written in the file's layout is contracted to the solver's cones; the solver's layout is accepted too
    WarmStart w;
    w.X = {10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    w.S = {20.0, 21.0, 22.0, 23.0, 24.0, 25.0};
    w.y = r.best.y;
    cuadmm::io::detail::write_h5_warm_start(path, w);
    const WarmStart v = io::read_warm_start(path, p);
    EXPECT_EQ(v.X, (std::vector<double>{10.0, 12.0, 13.0, 14.0, 15.0}));
    EXPECT_EQ(v.S, (std::vector<double>{20.0, 22.0, 23.0, 24.0, 25.0}));
    w.X = r.best.X;
    w.S = r.best.S;
    cuadmm::io::detail::write_h5_warm_start(path, w);
    EXPECT_EQ(io::read_warm_start(path, p).X, r.best.X);
    w.X.resize(4);
    cuadmm::io::detail::write_h5_warm_start(path, w);
    EXPECT_THROW(io::read_warm_start(path, p), std::invalid_argument);
    // a standalone solution.h5 of a problem read from another format names the original blocks (SDPT3-style split
    // cones and simplified blocks included): the simplifiable() problem of io_simplify_test.hpp
    ProblemData q = simplifiable();
    cuadmm::io::detail::apply_simplifications(q, true);
    const SolveResult rq = fake_result(q);
    const std::string standalone = temp_dir("h5/standalone");
    io::write_solution(standalone, q, rq, "h5");
    cuadmm::io::detail::read_h5_solution_blocks(standalone + "/solution.h5", kinds, sizes);
    EXPECT_EQ(kinds, (std::vector<char>{'s', 's', 's', 'l'}));
    EXPECT_EQ(sizes, (std::vector<std::int64_t>{1, 3, 2, 2}));
    cuadmm::io::detail::read_h5_solution(standalone + "/solution.h5", X, y, S, nullptr, nullptr);
    EXPECT_EQ(X.size(), 12u); // 1 + 6 + 3 + 2: the original svec layout
    EXPECT_EQ(X[7], rq.best.X[7]);  // (0,0) of the simplified s 2 block
    EXPECT_EQ(X[8], 0.0);           // (0,1): off-diagonal of a simplified block
    EXPECT_EQ(X[9], rq.best.X[8]);  // (1,1)
    EXPECT_EQ(X[10], rq.best.X[9]); // the l block follows
#endif
}

TEST(IoH5, ConvertedPlatoInstanceReadsBackIdentically)
{
#ifndef CUADMM_HAVE_HDF5
    GTEST_SKIP() << "HDF5 not compiled in";
#else
    const std::string dat = external("plato_sdpa/trto1.dat-s");
    IO_TEST_REQUIRE_FILE(dat);
    const ProblemData a = io::read_problem(dat, IoOptions());
    const std::string h5 = temp_dir("h5") + "/trto1.h5";
    io::write_problem(h5, a, 1);
    const ProblemData b = io::read_problem(h5, IoOptions());
    expect_bitwise(a, b, "trto1 .dat-s vs .h5");
    EXPECT_EQ(b.source.dimacs_kinds, a.source.dimacs_kinds);
    EXPECT_EQ(b.sense, Sense::Maximize);
#endif
}
