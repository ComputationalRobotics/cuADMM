/*

    io_sdpa_test.hpp  (CPU only)

    The SDPA reader on hand-made edge cases (header variants, comments, lower-triangle entries mirrored,
    duplicates rejected, explicit zeros dropped, diagonal blocks, out-of-range and non-finite entries, empty
    constraint matrices, gzip input) and against the reference conversions of the PLATO/Kocvara instances.

*/

#include <gtest/gtest.h>

#include <stdexcept>

#include "io_test_support.hpp"

#ifdef CUADMM_HAVE_ZLIB
#include <zlib.h>
#endif

namespace
{
using namespace io_test;

// min <C,X> for blocks s 2 and l 2 (diagonal block -2 in the file), m = 2
const char *kSdpaSmall =
    "\"comment line\n"
    "* another comment\n"
    "2 =mDIM\n"
    "2 = nBLOCK\n"
    "{2, -2}\n"
    "1.0 2.5\n"
    "0 1 1 1 -1.0\n"
    "0 1 2 1 0.5\n"   // lower triangle: mirrored to (1,2)
    "0 2 1 1 3.0\n"
    "1 1 1 1 1.0\n"
    "1 1 1 2 2.0\n"
    "1 2 2 2 4.0\n"
    "2 1 2 2 1.0\n"
    "2 2 1 1 0.0\n"   // explicit zero: dropped
    "2 2 2 2 -1.0\n";

ProblemData read_sdpa_text(const std::string &name, const std::string &text, bool simplify = true)
{
    const std::string path = write_file(temp_dir("sdpa") + "/" + name + ".dat-s", text);
    IoOptions o;
    o.simplify = simplify;
    return io::read_problem(path, o);
}
} // namespace

TEST(IoSdpa, SmallFileMappedAsTheConverter)
{
    const ProblemData p = read_sdpa_text("small", kSdpaSmall);
    EXPECT_EQ(p.source.format, "sdpa");
    EXPECT_EQ(p.sense, Sense::Maximize); // the file's value is max <F_0, Y> = -<C, X>
    ASSERT_EQ(p.cones.size(), 2u);
    EXPECT_EQ(p.cones[0].kind, ConeKind::Psd);
    EXPECT_EQ(p.cones[0].size, 2);
    EXPECT_EQ(p.cones[1].kind, ConeKind::Nonnegative);
    EXPECT_EQ(p.cones[1].size, 2);
    EXPECT_EQ(p.vec_len(), 5);
    EXPECT_EQ(p.num_constraints(), 2);
    EXPECT_EQ(p.source.dimacs_kinds, (std::vector<char>{'s', 'l'}));
    // b = c
    EXPECT_EQ(p.b.index, (std::vector<std::int64_t>{0, 1}));
    EXPECT_EQ(p.b.val, (std::vector<double>{1.0, 2.5}));
    // C = -F_0: (1,1) -> 1.0 at 0; (2,1) mirrored to (1,2) -> -0.5*sqrt2 at 1; block 2 (1,1) -> -3 at 3
    EXPECT_EQ(p.C.index, (std::vector<std::int64_t>{0, 1, 3}));
    EXPECT_EQ(p.C.val, (std::vector<double>{1.0, -0.5 * 1.4142135623730951, -3.0}));
    // At: constraint 1: F_1 = (1,1) 1, (1,2) 2 -> 2 sqrt2, block 2 (2,2) 4 -> index 4
    EXPECT_EQ(p.At.colptr, (std::vector<std::int64_t>{0, 3, 5}));
    EXPECT_EQ(p.At.rowind, (std::vector<std::int64_t>{0, 1, 4, 2, 4}));
    EXPECT_EQ(p.At.val, (std::vector<double>{1.0, 2.0 * 1.4142135623730951, 4.0, 1.0, -1.0}));
    bool mirrored = false, zeros = false;
    for (const std::string &line : p.source.report)
    {
        mirrored = mirrored || line.find("lower-triangle entries mirrored: 1") != std::string::npos;
        zeros = zeros || line.find("explicit zeros dropped: 1") != std::string::npos;
    }
    EXPECT_TRUE(mirrored);
    EXPECT_TRUE(zeros);
}

TEST(IoSdpa, HeaderVariantsAndValuesOverLines)
{
    const std::string text =
        "2\n2\n2 -2\n1.0\n2.5 extra_tokens_ignored\n"
        "0 1 1 1 -1.0\n1 1 1 1 1.0\n1 1 1 2 2.0\n1 2 2 2 4.0\n2 1 2 2 1.0\n2 2 2 2 -1.0\n";
    const ProblemData p = read_sdpa_text("variants", text);
    EXPECT_EQ(p.num_constraints(), 2);
    EXPECT_EQ(p.b.val, (std::vector<double>{1.0, 2.5}));
    const std::string parenthesized = "2\n2\n(2, -2)\n1.0, 2.5\n0 1 1 1 -1.0\n1 1 1 1 1.0\n1 2 1 1 1.0\n2 1 2 2 1.0\n";
    EXPECT_NO_THROW(read_sdpa_text("paren", parenthesized));
}

TEST(IoSdpa, RejectsWhatTheConverterRejects)
{
    const std::string head = "2\n2\n{2, -2}\n1.0 2.5\n";
    auto rejects = [&](const std::string &name, const std::string &entries, const std::string &needle)
    {
        try
        {
            read_sdpa_text(name, head + entries);
            ADD_FAILURE() << name << " was accepted";
        }
        catch (const std::invalid_argument &e)
        {
            EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
        }
    };
    rejects("dup", "1 1 1 2 1.0\n1 1 2 1 2.0\n2 2 1 1 1.0\n", "duplicate");           // (1,2) and (2,1) of the same matrix
    rejects("dup_zero", "1 1 1 2 1.0\n1 1 1 2 0.0\n2 2 1 1 1.0\n", "duplicate");      // a zero duplicate counts
    rejects("matno", "3 1 1 1 1.0\n1 1 1 1 1.0\n2 2 1 1 1.0\n", "matrix number 3");
    rejects("block", "1 3 1 1 1.0\n2 2 1 1 1.0\n", "block 3");
    rejects("index", "1 1 3 1 1.0\n2 2 1 1 1.0\n", "outside a block of size 2");
    rejects("offdiag_l", "1 1 1 1 1.0\n2 2 1 2 1.0\n", "off-diagonal entry");
    rejects("nonfinite", "1 1 1 1 inf\n2 2 1 1 1.0\n", "non-finite");
    rejects("fields", "1 1 1 1\n2 2 1 1 1.0\n", "expected 5 fields");
    rejects("empty_constraint", "1 1 1 1 1.0\n", "empty constraint matrices");
    EXPECT_THROW(read_sdpa_text("zero_block", "2\n2\n{0, 2}\n1.0 2.5\n1 2 1 1 1.0\n2 2 1 1 1.0\n"), std::invalid_argument); // a block of size 0
    EXPECT_THROW(read_sdpa_text("bad_header", "0\n1\n{2}\n1.0\n1 1 1 1 1.0\n"), std::invalid_argument);
    EXPECT_THROW(read_sdpa_text("truncated", "2\n2\n{2, -2}\n1.0\n"), std::invalid_argument);
}

TEST(IoSdpa, OneByOneBlocksAndUntouchedBlocksReported)
{
    const std::string text = "1\n3\n{1, 2, 3}\n1.0\n0 1 1 1 2.0\n1 1 1 1 1.0\n1 2 1 2 1.0\n";
    const ProblemData p = read_sdpa_text("one", text);
    ASSERT_EQ(p.cones.size(), 3u);
    EXPECT_EQ(p.cones[0].kind, ConeKind::Nonnegative); // 1x1 PSD -> l 1 (as the converter)
    EXPECT_EQ(p.cones[0].size, 1);
    EXPECT_EQ(p.cones[1].kind, ConeKind::Psd);
    EXPECT_EQ(p.cones[2].kind, ConeKind::Nonnegative); // untouched block: diagonal-only, simplified
    EXPECT_EQ(p.cones[2].size, 3);
    EXPECT_EQ(p.source.dimacs_kinds, (std::vector<char>{'s', 's', 's'}));
    bool untouched = false;
    for (const std::string &line : p.source.report)
        untouched = untouched || line.find("not touched by any F_i: 1 (3)") != std::string::npos;
    EXPECT_TRUE(untouched);
    const ProblemData q = read_sdpa_text("one_raw", text, false);
    EXPECT_EQ(q.cones[0].kind, ConeKind::Psd);
    EXPECT_EQ(q.cones[2].kind, ConeKind::Psd);
    EXPECT_EQ(q.vec_len(), 1 + 3 + 6);
}

TEST(IoSdpa, SenseOptionMustAgree)
{
    const std::string path = write_file(temp_dir("sdpa") + "/sense.dat-s", kSdpaSmall);
    IoOptions o;
    o.sense = Sense::Minimize;
    EXPECT_THROW(io::read_problem(path, o), std::invalid_argument);
    o.sense = Sense::Maximize;
    EXPECT_NO_THROW(io::read_problem(path, o));
}

TEST(IoSdpa, GzipInput)
{
    const std::string plain = write_file(temp_dir("sdpa") + "/gz_plain.dat-s", kSdpaSmall);
    const std::string gz = temp_dir("sdpa") + "/gz_copy.dat-s.gz";
#ifdef CUADMM_HAVE_ZLIB
    gzFile f = gzopen(gz.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    gzwrite(f, kSdpaSmall, static_cast<unsigned>(std::string(kSdpaSmall).size()));
    gzclose(f);
    IoOptions o;
    const ProblemData a = io::read_problem(plain, o), b = io::read_problem(gz, o);
    expect_bitwise(a, b, "gz vs plain");
    EXPECT_TRUE(io::format_available("sdpa.gz"));
#else
    write_file(gz, std::string("\x1f\x8b", 2) + "not really compressed");
    EXPECT_THROW(io::read_problem(gz, IoOptions()), std::invalid_argument); // a clear message instead of garbage
    EXPECT_FALSE(io::format_available("sdpa.gz"));
    GTEST_SKIP() << "zlib not compiled in";
#endif
}

TEST(IoSdpa, PlatoInstancesMatchTheReferenceConversion)
{
    const std::string txt_root = external("plato_kocvara/txt"), sdpa_root = external("plato_sdpa");
    IO_TEST_REQUIRE_FILE(txt_root);
    IO_TEST_REQUIRE_FILE(sdpa_root);
    for (const char *name : {"trto1", "mater-1", "buck3"})
    {
        const std::string dat = sdpa_root + "/" + std::string(name) + ".dat-s", txt = txt_root + "/" + name;
        if (!exists(dat) || !exists(txt))
        {
            std::printf("  (skipping %s: files missing)\n", name);
            continue;
        }
        IoOptions o;
        const ProblemData a = io::read_problem(dat, o);
        const ProblemData t = io::read_txt(txt);
        EXPECT_EQ(a.source.dimacs_kinds, t.source.dimacs_kinds) << name;
        ProblemData a_min = a;
        a_min.sense = Sense::Minimize; // the TXT conversion has no sense
        expect_bitwise(a_min, t, std::string(name) + " .dat-s vs TXT");
    }
    const std::string gz = sdpa_root + "/trto1.dat-s.gz";
    if (exists(gz) && io::format_available("sdpa.gz"))
        expect_bitwise(io::read_problem(gz, IoOptions()), io::read_problem(sdpa_root + "/trto1.dat-s", IoOptions()), "trto1 .gz vs plain");
}
