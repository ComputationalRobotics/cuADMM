/*

    io_builder_test.hpp  (CPU only)

    The CSC builder (in-place sort, duplicates, zeros, range checks), the svec helpers and SHA-256.

*/

#include <gtest/gtest.h>

#include <stdexcept>

#include "io/csc_builder.h"
#include "io/sha256.h"
#include "io/svec.h"

using namespace cuadmm;
using namespace cuadmm::io::detail;

TEST(IoBuilder, UnsortedColumnsAndRowsBecomeSortedCsc)
{
    CscBuilder b(5, 3);
    b.add(2, 4, 1.0);
    b.add(0, 3, 2.0);
    b.add(2, 0, 3.0);
    b.add(1, 1, 4.0);
    b.add(0, 1, 5.0);
    b.add(2, 2, 6.0);
    BuildStats st;
    const CscMatrix M = b.finish(DuplicatePolicy::Reject, true, st);
    EXPECT_FALSE(st.sorted_on_arrival);
    EXPECT_EQ(M.colptr, (std::vector<std::int64_t>{0, 2, 3, 6}));
    EXPECT_EQ(M.rowind, (std::vector<std::int64_t>{1, 3, 1, 0, 2, 4}));
    EXPECT_EQ(M.val, (std::vector<double>{5.0, 2.0, 4.0, 3.0, 6.0, 1.0}));
    EXPECT_EQ(st.duplicates, 0);
    EXPECT_EQ(st.zeros_dropped, 0);
    EXPECT_NO_THROW(([&]()
    {
        ProblemData p;
        p.At = M;
        p.b.size = 3;
        p.C.size = 5;
        p.cones = {{ConeKind::Nonnegative, 5, 0.0}};
        p.validate();
    })());
}

TEST(IoBuilder, SortedArrivalIsRecognizedAndEmptyColumnsKept)
{
    CscBuilder b(4, 4);
    b.add(1, 2, 1.0);
    b.add(1, 0, 2.0);
    b.add(3, 3, 3.0);
    BuildStats st;
    const CscMatrix M = b.finish(DuplicatePolicy::Sum, true, st);
    EXPECT_TRUE(st.sorted_on_arrival);
    EXPECT_EQ(M.colptr, (std::vector<std::int64_t>{0, 0, 2, 2, 3}));
    EXPECT_EQ(M.rowind, (std::vector<std::int64_t>{0, 2, 3}));
    EXPECT_EQ(M.val, (std::vector<double>{2.0, 1.0, 3.0}));
}

TEST(IoBuilder, DuplicatesSummedOrReported)
{
    {
        CscBuilder b(3, 2);
        b.add(1, 2, 1.0);
        b.add(0, 0, 1.0);
        b.add(1, 2, 0.5);
        b.add(1, 2, 0.25);
        b.add(0, 1, 2.0);
        b.add(0, 1, -2.0); // sums to zero: dropped
        BuildStats st;
        const CscMatrix M = b.finish(DuplicatePolicy::Sum, true, st);
        EXPECT_EQ(st.duplicates, 3);
        EXPECT_EQ(st.zeros_dropped, 1);
        EXPECT_EQ(M.colptr, (std::vector<std::int64_t>{0, 1, 2}));
        EXPECT_EQ(M.rowind, (std::vector<std::int64_t>{0, 2}));
        EXPECT_EQ(M.val, (std::vector<double>{1.0, 1.75}));
    }
    {
        CscBuilder b(3, 2);
        b.add(1, 2, 1.0);
        b.add(1, 0, 0.0);
        b.add(1, 2, 0.0); // a zero duplicate still counts (the SDPA converter rejects it)
        BuildStats st;
        const CscMatrix M = b.finish(DuplicatePolicy::Reject, true, st);
        EXPECT_EQ(st.duplicates, 1);
        EXPECT_EQ(st.first_duplicate_col, 1);
        EXPECT_EQ(st.first_duplicate_row, 2);
        EXPECT_EQ(M.nnz(), 2); // the duplicates are kept for the caller's message, the lone zero is dropped
    }
    {
        CscBuilder b(3, 2);
        b.add(0, 0, 0.0);
        b.add(1, 1, 0.0);
        BuildStats st;
        const CscMatrix M = b.finish(DuplicatePolicy::Reject, false, st);
        EXPECT_EQ(M.nnz(), 2); // zeros kept when asked
    }
}

TEST(IoBuilder, RangeChecksAndSparseVector)
{
    CscBuilder b(3, 2);
    EXPECT_THROW(b.add(2, 0, 1.0), std::invalid_argument);
    EXPECT_THROW(b.add(0, 3, 1.0), std::invalid_argument);
    EXPECT_THROW(b.add(-1, 0, 1.0), std::invalid_argument);
    SparseVectorBuilder v(10);
    v.add(7, 1.0);
    v.add(2, 2.0);
    v.add(7, 1.0);
    BuildStats st;
    const SparseVector s = v.finish(DuplicatePolicy::Sum, true, st);
    EXPECT_EQ(s.size, 10);
    EXPECT_EQ(s.index, (std::vector<std::int64_t>{2, 7}));
    EXPECT_EQ(s.val, (std::vector<double>{2.0, 2.0}));
}

TEST(IoBuilder, LargeUnsortedPermutationMatchesSortedInsertion)
{
    const std::int64_t rows = 1000, cols = 300, n = 50000;
    CscBuilder shuffled(rows, cols), ordered(rows, cols);
    std::vector<std::pair<std::int64_t, std::int64_t>> keys;
    std::uint64_t x = 99;
    for (std::int64_t k = 0; k < n; k++)
    {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        keys.push_back({static_cast<std::int64_t>((x >> 33) % cols), static_cast<std::int64_t>((x >> 13) % rows)});
    }
    for (std::size_t k = 0; k < keys.size(); k++)
        shuffled.add(keys[k].first, keys[k].second, static_cast<double>(k) + 1.0);
    std::vector<std::size_t> order(keys.size());
    for (std::size_t k = 0; k < order.size(); k++)
        order[k] = k;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });
    for (std::size_t k : order)
        ordered.add(keys[k].first, keys[k].second, static_cast<double>(k) + 1.0);
    BuildStats s1, s2;
    const CscMatrix A = shuffled.finish(DuplicatePolicy::Sum, true, s1), B = ordered.finish(DuplicatePolicy::Sum, true, s2);
    EXPECT_FALSE(s1.sorted_on_arrival);
    EXPECT_TRUE(s2.sorted_on_arrival);
    EXPECT_EQ(A.colptr, B.colptr);
    EXPECT_EQ(A.rowind, B.rowind);
    EXPECT_EQ(A.val, B.val); // duplicates summed in the same (stable) order on both sides
    EXPECT_EQ(s1.duplicates, s2.duplicates);
}

TEST(IoSvec, IndexHelpers)
{
    EXPECT_EQ(svec_index(0, 0, 0), 0);
    EXPECT_EQ(svec_index(0, 0, 1), 1);
    EXPECT_EQ(svec_index(0, 1, 0), 1); // mirrored
    EXPECT_EQ(svec_index(0, 1, 1), 2);
    EXPECT_EQ(svec_index(10, 2, 3), 10 + 6 + 2);
    for (std::int64_t n = 1; n <= 7; n++)
        for (std::int64_t l = 0; l < n; l++)
            for (std::int64_t k = 0; k <= l; k++)
            {
                const std::int64_t r = svec_index(0, k, l);
                std::int64_t kk, ll;
                svec_position(r, kk, ll);
                EXPECT_EQ(kk, k);
                EXPECT_EQ(ll, l);
                EXPECT_EQ(is_diagonal_position(r, n), k == l);
            }
    EXPECT_EQ(cone_length({ConeKind::Psd, 4, 0.0}), 10);
    EXPECT_EQ(cone_length({ConeKind::Free, 4, 0.0}), 4);
    EXPECT_EQ(kHalfSqrt2 * 2.0, kSqrt2);
    double a = 0.3;
    EXPECT_EQ(a * kHalfSqrt2 + a * kHalfSqrt2, a * kSqrt2); // the mirrored halves sum to the sqrt2 entry exactly
}

TEST(IoSha256, KnownVectors)
{
    Sha256 h;
    EXPECT_EQ(h.hex_digest(), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    Sha256 h2;
    h2.update(reinterpret_cast<const unsigned char *>("abc"), 3);
    EXPECT_EQ(h2.hex_digest(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    Sha256 h3;
    const std::string msg = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    h3.update(reinterpret_cast<const unsigned char *>(msg.data()), msg.size());
    EXPECT_EQ(h3.hex_digest(), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    Sha256 h4; // one million 'a' in uneven pieces
    const std::string piece(1000, 'a');
    for (int k = 0; k < 1000; k++)
        h4.update(reinterpret_cast<const unsigned char *>(piece.data()), piece.size());
    EXPECT_EQ(h4.hex_digest(), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}
