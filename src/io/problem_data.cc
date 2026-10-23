/*

    problem_data.cc

    ProblemData checks and the cone-code mapping.

*/

#include "cuadmm/problem.h"

#include <stdexcept>
#include <string>

namespace cuadmm
{

char cone_code(ConeKind kind)
{
    switch (kind)
    {
    case ConeKind::Psd:
        return 's';
    case ConeKind::Nonnegative:
        return 'l';
    case ConeKind::Free:
        return 'u';
    default:
        throw std::invalid_argument(std::string("cone ") + name_of(kind) + " is not supported by this build (plan Phase 9)");
    }
}

ConeKind cone_kind_of(char code)
{
    switch (code)
    {
    case 's':
        return ConeKind::Psd;
    case 'l':
        return ConeKind::Nonnegative;
    case 'u':
        return ConeKind::Free;
    case 'q':
        return ConeKind::SecondOrder;
    case 'r':
        return ConeKind::RotatedSecondOrder;
    default:
        throw std::invalid_argument(std::string("unknown cone code '") + code + "'");
    }
}

const char *name_of(ConeKind kind)
{
    switch (kind)
    {
    case ConeKind::Free:
        return "free";
    case ConeKind::Nonnegative:
        return "nonnegative";
    case ConeKind::Psd:
        return "psd";
    case ConeKind::SecondOrder:
        return "second_order";
    case ConeKind::RotatedSecondOrder:
        return "rotated_second_order";
    case ConeKind::Exponential:
        return "exponential";
    case ConeKind::Power:
        return "power";
    }
    return "?";
}

void ProblemData::validate() const
{
    auto fail = [](const std::string &why)
    { throw std::invalid_argument("ProblemData: " + why); };
    if (At.rows < 0 || At.cols < 0)
        fail("negative sizes");
    if (static_cast<std::int64_t>(At.colptr.size()) != At.cols + 1)
        fail("At.colptr must have cols + 1 entries");
    if (At.rowind.size() != At.val.size())
        fail("At.rowind and At.val differ in length");
    if (At.colptr.front() != 0 || At.colptr.back() != At.nnz())
        fail("At.colptr must start at 0 and end at nnz");
    for (std::int64_t j = 0; j < At.cols; j++)
    {
        if (At.colptr[j] > At.colptr[j + 1])
            fail("At.colptr must be non-decreasing");
        for (std::int64_t k = At.colptr[j]; k < At.colptr[j + 1]; k++)
        {
            if (At.rowind[k] < 0 || At.rowind[k] >= At.rows)
                fail("At row index out of range in column " + std::to_string(j));
            if (k > At.colptr[j] && At.rowind[k] <= At.rowind[k - 1])
                fail("At row indices must be sorted and unique within column " + std::to_string(j));
        }
    }
    if (b.size != At.cols || C.size != At.rows)
        fail("b must have m entries and C vec_len entries");
    if (b.index.size() != b.val.size() || C.index.size() != C.val.size())
        fail("sparse vector index and value lengths differ");
    for (std::int64_t i : b.index)
        if (i < 0 || i >= b.size)
            fail("b index out of range");
    for (std::int64_t i : C.index)
        if (i < 0 || i >= C.size)
            fail("C index out of range");
    std::int64_t len = 0;
    for (const ConeSpec &c : cones)
    {
        if (c.size <= 0)
            fail("cone sizes must be positive");
        if (c.kind == ConeKind::Psd)
            len += c.size * (c.size + 1) / 2;
        else if (c.kind == ConeKind::Free || c.kind == ConeKind::Nonnegative)
            len += c.size;
        else
            fail(std::string("cone ") + name_of(c.kind) + " is not supported by this build (plan Phase 9)");
    }
    if (len != At.rows)
        fail("the cones span " + std::to_string(len) + " entries, At has " + std::to_string(At.rows) + " rows");
    if (!source.dimacs_kinds.empty() && source.dimacs_kinds.size() != cones.size())
        fail("source.dimacs_kinds must have one entry per cone");
}

} // namespace cuadmm
