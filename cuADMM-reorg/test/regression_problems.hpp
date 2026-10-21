/*

    regression_problems.hpp

    Deterministic generator of SDP problems with an analytically known optimum, written in the
    TXT format read by Problem::from_txt (blk.txt, con_num.txt, At.txt, b.txt, C.txt).

    Block-wise, X* in K and S* in K* are built with X* S* = 0:
    - 's' blocks: X* = Q diag(x) Q^T and S* = Q diag(z) Q^T with a common orthogonal basis Q and
      complementary supports of x > 0 and z > 0 (rank X* + rank S* = n: strict complementarity).
      Q is a random permutation of a block-diagonal matrix of small random rotations, so that
      X* and S* stay sparse for large blocks;
    - 'l' blocks: X* and S* are positive on complementary supports;
    - 'u' blocks: X* is arbitrary and S* = 0.
    With a random sparse A (a quarter of the entries of each constraint on the support of X*, so
    that b != 0) and a random y*, we set b = A(X*) and C = A^T y* + S*. Then (X*, y*, S*) is
    primal-dual feasible with a zero duality gap, hence optimal: p* = <C, X*> = <b, y*>.

    Only the raw 32-bit outputs of std::mt19937 are used (no std distributions or std::shuffle,
    whose results are implementation-defined), so the problems are identical on every platform.

*/

#ifndef CUADMM_TEST_REGRESSION_PROBLEMS_HPP
#define CUADMM_TEST_REGRESSION_PROBLEMS_HPP

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace regression
{

/// @brief One block of a generated problem.
struct BlockSpec
{
    char type;       // 's', 'u' or 'l'
    int size;        // side of an 's' block, length of a 'u' or 'l' block
    int rank_X;      // 's': rank of X* (rank S* = size - rank_X); 'l': number of positive entries of X*
    int group;       // 's': size of the rotations composing Q (0: a single dense rotation)
    bool interleave; // 's': take rank_X / (number of rotations) columns of X* in every rotation,
                     //      instead of the first rank_X columns of Q
};

/// @brief Parameters of a generated problem.
struct ProblemSpec
{
    std::vector<BlockSpec> blocks;
    int con_num;       // number of constraints (0: middle of the generic nondegeneracy range)
    int nnz_per_con;   // number of entries of each constraint
    unsigned int seed; // seed of std::mt19937
};

/// @brief A generated problem and its optimal solution.
struct RegressionProblem
{
    std::vector<char> blk_types;
    std::vector<int> blk_sizes;
    int vec_len;
    int con_num;
    std::vector<int> At_rows;    // |
    std::vector<int> At_cols;    // |- At (vec_len x con_num) as COO triplets, grouped by constraint
    std::vector<double> At_vals; // |
    std::vector<double> b;       // dense, size con_num
    std::vector<double> C;       // dense, size vec_len
    std::vector<double> X_star;  // |
    std::vector<double> y_star;  // |- optimal solution
    std::vector<double> S_star;  // |
    double p_star;               // optimal value <C, X*> = <b, y*>
};

// uniform in (0, 1), from the raw output of std::mt19937 (which is fully specified)
inline double uniform01(std::mt19937 &gen)
{
    return (static_cast<double>(gen()) + 0.5) / 4294967296.0;
}

// uniform in (lo, hi)
inline double uniform(std::mt19937 &gen, double lo, double hi)
{
    return lo + (hi - lo) * uniform01(gen);
}

// uniform integer in [0, n)
inline int uniform_index(std::mt19937 &gen, int n)
{
    return static_cast<int>(gen() % static_cast<unsigned int>(n));
}

// random permutation of [0, n) (Fisher-Yates)
inline std::vector<int> random_permutation(std::mt19937 &gen, int n)
{
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    for (int i = n - 1; i > 0; i--)
        std::swap(perm[i], perm[uniform_index(gen, i + 1)]);
    return perm;
}

// random g x g orthogonal matrix (column-major), by Gram-Schmidt on a random matrix
inline std::vector<double> random_orthogonal(std::mt19937 &gen, int g)
{
    std::vector<double> Q(size_t(g) * g);
    for (auto &v : Q)
        v = uniform(gen, -1.0, 1.0);
    for (int j = 0; j < g; j++)
    {
        double *qj = Q.data() + size_t(j) * g;
        for (int pass = 0; pass < 2; pass++) // twice is enough for orthogonality to machine precision
        {
            for (int l = 0; l < j; l++)
            {
                const double *ql = Q.data() + size_t(l) * g;
                double dot = 0.0;
                for (int i = 0; i < g; i++)
                    dot += ql[i] * qj[i];
                for (int i = 0; i < g; i++)
                    qj[i] -= dot * ql[i];
            }
        }
        double norm = 0.0;
        for (int i = 0; i < g; i++)
            norm += qj[i] * qj[i];
        norm = std::sqrt(norm);
        for (int i = 0; i < g; i++)
            qj[i] /= norm;
    }
    return Q;
}

// index of the entry (i, j) of an 's' block in its svec (upper triangle, column by column)
inline int svec_index(int i, int j)
{
    const int r = std::min(i, j);
    const int c = std::max(i, j);
    return c * (c + 1) / 2 + r;
}

// X* and S* of an 's' block, written to X_star / S_star from offset (in svec form)
inline void fill_psd_block(
    std::mt19937 &gen, const BlockSpec &blk, int offset,
    std::vector<double> &X_star, std::vector<double> &S_star)
{
    const int n = blk.size;
    const int g = blk.group > 0 ? std::min(blk.group, n) : n;
    const int group_num = (n + g - 1) / g;
    if (blk.interleave && (n % g != 0 || blk.rank_X % group_num != 0))
        throw std::invalid_argument("interleaved blocks need equal rotations and rank_X divisible by their number");

    // columns of Q spanning the range of X* (the others span the range of S*)
    std::vector<char> is_X(n, 0);
    for (int j = 0; j < n; j++)
        is_X[j] = blk.interleave ? (j % g < blk.rank_X / group_num) : (j < blk.rank_X);

    // Q = P blkdiag(R_0, R_1, ...): rotation k acts on the indices perm[k g], ..., perm[k g + g - 1]
    const std::vector<int> perm = random_permutation(gen, n);
    const double sqrt2 = std::sqrt(2.0);
    for (int k = 0; k < group_num; k++)
    {
        const int start = k * g;
        const int gs = std::min(g, n - start);
        const std::vector<double> R = random_orthogonal(gen, gs);
        std::vector<double> x(gs, 0.0), z(gs, 0.0);
        for (int l = 0; l < gs; l++)
            (is_X[start + l] ? x[l] : z[l]) = uniform(gen, 1.0, 2.0);

        // local blocks R diag(x) R^T and R diag(z) R^T, scattered to the permuted indices
        for (int a = 0; a < gs; a++)
        {
            for (int c = a; c < gs; c++)
            {
                double X_ac = 0.0, S_ac = 0.0;
                for (int l = 0; l < gs; l++)
                {
                    const double RR = R[a + size_t(l) * gs] * R[c + size_t(l) * gs];
                    X_ac += RR * x[l];
                    S_ac += RR * z[l];
                }
                const double factor = (a == c) ? 1.0 : sqrt2; // svec scaling of off-diagonal entries
                const int idx = offset + svec_index(perm[start + a], perm[start + c]);
                X_star[idx] = factor * X_ac;
                S_star[idx] = factor * S_ac;
            }
        }
    }
}

/// @brief Generates the problem described by spec.
inline RegressionProblem generate(const ProblemSpec &spec)
{
    std::mt19937 gen(spec.seed);
    RegressionProblem p;

    // block layout and generic nondegeneracy range [lo, hi] of the number of constraints:
    // lo = dimension of the face of X* (primal uniqueness), vec_len - hi = that of S* (dual uniqueness)
    p.vec_len = 0;
    int lo = 0, hi = 0;
    for (const BlockSpec &blk : spec.blocks)
    {
        p.blk_types.push_back(blk.type);
        p.blk_sizes.push_back(blk.size);
        if (blk.type == 's')
        {
            const int rS = blk.size - blk.rank_X;
            p.vec_len += blk.size * (blk.size + 1) / 2;
            lo += blk.rank_X * (blk.rank_X + 1) / 2;
            hi += blk.size * (blk.size + 1) / 2 - rS * (rS + 1) / 2;
        }
        else if (blk.type == 'u')
        {
            p.vec_len += blk.size;
            lo += blk.size;
            hi += blk.size;
        }
        else if (blk.type == 'l')
        {
            p.vec_len += blk.size;
            lo += blk.rank_X;
            hi += blk.rank_X;
        }
        else
            throw std::invalid_argument("unknown block type");
    }
    p.con_num = spec.con_num > 0 ? spec.con_num : (lo + hi) / 2;

    // optimal X* and S*
    p.X_star.assign(p.vec_len, 0.0);
    p.S_star.assign(p.vec_len, 0.0);
    int offset = 0;
    for (const BlockSpec &blk : spec.blocks)
    {
        if (blk.type == 's')
        {
            fill_psd_block(gen, blk, offset, p.X_star, p.S_star);
            offset += blk.size * (blk.size + 1) / 2;
        }
        else if (blk.type == 'u')
        {
            for (int i = 0; i < blk.size; i++)
                p.X_star[offset + i] = uniform(gen, -1.0, 1.0);
            offset += blk.size;
        }
        else
        {
            const std::vector<int> perm = random_permutation(gen, blk.size);
            for (int i = 0; i < blk.size; i++)
                (i < blk.rank_X ? p.X_star : p.S_star)[offset + perm[i]] = uniform(gen, 1.0, 2.0);
            offset += blk.size;
        }
    }

    // support of X*
    std::vector<int> support;
    for (int i = 0; i < p.vec_len; i++)
        if (p.X_star[i] != 0.0)
            support.push_back(i);

    // sparse A: a quarter of the entries of each constraint on the support of X*, the others anywhere
    const int nnz_support = std::max(1, spec.nnz_per_con / 4);
    for (int con = 0; con < p.con_num; con++)
    {
        std::vector<int> ids;
        while (int(ids.size()) < spec.nnz_per_con)
        {
            const int id = int(ids.size()) < nnz_support
                               ? support[uniform_index(gen, int(support.size()))]
                               : uniform_index(gen, p.vec_len);
            if (std::find(ids.begin(), ids.end(), id) != ids.end())
                continue;
            ids.push_back(id);
            const double sign = uniform01(gen) < 0.5 ? -1.0 : 1.0;
            p.At_rows.push_back(id);
            p.At_cols.push_back(con);
            p.At_vals.push_back(sign * uniform(gen, 0.5, 1.5));
        }
    }

    // make At reach the last index of X (Problem::from_txt warns otherwise), with the last entry drawn anywhere
    if (*std::max_element(p.At_rows.begin(), p.At_rows.end()) != p.vec_len - 1)
        p.At_rows.back() = p.vec_len - 1;

    // y*, b = A(X*), C = A^T y* + S*
    p.y_star.resize(p.con_num);
    for (double &v : p.y_star)
        v = uniform(gen, -1.0, 1.0);
    p.b.assign(p.con_num, 0.0);
    p.C = p.S_star;
    for (size_t e = 0; e < p.At_vals.size(); e++)
    {
        p.b[p.At_cols[e]] += p.At_vals[e] * p.X_star[p.At_rows[e]];
        p.C[p.At_rows[e]] += p.At_vals[e] * p.y_star[p.At_cols[e]];
    }
    p.p_star = std::inner_product(p.b.begin(), p.b.end(), p.y_star.begin(), 0.0);

    return p;
}

// writes a file atomically (temporary file + rename), so that concurrent writers of the same
// deterministic content never expose a partially written file
inline void write_file_atomically(const std::string &filename, const std::function<void(std::ostream &)> &write)
{
    const std::string tmp = filename + ".tmp." + std::to_string(getpid());
    {
        std::ofstream file(tmp);
        if (!file.is_open())
            throw std::runtime_error("could not open " + tmp);
        file << std::setprecision(17);
        write(file);
        if (!file.good())
            throw std::runtime_error("could not write " + tmp);
    }
    std::filesystem::rename(tmp, filename);
}

/// @brief Writes the problem as a TXT directory (0-based indices, like examples/sedumi_to_txt.m).
inline void write_txt(const RegressionProblem &p, const std::string &dir)
{
    std::filesystem::create_directories(dir);
    write_file_atomically(dir + "/blk.txt", [&](std::ostream &os)
    {
        for (size_t k = 0; k < p.blk_types.size(); k++)
            os << p.blk_types[k] << " " << p.blk_sizes[k] << "\n";
    });
    write_file_atomically(dir + "/con_num.txt", [&](std::ostream &os)
    {
        os << p.con_num << "\n";
    });
    write_file_atomically(dir + "/At.txt", [&](std::ostream &os)
    {
        for (size_t e = 0; e < p.At_vals.size(); e++)
            os << p.At_rows[e] << " " << p.At_cols[e] << " " << p.At_vals[e] << "\n";
    });
    write_file_atomically(dir + "/b.txt", [&](std::ostream &os)
    {
        for (int i = 0; i < p.con_num; i++)
            if (p.b[i] != 0.0)
                os << i << " 0 " << p.b[i] << "\n";
    });
    write_file_atomically(dir + "/C.txt", [&](std::ostream &os)
    {
        for (int i = 0; i < p.vec_len; i++)
            if (p.C[i] != 0.0)
                os << i << " 0 " << p.C[i] << "\n";
    });
}

/// @brief Parameters of the named regression problems.
inline ProblemSpec problem_spec(const std::string &name)
{
    // {type, size, rank_X, group, interleave}
    if (name == "PSD")
        return {{{'s', 24, 12, 0, false}}, 0, 12, 1001};
    if (name == "Free")
        return {{{'u', 30, 0, 0, false}}, 15, 6, 1002};
    if (name == "PSD_Free")
        return {{{'s', 20, 10, 0, false}, {'u', 10, 0, 0, false}}, 0, 12, 1003};
    if (name == "Free_PSD")
        return {{{'u', 10, 0, 0, false}, {'s', 20, 10, 0, false}}, 0, 12, 1004};
    if (name == "PSD_Free_PSD")
        return {{{'s', 12, 6, 0, false}, {'u', 8, 0, 0, false}, {'s', 16, 8, 0, false}}, 0, 12, 1005};
    if (name == "Nonneg_Free_PSD")
        return {{{'l', 20, 10, 0, false}, {'u', 8, 0, 0, false}, {'s', 16, 8, 0, false}}, 0, 12, 1006};
    if (name == "Free_Free_PSD")
        return {{{'u', 6, 0, 0, false}, {'u', 9, 0, 0, false}, {'s', 16, 8, 0, false}}, 0, 12, 1007};
    if (name == "PSD_Nonneg")
        return {{{'s', 16, 8, 0, false}, {'l', 20, 10, 0, false}}, 0, 12, 1008};
    if (name == "SmallBatch_Free")
    {
        // six 8 x 8 blocks: batched Jacobi path
        std::vector<BlockSpec> blocks(6, {'s', 8, 4, 0, false});
        blocks.push_back({'u', 10, 0, 0, false});
        return {blocks, 0, 12, 1009};
    }
    if (name == "Medium_Free") // one 40 x 40 block: streamed cuSOLVER path
        return {{{'s', 40, 20, 0, false}, {'u', 10, 0, 0, false}}, 0, 12, 1010};
    // n = 1100 (large: full EVD or LOBPCG), Q made of 10 x 10 rotations
    if (name == "Large_LowRankX") // rank X* = 3: few negative eigenvalues, LOBPCG negative branch
        return {{{'s', 1100, 3, 10, false}}, 1000, 8, 1011};
    if (name == "Large_LowRankS") // rank S* = 3: few positive eigenvalues, LOBPCG positive branch
        return {{{'s', 1100, 1097, 10, false}}, 1000, 8, 1012};
    if (name == "Large_HalfRank") // rank X* = rank S* = 550: full EVD
        return {{{'s', 1100, 550, 10, true}}, 1000, 8, 1013};
    throw std::invalid_argument("unknown regression problem " + name);
}

/// @brief Generates the named regression problem.
inline RegressionProblem generate(const std::string &name)
{
    return generate(problem_spec(name));
}

} // namespace regression

#endif // CUADMM_TEST_REGRESSION_PROBLEMS_HPP
