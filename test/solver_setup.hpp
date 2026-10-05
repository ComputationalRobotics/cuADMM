/*

    solver_setup.hpp

    Helpers shared by the GPU tests that run SDPSolver on generated problems (test/regression_problems.hpp).

*/

#ifndef CUADMM_TEST_SOLVER_SETUP_HPP
#define CUADMM_TEST_SOLVER_SETUP_HPP

#include <vector>

#include "cuadmm/solver.h"
#include "cuadmm/io.h"
#include "regression_problems.hpp"

namespace solver_setup
{

// Initializes the solver on a generated problem (A given as COO triplets of At).
inline void init_solver(const regression::RegressionProblem &p, SDPSolver &solver, bool use_lobpcg = true, double sig = 1e2)
{
    std::vector<int> rows = p.At_rows, cols = p.At_cols, col_ptrs;
    std::vector<double> vals = p.At_vals;
    COO_to_CSC(col_ptrs, cols, rows, vals, (int)vals.size(), p.con_num);

    std::vector<int> b_idx, C_idx;
    std::vector<double> b_val, C_val;
    for (int i = 0; i < p.con_num; i++)
        if (p.b[i] != 0.0)
        {
            b_idx.push_back(i);
            b_val.push_back(p.b[i]);
        }
    for (int i = 0; i < p.vec_len; i++)
        if (p.C[i] != 0.0)
        {
            C_idx.push_back(i);
            C_val.push_back(p.C[i]);
        }
    std::vector<char> types = p.blk_types;
    std::vector<int> sizes = p.blk_sizes;
    solver.init(
        15, p.vec_len, p.con_num,
        col_ptrs.data(), rows.data(), vals.data(), (int)vals.size(),
        b_idx.data(), b_val.data(), (int)b_idx.size(),
        C_idx.data(), C_val.data(), (int)C_idx.size(),
        types.data(), sizes.data(), (int)types.size(),
        ProjectionMethod::EIG_FP64, ProjectionMethod::EIG_FP64,
        nullptr, nullptr, nullptr, sig, use_lobpcg);
}

} // namespace solver_setup

#endif // CUADMM_TEST_SOLVER_SETUP_HPP
