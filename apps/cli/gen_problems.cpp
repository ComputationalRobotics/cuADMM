/*

    gen_problems.cpp

    Writes the known-optimum generator problems of test/regression_problems.hpp as TXT directories, so that the
    reference solvers (which read only TXT) and the new build can be run on identical data.

    usage: cuadmm_gen_problems <output directory> [problem names...]
    Without names, all problems are written. Each problem goes to <output directory>/<name>/ with blk.txt,
    con_num.txt, At.txt, b.txt, C.txt (17 significant digits) and optimum.txt (p*, one line).

*/

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "regression_problems.hpp"

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <output directory> [problem names...]\n";
        return 1;
    }
    const std::string out = argv[1];
    std::vector<std::string> names;
    for (int i = 2; i < argc; i++)
        names.push_back(argv[i]);
    if (names.empty())
        names = {"PSD", "Free", "PSD_Free", "Free_PSD", "PSD_Free_PSD", "Nonneg_Free_PSD", "Free_Free_PSD", "PSD_Nonneg",
                 "SmallBatch_Free", "Medium_Free", "Large_LowRankX", "Large_LowRankS", "Large_HalfRank"};
    try
    {
        for (const std::string &name : names)
        {
            const regression::RegressionProblem p = regression::generate(name);
            const std::string dir = out + "/" + name;
            regression::write_txt(p, dir);
            std::ofstream opt(dir + "/optimum.txt");
            opt << std::setprecision(17) << p.p_star << "\n";
            std::printf("%s: vec_len %d, con_num %d, nnz(At) %zu, p* = %.17g\n", name.c_str(), p.vec_len, p.con_num,
                        p.At_vals.size(), p.p_star);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
