/*

    bench/linsys_bench.cu

    Phase 7 (plan Section 11.6): the normal-equation backends on the systems A A^T + eps I of problem
    directories (the TXT format of main): the factorization time, the time per solve over `solves` solves with
    random right-hand sides, the relative residual of the last solve on the host, and the device / host memory
    the factorization holds, for

        cholmod       CholmodSolver with today's rule for the dense-factor variant (the default of the solver)
        cholmod_cpu   CholmodSolver without the variant (CHOLMOD solves on the host)
        dense         DenseCusolverSolver (cuSOLVER Cholesky on the device)
        cudss         CudssSolver (when compiled in)

    usage: cuadmm_linsys_bench <out.csv> <problem dir> [<problem dir> ...]  (--solves N, --eps X before the dirs)
    The CSV has one row per (problem, backend): problem,m,n,nnz,backend,kind,factor_s,ms_per_solve,rel_residual,
    device_bytes,host_bytes,nnz_L,error. docs/phase7_linsys_benchmark.md holds the table of the recorded run.

*/

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "core/check.h"
#include "core/device_buffer.h"
#include "cuadmm_legacy/problem.h"
#include "linsys/direct/cholmod/cholmod_solver.h"
#include "linsys/direct/dense/dense_cusolver_solver.h"
#ifdef CUADMM_HAVE_CUDSS
#include "linsys/direct/cudss/cudss_solver.h"
#endif
#include "linsys/normal_system.h"

namespace
{
struct Row
{
    std::string problem, backend, kind, error;
    int m = 0, n = 0;
    long long nnz = 0;
    double factor_s = 0.0, ms_per_solve = 0.0, rel_residual = 0.0, nnz_L = 0.0;
    std::size_t device_bytes = 0, host_bytes = 0;
};

double seconds_since(const std::chrono::steady_clock::time_point &t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

Row run_backend(const std::string &problem, const std::string &backend, cuadmm::linsys::NormalEquationSolver &solver,
                const cuadmm::linsys::NormalSystem &sys, int solves, std::mt19937 &gen)
{
    Row r;
    r.problem = problem;
    r.backend = backend;
    r.m = sys.m();
    r.n = sys.n();
    r.nnz = static_cast<long long>(sys.at_colptr()[sys.m()]);
    try
    {
        auto t0 = std::chrono::steady_clock::now();
        solver.analyze(sys);
        const cuadmm::linsys::FactorInfo f = solver.factorize(sys);
        r.factor_s = seconds_since(t0);
        r.kind = f.kind;
        r.device_bytes = solver.device_bytes();
        r.host_bytes = solver.host_bytes();
        if (backend.rfind("cholmod", 0) == 0)
            r.nnz_L = static_cast<cuadmm::linsys::CholmodSolver &>(solver).symbolic_info().nnz_L;

        const int m = sys.m();
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        std::vector<double> rhs(m), y(m), check(m);
        for (double &x : rhs)
            x = u(gen);
        cuadmm::core::DeviceBuffer<double> d_rhs(m), d_y(m);
        CUADMM_CHECK_CUDA(cudaMemcpy(d_rhs.data(), rhs.data(), sizeof(double) * m, cudaMemcpyHostToDevice));
        cuadmm::linsys::SolveRequest req;
        // a warm-up solve, then the timed ones
        solver.solve(d_rhs.data(), d_y.data(), m, req, nullptr);
        CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < solves; i++)
            solver.solve(d_rhs.data(), d_y.data(), m, req, nullptr);
        CUADMM_CHECK_CUDA(cudaDeviceSynchronize());
        r.ms_per_solve = 1000.0 * seconds_since(t0) / solves;
        CUADMM_CHECK_CUDA(cudaMemcpy(y.data(), d_y.data(), sizeof(double) * m, cudaMemcpyDeviceToHost));
        sys.apply_host(y.data(), check.data());
        double num = 0.0, den = 0.0;
        for (int i = 0; i < m; i++)
        {
            num += (check[i] - rhs[i]) * (check[i] - rhs[i]);
            den += rhs[i] * rhs[i];
        }
        r.rel_residual = std::sqrt(num) / std::sqrt(den);
    }
    catch (const std::exception &e)
    {
        r.error = e.what();
    }
    return r;
}
} // namespace

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s <out.csv> [--solves N] [--eps X] <problem dir> [<problem dir> ...]\n", argv[0]);
        return 2;
    }
    const std::string out = argv[1];
    int solves = 50;
    double eps = 1e-15;
    std::vector<std::string> dirs;
    for (int i = 2; i < argc; i++)
    {
        if (std::strcmp(argv[i], "--solves") == 0 && i + 1 < argc)
            solves = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--eps") == 0 && i + 1 < argc)
            eps = std::atof(argv[++i]);
        else
            dirs.push_back(argv[i]);
    }
    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    CUADMM_CHECK_CUBLAS(cublasCreate(&cublas));
    CUADMM_CHECK_CUBLAS(cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH));
    CUADMM_CHECK_CUSOLVER(cusolverDnCreate(&cusolver));
    std::ofstream csv(out);
    csv << "problem,m,n,nnz,backend,kind,factor_s,ms_per_solve,rel_residual,device_bytes,host_bytes,nnz_L,error\n";
    std::printf("cuADMM-reorg linsys benchmark, %d solves per backend, eps %g\n", solves, eps);
    std::printf("| problem | m | nnz(A) | backend | kind | factor s | ms / solve | rel. residual | device MiB |\n|---|---|---|---|---|---|---|---|---|\n");
    std::mt19937 gen(20261023);
    for (const std::string &dir : dirs)
    {
        Problem problem;
        std::string name = dir;
        while (!name.empty() && name.back() == '/')
            name.pop_back();
        name = name.substr(name.find_last_of('/') + 1);
        try
        {
            problem.from_txt(const_cast<std::string &>(dir));
        }
        catch (const std::exception &e)
        {
            std::printf("| %s | load failed: %s |\n", name.c_str(), e.what());
            continue;
        }
        const cuadmm::linsys::NormalSystem sys(problem.con_num, problem.vec_len, problem.At_csc_col_ptrs.data(), problem.At_csc_row_ids.data(),
                                               problem.At_csc_vals.data(), eps);
        std::vector<std::pair<std::string, std::unique_ptr<cuadmm::linsys::NormalEquationSolver>>> backends;
        backends.emplace_back("cholmod", std::unique_ptr<cuadmm::linsys::NormalEquationSolver>(new cuadmm::linsys::CholmodSolver(cublas)));
        cuadmm::linsys::CholmodSolver::Options cpu_only;
        cpu_only.dense_gpu_auto = false;
        backends.emplace_back("cholmod_cpu", std::unique_ptr<cuadmm::linsys::NormalEquationSolver>(new cuadmm::linsys::CholmodSolver(cublas, cpu_only)));
        cuadmm::linsys::DenseCusolverSolver::Options dense_options;
        dense_options.memory_fraction = 0.5;
        backends.emplace_back("dense", std::unique_ptr<cuadmm::linsys::NormalEquationSolver>(new cuadmm::linsys::DenseCusolverSolver(cusolver, dense_options)));
#ifdef CUADMM_HAVE_CUDSS
        cuadmm::linsys::CudssSolver::Options cudss_options;
        cudss_options.determinism_strict = true;
        backends.emplace_back("cudss", std::unique_ptr<cuadmm::linsys::NormalEquationSolver>(new cuadmm::linsys::CudssSolver(cudss_options)));
#endif
        for (auto &b : backends)
        {
            const Row r = run_backend(name, b.first, *b.second, sys, solves, gen);
            csv << r.problem << ',' << r.m << ',' << r.n << ',' << r.nnz << ',' << r.backend << ",\"" << r.kind << "\"," << r.factor_s << ',' << r.ms_per_solve << ','
                << r.rel_residual << ',' << r.device_bytes << ',' << r.host_bytes << ',' << r.nnz_L << ",\"" << r.error << "\"\n";
            csv.flush();
            if (r.error.empty())
                std::printf("| %s | %d | %lld | %s | %s | %.3f | %.3f | %.1e | %.1f |\n", r.problem.c_str(), r.m, r.nnz, r.backend.c_str(), r.kind.c_str(), r.factor_s,
                            r.ms_per_solve, r.rel_residual, r.device_bytes / 1048576.0);
            else
                std::printf("| %s | %d | %lld | %s | failed: %s | | | | |\n", r.problem.c_str(), r.m, r.nnz, r.backend.c_str(), r.error.c_str());
            std::fflush(stdout);
            b.second.reset(); // free the factorization before the next backend
        }
    }
    CUADMM_CHECK_CUSOLVER_NOTHROW(cusolverDnDestroy(cusolver));
    CUADMM_CHECK_CUBLAS_NOTHROW(cublasDestroy(cublas));
    return 0;
}
