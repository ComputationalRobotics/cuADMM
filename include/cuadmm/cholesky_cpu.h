/*

    cholesky_cpu.h

    A wrapper for the linear system solver that uses the Cholesky
    decomposition to store AA^T. Based on CHOLDMOD from SuiteSparse.

*/

#ifndef CUADMM_CHOLESKY_CPU_H
#define CUADMM_CHOLESKY_CPU_H

#include <cholmod.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
// #include <cblas.h>

// A wrapper for the Cholesky decomposition and forward/bakward
// linear system solver. Based on CHOLDMOD from SuiteSparse.
class CholeskySolverCPU
{
public:
    cholmod_common cc;
    cholmod_sparse *chol_sp_A;   // A matrix in sparse format
    cholmod_sparse *chol_sp_AAt; // AAt matrix in sparse format
    cholmod_factor *chol_fac_L;  // L matrix in factorized format
    cholmod_dense *chol_dn_rhs;  // rhs vector in dense format
    cholmod_dense *chol_dn_res;  // res vector in dense format
    int A_row_size;              // number of rows in A
    int A_col_size;              // number of columns in A
    int AAt_size;                // number of rows=columns in AAt
    int A_nnz;                   // number of non-zero elements in A
    cholmod_dense *Y;            // first workspace
    cholmod_dense *E;            // second workspace
    bool supernodal_used;        // whether factorize() used the supernodal method (see factorize())
    double min_pivot;            // |
    double max_pivot;            // |- pivots D_jj of the LDL^T factor; tiny (|D_jj| < 1e-12 max |D|) or
    int tiny_pivots;             // |  nonpositive pivots mean that A does not have full row rank
    int nonpositive_pivots;      // |

    CholeskySolverCPU() : chol_sp_A(nullptr), chol_sp_AAt(nullptr), chol_fac_L(nullptr),
                          chol_dn_rhs(nullptr), chol_dn_res(nullptr),
                          A_row_size(0), A_col_size(0), AAt_size(0), A_nnz(0),
                          Y(nullptr), E(nullptr), supernodal_used(false),
                          min_pivot(0), max_pivot(0), tiny_pivots(0), nonpositive_pivots(0)
    {
        cholmod_start(&this->cc);
        this->cc.final_asis = false;
        this->cc.final_super = false;
    }

    CholeskySolverCPU(
        int *A_col_ptrs, int *A_row_ids, double *A_vals,
        const int A_row_size, const int A_col_size, const int A_nnz,
        bool if_from_device = false,
        double eps = 1e-16) : chol_fac_L(nullptr), Y(nullptr), E(nullptr), supernodal_used(false),
                              min_pivot(0), max_pivot(0), tiny_pivots(0), nonpositive_pivots(0)
    {
        cholmod_start(&this->cc);
        this->cc.final_asis = false;
        this->cc.final_super = false;

        this->get_A(A_col_ptrs, A_row_ids, A_vals, A_row_size, A_col_size, A_nnz, if_from_device, eps);
        this->factorize();
    }

    // Copy the A matrix in CSC to the solver storage, and computes AAt.
    // If if_from_device is true, the data is copied from device to host.
    inline void get_A(
        int *A_col_ptrs, int *A_row_ids, double *A_vals,
        const int A_row_size, const int A_col_size, const int A_nnz,
        bool if_from_device = false, double eps = 1e-16)
    {
        this->A_row_size = A_row_size;
        this->A_col_size = A_col_size;
        this->A_nnz = A_nnz;

        // Allocate the sparse matrix
        this->chol_sp_A = cholmod_allocate_sparse(
            this->A_row_size, this->A_col_size, this->A_nnz, 1, 1,
            0, CHOLMOD_DOUBLE + CHOLMOD_REAL, &this->cc);
        // Copy the data to the sparse matrix (from GPU or CPU)
        if (if_from_device)
        {
            CHECK_CUDA(cudaMemcpy(this->chol_sp_A->p, A_col_ptrs, sizeof(int) * (this->A_col_size + 1), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(this->chol_sp_A->i, A_row_ids, sizeof(int) * this->A_nnz, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(this->chol_sp_A->x, A_vals, sizeof(double) * this->A_nnz, cudaMemcpyDeviceToHost));
        }
        else
        {
            memcpy(this->chol_sp_A->p, A_col_ptrs, sizeof(int) * (this->A_col_size + 1));
            memcpy(this->chol_sp_A->i, A_row_ids, sizeof(int) * this->A_nnz);
            memcpy(this->chol_sp_A->x, A_vals, sizeof(double) * this->A_nnz);
        }

        // Compute the AAt matrix:
        // AAt <-- A * At
        cholmod_sparse *chol_sp_AAt_1 = cholmod_aat(
            this->chol_sp_A, NULL, 0, 1, &this->cc);
        chol_sp_AAt_1->stype = 1;
        this->AAt_size = this->A_row_size;

        // Stabilize the system by adding a small value to the diagonal:
        // AAt <-- 1.0 * AAt + eps * I
        double alpha = 1.0;
        cholmod_sparse *chol_sp_I = cholmod_speye(this->AAt_size, this->AAt_size, CHOLMOD_DOUBLE + CHOLMOD_REAL, &this->cc);
        cholmod_sparse *chol_sp_AAt_2 = cholmod_add(
            chol_sp_AAt_1, chol_sp_I, &alpha, &eps, 1, 1, &this->cc);
        this->chol_sp_AAt = cholmod_copy(chol_sp_AAt_2, 1, 1, &this->cc);
        this->chol_sp_AAt->dtype = CHOLMOD_DOUBLE;
        this->chol_sp_AAt->xtype = CHOLMOD_REAL;
        this->chol_sp_AAt->itype = CHOLMOD_INT;

        // Allocate the dense vectors for the rhs and res
        chol_dn_rhs = cholmod_allocate_dense(
            this->A_row_size, 1, this->A_row_size, CHOLMOD_REAL, &this->cc);
        chol_dn_res = cholmod_allocate_dense(
            this->A_row_size, 1, this->A_row_size, CHOLMOD_REAL, &this->cc);

        this->Y = NULL;
        this->E = NULL;

        // Free the temporary matrices
        cholmod_free_sparse(&chol_sp_AAt_1, &this->cc);
        cholmod_free_sparse(&chol_sp_AAt_2, &this->cc);
        cholmod_free_sparse(&chol_sp_I, &this->cc);

        return;
    }

    // Factorize the AAt=AA^T matrix.
    // This function should be called after get_A() and before solve().
    // CHOLMOD chooses the method (CHOLMOD_AUTO): supernodal LL^T with BLAS-3 when the factor is dense enough
    // (70x faster on neosfrbr25), simplicial otherwise. The factor is always converted to the simplicial
    // LDL^T form (final_super = final_ll = false). The supernodal method needs eps*I + AA^T to be numerically
    // positive definite; when A does not have full row rank (the SPOT moment relaxations, for example) it fails
    // and the simplicial LDL^T factorization is used, which tolerates the tiny (even slightly negative) pivots:
    // the right-hand sides of the ADMM steps are in range(A).
    inline void factorize()
    {
        cholmod_free_factor(&this->chol_fac_L, &this->cc);
        this->cc.supernodal = CHOLMOD_AUTO;
        this->chol_fac_L = cholmod_analyze(this->chol_sp_AAt, &this->cc);
        this->supernodal_used = this->chol_fac_L != nullptr && this->chol_fac_L->is_super;
        if (this->supernodal_used)
        {
            const int print_level = this->cc.print;
            this->cc.print = 0; // no "not positive definite" message (printed as an error at level 1): handled below
            cholmod_factorize(this->chol_sp_AAt, this->chol_fac_L, &this->cc);
            this->cc.print = print_level;
            if (this->cc.status != CHOLMOD_OK) // not positive definite (or out of memory for the supernodes, ...)
            {
                cholmod_free_factor(&this->chol_fac_L, &this->cc);
                this->supernodal_used = false;
            }
        }
        if (!this->supernodal_used)
        {
            cholmod_free_factor(&this->chol_fac_L, &this->cc);
            this->cc.status = CHOLMOD_OK;
            this->cc.supernodal = CHOLMOD_SIMPLICIAL;
            this->chol_fac_L = cholmod_analyze(this->chol_sp_AAt, &this->cc);
            if (this->chol_fac_L != nullptr)
                cholmod_factorize(this->chol_sp_AAt, this->chol_fac_L, &this->cc);
        }
        if (this->chol_fac_L == nullptr || this->cc.status != CHOLMOD_OK)
        {
            throw std::runtime_error(
                "CHOLMOD factorization of eps*I + AA^T failed (status " + std::to_string(this->cc.status) +
                "): A may not have full row rank");
        }
        this->pivot_statistics();
    }

    // Computes min_pivot, max_pivot, tiny_pivots and nonpositive_pivots from the simplicial LDL^T factor
    // (D_jj is the first entry of column j of L).
    inline void pivot_statistics()
    {
        this->min_pivot = 0.0;
        this->max_pivot = 0.0;
        this->tiny_pivots = 0;
        this->nonpositive_pivots = 0;
        if (this->chol_fac_L == nullptr || this->chol_fac_L->is_super || this->chol_fac_L->is_ll || this->AAt_size == 0)
            return;
        const int *Lp = (const int *)this->chol_fac_L->p;
        const double *Lx = (const double *)this->chol_fac_L->x;
        double max_abs = 0.0;
        this->min_pivot = Lx[Lp[0]];
        this->max_pivot = Lx[Lp[0]];
        for (int j = 0; j < this->AAt_size; j++)
        {
            const double d = Lx[Lp[j]];
            this->min_pivot = std::min(this->min_pivot, d);
            this->max_pivot = std::max(this->max_pivot, d);
            max_abs = std::max(max_abs, std::abs(d));
            if (d <= 0.0)
                this->nonpositive_pivots++;
        }
        for (int j = 0; j < this->AAt_size; j++)
            if (std::abs(Lx[Lp[j]]) < 1e-12 * max_abs)
                this->tiny_pivots++;
    }

    // Solve the linear system AA^T * x = b, where b is stored in this->chol_dn_rhs.
    // The result will be stored in this->chol_dn_res.
    // This function should be called after factorize().
    inline void solve()
    {
        // here we suppose rhs data already stored in this->chol_dn_rhs
        // after this function call, we need to fetch the result in this->chol_dn_res manually
        cholmod_solve2(
            CHOLMOD_LDLt, this->chol_fac_L, this->chol_dn_rhs, NULL, // input
            &this->chol_dn_res, NULL,                                // ouput
            &this->Y, &this->E, &this->cc                            // workspace and common
        );
        return;
    }

    // Compute the norm of the `rhs` vector.
    // inline double get_rhs_norm() {
    //     return cblas_dnrm2(this->AAt_size, (double*) this->chol_dn_rhs->x, 1);
    // }

    // Compute the norm of the `res` vector.
    // inline double get_res_norm() {
    //     return cblas_dnrm2(this->AAt_size, (double*) this->chol_dn_res->x, 1);
    // }

    ~CholeskySolverCPU()
    {
        cholmod_free_sparse(&this->chol_sp_AAt, &this->cc);
        cholmod_free_factor(&this->chol_fac_L, &this->cc);
        cholmod_free_dense(&this->chol_dn_rhs, &this->cc);
        cholmod_free_dense(&this->chol_dn_res, &this->cc);
        cholmod_free_dense(&this->Y, &this->cc);
        cholmod_free_dense(&this->E, &this->cc);
        cholmod_finish(&this->cc);
        // std::cout << "CholeskySolverCPU destructor called!" << std::endl;
    }
};

#endif // CUADMM_CHOLESKY_CPU_H