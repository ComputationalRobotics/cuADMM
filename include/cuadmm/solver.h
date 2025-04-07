#ifndef CUADMM_SOLVER_H
#define CUADMM_SOLVER_H

#include "cuadmm/check.h"
#include "cuadmm/eig_cpu.h"
#include "cuadmm/io.h"
#include "cuadmm/kernels.h"
#include "cuadmm/memory.h"
#include "cuadmm/utils.h"

// Main solver class for the SDP problem.
// Uses the sGS-ADMM algorithm to solve the problem:
//      min(X) <C, X>  s.t. A(X) = b,       X >= 0,
// of dual:
//   max(y, S) <b, y>  s.t. At(y) + S = C,  S >= 0
class SDPSolver {
    public:
        // problem data
        DeviceSparseMatrixDoubleCSC At_csc; // |
        DeviceSparseMatrixDoubleCSR At_csr; // | constraint matrix
        DeviceSparseMatrixDoubleCSR A_csr;  // |
        DeviceSparseVector<double> b;       // constraint vector
        DeviceSparseVector<double> C;       // cost matrix
        DeviceDenseVector<double> blk;      // block sizes
        DeviceDenseVector<double> X;        // primal variable
        DeviceDenseVector<double> S;        // dual variable 2
        DeviceDenseVector<double> y;        // dual variable 1
        
        // hyperparameters
        double sigma; // Lagrangian penalty
        int vec_len;  // length of X in vector form
        int con_num;  // number of constraints (length of y)

        // scaling
        DeviceDenseVector<double> normA; // norm of A
        DeviceDenseVector<double> borg;
        DeviceDenseVector<double> Corg;

        // KKT residual
        DeviceDenseVector<double> Aty;
        DeviceDenseVector<double> Rp;
        DeviceDenseVector<double> SmC;
        DeviceDenseVector<double> Rd;
        DeviceDenseVector<double> Rporg;
        DeviceDenseVector<double> Rdorg;
        double norm_borg;
        double norm_Corg;
        double bscale;
        double Cscale;
        double objscale;
        double errRp;
        double errRd;
        double axfeas;
        double pobj;
        double dobj;
        double relgap;
        size_t SparseMV_Aty_buffer_size;
        DeviceDenseVector<double> SparseMV_Aty_buffer;
        size_t SparseMV_AX_buffer_size;
        DeviceDenseVector<double> SparseMV_AX_buffer;
        size_t SparseVV_CtX_buffer_size;
        DeviceDenseVector<double> SparseVV_CtX_buffer;
        size_t SparseVV_bty_buffer_size;
        DeviceDenseVector<double> SparseVV_bty_buffer;

        // Cholesky decomposition on CPU
        
};

#endif // CUADMM_SOLVER_H