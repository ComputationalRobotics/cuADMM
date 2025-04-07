/*

    eig_cpu.h

    Compute the eigenvalues and eigenvectors of a symmetric matrix using LAPACK. (on CPU)

*/

#ifndef CUADMM_EIG_CPU_H
#define CUADMM_EIG_CPU_H

#include <lapack.h>

#include "cuadmm/memory.h"

// Compute the eigenvalues and eigenvectors of a symmetric matrix using LAPACK.
// Parameters:
// - mat: the input matrix, of size (LDA, N), given as the upper triangle. On exit, mat contains the orthonormal eigenvectors.
// - W: the output eigenvalues (in ascending order)
// - workspace: workspace array for LAPACK
// - workspace2: second workspace array for LAPACK
// - info: array to store the output info from LAPACK
// - mat_size: size of the matrix
// - lwork: size of the workspace array
// - lwork2: size of the second workspace array
// - mat_offset: offset for the matrix
// - W_offset: offset for the eigenvalues
// - workspace_offset: offset for the workspace
// - workspace2_offset: offset for the second workspace
// - info_offset: offset for the info array
inline void single_eig_lapack(
    HostDenseVector<double>& mat, HostDenseVector<double>& W,
    HostDenseVector<double>& workspace, HostDenseVector<ptrdiff_t>& workspace2, HostDenseVector<ptrdiff_t>& info,
    const ptrdiff_t mat_size, const ptrdiff_t lwork, const ptrdiff_t lwork2, 
    const int mat_offset = 0, const int W_offset = 0, 
    const int workspace_offset = 0, const int workspace2_offset = 0, const int info_offset = 0
) {
    const char jobz = 'V'; // compute eigenvalues and eigenvectors
    const char uplo = 'U'; // upper triangle of mat is provided

    // LAPACK function to compute eigenvalues and eigenvectors
    dsyevd(
        &jobz, &uplo, 
        &mat_size, mat.vals + mat_offset, 
        &mat_size, W.vals + W_offset,
        workspace.vals + workspace_offset, &lwork,
        workspace2.vals + workspace2_offset, &lwork2,
        info.vals + info_offset
    );
    return;
}


#endif // CUADMM_EIG_CPU_H