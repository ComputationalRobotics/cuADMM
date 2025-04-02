/*

    kernels/vec_mat_conversion.cu

    Converts a dense vector to a dense matrix and vice versa. This uses the vectorized representation of symmetric matrices, where some coefficients are multiplied by the square root of 2 to preserve the scalar product.

*/

#include "cuadmm/kernels.h"

__global__ void vector_to_matrices_kernel(
    double* Xb_vals, double* mom_mat_vals, double* loc_mat_vals,
    int* map_B_vals, int* map_M1_vals, int* map_M2_vals,
    int vec_len
){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int b, m1, m2;
    bool if_diag;
    if (idx < vec_len) {
        b = map_B_vals[idx];
        m1 = map_M1_vals[idx];
        m2 = map_M2_vals[idx];
        if_diag = (m1 == m2);
        if (b == 0) {
            // if the coefficient is not on the diagonal, we need to multiply by 1/sqrt(2)
            mom_mat_vals[m1] = Xb_vals[idx] * (SQRT2INV + int(if_diag) * (1 - SQRT2INV));
            mom_mat_vals[m2] = mom_mat_vals[m1];
        } else {
            loc_mat_vals[m1] = Xb_vals[idx] * (SQRT2INV + int(if_diag) * (1 - SQRT2INV));
            loc_mat_vals[m2] = loc_mat_vals[m1];
        }
    }
    return;
}

__global__ void matrices_to_vector_kernel(
    double* Xb_vals, double* mom_mat_vals, double* loc_mat_vals,
    int* map_B_vals, int* map_M1_vals, int* map_M2_vals,
    int vec_len
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int b, m1, m2;
    bool if_diag;
    if (idx < vec_len) {
        b = map_B_vals[idx];
        m1 = map_M1_vals[idx];
        m2 = map_M2_vals[idx];
        if_diag = (m1 == m2);
        if (b == 0) { // from moment matrix
            // if the coefficient is not on the diagonal, we need to multiply by sqrt(2)
            Xb_vals[idx] = mom_mat_vals[m1] * (SQRT2 + int(if_diag) * (1 - SQRT2));
        } else { // from localizing matrix
            Xb_vals[idx] = loc_mat_vals[m1] * (SQRT2 + int(if_diag) * (1 - SQRT2));
        }
    }
    return;
}

// Convert the vector Xb to the matrices mom_mat and loc_mat using the mapping provided by map_B, map_M1, and map_M2.
// - map_B is used to determine which coefficients of the vector correspond to which elements of the matrices. If map_B[idx] is 0, the coefficient is used for the momentum matrix (mom_mat), otherwise it is used for the localizing matrix (loc_mat).
// - map_M1 is used to determine the row index of the coefficient in the matrices. For instance, if map_B[idx] is 0, the coefficient Xb[idx] is set to the one of moment matrix at index map_M1_vals[idx].
// - map_M2 is used to determine the column index of the coefficient in the matrices. If map_M1[idx] == map_M2[idx], the coefficient is on the diagonal, otherwise it is off-diagonal.
// The coefficients are multiplied by 1/sqrt(2) if they are off-diagonal, to preserve the scalar product.
void vector_to_matrices(
    DeviceDenseVector<double>& Xb, DeviceDenseVector<double>& mom_mat, DeviceDenseVector<double>& loc_mat,
    DeviceDenseVector<int>& map_B, DeviceDenseVector<int>& map_M1, DeviceDenseVector<int>& map_M2,
    const cudaStream_t& stream, int block_size
) {
    int vec_len = Xb.size;
    int num_block = (vec_len + block_size - 1) / block_size;
    vector_to_matrices_kernel<<<num_block, block_size, 0, stream>>>(
        Xb.vals, mom_mat.vals, loc_mat.vals,
        map_B.vals, map_M1.vals, map_M2.vals,
        vec_len
    );
    return;
}


// Convert the matrices mom_mat and loc_mat to the vector Xb using the mapping provided by map_B, map_M1, and map_M2.
// - map_B is used to determine which matrix to use for a coefficient of the vector. If map_B[idx] is 0, the coefficient used is the one from the moment matrix (mom_mat), otherwise it uses the one from the localizing matrix (loc_mat).
// - map_M1 is used to determine the position of the coefficient in the matrices. For instance, if map_B[idx] is 0, the coefficient Xb[idx] is set to the one of moment matrix at index map_M1_vals[idx].
// - map_M2 is used to determine if the coefficient is on the diagonal or off-diagonal. If map_M1[idx] == map_M2[idx], the coefficient is on the diagonal, otherwise it is off-diagonal.
// The coefficients are multiplied by sqrt(2) if they are off-diagonal, to compensate for the previous multiplication by 1/sqrt(2) when converting from vector to matrix.
void matrices_to_vector(
    DeviceDenseVector<double>& Xb, DeviceDenseVector<double>& mom_mat, DeviceDenseVector<double>& loc_mat,
    DeviceDenseVector<int>& map_B, DeviceDenseVector<int>& map_M1, DeviceDenseVector<int>& map_M2,
    const cudaStream_t& stream, int block_size
) {
    int vec_len = Xb.size;
    int num_block = (vec_len + block_size - 1) / block_size;
    matrices_to_vector_kernel<<<num_block, block_size, 0, stream>>>(
        Xb.vals, mom_mat.vals, loc_mat.vals,
        map_B.vals, map_M1.vals, map_M2.vals,
        vec_len
    );
    return;
}