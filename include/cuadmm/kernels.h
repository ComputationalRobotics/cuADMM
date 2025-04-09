#ifndef CUADMM_KERNELS_H
#define CUADMM_KERNELS_H

#include "cuadmm/memory.h"

/*
    Dense-dense operations (kernels/dense_dense.cu)
*/

// vec1 = alpha * vec1 + beta * vec2
void dense_vector_add_dense_vector(
    DeviceDenseVector<double>& vec1, const DeviceDenseVector<double>& vec2,
    double alpha = 1.0, double beta = 1.0,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

// vec1 = alpha * vec2 + beta * vec3
void dense_vector_add_dense_vector(
    DeviceDenseVector<double>& vec1, const DeviceDenseVector<double>& vec2, DeviceDenseVector<double>& vec3,
    double alpha = 1.0, double beta = 1.0,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

// vec1 <-- vec2 + vec3 * scalar
void dense_vector_plus_dense_vector_mul_scalar(
    DeviceDenseVector<double>& vec1, const DeviceDenseVector<double>& vec2, const DeviceDenseVector<double>& vec3,
    const double scalar,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

// vec1 *= vec2
void dense_vector_mul_dense_vector(
    DeviceDenseVector<double>& vec1, const DeviceDenseVector<double>& vec2,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

// vec1 <-- vec2 * vec3 * scalar
void dense_vector_mul_dense_vector_mul_scalar(
    DeviceDenseVector<double>& vec1, const DeviceDenseVector<double>& vec2, const DeviceDenseVector<double>& vec3,
    const double scalar,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

// vec1 <-- vec1 / vec2 * scalar
void dense_vector_div_dense_vector_mul_scalar(
    DeviceDenseVector<double>& vec1, const DeviceDenseVector<double>& vec2,
    const double scalar,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);


/*
    Dense-scalar operations (kernels/dense_scalar.cu)
*/

// Multiply in place a vector by a scalar:
// vec *= scalar
void dense_vector_mul_scalar(DeviceDenseVector<double>& vec, double scalar, const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024);

// Divide in place a vector by a scalar:
// vec *= scalar
void dense_vector_div_scalar(DeviceDenseVector<double>& vec, double scalar, const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024);

// Multiply a vector by a scalar and store the result in another vector:
// vec1 = vec2 * scalar
void dense_vector_mul_scalar(
    DeviceDenseVector<double>& vec1, DeviceDenseVector<double>& vec2, double scalar,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

// Set a vector to its positive part coefficient-wise:
// vec = max(vec, 0)
void max_dense_vector_zero(
    DeviceDenseVector<double>& vec,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

// Set a vector to its positive part coefficient-wise and multiply by a mask:
// vec = max(vec, 0) .* mask
void max_dense_vector_zero_mask(
    DeviceDenseVector<double>& vec, DeviceDenseVector<int>& mask,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);


/*
    Sparse-scalar operations (kernels/sparse_scalar.cu)
*/

// Multiply in place a sparse vector by a scalar:
// vec *= scalar
void sparse_vector_mul_scalar(
    DeviceSparseVector<double>& vec, double scalar,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

// Divide in place a sparse vector by a scalar:
// vec /= scalar
void sparse_vector_div_scalar(
    DeviceSparseVector<double>& vec, double scalar,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);


/*
    Sparse-dense operations (kernels/sparse_dense.cu)
*/

// Divide a sparse vector by a dense vector, element-wise and in-place
// sp_vec <-- sp_vec / dn_vec
void sparse_vector_div_dense_vector(
    DeviceSparseVector<double>& spvec, const DeviceDenseVector<double>& dnvec,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);


/*
    Inverse permutation (kernels/inverse_permutation.cu)
*/

// Perform a permutation on a dense vector.
// vec1 <-- vec2[perm]
void perform_permutation(
    DeviceDenseVector<double>& vec1,
    const DeviceDenseVector<double>& vec2,
    const DeviceDenseVector<int>& perm,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

/*
    Sparse matrix operations (kernels/sparse_matrix.cu)
*/

// Compute the norm (of size (con_num, 1)) of a CSC matrix At and normalize it.
void get_normA(
    DeviceSparseMatrixDoubleCSC& At, DeviceDenseVector<double>& normA,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

// Computes the multiplication of a dense matrix by a batch of dense vectors as diagonal matrices:
// mat1[i] = mat2[i] * diag(vec[i])
void dense_matrix_mul_diag_batch(
    DeviceDenseVector<double>& dnmat1,
    const DeviceDenseVector<double>& dnmat2,
    const DeviceDenseVector<double>& dnvec,
    const int mat_size,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

/*
    Vector-matrices conversion (kernels/vec_mat_conversion.cu)
*/

// Compute the square root of 2 using the Newton-Raphson method.
// This is a constexpr function, so it can be evaluated at compile time.
double constexpr sqrtNewtonRaphson(double x, double curr, double prev)
    {
        return curr == prev
            ? curr
            : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
    }

constexpr double SQRT2 = sqrtNewtonRaphson(2.0, 2.0, 0.0);
constexpr double SQRT2INV = 1.0/SQRT2;

// Convert the vector Xb to the matrices mom_mat and loc_mat using the mapping provided by map_B, map_M1, and map_M2.
void vector_to_matrices(
    DeviceDenseVector<double>& Xb, DeviceDenseVector<double>& mom_mat, DeviceDenseVector<double>& loc_mat,
    DeviceDenseVector<int>& map_B, DeviceDenseVector<int>& map_M1, DeviceDenseVector<int>& map_M2,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

// Convert the matrices mom_mat and loc_mat to the vector Xb using the mapping provided by map_B, map_M1, and map_M2.
void matrices_to_vector(
    DeviceDenseVector<double>& Xb, DeviceDenseVector<double>& mom_mat, DeviceDenseVector<double>& loc_mat,
    DeviceDenseVector<int>& map_B, DeviceDenseVector<int>& map_M1, DeviceDenseVector<int>& map_M2,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

/* 
    Type conversion (kernels/type_conversion.cu)
*/

// Convert a dense vector of type int to a dense vector of type long int.
void long_int_to_int(
    DeviceDenseVector<int>& vec_int, const DeviceDenseVector<size_t>& vec_long_int,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

// Convert a dense vector of type long int to a dense vector of type int.
void int_to_long_int(
    DeviceDenseVector<size_t>& vec_long_int, const DeviceDenseVector<int>& vec_int,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

// Convert a dense vector of type double to a dense vector of type float.
void double_to_float(
    DeviceDenseVector<float>& vec_float, const DeviceDenseVector<double>& vec_double,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

// Convert a dense vector of type float to a dense vector of type double.
void float_to_double(
    DeviceDenseVector<double>& vec_double, const DeviceDenseVector<float>& vec_float,
    const cudaStream_t& stream = (cudaStream_t) 0, int block_size = 1024
);

#endif