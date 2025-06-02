/*

    utils/projection.cu

    Defines the PSD cone projection of large matrices.

*/

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

#include "cuadmm/utils.h"
#include "cuadmm/projection.h"

// Print an n×n matrix of doubles
void printMatrixDouble(const double* dM, int n) {
    size_t N = size_t(n)*n;
    std::vector<double> hM(N);
    CHECK_CUDA(cudaMemcpy(hM.data(), dM, N*sizeof(double), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setprecision(6)
                      << hM[i*n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Print an n×n matrix of floats
void printMatrixFloat(const float* dM, int n) {
    size_t N = size_t(n)*n;
    std::vector<float> hM(N);
    CHECK_CUDA(cudaMemcpy(hM.data(), dM, N*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setprecision(6)
                      << hM[i*n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Print an n×n matrix of __half
void printMatrixHalf(const __half* dM, int n) {
    size_t N = size_t(n)*n;
    std::vector<__half> hM(N);
    CHECK_CUDA(cudaMemcpy(hM.data(), dM, N*sizeof(__half), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            float v = __half2float(hM[i*n + j]);
            std::cout << std::fixed << std::setprecision(6)
                      << v << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Helper kernel: convert float array -> half array
__global__ void float2half_kernel(const float* A, __half* B, size_t N) {
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) B[i] = __float2half_rn(A[i]);
}

// Kernel: each thread handles 4 floats at once, writing 4 halves
__global__ void float4_to_half_kernel(
    const float4* __restrict__ A4,
    __half2 * __restrict__ B2,
    size_t N4
) {
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N4) return;

    // load 4 floats
    float4 v = A4[idx];

    // pack low two floats into half2
    B2[2*idx + 0] = __float22half2_rn(make_float2(v.x, v.y));
    // pack high two floats into half2
    B2[2*idx + 1] = __float22half2_rn(make_float2(v.z, v.w));
}

void convertFloatToHalf4(const float* dA, __half* dB, size_t N) {
    size_t N4 = (N + 3)/4;  // how many float4’s
    auto A4 = reinterpret_cast<const float4*>(dA);
    auto B2 = reinterpret_cast<__half2*>(dB);

    const int blk = 1024;
    int grid = (N4 + blk - 1)/blk;
    float4_to_half_kernel<<<grid,blk>>>(A4, B2, N4);
    // cudaDeviceSynchronize();
}

void projection_TF16(
    cublasHandle_t cublasH,
    DeviceDenseVector<double>& mat,
    const int n,
    const int mat_offset,
    const int n_iter
) {
    const int nn = n * n;

    // copy the double matrix back to the host
    std::vector<double> A_h_d(nn);
    CHECK_CUDA( cudaMemcpy(A_h_d.data(), mat.vals + mat_offset, nn * sizeof(double), D2H) );

    // convert the host matrix to float
    std::vector<float> A_h(nn);
    for (int i = 0; i < nn; i++) {
        A_h[i] = static_cast<float>(A_h_d[i]);
    }

    // allocate the device buffers
    float *dA_orig, *dA_our, *dTmp, *dI, *dT1, *dT2, *dF, *dDiff;
    CHECK_CUDA( cudaMalloc(&dA_orig, nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&dA_our,  nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&dTmp,    nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&dI,      nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&dT1,     nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&dT2,     nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&dF,      nn * sizeof(float)) );
    CHECK_CUDA( cudaMalloc(&dDiff,   nn * sizeof(float)) );

    // copy the float host matrix to the device
    CHECK_CUDA( cudaMemcpy(dA_orig, A_h.data(), nn * sizeof(float), H2D) );
    CHECK_CUDA( cudaMemcpy(dA_our,  A_h.data(), nn * sizeof(float), H2D) );

    // build identity I on device
    std::vector<float> I_h(nn, 0.0f);
    for (int i = 0; i < n; i++) I_h[i*n + i] = 1.0f;
    CHECK_CUDA( cudaMemcpy(dI, I_h.data(), nn * sizeof(float), H2D) );

    // half buffers
    __half *dT3_half, *dT4_half;
    CHECK_CUDA( cudaMalloc(&dT3_half, nn*sizeof(__half)) );
    CHECK_CUDA( cudaMalloc(&dT4_half, nn*sizeof(__half)) );

    const float one = 1.0f, zero = 0.0f, neg1 = -1.0f;
    float half = 0.5f;

    // iterative algorithm in float
    // auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 1; iter <= n_iter; iter++) {
        // T1 = A_our * A_our
        // float2half_kernel<<<blocks,threads>>>(dA_our, dT3_half, nn);
        convertFloatToHalf4(dA_our, dT3_half, nn);
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT3_half, CUDA_R_16F, n,
            &zero,
            dT1,      CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
        );

        // CHECK_CUDA( cudaDeviceSynchronize() );
        // printMatrixHalf(dT3_half, n);
        // printMatrixFloat(dT1, n);

        // T2 = I - T1
        CHECK_CUBLAS( cublasSgeam(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n,
            &one,  dI,  n,
            &neg1, dT1, n,
            dT2,       n) );

        // T1 = T2 * T2
        // float2half_kernel<<<blocks,threads>>>(dT2, dT3_half, nn);
        convertFloatToHalf4(dT2, dT3_half, nn);
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT3_half, CUDA_R_16F, n,
            &zero,
            dT1,      CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printMatrixHalf(dT3_half, n);
        // printMatrixFloat(dT1, n);

        // F = I + log(iter+10)*T1
        float logv = std::log(iter + 10.0f);
        CHECK_CUBLAS( cublasSgeam(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n,
            &one,  dI,  n,
            &logv, dT1, n,
            dF,      n) );

        // CUDA_CHECK(cudaDeviceSynchronize());
        // printMatrixFloat(dF, n);

        // A_our = A_our * F (to dTmp, then copy back)
        // float2half_kernel<<<blocks,threads>>>(dA_our, dT3_half, nn);
        // float2half_kernel<<<blocks,threads>>>(dF, dT4_half, nn);
        convertFloatToHalf4(dA_our, dT3_half, nn);
        convertFloatToHalf4(dF, dT4_half, nn);
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT4_half, CUDA_R_16F, n,
            &zero,
            dTmp,     CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

        // CUDA_CHECK(cudaDeviceSynchronize());
        // printMatrixHalf(dT3_half, n);
        // printMatrixHalf(dT4_half, n);
        // printMatrixFloat(dTmp, n);

        CHECK_CUDA( cudaMemcpy(dA_our, dTmp, nn*sizeof(float), cudaMemcpyDeviceToDevice) );
        // T1 = A_our^2, T2 = I - T1
        // float2half_kernel<<<blocks,threads>>>(dA_our, dT3_half, nn);
        convertFloatToHalf4(dA_our, dT3_half, nn);
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT3_half, CUDA_R_16F, n,
            &zero,
            dT1,      CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

        // CUDA_CHECK(cudaDeviceSynchronize());
        // printMatrixHalf(dT3_half, n);
        // printMatrixFloat(dT1, n);

        CHECK_CUBLAS( cublasSgeam(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n,
            &one,  dI,  n,
            &neg1, dT1, n,
            dT2,       n) );
        // F = I + (1/sqrt(iter))*T2
        float invs = 1.0f / std::sqrt((float)iter);
        CHECK_CUBLAS( cublasSgeam(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n,
            &one,  dI,  n,
            &invs, dT2, n,
            dF,      n) );

        // A_our = A_our * F (to dTmp)
        // float2half_kernel<<<blocks,threads>>>(dA_our, dT3_half, nn);
        // float2half_kernel<<<blocks,threads>>>(dF, dT4_half, nn);
        convertFloatToHalf4(dA_our, dT3_half, nn);
        convertFloatToHalf4(dF, dT4_half, nn);
        CHECK_CUBLAS( cublasGemmEx(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &one,
            dT3_half, CUDA_R_16F, n,
            dT4_half, CUDA_R_16F, n,
            &zero,
            dTmp,   CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

        // CUDA_CHECK(cudaDeviceSynchronize());
        // printMatrixHalf(dT3_half, n);
        // printMatrixHalf(dT4_half, n);
        // printMatrixFloat(dTmp, n);

        // A_our <-- Tmp
        CHECK_CUDA( cudaMemcpy(dA_our, dTmp, nn*sizeof(float), cudaMemcpyDeviceToDevice) );

        // force symmetry: A_our <-- 0.5 * (A_our + Tmp')
        CHECK_CUBLAS( cublasSgeam(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_T,
            n, n,
            &half, dA_our, n,
            &half, dTmp, n,
            dA_our, n) );

        // Compute Frobenius norm ||A_our||_F
        // float fro_err = 0.0f;
        // CHECK_CUBLAS( cublasSnrm2(cublasH, nn, dA_our, 1, &fro_err) );

        // printf("Iter: %d | Fro norm = %.10f \n", iter, fro_err);
    }

    // final combine: A_our = A_orig * (A_our + I) / 2
    CHECK_CUBLAS( cublasSgeam(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n,
        &one, dA_our, n,
        &one, dI,     n,
        dF,            n) );

    // float2half_kernel<<<blocks,threads>>>(dA_orig, dT3_half, nn);
    // float2half_kernel<<<blocks,threads>>>(dF, dT4_half, nn);
    convertFloatToHalf4(dA_orig, dT3_half, nn);
    convertFloatToHalf4(dF, dT4_half, nn);
    CHECK_CUBLAS( cublasGemmEx(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &one,
        dT3_half, CUDA_R_16F, n,
        dT4_half, CUDA_R_16F, n,
        &zero,
        dTmp,    CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    CHECK_CUDA( cudaMemcpy(dA_our, dTmp, nn*sizeof(float), cudaMemcpyDeviceToDevice) );
    CHECK_CUBLAS( cublasSscal(cublasH, nn, &half, dA_our, 1) );
    CHECK_CUDA( cudaDeviceSynchronize() );

    // copy the result back to mat by converting the float matrix to double using the host
    std::vector<float> A_our_h_f(nn);
    std::vector<double> A_our_h_d(nn);
    CHECK_CUDA( cudaMemcpy(A_our_h_f.data(), dA_our, nn * sizeof(float), D2H) );
    for (size_t i = 0; i < nn; i++) {
        A_our_h_d[i] = static_cast<double>(A_our_h_f[i]);
    }
    CHECK_CUDA( cudaMemcpy(mat.vals + mat_offset, A_our_h_d.data(), nn * sizeof(double), H2D) );

    CHECK_CUDA( cudaDeviceSynchronize() );

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<float> elapsed = end - start;
    // std::cout << "PSD projection total time: " << elapsed.count() << " seconds\n";

    // cleanup
    cudaFree(dA_orig);
    cudaFree(dA_our);
    cudaFree(dTmp);
    cudaFree(dI);
    cudaFree(dT1);
    cudaFree(dT2);
    cudaFree(dF);
    cudaFree(dDiff);
}