#include "matmul.h"
#include <stdexcept>
#include <immintrin.h>  // AVX2 intrinsics

namespace laylow {

// ---------------------------------------------------------------------
// Scalar baseline
// Three nested loops - textbook matrix multiply.
// O(M*K*N) - slow, but trivially correct.
// ---------------------------------------------------------------------
void matmul_scalar(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ---------------------------------------------------------------------
// AVX2 SIMD kernel
// Key idea: instead of multiplying one float at a time, we load 8 floats
// at once into a 256-bit register and multiply all 8 simultaneously.
// That's an 8x theoretical speedup on the inner loop.
//
// _mm256_loadu_ps  - load 8 floats from memory into a ymm register
// _mm256_fmadd_ps  - fused multiply-add: acc = a*b + acc (one instruction)
// _mm256_storeu_ps - write 8 floats from a ymm register back to memory
// ---------------------------------------------------------------------
void matmul_avx2(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 8) {
            // Accumulator: holds 8 partial sums simultaneously
            __m256 acc = _mm256_setzero_ps();

            for (int k = 0; k < K; k++) {
                // Broadcast single value A[i][k] into all 8 lanes
                __m256 a = _mm256_set1_ps(A[i * K + k]);

                // Load 8 consecutive values from row k of B
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);

                // acc += a * b  (8 multiplies + 8 adds in one instruction)
                acc = _mm256_fmadd_ps(a, b, acc);
            }

            // Write all 8 results back to C
            _mm256_storeu_ps(&C[i * N + j], acc);
        }
    }
}

// ---------------------------------------------------------------------
// Tensor-level wrapper
// Validates shapes, picks the right implementation, runs it.
// ---------------------------------------------------------------------
void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.dtype != DType::F32 || B.dtype != DType::F32 || C.dtype != DType::F32)
        throw std::runtime_error("matmul: only F32 tensors supported");

    if (A.shape.size() != 2 || B.shape.size() != 2 || C.shape.size() != 2)
        throw std::runtime_error("matmul: tensors must be 2D");

    int M = A.dim(0), K = A.dim(1);
    int K2 = B.dim(0), N = B.dim(1);

    if (K != K2)
        throw std::runtime_error("matmul: incompatible shapes");

    if (C.dim(0) != M || C.dim(1) != N)
        throw std::runtime_error("matmul: output tensor wrong shape");

    // Use AVX2 if N is divisible by 8, otherwise fall back to scalar
    if (N % 8 == 0) {
        matmul_avx2(
            static_cast<const float*>(A.data),
            static_cast<const float*>(B.data),
            static_cast<float*>(C.data),
            M, K, N
        );
    } else {
        matmul_scalar(
            static_cast<const float*>(A.data),
            static_cast<const float*>(B.data),
            static_cast<float*>(C.data),
            M, K, N
        );
    }
}

} // namespace laylow