#pragma once
#include "tensor.h"

namespace laylow {

// Scalar (baseline) matrix multiply - correct but slow
// C = A * B
// A: [M x K], B: [K x N], C: [M x N]
void matmul_scalar(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
);

// AVX2 SIMD matrix multiply - fast version we'll verify against scalar
void matmul_avx2(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
);

// High-level tensor wrapper - picks the best available implementation
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

} // namespace laylow