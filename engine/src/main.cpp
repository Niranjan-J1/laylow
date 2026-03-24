#include <iostream>
#include <cmath>
#include <cstring>
#include "tensor.h"
#include "matmul.h"

// Fill a float buffer with a simple pattern we can reason about
static void fill_pattern(float* data, int n, float start, float step) {
    for (int i = 0; i < n; i++)
        data[i] = start + i * step;
}

// Check scalar and AVX2 outputs match within floating point tolerance
static bool outputs_match(const float* ref, const float* fast, int n) {
    for (int i = 0; i < n; i++) {
        if (std::fabs(ref[i] - fast[i]) > 1e-1f) {
            std::cout << "MISMATCH at [" << i << "]: scalar="
                      << ref[i] << " avx2=" << fast[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "laylow v0.1.0 - local LLM inference engine" << std::endl;
    std::cout << "AVX2 SIMD enabled" << std::endl;
    std::cout << std::endl;

    // Matrix dimensions: A[32 x 64] * B[64 x 32] = C[32 x 32]
    // These are small but representative of transformer weight shapes
    int M = 32, K = 64, N = 32;

    auto A      = laylow::Tensor::empty("A",           laylow::DType::F32, {M, K});
    auto B      = laylow::Tensor::empty("B",           laylow::DType::F32, {K, N});
    auto C_ref  = laylow::Tensor::empty("C_scalar",    laylow::DType::F32, {M, N});
    auto C_fast = laylow::Tensor::empty("C_avx2",      laylow::DType::F32, {M, N});

    // Fill A and B with known values
    fill_pattern(static_cast<float*>(A.data), M * K, 0.0f, 0.01f);
    fill_pattern(static_cast<float*>(B.data), K * N, 0.0f, 0.01f);

    // Run both implementations
    laylow::matmul_scalar(
        static_cast<float*>(A.data),
        static_cast<float*>(B.data),
        static_cast<float*>(C_ref.data),
        M, K, N
    );

    laylow::matmul_avx2(
        static_cast<float*>(A.data),
        static_cast<float*>(B.data),
        static_cast<float*>(C_fast.data),
        M, K, N
    );

    // Verify they agree
    bool ok = outputs_match(
        static_cast<float*>(C_ref.data),
        static_cast<float*>(C_fast.data),
        M * N
    );

    if (ok) {
        std::cout << "matmul [" << M << "x" << K << "] * ["
                  << K << "x" << N << "] = [" << M << "x" << N << "]" << std::endl;
        std::cout << "scalar vs AVX2: MATCH - SIMD kernel is correct" << std::endl;
    } else {
        std::cout << "FAIL: scalar and AVX2 outputs differ" << std::endl;
    }

    A.free_data(); B.free_data(); C_ref.free_data(); C_fast.free_data();
    return ok ? 0 : 1;
}