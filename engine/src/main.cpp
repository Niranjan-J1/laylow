#include <iostream>
#include "tensor.h"

int main() {
    std::cout << "laylow v0.1.0 - local LLM inference engine" << std::endl;
    std::cout << "AVX2 SIMD enabled" << std::endl;

    // Allocate a [4 x 64] float32 tensor - same shape as a tiny weight matrix
    auto t = laylow::Tensor::empty("test_weight", laylow::DType::F32, {4, 64});

    std::cout << "Tensor '" << t.name << "' allocated: "
              << t.dim(0) << "x" << t.dim(1)
              << " (" << t.nbytes << " bytes, "
              << t.numel() << " elements)" << std::endl;

    t.free_data();
    std::cout << "Tensor freed OK" << std::endl;

    return 0;
}