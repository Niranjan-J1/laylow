#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>

namespace laylow {

enum class DType {
    F32,   // 32-bit float  - used for computation
    F16,   // 16-bit float  - used in some model weights
    Q8_0,  // 8-bit quantized
    Q4_0,  // 4-bit quantized (block size 32, 18 bytes/block)
    Q6_K,  // 6-bit quantized, super-block (block size 256, 210 bytes/block)
};

struct Tensor {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    void* data   = nullptr;
    size_t nbytes = 0;

    // Total number of elements across all dimensions
    int64_t numel() const;

    // Size of a specific dimension
    int64_t dim(int i) const;

    // True if this tensor stores quantized (non-float) data
    bool is_quantized() const;

    // Allocate aligned memory for a new tensor
    static Tensor empty(const std::string& name, DType dtype,
                        std::vector<int64_t> shape);

    // Release the memory
    void free_data();
};

// Dequantize a Q4_0 tensor into a new F32 tensor
Tensor dequantize_q4(const Tensor& src);

// Dequantize a Q8_0 tensor into a new F32 tensor
Tensor dequantize_q8(const Tensor& src);

// Dequantize a Q6_K tensor into a new F32 tensor
Tensor dequantize_q6k(const Tensor& src);

} // namespace laylow