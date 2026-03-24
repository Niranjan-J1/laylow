#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>

namespace laylow {

enum class DType {
    F32,   // 32-bit float  - used for computation
    F16,   // 16-bit float  - used in some model weights
    Q8_0,  // 8-bit quantized - smaller, slightly slower
    Q4_0,  // 4-bit quantized - smallest, what we'll use for Phi-3
};

struct Tensor {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    void* data;
    size_t nbytes;

    // Total number of elements across all dimensions
    int64_t numel() const;

    // Size of a specific dimension
    int64_t dim(int i) const;

    // True if this tensor uses Q4 or Q8 compression
    bool is_quantized() const;

    // Allocate aligned memory for a new tensor
    static Tensor empty(const std::string& name, DType dtype, std::vector<int64_t> shape);

    // Release the memory
    void free_data();
};

} // namespace laylow