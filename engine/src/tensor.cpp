#include "tensor.h"
#include <stdexcept>
#include <numeric>

namespace laylow {

int64_t Tensor::numel() const {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
}

int64_t Tensor::dim(int i) const {
    if (i < 0 || i >= (int)shape.size())
        throw std::out_of_range("Tensor::dim index out of range");
    return shape[i];
}

bool Tensor::is_quantized() const {
    return dtype == DType::Q4_0 || dtype == DType::Q8_0;
}

static size_t dtype_nbytes(DType dtype, int64_t numel) {
    switch (dtype) {
        case DType::F32:  return numel * 4;
        case DType::F16:  return numel * 2;
        case DType::Q8_0: return numel;
        case DType::Q4_0: return (numel / 2) + (numel / 32) * 4;
        default: throw std::runtime_error("Unknown dtype");
    }
}

Tensor Tensor::empty(const std::string& name, DType dtype, std::vector<int64_t> shape) {
    Tensor t;
    t.name   = name;
    t.dtype  = dtype;
    t.shape  = shape;
    t.nbytes = dtype_nbytes(dtype, t.numel());
    t.data   = _aligned_malloc(t.nbytes, 32);
    if (!t.data)
        throw std::runtime_error("Failed to allocate tensor: " + name);
    return t;
}

void Tensor::free_data() {
    if (data) {
        _aligned_free(data);
        data = nullptr;
    }
}

} // namespace laylow