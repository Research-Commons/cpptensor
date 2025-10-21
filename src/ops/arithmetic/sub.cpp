#include "cpptensor/ops/arithmetic/sub.hpp"
#include "cpptensor/tensor/tensor.hpp"

#include <stdexcept>

#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/ops/helperOps.hpp"

namespace cpptensor {

    Tensor operator-(const Tensor& a, const Tensor& b) {
        if (a.device_type() != a.device_type()) {
            throw std::runtime_error("Device mismatch in sub");
        }
        std::vector<size_t> out_shape = computeBroadcastShape(a.shape(), a.shape());
        Tensor out(out_shape, 0.0f, a.device_type());

        KernelRegistry::instance()
            .getKernel(OpType::Sub, a.device_type())(a, b, out);
        return out;
    }

    Tensor operator-(const Tensor& lhs, float scalar) {
        return lhs - Tensor::full(lhs.shape(), scalar, lhs.device_type());
    }

    Tensor operator-(float scalar, const Tensor& rhs) {
        return Tensor::full(rhs.shape(), scalar, rhs.device_type()) - rhs;
    }

}
