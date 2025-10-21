#include "cpptensor/ops/arithmetic/add.hpp"
#include "cpptensor/tensor/tensor.hpp"

#include <stdexcept>

#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/ops/helperOps.hpp"


namespace cpptensor {

    Tensor operator+(const Tensor& a, const Tensor& b) {
        if (a.device_type() != b.device_type()) {
            throw std::runtime_error("Device mismatch in add");
        }
        // Compute broadcasted shape and create output tensor
        std::vector<size_t> out_shape = computeBroadcastShape(a.shape(), b.shape());
        Tensor out(out_shape, 0.0f, a.device_type());

        // Lookup and call the registered kernel
        KernelRegistry::instance()
            .getKernel(OpType::Add, a.device_type())(a, b, out);
        return out;
    }


    Tensor operator+(const Tensor& lhs, float scalar) {
        return lhs + Tensor::full(lhs.shape(), scalar, lhs.device_type());
    }
    Tensor operator+(float scalar, const Tensor& rhs) {
        return rhs + Tensor::full(rhs.shape(), scalar, rhs.device_type());
    }
}
