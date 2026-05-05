#include "../../../include/cpptensor/ops/arithmetic/mul.hpp"
#include "cpptensor/tensor/tensor.hpp"

#include <stdexcept>

#include "cpptensor/backend/cpu_backend.h"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/ops/helperOps.hpp"

namespace cpptensor {

    Tensor operator*(const Tensor& a, const Tensor& b) {
        if (a.device_type() != b.device_type()) {
            throw std::runtime_error("Device mismatch in mul");
        }
        std::vector<size_t> out_shape = computeBroadcastShape(a.shape(), b.shape());
        Tensor out(out_shape, 0.0f, a.device_type());

        if (a.device_type() == DeviceType::CPU && needsBroadcast(a.shape(), b.shape())) {
            CPU::mulKernel(a, b, out);
        } else {
            KernelRegistry::instance()
                .getKernel(OpType::Mul, a.device_type())(a, b, out);
        }
        return out;
    }

    Tensor operator*(const Tensor& lhs, float scalar) {
        return lhs * Tensor::full(lhs.shape(), scalar, lhs.device_type());
    }

    Tensor operator*(float scalar, const Tensor& rhs) {
        return rhs * Tensor::full(rhs.shape(), scalar, rhs.device_type());
    }
}
