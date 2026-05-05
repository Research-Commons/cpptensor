#include "cpptensor/ops/arithmetic/sub.hpp"
#include "cpptensor/tensor/tensor.hpp"

#include <stdexcept>

#include "cpptensor/backend/cpu_backend.h"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/ops/helperOps.hpp"

namespace cpptensor {

    Tensor operator-(const Tensor& a, const Tensor& b) {
        if (a.device_type() != b.device_type()) {
            throw std::runtime_error("Device mismatch in sub");
        }
        std::vector<size_t> out_shape = computeBroadcastShape(a.shape(), b.shape());
        Tensor out(out_shape, 0.0f, a.device_type());
        const Tensor lhs = materialize_for_backend_input(a);
        const Tensor rhs = materialize_for_backend_input(b);

        if (a.device_type() == DeviceType::CPU && needsBroadcast(a.shape(), b.shape())) {
            CPU::subKernel(lhs, rhs, out);
        } else {
            KernelRegistry::instance()
                .getKernel(OpType::Sub, a.device_type())(lhs, rhs, out);
        }
        return out;
    }

    Tensor operator-(const Tensor& lhs, float scalar) {
        return lhs - Tensor::full(lhs.shape(), scalar, lhs.device_type());
    }

    Tensor operator-(float scalar, const Tensor& rhs) {
        return Tensor::full(rhs.shape(), scalar, rhs.device_type()) - rhs;
    }

}
