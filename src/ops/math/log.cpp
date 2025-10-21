#include "ops/math/log.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"

namespace cpptensor {

    Tensor log(const Tensor& a) {
        Tensor out = Tensor::full(a.shape(), 0.f , a.device_type());

        KernelRegistry::instance().getUnaryKernel(OpType::Log, a.device_type())(a, out);

        return out;
    }

}