#include "ops/sin.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"

namespace cpptensor {

    Tensor sin(const Tensor& a) {
        Tensor out = Tensor::full(a.shape(), 0.f , a.requires_grad());

        KernelRegistry::instance().getUnaryKernel(OpType::Sin, a.device_type())(a, out);

        return out;
    }

}