#include "ops/tan.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"

namespace cpptensor {

    Tensor tan(const Tensor& a) {
        Tensor out = Tensor::full(a.shape(), 0.f , a.requires_grad());

        KernelRegistry::instance().getUnaryKernel(OpType::Tan, a.device_type())(a, out);

        return out;
    }

}