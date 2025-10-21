#include "ops/cos.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"

namespace cpptensor {

    Tensor cos(const Tensor& a) {
        Tensor out = Tensor::full(a.shape(), 0.f , a.requires_grad());

        KernelRegistry::instance().getUnaryKernel(OpType::Cos, a.device_type())(a, out);

        return out;
    }

}