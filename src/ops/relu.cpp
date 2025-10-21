#include "ops/relu.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"

namespace cpptensor {

    Tensor relu(const Tensor& a) {
        Tensor out = Tensor::full(a.shape(), 0.f , a.requires_grad());

        KernelRegistry::instance().getUnaryKernel(OpType::Relu, a.device_type())(a, out);

        return out;
    }

}