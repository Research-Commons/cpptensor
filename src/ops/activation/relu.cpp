#include "ops/activation/relu.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"

namespace cpptensor {

    Tensor relu(const Tensor& a) {
        Tensor out = Tensor::full(a.shape(), 0.f , a.device_type());

        KernelRegistry::instance().getUnaryKernel(OpType::Relu, a.device_type())(a, out);

        return out;
    }

}