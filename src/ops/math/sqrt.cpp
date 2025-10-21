#include "ops/math/sqrt.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"

namespace cpptensor {

    Tensor sqrt(const Tensor& a) {
        Tensor out = Tensor::full(a.shape(), 0.f , a.device_type());

        KernelRegistry::instance().getUnaryKernel(OpType::Sqrt, a.device_type())(a, out);

        return out;
    }

}