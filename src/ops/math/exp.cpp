#include "ops/math/exp.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"

namespace cpptensor {

    Tensor exp(const Tensor& a) {
        Tensor out = Tensor::full(a.shape(), 0.f , a.device_type());

        KernelRegistry::instance().getUnaryKernel(OpType::Exp, a.device_type())(a, out);

        return out;
    }

}