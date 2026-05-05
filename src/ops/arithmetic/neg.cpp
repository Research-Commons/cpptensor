#include "ops/arithmetic/neg.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"


namespace cpptensor {

    Tensor operator-(const Tensor& a) {
        Tensor out = Tensor::full(a.shape(), 0.0f, a.device_type());
        KernelRegistry::instance().getUnaryKernel(OpType::Neg, a.device_type())(a, out);
        return out;
    }

}
